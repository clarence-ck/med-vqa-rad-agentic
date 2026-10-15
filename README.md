# MedVisionQuery — an agentic multimodal system for medical visual question answering

MedVisionQuery answers plain-language questions about radiology images, including X-rays, CT scans, and MRI scans.
It uses a **Haystack Agent controller** to decide how to inspect an image, answer a question, check the answer against
the image, and choose whether to inspect a region again, revise the answer, abstain, or finish. We evaluated it on
300 VQA-RAD questions and compared it with a single-prompt baseline, a hand-designed workflow, and a fixed pipeline.

> ⚠️ **For education and research only. Not for clinical use.**

Built for SUTD 51.511 Multimodal Generative AI (Template 1: Agentic Multimodal System). Report:
[`docs/report/MGAI_Final_Report_Group9_revised.pdf`](docs/report/MGAI_Final_Report_Group9_revised.pdf)
(LaTeX source in `docs/report/`; build with `tectonic main.tex`).

---

## Contents

- [Results at a glance](#results-at-a-glance)
- [Problem and task decomposition](#problem-and-task-decomposition)
- [Agent architecture](#agent-architecture)
- [Multimodal grounding](#multimodal-grounding)
- [Control and failure handling](#control-and-failure-handling)
- [Evaluation design](#evaluation-design)
- [Results](#results)
- [Failure analysis](#failure-analysis)
- [Reflection and limitations](#reflection-and-limitations)
- [User interface](#user-interface)
- [Quick start](#quick-start)
- [Reproducing the evaluation](#reproducing-the-evaluation)
- [Project structure](#project-structure)
- [API reference](#api-reference)

---

## Results at a glance

<!-- generated:glance -->
Run `2026-10-15_1641` used a seeded, stratified sample of 300 VQA-RAD questions: 160 close-ended and 140 open-ended.
All systems used the same answering model, `gpt-6-luna`. The agentic loop and fixed pipeline also used `gpt-6.1-sol`
as a verifier. Four cases ran at the same time. The 63 cases that required another image inspection took the longest
and raised the mean time. See [Run timings](#run-timings) for the full timing data. “Strict” uses rule-based scoring.
“Judged” uses `gpt-6-astra` as a blind judge for open-ended answers. See [Evaluation design](#evaluation-design).

| System | Strict accuracy | Judged accuracy | Time / case (mean · median) | Cost / case |
|---|---|---|---|---|
| Single-shot baseline | 64.0% | 70.0% | 4.3 s · 3.6 s | $0.0002 |
| Workflow baseline (hand-designed) | 28.0% | 29.3% | 9.3 s · 8.8 s | $0.0005 |
| **Agentic controller loop** | 59.7% | 64.7% | 48.1 s · 34.2 s | $0.0132 |
| Fixed pipeline (ablation) | 58.0% | 62.0% | 34.5 s · 27.9 s | $0.0068 |

- **Agentic vs single-shot:** single-shot performed better. The difference was significant only for the judged score
  (-4.3 percentage points strict, p = 0.079; -5.3 judged, p = 0.033; exact McNemar test).
- **Agentic vs fixed pipeline:** the agentic loop was 1.7 points better on strict accuracy (p = 0.58) and 2.7 points
  better on judged accuracy (p = 0.35).
- **Agentic vs its verifier alone:** the agentic loop scored 59.7% strict accuracy. The verifier model's single blind
  reading scored 67.0% (p = 0.008).
- **Confidence and stability:** answers marked as verified were correct 71% of the time. Other non-abstained answers
  were correct 40% of the time. When the same 100 cases were run a second time, 12% of agentic answers changed from
  right to wrong or wrong to right, compared with 6% for the single-shot baseline.
<!-- /generated:glance -->

![Evaluation dashboard](eval-dashboard-screenshots/01-eval-kpi-cards.png)

---

## Problem and task decomposition

**Input:** a radiology image in JPEG or PNG format and a plain-text question. **Output:** an answer, an explanation
based on visible evidence, a verification status, uncertainty signals, and a record of every decision.

A single model call combines several tasks that can fail for different reasons. This system separates those tasks:

| Sub-task | Question it answers | Component |
|---|---|---|
| Question understanding | What kind of answer is needed, such as yes/no, location, or image type? | `analyze_question` |
| Perception | What does the image show that is relevant to the question? | `look(focus)` |
| Synthesis | What is the best answer based on the image and observations? | `answer(guidance)` |
| Verification | Does the image support the answer, and what should happen next? | `check` |
| Planning / control | Should the system inspect a region again, revise, abstain, or finish? | Controller (Haystack Agent) |

---

## Agent architecture

![System diagram](docs/report/system_diagram.png)

### Controller loop

The system creates a new Haystack `Agent` for each request in `backend/agents/agentic_loop.py`. Each request has its
own working memory, which stores the image, question analysis, image inspections, answers, and checks. The controller
sees only short text results from the tools and decides which tool to call next. There is one agent: the controller.
The five tools are functions, not separate agents. The `check` tool recommends a next step, but the controller makes
the final decision. In the fixed-pipeline comparison, all steps run in a fixed order.

| Tool | Role | Model (default) | Notes |
|---|---|---|---|
| `analyze_question()` | Understand the task | gpt-6-luna | Finds the question type and expected answer format. Keyword rules match whole words. |
| `look(focus)` | Inspect the image | gpt-6-luna, vision | Examines the image with the question in mind. When given a `focus`, it finds, crops, enlarges, and inspects that region beside the full image. If it cannot find the region, it inspects the full image again. The first look also rejects images that are not radiology images. |
| `answer(guidance)` | Create an answer | gpt-6-luna, vision | Receives the image and all observations. `guidance` explains what the verifier asked it to fix. |
| `check()` | Check the answer | gpt-6.1-sol, vision | First reads the image without seeing the proposed answer, then reviews the answer and recommends the next step. |
| `finish(abstain)` | End the process | — | Ends the case. A normal finish is blocked if the latest answer has not been checked, but abstaining is still allowed. |

**How `check` works.** It has two parts, in this order:

1. **Blind reading:** once per case, the verifier answers using only the image and question. It does not see the
   proposed answer or the image-inspection notes.
2. **Review:** the verifier checks whether the answer matches the image, contradicts visible evidence, or uses the
   wrong format. It returns a status, a list of problems, and one recommended action: `accept`, `relook` with a focus,
   `reanswer` with guidance, or `abstain`. The guidance may include a suggested corrected answer.

An answer is marked as *verified* only when all three conditions are met: the verifier returns `verified`, no format
or question-type check fails, and the blind reading agrees with the proposed answer. Word-based rules are checked
first, with exact matching for yes/no answers. For open-ended answers, the verifier decides whether the meanings are
the same. The controller learns whether the answers agree, but it never sees the blind answer. However, the review
guidance may still include a corrected answer.

The limits are 2 focused re-inspections, 3 answers, and 16 agent steps. If the controller stops making progress, a
safety rule creates and/or checks an answer before finishing. The current run used these paths:

<!-- generated:paths -->
| Path | Cases | Tool sequence |
|---|---|---|
| direct | 187 | analyze → look → answer → check → finish |
| relook | 63 | … → check → look(focus) → answer(guided) → check → finish |
| reanswer | 25 | … → check → answer(guided) → check → finish |
| abstained | 25 | … → check → finish(abstain) |
<!-- /generated:paths -->

### Why these models and tools

- **Haystack `Agent` + `Tool`:** provides a tool-calling loop with defined input types and stopping rules. The image
  stays in request memory instead of the controller's context, which lowers controller cost.
- **One OpenAI model family with different models for different jobs:** controller decisions and question analysis
  are simple, so they use `gpt-6-luna` with low or no reasoning effort. Answering uses `gpt-6-luna` with medium effort.
  The answerer always sees the image because a text description can lose important visual details.
- **A stronger model verifies and a third model judges:** `check` uses `gpt-6.1-sol`, which costs 20 times more per
  token than `gpt-6-luna`. This prevents the answering model from checking itself. Because the verifier can suggest
  a corrected answer, the evaluation also measures the verifier's blind answer by itself. The evaluation judge,
  `gpt-6-astra`, is not used by any system being tested.
- All roles can be changed through `MODEL_<ROLE>` and `REASONING_EFFORT_<ROLE>` in `.env`. The system records token
  use and cost for every model call.

### Baselines and ablation

| System | File | What it does |
|---|---|---|
| Single-shot baseline | `backend/agents/baseline.py` | Uses one vision prompt. It gives a short answer and starts with yes or no when needed. |
| Workflow baseline | `backend/agents/workflow_baseline.py` | Uses a fixed chain: a general image description, question analysis, and hand-written rules for each question type. |
| Fixed pipeline (ablation) | `backend/agents/orchestrator.py` | Calls six tools in a fixed order: `interpret_image` → `analyze_question` → `filter_image_context` → `route_difficulty` → `reason_answer` → `verify_answer`. It repeats the sequence once if verification finds a serious or fixable problem. It uses the same models, but has a general image description, no zoom, no blind check, and a text-only verifier. The comparison therefore measures the full agentic design, not only controller freedom. |

---

## Multimodal grounding

The system uses two types of input: the image and the text question. They work together at every important step:

| Step | Image | Text |
|---|---|---|
| `analyze_question()` | ✗ | The question and any observations made so far |
| `look(focus)` | ✓ Full image at high detail, plus an enlarged crop when a focus is given | The question and focus tell the model what to inspect |
| `answer(guidance)` | ✓ Sent directly | Question analysis, all observations, and verifier guidance |
| `check()` | ✓ Sent directly; the blind reading sees only the image and question | The review also sees the answer, explanation, and observations |
| Controller | ✗ | Short tool results only |

Every model that inspects, answers, or verifies sees the image directly. This is important because a text description
can leave out visual details. Only the question analyzer and controller use text alone. The LLM evaluation judge
compares text answers with the reference answer and does not see the image.

---

## Control and failure handling

| Mechanism | What it does |
|---|---|
| Image-grounded verification | `check` makes one independent blind reading, checks the answer against the image, and compares both answers. It returns five checks, a status (`verified`, `partially_verified`, `low_confidence`, or `unsupported`), and a recommended next action. |
| Targeted recovery | The controller can zoom into one region with `look(focus)` or revise with `answer(guidance)` instead of repeating the full process. |
| Best-verified answer | A later answer replaces an earlier answer only if its check result is clearly better. This helps because repeated inspections can create false details. |
| Self-consistency | After finishing, the answerer answers the question independently two more times. Low agreement adds `self_inconsistent` and marks the case for review. |
| Abstention | The controller may choose not to answer. Abstentions are recorded as abstentions and are never changed to “no.” |
| Guardrail | Records failure categories (`unsupported_claim`, `low_confidence`, `visual_contradiction`, `partially_verified`, `question_answer_mismatch`, `reasoning_generation_unstable`, `parser_failure`, `self_inconsistent`), adds a disclaimer, and marks cases that need review. |
| Input rejection | The first inspection rejects non-radiology images. If normal image reading fails, the system tries the original bytes. If the first inspection still fails, it stops instead of guessing. |

---

## Evaluation design

Each evaluation is stored as a separate **run** in `experiments/runs/<run_id>/`. A run contains cases, judge results,
repeat runs, metrics, timings, and costs. A new run never changes an old one. `experiments/runs/README.md` lists all
runs. Every number in this README and the report comes from the latest completed run. The scripts
`scripts/final_metrics.py` and `scripts/update_docs.py` create the text between generated markers and update the report
numbers. The failure-pattern and reflection counts come from the same run files. Thresholds and limits come from code.

- **Data:** VQA-RAD contains 2,244 questions, each with a reference answer. We removed one known bad item and the 300
  test questions used in an earlier evaluation. From the remaining 1,943 questions, we selected 300 at random with
  seed 51511 while keeping the same mix of close-ended and open-ended questions: 160 close-ended and 140 open-ended.
  No system was trained, so the dataset's train/test labels do not affect the systems. Because all excluded questions
  were from the test split, 277 selected questions have a train label. The list is in `experiments/eval_set.json`.
  It was created with `python scripts/make_eval_set.py --size 300 --split all --seed 51511 --exclude-run 2026-10-15_1308`.
  Results for question types with few examples are less reliable.
- **Systems:** the single-shot baseline, workflow baseline, agentic loop, and fixed-pipeline ablation all answer the
  same questions with the same answering model.
- **Strict scorer** (`backend/utils/scoring.py`):
  - yes/no reference: the answer must start with the correct word;
  - paired terms such as normal/abnormal or present/absent: the answer must contain the correct term and not its opposite;
  - counting: the numbers must match;
  - open-ended: every important word from the reference must appear, or at least 60% for references longer than three
    words. The scorer handles aliases such as CT and computed tomography and treats singular and plural forms as the
    same. Left/right conflicts are wrong. An answer that starts with “no” is wrong when the reference does not.
    S-PubMedBert-MS-MARCO similarity is used only as a fallback, with thresholds of 0.95 for short answers and 0.90
    for long answers.
- **LLM judge** (`scripts/judge_open_ended.py`): `gpt-6-astra` judges every open-ended answer from every system,
  including answers marked right or wrong by the strict scorer. It checks whether each answer has the same meaning as
  the reference. System names are hidden, and answers are shuffled. Its result becomes the “judged” score for
  open-ended answers. Close-ended answers keep the rule-based score. Astra does not answer questions or verify answers
  in any tested system. In a 30-result review, it agreed with 29 credit decisions. In case 22, it was too strict with
  an agentic-loop answer. See `judge_review_sample.md` in the run folder. This was an AI-assisted review, not human
  annotation.
- **Statistics:** at n = 300, the 95% intervals are about ±5.6 percentage points and never wider than ±5.7 points.
  Paired comparisons use the exact McNemar test.
- **Variance:** 100 questions from the main evaluation are selected with seed 42. The original answers are treated as
  the first pass, then the single-shot baseline and agentic loop answer the same questions again in an independent
  repeat. Both passes belong to the same evaluation run folder.
- **Code changes after the run:** the run used commit `2b1d54f`. The yes/no normalizer now checks negative phrases
  before positive words, because “no visible fracture” was previously read as “yes.” Also, `check` now makes the blind
  reading before the review, as described in the controller prompt. Neither change affects this run's results. Replaying
  all 4,519 recorded answers gave the same normalized output, and the two `check` calls are independent, so their order
  does not change either model's input.
<!-- generated:cost -->
- **Cost:** every OpenAI call is recorded by `backend/utils/usage.py`. The evaluation cost $6.20, repeats cost $1.31,
  and judging cost $1.22, for a total of $8.73.

### Run timings

These are the wall-clock times for each stage of run `2026-10-15_1641`, recorded by `scripts/run_all.py` in
`experiments/runs/2026-10-15_1641/run.json`:

| Stage | Started | Finished | Wall clock | Work | Cost |
|---|---|---|---|---|---|
| Main evaluation (4 systems) | 2026-10-15 16:41:12 | 17:48:42 | 67 min 29 s | 300 cases × 4 systems, 4 at a time | $6.20 |
| LLM judge (`gpt-6-astra`) | 2026-10-15 17:48:43 | 17:59:33 | 10 min 50 s | 140 open-ended cases | $1.22 |
| Variance repeats (baseline + agentic) | 2026-10-15 17:59:33 | 19:22:22 | 82 min 48 s | 100 cases, one at a time | $1.31 |

The per-case times in the results tables show each system's own response time. Because all four systems ran in
parallel, a full case took 52.9 seconds on average.
<!-- /generated:cost -->

---

## Results

<!-- generated:results -->
### Overall

| System | Strict | Judged | Close-ended (160) | Open-ended (140), strict → judged |
|---|---|---|---|---|
| Single-shot baseline | 64.0% | 70.0% | 81.9% | 43.6% → 56.4% |
| Workflow baseline | 28.0% | 29.3% | 44.4% | 9.3% → 12.1% |
| **Agentic loop** | 59.7% | 64.7% | 73.8% | 43.6% → 54.3% |
| Fixed pipeline (ablation) | 58.0% | 62.0% | 73.1% | 40.7% → 49.3% |

Paired comparisons count cases where one system was right and the other was wrong:

| Comparison | Strict | Judged |
|---|---|---|
| Agentic vs single-shot | 17 vs 30, p = 0.079 | 17 vs 33, p = 0.033 |
| Agentic vs fixed pipeline | 28 vs 23, p = 0.58 | 32 vs 24, p = 0.35 |

### By controller path

| Path | Cases | Agentic | Single-shot | Fixed pipeline |
|---|---|---|---|---|
| direct | 187 | 75% | 74% | 70% |
| relook | 63 | 38% | 41% | 33% |
| reanswer | 25 | 56% | 64% | 52% |
| abstained | 25 | — (abstained) | 44% | 40% |

### Uncertainty signals predict errors

| Signal | Levels | Agentic accuracy |
|---|---|---|
| Verification (answered questions) | verified (220) / not verified (55) | 71% / 40% |
| Self-consistency | high (242) / medium or low (33) | 69% / 39–40% |
| Abstention | 25 cases; the single-shot baseline was wrong on 56% of them | — |

### Stability across runs (same 100 cases)

This table uses strict scoring. “Original evaluation” is the first answer recorded during the 300-case evaluation.
“Independent repeat” is a new answer for the same case during the variance stage of the same run.

| System | Original evaluation | Independent repeat | Answers that flipped right ↔ wrong |
|---|---|---|---|
| Single-shot baseline | 60/100 | 60/100 | 6 (6%) |
| Agentic loop | 59/100 | 55/100 | 12 (12%) |

### What the loop's mechanisms did

| Mechanism | Result in this run |
|---|---|
| Revision (re-look or re-answer) | Used on 88 questions. It fixed 11 wrong first answers and broke 6 right first answers under strict scoring. |
| Blind check | Rejected 82 first answers, and 59 of those answers (72%) were wrong. When word rules could not decide agreement, the verifier compared meanings for 66 answers. |
| Zoom | Used in 63 cases; 5 later abstained. On the remaining 58, agentic accuracy was 36.2% and single-shot accuracy was 39.7%. |
| Abstention | Used on 25 questions. The loop's last answer was right in 7 cases, and the single-shot answer was right in 11. |
| Answered questions only | On the same 275 answered questions, agentic accuracy was 65.1% and single-shot accuracy was 65.8%. |

The blind reading is the verifier model, `gpt-6.1-sol`, answering directly from the image. By itself, it reached
67.0% strict accuracy. Agentic vs blind reading was 21 vs 43, p = 0.008. Blind reading vs single-shot was 29 vs 20,
p = 0.25.
<!-- /generated:results -->

---

## Failure analysis

The dashboard summarizes the four analyses required by Template 1 for every run.

<!-- generated:failure -->
- **Self-consistency checks:** the answerer gives two independent extra answers per case. Low agreement marks the case
  for review. Low consistency means uncertainty, not certain failure: 39% of low-consistency answers were still right.
- **Cross-modal agreement:** the verifier sees the image. It checks whether the explanation is supported by visible
  evidence and whether the answer contradicts the image, rather than checking only a text description.
- **Verifier model:** a different, stronger model, `gpt-6.1-sol`, checks answers from `gpt-6-luna`. After checking,
  113 of the 300 cases left the direct path.
- **Disagreement analysis:** under the same scoring rules, the three systems disagreed on 69% of cases. When only one
  system was right, the agentic system won 15 cases, single-shot won 20, and workflow won 12.

The following examples were selected automatically by `scripts/final_metrics.py`. Screenshots are in
`eval-dashboard-screenshots/`.

| Case | What happened |
|---|---|
| 503 “is there a pleural effusion present?” | A focused second look answered correctly, while single-shot did not. Agentic: “no”; single-shot: “Yes, there is a small left pleural effu…”; reference: “no”. |
| 457 “what brain region demonstrates edema?” | A verifier-guided revision answered correctly, while single-shot did not. Agentic: “Bilateral basal ganglia”; single-shot: “Both thalami demonstrate edema.”; reference: “basal ganglia”. |
| 8 “where is the mass?” | An intermediate answer was right, but a later revision replaced it with a wrong answer that passed verification. Single-shot was right. Agentic: “Around the posterior third ventricle, e…”; single-shot: “The mass is in the midline pineal regio…”; reference: “pineal region”. |
| 19 “what plane is seen?” | Low self-consistency marked the case for review, although the final answer was right. Agentic: “Frontal (PA) view”; single-shot: “Frontal (coronal) plane.”; reference: “pa”. |
<!-- /generated:failure -->

**Systematic failure patterns**

1. **Shared perception errors:** 65 questions (22%) were answered incorrectly by the single-shot model, agentic loop,
   and fixed pipeline under judged scoring. Of these, 49 were open-ended and usually asked for a location, finding, or
   diagnosis. The check cannot find an error when the image inspector and verifier make the same mistake. The
   verifier's blind reading was correct in only 16 of the 65 cases, and those correct readings still did not become the
   final answer. Self-consistency was not useful here: 51 of the 59 answered cases had high consistency. In other
   words, the models often gave the same wrong answer with confidence.
2. **Abstention removes some correct answers:** the loop did not answer 25 questions, including 16 close-ended
   questions. The verifier recommended abstaining in 21 cases. In 14 of those cases, its blind reading agreed with the
   loop's best checked answer. On the 25 abstained cases, single-shot was correct 11 times, the loop's last answer was
   correct 7 times, and the blind reading was correct 18 times. This explains most of the agentic loop's lower score.
   On the 275 questions it answered, its accuracy was almost the same as single-shot: 65.1% compared with 65.8%.
   Abstention caused 11 of single-shot's 13-question lead under strict scoring.
3. **Another look does not solve the hardest cases:** the controller mainly uses zoom for difficult questions, but
   zoom did not make it more accurate than single-shot on those cases. Verifier-guided answer revisions also remained
   behind single-shot. Revisions fixed 11 wrong first answers but changed 6 correct first answers into wrong answers.
   Keeping the best-verified answer reduces this risk but does not remove it.
4. **Some dataset questions are unclear:** a few questions have grammar problems, such as “is the gyral enhancement?”,
   or ask for detail that one image may not clearly show, such as “below the 7th rib in the right lung.” The dataset
   still provides one reference answer, so an abstention or cautious answer is scored as wrong.
5. **Fixed workflow rules can be too general:** the workflow may answer “Abdomen” or “Chest” when the question asks
   for a specific finding or side.

---

## Reflection and limitations

**When is agentic behaviour beneficial?** In this run, it did not improve accuracy. Single-shot was more accurate
than the agentic loop. The difference was significant for judged accuracy but not strict accuracy. The stronger
verifier model also performed significantly better when answering alone than the full loop. Most of the loss came
from abstention. On the 275 questions it answered, the loop almost matched single-shot at 65.1% vs 65.8%, but all 25
abstentions scored zero under strict scoring. Revisions fixed slightly more answers than they broke, 11 vs 6, but the
difference was not significant. Some fixes came from corrected answers suggested by the verifier. When the blind
check rejected a first answer, that answer was usually wrong: 59 of 82 cases. However, the controller did not use this
signal well in the most important cases. It often abstained when the blind reading was right: 18 of 25 abstentions.
When the loop and blind reading disagreed but the loop still answered, they were right about equally often: 15 vs 17
of 43 cases.

A later analysis used a rule chosen after seeing the results, so it is not an official system result. If the system
had used the blind reading whenever it abstained or disagreed with it, strict accuracy would have increased from
59.7% to 66.3%, close to the verifier's own 67.0%. Most of the gain would come from replacing abstentions, making the
loop behave much like the verifier alone. The loop was also less stable: 12 of 100 answers changed between right and
wrong across runs, compared with 6 for single-shot. This difference was not significant (Fisher p = 0.22).

The main benefit is **useful confidence information**. Verified answers were right 71% of the time, while other
answers were right 40% of the time. This makes the verification status useful for deciding which answers need review,
and the full decision history can be inspected. The trade-off is cost: the agentic loop costs about 58 times more
than one prompt. The next step suggested by this run is to answer instead of abstaining, add a low-confidence warning,
and test that change on new questions. The blind reading may help because it was right on 18 of 25 abstained cases.

**Limitations**

<!-- generated:limit-cost -->
- The agentic loop costs about 58 times more and has about 10 times the median delay of a single prompt.
<!-- /generated:limit-cost -->
- The verifier has many of the same vision weaknesses as the answering model. Agreement does not guarantee a correct
  answer. In case 8, it approved a wrong revision that replaced a right answer.
- The loop's accuracy cannot be fully separated from the verifier's accuracy because verifier guidance may suggest an
  answer. A verifier that reports only problems, without suggesting answers, would better measure the loop itself.
- Abstentions are scored as wrong. The judge gave credit to one because its reference answer was “maybe.” This means
  the abstention policy reduces accuracy by design.
- The loop's answers changed between runs more often than single-shot answers: 12 vs 6 of 100, though the difference
  was not significant.
- With 300 questions, confidence intervals can be as wide as ±5.7 percentage points. Results for small question types
  are less reliable.
- The sample comes from the full dataset and mostly contains train-labelled questions. The scores therefore cannot be
  compared directly with published VQA-RAD test-split results.
- The strict scorer misses some correct paraphrases and may accept vague answers that contain the reference word. The
  LLM judge helps correct both problems, but it is still a model and was spot-checked rather than human-annotated.
- The workflow baseline intentionally uses a general image description that is not based on the question.
- The system has no retrieval step, so it cannot use knowledge that is not visible in the image.

---

## User interface

- **Dashboard:** upload an image, ask a question, and compare the agentic loop with both baselines. The agentic view
  shows the answer, explanation, verification checks, failure categories, and full decision history.
- **Case Explorer:** browse evaluation questions and cached answers from single-shot, workflow, and agentic systems.
  You can also inspect the agentic trace. Fixed-pipeline answers appear in the Evaluation table. Search by question
  text or ID; an exact ID match appears first. If a question is not in the current run, or you select “Refresh
  analysis,” it runs live and is labelled “Live result, not from the evaluation run.”
- **Evaluation:** view strict and LLM-judged accuracy for all four systems, close-ended and open-ended accuracy,
  accuracy by question type and controller path, system disagreements, and a filterable table of every case.

---

## Quick start

**Prerequisites:** Python 3.12, Node.js 20.19+ or 22.12+ for Vite 8, and an OpenAI API key that can use the models
listed in `.env`.

```bash
git clone https://github.com/clarence-ck/med-vqa-rad-agentic.git
cd med-vqa-rad-agentic
cp .env.example .env            # set OPENAI_API_KEY; per-role models are documented in the file

conda env create -f environment.yml   # recommended for native dependencies on Windows
conda activate medvisionquery
# or: pip install -r backend/requirements.txt

python scripts/download_dataset.py    # downloads VQA-RAD to data/vqa-rad/

cd backend && uvicorn main:app --port 8000      # terminal 1
cd frontend && npm install && npm run dev       # terminal 2
```

<!-- generated:question-cost -->
Open http://localhost:5173. Each new question runs all three systems and costs about $0.014 with the default models.
<!-- /generated:question-cost -->

From `backend/`, run `python -m unittest discover tests` to run tests without making API calls. To rebuild all result
files, figures, screenshots, and the report from cached results without spending money, run
`python scripts/run_all.py --stages metrics,docs,diagram,screenshots,report`.

---

## Reproducing the evaluation

<!-- generated:reproduce -->
```bash
python scripts/run_all.py --dry-run         # shows stages and estimated cost; spends nothing
python scripts/run_all.py                   # runs evaluation, repeats, judging, metrics, docs, figures, and report
python scripts/run_all.py --stages metrics,docs,diagram,report   # rebuilds outputs from cached results for free
```

| Stage | What it runs | Last cost | Last wall clock |
|---|---|---|---|
| `eval` | `run_case_explorer_cache.py --ablation --concurrency 4` (300 questions × 4 systems) | $6.20 | ~67 min |
| `variance` | `repeat_cases.py --from-eval-sample 300 --n 100` (baseline + agentic loop again) | $1.31 | ~83 min |
| `judge` | `judge_open_ended.py` | $1.22 | ~11 min |
| `metrics` | `final_metrics.py` → `final_metrics.json` and `cases.csv` in the run folder | free | seconds |
| `docs` | `update_docs.py` → README blocks and `docs/report/results.tex` | free | seconds |
| `diagram` | `system_diagram.py` → `docs/report/system_diagram.png` for the README and report | free | seconds |
| `screenshots` | Starts the backend and frontend, then runs `capture-screenshots.mjs` for example cases | free | ~1 min |
| `report` | `tectonic main.tex` → `docs/report/MGAI_Final_Report_Group9_revised.pdf` | free | ~1 min |
<!-- /generated:reproduce -->

- Paid stages ask for confirmation unless you use `--yes`. All stages share one spending limit set with `--budget`,
  which defaults to $10. Judging runs before variance repeats, so only repeats are reduced if the budget is low.
- Continue an interrupted run with `python scripts/run_all.py --resume`. Completed cases are kept. The README and
  report are rebuilt only after the full run finishes.
- The `screenshots` stage needs Node.js and a local Chrome or Edge installation. The `report` stage needs `tectonic`,
  which is included in `environment.yml`.
- After a full run, two manual checks remain: review the judge sample in `judge_review_sample.md` in the run folder,
  and reread the *Systematic failure patterns* section and the report.
- Every cached result stores the git commit, models, reasoning settings, and token use. Failed API calls are not cached.
  The runner retries them and includes their cost in the spending limit.

---

## Project structure

```text
backend/
  main.py                      FastAPI app
  config.py                    per-role models and reasoning effort, AGENT_MODE, timeouts, code version
  agents/
    agentic_loop.py            controller loop (the agentic system)
    look_agent.py              question-aware perception, focused re-looks, radiology check
    question_analyzer.py       question type and expected answer format
    reasoning_agent.py         answer synthesis (image + observations + guidance)
    verification_agent.py      image-grounded verification and next-action recommendation
    orchestrator.py            fixed six-stage pipeline (ablation)
    image_interpreter.py       question-agnostic description (workflow baseline, ablation)
    difficulty_router.py       difficulty routing (ablation only)
    baseline.py                single-shot baseline
    workflow_baseline.py       hand-designed workflow baseline
  logger.py                    session log (backend/logs/), lines tagged with the case being run
  routes/                      /api/analyze*, /api/dataset/* (samples, analysis, evaluation)
  utils/                       scoring, answer normalisation, image zoom, usage tracking, data loading
  utils/runs.py                where evaluation runs live; which run is current
  tests/                       unit tests (no API calls)
frontend/                      React 19 + Vite UI; scripts/capture-screenshots.mjs
scripts/                       run_all.py (one command for everything) and the stages it runs
experiments/runs/              one folder per evaluation run (cases/, judgments, variance, metrics, cases.csv)
experiments/eval_set.json      the evaluation questions (scripts/make_eval_set.py)
eval-dashboard-screenshots/    dashboard screenshots (regenerated by scripts/run_all.py)
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check and endpoint list |
| POST | `/api/analyze` | Runs the agentic system. `AGENT_MODE=loop` is the default; use `pipeline` for the ablation. Send multipart fields `image` and `question`. |
| POST | `/api/analyze/baseline` | Runs the single-shot baseline |
| POST | `/api/analyze/workflow` | Runs the workflow baseline |
| GET | `/api/dataset/samples` | Returns the evaluation questions from `experiments/eval_set.json` |
| GET | `/api/dataset/sample/{id}` | Returns one case and its image |
| GET | `/api/dataset/sample/{id}/analysis?refresh=&ablation=` | Runs all systems on one case or loads cached results |
| GET | `/api/dataset/evaluation?run=` | Returns run metrics and per-case rows. The default is the current run. |
