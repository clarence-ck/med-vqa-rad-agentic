# MedVisionQuery — an agentic multimodal system for medical visual question answering

MedVisionQuery answers natural-language questions about radiology images, such as X-rays, CT scans and MRI scans.
Instead of asking a model once, it lets a **controller** (a Haystack Agent) work step by step: look at the image,
answer, check the answer against the image, and then decide whether to look again more closely, rewrite the answer,
decline to answer (*abstain*), or finish. We tested it on 300 questions from the VQA-RAD dataset and compared it with
three simpler systems: a single prompt, a hand-designed workflow and a fixed pipeline.

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
Run `2026-10-15_1641` tested 300 VQA-RAD questions, picked at random with a fixed seed so the same pick can be made
again. The pick keeps the dataset's mix of question kinds: 160 close-ended (yes/no, or one of two terms such as
normal/abnormal) and 140 open-ended. All four systems used the same answering model, `gpt-6-luna`. The agentic loop
and the fixed pipeline also used a second model, `gpt-6.1-sol`, to check answers.

How to read the table:

- **Strict accuracy:** answers are marked by fixed rules. For example, a yes/no answer must start with the right word.
- **Judged accuracy:** the same, except that open-ended answers are marked by a third model, `gpt-6-astra`, which
  decides whether an answer means the same as the reference answer. It is not told which system wrote which answer.
- **Time:** four questions ran at the same time. The 63 questions where the loop took a second, closer look were the
  slowest and pull the mean up; the median is the middle value. Full timings are in [Run timings](#run-timings).

| System | Strict accuracy | Judged accuracy | Time / case (mean · median) | Cost / case |
|---|---|---|---|---|
| Single-shot baseline | 64.0% | 70.0% | 4.3 s · 3.6 s | $0.0002 |
| Workflow baseline (hand-designed) | 28.0% | 29.3% | 9.3 s · 8.8 s | $0.0005 |
| **Agentic controller loop** | 59.7% | 64.7% | 48.1 s · 34.2 s | $0.0132 |
| Fixed pipeline (ablation) | 58.0% | 62.0% | 34.5 s · 27.9 s | $0.0068 |

**How to read the p-values.** Every system answered the same questions, so two systems are compared question by
question (exact McNemar test). The p-value is the chance of seeing a gap at least this large if the two systems were
really equally good. Below 0.05, the usual cut-off, the gap is called *significant*: it is unlikely to be luck. Above
0.05, the gap could be chance.

- **Agentic vs single-shot: single-shot ahead, significant on the judged score only.** The agentic loop scored 4.3
  points lower on strict accuracy (p = 0.079, could be chance) and 5.3 points lower on judged accuracy (p = 0.033,
  significant).
- **Agentic vs fixed pipeline:** the agentic loop scored 1.7 points higher on strict accuracy (p = 0.58) and 2.7
  points higher on judged accuracy (p = 0.35). Both gaps could be chance.
- **Agentic vs its verifier model alone:** inside `check`, the verifier model `gpt-6.1-sol` first answers the question
  by itself (the *blind reading*). Scored on its own, that answer reached 67.0% strict accuracy, against 59.7% for the
  full loop (p = 0.008, significant).
- **Confidence and stability:** answers the loop marked as *verified* were right 71% of the time; its other answers
  (abstentions not counted) were right 40% of the time. When the same 100 questions were run a second time, 12% of
  the loop's answers changed between right and wrong, against 6% for single-shot.
<!-- /generated:glance -->

![Evaluation dashboard](eval-dashboard-screenshots/01-eval-kpi-cards.png)

---

## Problem and task decomposition

**Input:** a radiology image (JPEG or PNG) and a question in plain text. **Output:** an answer, an explanation based
on what is visible in the image, a verification status, warning signs of uncertainty, and a record of every decision.

One model call does several jobs at once, and each job can go wrong in its own way. This system splits them up:

| Sub-task | Question it answers | Component |
|---|---|---|
| Question understanding | What kind of answer is needed (yes/no, a location, the image type, ...)? | `analyze_question` |
| Perception | What does the image show that matters for this question? | `look(focus)` |
| Synthesis | What is the best answer, given the image and what was seen? | `answer(guidance)` |
| Verification | Does the image support the answer, and what should happen next? | `check` |
| Planning / control | Look again, rewrite, abstain, or finish? | Controller (Haystack Agent) |

---

## Agent architecture

![System diagram](docs/report/system_diagram.png)

### Controller loop

For each question, `backend/agents/agentic_loop.py` creates a new Haystack `Agent`. Each question gets its own working
memory that holds the image, the question analysis, what each look found, each answer and each check. The controller
itself never sees the image; it sees only short text results from the tools and picks the next tool to call. There is
only one agent, the controller. The five tools are ordinary functions, not separate agents. `check` recommends a next
step, but the controller makes the decision. In the fixed-pipeline comparison, the steps instead run in a set order.

| Tool | Role | Model (default) | Notes |
|---|---|---|---|
| `analyze_question()` | Understand the task | gpt-6-luna | Works out the question type and the expected answer format. Its keyword rules match whole words only. |
| `look(focus)` | Inspect the image | gpt-6-luna, vision | Looks at the image with the question in mind. Given a `focus`, it finds that region, crops and enlarges it, and looks at the crop next to the full image. If it cannot find the region, it looks at the full image again. The first look also rejects images that are not radiology images. |
| `answer(guidance)` | Write an answer | gpt-6-luna, vision | Sees the image and everything the looks found. `guidance` says what the verifier asked to fix. |
| `check()` | Check the answer | gpt-6.1-sol, vision | First answers the question itself without seeing the proposed answer, then reviews the proposed answer and recommends a next step. |
| `finish(abstain)` | Stop | — | Ends the case. The controller cannot finish normally while the latest answer is unchecked, but it can always abstain. |

**How `check` works.** It has two parts, in this order:

1. **Blind reading:** once per case, the verifier answers using only the image and the question. It does not see the
   proposed answer or the notes from the looks, so it cannot be swayed by them.
2. **Review:** the verifier checks whether the proposed answer fits the image, contradicts anything visible, or has
   the wrong format. It returns a status, a list of problems and one recommended action: `accept`, `relook` (with a
   region to focus on), `reanswer` (with guidance) or `abstain`. The guidance may suggest a corrected answer.

An answer counts as *verified* only if all three of these hold: the verifier returns `verified`, no format or
question-type check fails, and the blind reading agrees with the answer. Agreement is first tested with word rules
(yes/no answers must match exactly); for open-ended answers the rules cannot settle, the verifier decides whether the
two mean the same. The controller is told only whether they agree, never what the blind answer was. The review's
guidance, however, may still name a corrected answer.

Limits: at most 2 focused re-looks, 3 answers and 16 agent steps. If the controller stops making progress, a safety
rule makes sure there is a checked answer before the case ends. The questions in this run took these paths:

<!-- generated:paths -->
| Path | Cases | Tool sequence |
|---|---|---|
| direct | 187 | analyze → look → answer → check → finish |
| relook | 63 | … → check → look(focus) → answer(guided) → check → finish |
| reanswer | 25 | … → check → answer(guided) → check → finish |
| abstained | 25 | … → check → finish(abstain) |
<!-- /generated:paths -->

### Why these models and tools

- **Haystack `Agent` + `Tool`:** gives a ready-made tool-calling loop with typed inputs and stopping rules. The image
  stays in the question's memory rather than in the controller's prompt, which keeps the controller cheap.
- **One OpenAI model family, with a different model for each job:** controller decisions and question analysis are
  simple, so they use `gpt-6-luna` with little or no reasoning effort. Answering uses `gpt-6-luna` with medium effort.
  The answerer always sees the image itself, because a text description can lose visual detail.
- **A stronger model checks and a third model judges:** `check` uses `gpt-6.1-sol`, which costs 20 times more per
  token than `gpt-6-luna`, so the answering model never marks its own work. Because the verifier can suggest a
  corrected answer, the evaluation also scores the verifier's blind answer on its own. The evaluation judge,
  `gpt-6-astra`, is not used by any of the systems being tested.
- Every role's model can be changed with `MODEL_<ROLE>` and `REASONING_EFFORT_<ROLE>` in `.env`. Token use and cost
  are recorded for every model call.

### Baselines and ablation

| System | File | What it does |
|---|---|---|
| Single-shot baseline | `backend/agents/baseline.py` | One prompt with the image. It gives a short answer and starts with yes or no when the question needs it. |
| Workflow baseline | `backend/agents/workflow_baseline.py` | A fixed chain: a general description of the image, a question analysis, then hand-written rules for each question type. |
| Fixed pipeline (ablation) | `backend/agents/orchestrator.py` | Six tools always called in the same order: `interpret_image` → `analyze_question` → `filter_image_context` → `route_difficulty` → `reason_answer` → `verify_answer`. It runs once more if the check finds a serious or fixable problem. Same models, but a general image description, no zoom, no blind reading, and a verifier that reads text only. So the comparison tests the whole agentic design, not just the controller's freedom to choose. |

---

## Multimodal grounding

The system works with two kinds of input, the image and the question text, and combines them at each important step:

| Step | Image | Text |
|---|---|---|
| `analyze_question()` | ✗ | The question and anything seen so far |
| `look(focus)` | ✓ Full image at high detail, plus an enlarged crop when a focus is given | The question and focus say what to look for |
| `answer(guidance)` | ✓ Sent directly | Question analysis, everything seen, and the verifier's guidance |
| `check()` | ✓ Sent directly; the blind reading sees only the image and the question | The review also sees the answer, its explanation and what the looks found |
| Controller | ✗ | Short tool results only |

Every model that looks, answers or checks sees the image itself, because a text description can leave out visual
details. Only the question analyser and the controller work from text alone. The evaluation judge compares text
answers with the reference answer and does not see the image.

---

## Control and failure handling

| Mechanism | What it does |
|---|---|
| Checking against the image | `check` makes one independent blind reading, checks the answer against the image, and compares the two answers. It returns five checks, a status (`verified`, `partially_verified`, `low_confidence` or `unsupported`) and a recommended next step. |
| Targeted recovery | Instead of starting over, the controller can zoom into one region with `look(focus)` or rewrite the answer with `answer(guidance)`. |
| Best-verified answer | A later answer replaces an earlier one only if its check result is clearly better. This matters because looking again can make the model "see" details that are not there. |
| Self-consistency | After finishing, the answerer answers the question two more times on its own. If the answers disagree, the case gets a `self_inconsistent` flag and is marked for review. |
| Abstention | The controller may decline to answer. An abstention is recorded as an abstention and is never turned into "no". |
| Guardrail | Records what went wrong (`unsupported_claim`, `low_confidence`, `visual_contradiction`, `partially_verified`, `question_answer_mismatch`, `reasoning_generation_unstable`, `parser_failure`, `self_inconsistent`), adds a disclaimer, and marks cases that need review. |
| Input rejection | The first look rejects images that are not radiology images. If the image cannot be read normally, the system tries the original file bytes. If the first look still fails, it stops rather than guess. |

---

## Evaluation design

Each evaluation is saved as its own **run** in `experiments/runs/<run_id>/`: the answers for every case, the judge's
marks, the repeat runs, the metrics, timings and costs. A new run never changes an old one, and
`experiments/runs/README.md` lists them all. Every number in this README and in the report comes from the latest run.
`scripts/final_metrics.py` computes the numbers and `scripts/update_docs.py` writes them into the marked sections of
this README and into the report. The counts in the failure-pattern and reflection sections come from the same run
files; thresholds and limits come from the code.

- **Data:** VQA-RAD has 2,244 questions, each with a reference answer. We removed one known bad item and the 300 test
  questions used in an earlier evaluation. From the remaining 1,943 questions we picked 300 at random (seed 51511),
  keeping the dataset's mix: 160 close-ended and 140 open-ended. No system was trained on the data, so the dataset's
  train/test labels make no difference here. Because all the removed questions were test questions, 277 of the 300
  carry a "train" label. The list is in `experiments/eval_set.json` and was made with
  `python scripts/make_eval_set.py --size 300 --split all --seed 51511 --exclude-run 2026-10-15_1308`.
  Question types with only a few examples give less reliable results.
- **Systems:** the single-shot baseline, workflow baseline, agentic loop and fixed pipeline all answer the same
  questions with the same answering model.
- **Strict scorer** (`backend/utils/scoring.py`) marks an answer right when:
  - the reference is yes/no: the answer starts with the right word;
  - the reference is one of a pair, such as normal/abnormal or present/absent: the answer contains the right term and
    not its opposite;
  - the reference is a count: the numbers match;
  - the reference is open-ended: every important word of the reference appears (60% of them for references longer
    than three words). Aliases such as CT and computed tomography count as the same, and so do singular and plural.
    A left/right mix-up is wrong, and so is an answer starting with "no" when the reference does not. Only as a last
    resort, a similarity score from the S-PubMedBert-MS-MARCO model is used: 0.95 or higher for short answers, 0.90
    for long ones.
- **LLM judge** (`scripts/judge_open_ended.py`): `gpt-6-astra` marks every open-ended answer from every system,
  including ones the strict scorer already marked. It decides whether each answer means the same as the reference.
  System names are hidden and the answers are shuffled. Its marks give the "judged" score for open-ended answers;
  close-ended answers keep their strict mark. Astra does not answer or check anything in the systems being tested. In
  a review of 30 of its marks, 29 were right; in case 22 it was too harsh on an agentic-loop answer. See
  `judge_review_sample.md` in the run folder. That review was done with AI help, not by human annotators.
- **Statistics:** with 300 questions, each accuracy is uncertain by about ±5.6 percentage points (95% confidence
  interval), and never more than ±5.7. Two systems are compared question by question with the exact McNemar test,
  which looks only at questions where one system was right and the other wrong. A p-value below 0.05 counts as
  significant.
- **Variance:** to see how much answers change from one run to the next, 100 of the evaluation questions are picked
  (seed 42). Their original answers count as the first pass; the single-shot baseline and the agentic loop then answer
  them again. Both passes belong to the same run folder.
- **Code changes after the run:** the run used commit `2b1d54f`. Since then, two things changed. The yes/no answer
  cleaner now checks negative phrases before positive words, because "no visible fracture" used to be read as "yes".
  And `check` now makes the blind reading before the review, as the controller prompt describes. Neither change
  alters this run's results: re-cleaning all 4,519 recorded answers gave the same output, and the two `check` calls
  do not depend on each other, so their order does not change what either model sees.
<!-- generated:cost -->
- **Cost:** every OpenAI call is recorded by `backend/utils/usage.py`. The evaluation cost $6.20, the repeats $1.31
  and the judging $1.22, a total of $8.73.

### Run timings

How long each stage of run `2026-10-15_1641` took, as recorded by `scripts/run_all.py` in
`experiments/runs/2026-10-15_1641/run.json`:

| Stage | Started | Finished | Wall clock | Work | Cost |
|---|---|---|---|---|---|
| Main evaluation (4 systems) | 2026-10-15 16:41:12 | 17:48:42 | 67 min 29 s | 300 cases × 4 systems, 4 at a time | $6.20 |
| LLM judge (`gpt-6-astra`) | 2026-10-15 17:48:43 | 17:59:33 | 10 min 50 s | 140 open-ended cases | $1.22 |
| Variance repeats (baseline + agentic) | 2026-10-15 17:59:33 | 19:22:22 | 82 min 48 s | 100 cases, one at a time | $1.31 |

The per-case times in the results tables are each system's own response time. The four systems ran side by side,
so one whole case took 52.9 seconds on average.
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

Head-to-head comparisons. "17 vs 30" means 17 questions only the first system got right and 30 only the second got
right; questions both got right, or both got wrong, are left out. A p-value below 0.05 means the gap is significant.

| Comparison | Strict | Judged |
|---|---|---|
| Agentic vs single-shot | 17 vs 30, p = 0.079 | 17 vs 33, p = 0.033 |
| Agentic vs fixed pipeline | 28 vs 23, p = 0.58 | 32 vs 24, p = 0.35 |

### By controller path

The path is the route the loop took on a question (see [Controller loop](#controller-loop)). The other columns show
how the other systems did on the same questions.

| Path | Cases | Agentic | Single-shot | Fixed pipeline |
|---|---|---|---|---|
| direct | 187 | 75% | 74% | 70% |
| relook | 63 | 38% | 41% | 33% |
| reanswer | 25 | 56% | 64% | 52% |
| abstained | 25 | — (abstained) | 44% | 40% |

### Uncertainty signals predict errors

These signals tell you how far to trust an answer: answers with a good signal were right much more often.

| Signal | Levels | Agentic accuracy |
|---|---|---|
| Verification (answered questions) | verified (220) / not verified (55) | 71% / 40% |
| Self-consistency | high (242) / medium or low (33) | 69% / 39–40% |
| Abstention | 25 cases; the single-shot baseline was wrong on 56% of them | — |

### Stability across runs (same 100 cases)

Strict scoring. "Original evaluation" is the answer from the main 300-question evaluation. "Independent repeat" is a
fresh answer to the same question, made later in the same run.

| System | Original evaluation | Independent repeat | Answers that flipped right ↔ wrong |
|---|---|---|---|
| Single-shot baseline | 60/100 | 60/100 | 6 (6%) |
| Agentic loop | 59/100 | 55/100 | 12 (12%) |

### What the loop's mechanisms did

| Mechanism | Result in this run |
|---|---|
| Revision (re-look or re-answer) | Used on 88 questions. It turned 11 wrong first answers into right ones and 6 right first answers into wrong ones (strict scoring). |
| Blind check | Disagreed with 82 first answers, and 59 of those (72%) were indeed wrong. For 66 answers the word rules could not tell whether the two agreed, so the verifier compared their meanings. |
| Zoom | Used on 63 questions; 5 of them later ended in abstention. On the other 58, the loop was right 36.2% of the time and single-shot 39.7%. |
| Abstention | Used on 25 questions. The loop's last answer before abstaining would have been right in 7 of them; single-shot was right in 11. |
| Answered questions only | On the 275 questions the loop answered, it was right 65.1% of the time and single-shot 65.8%. |

The blind reading is the verifier model, `gpt-6.1-sol`, answering straight from the image. Scored on its own, it
reached 67.0% strict accuracy. Against the loop: 21 vs 43 (p = 0.008, significant). Against single-shot: 29 vs 20
(p = 0.25, could be chance).
<!-- /generated:results -->

---

## Failure analysis

The dashboard shows the four analyses that Template 1 asks for, for every run.

<!-- generated:failure -->
- **Self-consistency checks:** the answerer gives two extra answers of its own for each case. If they disagree, the
  case is marked for review. Disagreement means "uncertain", not "wrong": 39% of low-consistency answers were still
  right.
- **Cross-modal agreement:** the verifier looks at the image itself. It checks whether the explanation matches what is
  visible and whether the answer contradicts the image, instead of only reading a text description.
- **Verifier model:** a different, stronger model, `gpt-6.1-sol`, checks the answers of `gpt-6-luna`. After its
  checks, 113 of the 300 cases left the direct path.
- **Disagreement analysis:** compared with the same rules used for scoring, the three systems' answers did not all
  match on 69% of cases. Where only one system was right, it was the agentic loop in 15 cases, single-shot in 20 and
  the workflow in 12.

These examples were picked automatically by `scripts/final_metrics.py`. Screenshots are in
`eval-dashboard-screenshots/`.

| Case | What happened |
|---|---|
| 503 "is there a pleural effusion present?" | A closer second look got it right; single-shot got it wrong. Agentic: "no"; single-shot: "Yes, there is a small left pleural effu…"; reference: "no". |
| 457 "what brain region demonstrates edema?" | An answer rewritten on the verifier's advice got it right; single-shot got it wrong. Agentic: "Bilateral basal ganglia"; single-shot: "Both thalami demonstrate edema."; reference: "basal ganglia". |
| 8 "where is the mass?" | An earlier answer was right, but a later rewrite replaced it with a wrong one, and the check approved the wrong one. Single-shot was right. Agentic: "Around the posterior third ventricle, e…"; single-shot: "The mass is in the midline pineal regio…"; reference: "pineal region". |
| 19 "what plane is seen?" | The extra answers disagreed, so the case was marked for review, but the final answer was right. Agentic: "Frontal (PA) view"; single-shot: "Frontal (coronal) plane."; reference: "pa". |
<!-- /generated:failure -->

**Systematic failure patterns**

1. **All models miss the same things:** 65 questions (22%) were wrong for single-shot, the agentic loop and the fixed
   pipeline alike (judged scoring). 49 of them were open-ended, mostly asking for a location, a finding or a
   diagnosis. A check cannot catch a mistake that the looker and the verifier both make. The verifier's blind reading
   was right on only 16 of these 65, and even then its answer did not become the final one. Self-consistency did not
   help either: 51 of the 59 answered cases were highly consistent. The models often gave the same wrong answer, and
   gave it confidently.
2. **Abstaining throws away some right answers:** the loop declined to answer 25 questions, 16 of them close-ended.
   The verifier recommended abstaining in 21 of these, and in 14 of those 21 its own blind reading agreed with the
   loop's best checked answer. On the 25 abstained questions, single-shot was right 11 times, the loop's last answer
   7 times and the blind reading 18 times. This explains most of the loop's lower score. On the 275 questions it did
   answer, it was about as accurate as single-shot (65.1% vs 65.8%). Of single-shot's 13-question lead on strict
   scoring, 11 questions come from abstentions.
3. **Looking again does not fix the hardest questions:** the controller mostly zooms in on hard questions, but zooming
   did not make it beat single-shot on them. Answers rewritten on the verifier's advice also stayed behind
   single-shot. Rewrites fixed 11 wrong first answers but broke 6 right ones. Keeping the best-verified answer lowers
   this risk but does not remove it.
4. **Some dataset questions are unclear:** a few are ungrammatical, such as "is the gyral enhancement?", or ask for
   detail that one image may not clearly show, such as "below the 7th rib in the right lung". The dataset still gives
   one reference answer, so an abstention or a cautious answer is marked wrong.
5. **Fixed workflow rules are too general:** the workflow may answer "Abdomen" or "Chest" when the question asks for
   a specific finding or side.

---

## Reflection and limitations

**When does agentic behaviour help?** In this run, it did not make answers more accurate. Single-shot beat the
agentic loop; the gap was significant on judged accuracy but not on strict accuracy. The verifier model answering
alone also beat the full loop, and that gap was significant. Most of the loss came from abstaining: on the 275
questions it answered, the loop nearly matched single-shot (65.1% vs 65.8%), but all 25 abstentions count as wrong
under strict scoring. Rewrites fixed slightly more answers than they broke (11 vs 6), but that gap could be chance,
and some of the fixes came from corrected answers that the verifier suggested. When the blind check disagreed with a
first answer, that answer was usually wrong (59 of 82 cases). But the controller did not use this signal well where
it mattered most: it often abstained when the blind reading was right (18 of 25 abstentions). Where the loop disagreed
with the blind reading but answered anyway, the two were right about equally often (15 vs 17 of 43 cases).

One extra analysis uses a rule we chose *after* seeing the results, so it is not an official result. If the loop had
used the blind reading whenever it abstained or disagreed with it, strict accuracy would have risen from 59.7% to
66.3%, close to the verifier's own 67.0%. Most of that gain comes from replacing abstentions, which would make the loop
behave much like the verifier alone. The loop was also less stable: 12 of 100 answers changed between right and wrong
from one run to the next, against 6 for single-shot. That gap could be chance (Fisher's exact test, p = 0.22).

The main benefit is **knowing which answers to trust**. Verified answers were right 71% of the time, other answers
40%. So the verification status is a useful guide to which answers need a second look, and every decision the loop
made can be inspected. The price is cost: the loop costs about 58 times more than a single prompt. The next step this
run points to is to answer instead of abstaining, add a low-confidence warning, and test that change on new questions.
The blind reading may help here, as it was right on 18 of the 25 abstained questions.

**Limitations**

<!-- generated:limit-cost -->
- The agentic loop costs about 58 times more than a single prompt, and its median response time is about 10 times
  longer.
<!-- /generated:limit-cost -->
- The verifier shares many of the answering model's vision weaknesses, so agreement does not guarantee a right
  answer. In case 8 it approved a wrong rewrite that replaced a right answer.
- The loop's accuracy cannot be fully separated from the verifier's, because the verifier's guidance may suggest an
  answer. A verifier that only points out problems, without suggesting answers, would measure the loop itself better.
- Abstentions count as wrong. The judge gave credit to just one, whose reference answer was "maybe". So the abstention
  policy lowers accuracy by design.
- The loop's answers changed between runs more often than single-shot's (12 vs 6 of 100), though that gap could be
  chance.
- With 300 questions, each accuracy can be off by up to ±5.7 percentage points (95% confidence interval). Results for
  small question types are less reliable.
- The questions come from the whole dataset and are mostly labelled "train", so the scores cannot be compared directly
  with published VQA-RAD results, which use the test split.
- The strict scorer misses some right answers that are worded differently, and may accept vague answers that happen to
  contain the reference word. The LLM judge corrects for both, but it is still a model, and it was spot-checked rather
  than checked by human annotators.
- The workflow baseline deliberately uses a general image description that ignores the question.
- The system does not look anything up, so it cannot use knowledge that is not visible in the image.

---

## User interface

- **Dashboard:** upload an image, ask a question, and compare the agentic loop with both baselines. The agentic view
  shows the answer, explanation, verification checks, what went wrong, and every decision the loop made.
- **Case Explorer:** browse the evaluation questions and the saved answers from single-shot, workflow and the agentic
  loop, and step through the loop's decisions. Fixed-pipeline answers are in the Evaluation table. Search by question
  text or ID; an exact ID match is listed first. If a question is not in the current run, or you click "Refresh
  analysis", it runs live and is tagged "Live result, not from the evaluation run".
- **Evaluation:** strict and judged accuracy for all four systems, close-ended and open-ended accuracy, accuracy by
  question type and by controller path, where the systems disagree, and a filterable table of every case.

---

## Quick start

**You need:** Python 3.12, Node.js 20.19+ or 22.12+ (for Vite 8), and an OpenAI API key with access to the models
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

To run the tests (no API calls), run `python -m unittest discover tests` from `backend/`. To rebuild every result
file, figure, screenshot and the report from saved results, at no cost, run
`python scripts/run_all.py --stages metrics,docs,diagram,screenshots,report`.

---

## Reproducing the evaluation

<!-- generated:reproduce -->
```bash
python scripts/run_all.py --dry-run         # lists the stages and estimated cost; spends nothing
python scripts/run_all.py                   # runs evaluation, repeats, judging, metrics, docs, figures and report
python scripts/run_all.py --stages metrics,docs,diagram,report   # rebuilds outputs from saved results, at no cost
```

| Stage | What it runs | Last cost | Last wall clock |
|---|---|---|---|
| `eval` | `run_case_explorer_cache.py --ablation --concurrency 4` (300 questions × 4 systems) | $6.20 | ~67 min |
| `variance` | `repeat_cases.py --from-eval-sample 300 --n 100` (baseline + agentic loop again) | $1.31 | ~83 min |
| `judge` | `judge_open_ended.py` | $1.22 | ~11 min |
| `metrics` | `final_metrics.py` → `final_metrics.json` and `cases.csv` in the run folder | free | seconds |
| `docs` | `update_docs.py` → README sections and `docs/report/results.tex` | free | seconds |
| `diagram` | `system_diagram.py` → `docs/report/system_diagram.png` for the README and report | free | seconds |
| `screenshots` | Starts the backend and frontend, then runs `capture-screenshots.mjs` for example cases | free | ~1 min |
| `report` | `tectonic main.tex` → `docs/report/MGAI_Final_Report_Group9_revised.pdf` | free | ~1 min |
<!-- /generated:reproduce -->

- Stages that cost money ask before they start, unless you pass `--yes`. All stages share one spending limit, set
  with `--budget` (default $10). Judging runs before the repeats, so if money runs short, only the repeats are cut.
- To continue a run that was interrupted, use `python scripts/run_all.py --resume`. Finished cases are kept. The
  README and report are rebuilt only once the whole run has finished.
- The `screenshots` stage needs Node.js and Chrome or Edge installed. The `report` stage needs `tectonic`, which is in
  `environment.yml`.
- After a full run, two things are still done by hand: review the judge sample in `judge_review_sample.md` in the run
  folder, and reread the *Systematic failure patterns* section and the report.
- Every saved result records the git commit, models, reasoning settings and token use. Failed API calls are not
  saved; the runner tries them again and counts their cost towards the spending limit.

---

## Project structure

```text
backend/
  main.py                      FastAPI app
  config.py                    per-role models and reasoning effort, AGENT_MODE, timeouts, code version
  agents/
    agentic_loop.py            controller loop (the agentic system)
    look_agent.py              looks at the image for the question, focused re-looks, radiology check
    question_analyzer.py       question type and expected answer format
    reasoning_agent.py         writes the answer (image + what was seen + guidance)
    verification_agent.py      checks the answer against the image and recommends the next step
    orchestrator.py            fixed six-stage pipeline (ablation)
    image_interpreter.py       general image description (workflow baseline, ablation)
    difficulty_router.py       difficulty routing (ablation only)
    baseline.py                single-shot baseline
    workflow_baseline.py       hand-designed workflow baseline
  logger.py                    session log (backend/logs/), each line tagged with the case being run
  routes/                      /api/analyze*, /api/dataset/* (samples, analysis, evaluation)
  utils/                       scoring, answer cleaning, image zoom, usage tracking, data loading
  utils/runs.py                where evaluation runs are stored and which one is current
  tests/                       unit tests (no API calls)
frontend/                      React 19 + Vite UI; scripts/capture-screenshots.mjs
scripts/                       run_all.py (one command for everything) and the stages it runs
experiments/runs/              one folder per evaluation run (cases/, judgments, variance, metrics, cases.csv)
experiments/eval_set.json      the evaluation questions (made by scripts/make_eval_set.py)
eval-dashboard-screenshots/    dashboard screenshots (remade by scripts/run_all.py)
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check and list of endpoints |
| POST | `/api/analyze` | Runs the agentic system. `AGENT_MODE=loop` is the default; set `pipeline` for the ablation. Send the multipart fields `image` and `question`. |
| POST | `/api/analyze/baseline` | Runs the single-shot baseline |
| POST | `/api/analyze/workflow` | Runs the workflow baseline |
| GET | `/api/dataset/samples` | Returns the evaluation questions from `experiments/eval_set.json` |
| GET | `/api/dataset/sample/{id}` | Returns one case and its image |
| GET | `/api/dataset/sample/{id}/analysis?refresh=&ablation=` | Loads the saved results for one case, or runs all systems on it |
| GET | `/api/dataset/evaluation?run=` | Returns run metrics and per-case rows (the current run by default) |
