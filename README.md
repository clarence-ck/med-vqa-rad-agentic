# MedVisionQuery — an agentic multimodal system for medical visual question answering

MedVisionQuery answers natural-language questions about radiology images (X-ray, CT, MRI). Its agentic system is a
**Haystack Agent controller** that decides, case by case, how to look at the image, answer, check the answer *against
the image*, and whether to re-examine a region, revise, abstain or finish. It is evaluated against a single-prompt
baseline, a hand-designed workflow and a fixed-pipeline ablation on 300 VQA-RAD test questions.

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
Run `2026-10-15_1308`: 300 seeded-random VQA-RAD test questions (169 close-ended, 131 open-ended), all systems on the same answering model
(`gpt-6-luna`; the agentic loop and the fixed pipeline also use `gpt-6.1-sol` as verifier). Per-case times were
measured with four cases running at once; the mean is pulled up by a few long re-look cases (actual run times are in
[Run timings](#run-timings)). "Strict" uses the rule-based scorer; "judged" scores open-ended answers with a blinded LLM
judge, `gpt-6-astra` (see [Evaluation design](#evaluation-design)).

| System | Strict accuracy | Judged accuracy | Time / case (mean · median) | Cost / case |
|---|---|---|---|---|
| Single-shot baseline | 68.3% | 70.7% | 4.7 s · 3.8 s | $0.0002 |
| Workflow baseline (hand-designed) | 32.7% | 34.0% | 9.7 s · 9.4 s | $0.0005 |
| **Agentic controller loop** | 66.3% | 70.3% | 51.7 s · 36.6 s | $0.0135 |
| Fixed pipeline (ablation) | 67.0% | 68.7% | 36.5 s · 29.6 s | $0.0066 |

- **Agentic vs single-shot: no significant difference** (-2.0 points strict, p = 0.52; -0.4 judged, p = 1.00; exact McNemar).
- **Agentic vs fixed pipeline:** -0.7 points strict (p = 0.89), +1.6 judged (p = 0.56).
- **Agentic vs its verifier model alone** (`gpt-6.1-sol` answering once, the check's blind reading): 66.3% vs 72.0% strict (p = 0.024).
- **Confidence and stability:** answers the check verified are 79% correct, the rest
  29%; between two runs, 15% of the loop's answers change right ↔ wrong vs
  9% for the single prompt.
<!-- /generated:glance -->

![Evaluation dashboard](eval-dashboard-screenshots/01-eval-kpi-cards.png)

---

## Problem and task decomposition

**Input:** a radiology image (JPEG/PNG) and a free-text question. **Output:** an answer, an explanation grounded in
visible evidence, a verification status, uncertainty signals, and a trace of every decision.

A single model call conflates steps that fail in different ways. The system separates them:

| Sub-task | Question it answers | Component |
|---|---|---|
| Question understanding | What kind of answer is expected (yes/no, location, modality…)? | `analyze_question` |
| Perception | What does the image show *that bears on this question*? | `look(focus)` |
| Synthesis | What is the best answer given the image and observations? | `answer(guidance)` |
| Verification | Does the image itself support this answer? What should happen next? | `check` |
| Planning / control | Re-examine a region, revise, abstain or finish? | Controller (Haystack Agent) |

---

## Agent architecture

![System diagram](docs/report/system_diagram.png)

### Controller loop

A fresh Haystack `Agent` is created per request (`backend/agents/agentic_loop.py`). Its tools are bound to a
per-request working memory that holds the image, the question analysis, every look, every answer and every check.
The controller sees only short text results from each tool and chooses the next call. There is one agent (the
controller); the five tools are functions it calls, not agents. `check` only *recommends* the next step; the
controller decides whether to follow it; in the fixed-pipeline ablation every step runs in a prescribed order.

| Tool | Role | Model (default) | Notes |
|---|---|---|---|
| `analyze_question()` | Task understanding | gpt-6-luna | Question type and expected answer format; keyword rules use whole-word matching |
| `look(focus)` | Perception (executor) | gpt-6-luna, vision | Looks *with the question in mind*; with `focus` it **zooms**: the named region is located, cropped and enlarged, and examined next to the full image. The first look also rejects non-radiology images |
| `answer(guidance)` | Synthesis (executor) | gpt-6-luna, vision | Receives the image plus all observations; `guidance` carries what the verifier asked to fix |
| `check()` | Evaluator (a larger model) | gpt-6.1-sol, vision | Two parts. **Blind reading** (independent): once per episode, the verifier answers from the image and question alone, without the proposed answer or the looker's notes. **Review** (critic): it checks the answer against the image (grounding, contradictions, format) and returns a status, the issues found and a next step (`accept`, `relook` with a focus, `reanswer` with guidance, `abstain`); the guidance may include a suggested corrected answer. An answer is *verified* only if the review passes **and** the blind reading agrees (word rules first, exact for yes/no; for open-ended answers the verifier judges whether the two mean the same). The controller learns only whether they agree, never the blind answer |
| `finish(abstain)` | Decision | — | Ends the episode; refused while the latest answer is unchecked |

Budgets: at most 2 focused re-looks, 3 answers and 16 agent steps. If the controller stalls, a safety net answers
and/or checks before finishing. Paths observed in the current run:

<!-- generated:paths -->
| Path | Cases | Tool sequence |
|---|---|---|
| direct | 186 | analyze → look → answer → check → finish |
| relook | 57 | … → check → look(focus) → answer(guided) → check → finish |
| reanswer | 35 | … → check → answer(guided) → check → finish |
| abstained | 22 | … → check → finish(abstain) |
<!-- /generated:paths -->

### Why these models and tools

- **Haystack `Agent` + `Tool`** gives a tool-calling loop with typed tools and exit conditions. Keeping the image in
  per-request memory (not in the controller's context) keeps the controller cheap.
- **One model family (OpenAI), roles sized by need.** The controller and question analysis are simple decisions, so
  they use `gpt-6-luna` with low or no reasoning effort. Answering uses `gpt-6-luna` with medium effort, and the answerer always sees
  the image: a text description of the image loses the visual detail many questions depend on.
- **A different, larger model verifies; a third judges.** `check` uses `gpt-6.1-sol`, so the answering model never
  verifies itself and the critique comes from a stronger reader. Because the review's guidance can suggest a
  corrected answer, part of the loop's accuracy can come from the verifier, so every run also scores the verifier
  answering alone (its blind reading) as a reference. The evaluation judge, `gpt-6-astra`, takes part in no system.
- All roles are configurable (`MODEL_<ROLE>`, `REASONING_EFFORT_<ROLE>` in `.env`); every call's tokens and cost are
  recorded.

### Baselines and ablation

| System | File | What it does |
|---|---|---|
| Single-shot baseline | `backend/agents/baseline.py` | One vision prompt; asked for a concise answer that starts with yes/no where applicable |
| Workflow baseline | `backend/agents/workflow_baseline.py` | Fixed chain: question-agnostic image description → question analysis → hand-written rules per question type |
| Fixed pipeline (ablation) | `backend/agents/orchestrator.py` | A controller instructed to call six tools in a prescribed order (`interpret_image` → `analyze_question` → `filter_image_context` → `route_difficulty` → `reason_answer` → `verify_answer`), re-running the sequence once when verification flags a fixable problem. Same models, but a question-agnostic image description, no zoom and no blind check, so the comparison measures the agentic design as a whole (control, question-aware and zoomed looks, blind check), not control alone |

---

## Multimodal grounding

The two modalities are the image and the text question. They interact at every step that matters:

| Step | Image | Text |
|---|---|---|
| `look(focus)` | ✓ sent at high detail, plus an enlarged crop of the focus region | the question (and focus) decide what is examined |
| `answer(guidance)` | ✓ sent directly | question analysis, all observations, verifier guidance |
| `check()` | ✓ sent directly (blind reading: image + question only) | answer, explanation, observations (review step only) |
| Controller | ✗ | short tool results only |

Every model that answers or judges sees the image itself, because a text description of the image loses the visual
detail many questions depend on.

---

## Control and failure handling

| Mechanism | What it does |
|---|---|
| Image-grounded verification | `check` compares the answer with an independent blind reading of the image, then judges it against the image: 5 checks, a status (`verified`, `partially_verified`, `low_confidence`, `unsupported`) and a recommended next action |
| Targeted recovery | The controller zooms into a region (`look(focus)`) or revises (`answer(guidance)`) instead of re-running everything |
| Best-verified answer | A later answer replaces an earlier one only if its check is strictly better (re-looks can over-read) |
| Self-consistency | After finishing, the answerer is re-asked independently (2×); low agreement adds `self_inconsistent` and flags the case for review |
| Abstention | The controller may abstain; abstentions are reported as such (never normalised into "no") |
| Guardrail | Failure categories (`unsupported_claim`, `low_confidence`, `visual_contradiction`, `partially_verified`, `question_answer_mismatch`, `reasoning_generation_unstable`, `parser_failure`, `self_inconsistent`), a disclaimer and a review flag |
| Input rejection | Non-radiology images are rejected on the first look. Unreadable images fall back to the raw bytes; if the first look still fails, the system stops instead of answering blind |

---

## Evaluation design

Every evaluation is a **run** with its own folder, `experiments/runs/<run_id>/` (cases, judge verdicts, variance
repeats, metrics, timings, cost); a later run never changes an earlier one, and `experiments/runs/README.md` lists
them all. Every number in this README and the report comes from the current (latest completed) run via
`scripts/final_metrics.py` and `scripts/update_docs.py`; text between `<!-- generated -->` markers is not edited by
hand.

- **Data:** 300 of the VQA-RAD test split's 450 usable questions (one known-bad item excluded), drawn at random
  with seed 42 while keeping the split's mix of 56% yes/no (169 close-ended, 131 open-ended). The list is
  `experiments/eval_set.json` (`scripts/make_eval_set.py`); per-type numbers for small categories are noisy.
- **Systems:** single-shot baseline, workflow baseline, agentic loop and fixed-pipeline ablation, all on the same cases
  and the same answering model.
- **Strict scorer** (`backend/utils/scoring.py`):
  - yes/no ground truth: the answer must start with the correct word;
  - binary terms (normal/abnormal, present/absent, …): the correct term and not its opposite;
  - counting: the numbers must match;
  - open-ended: every content word of the ground truth must appear (aliases such as CT = computed tomography; plurals
    folded), or at least 60% of them for ground truths longer than 3 words. Left/right conflicts, and answers starting
    with "no" when the ground truth does not, are wrong. S-PubMedBert-MS-MARCO similarity is only a strict fallback
    (≥ 0.95 short, ≥ 0.90 long).
- **LLM judge** (`scripts/judge_open_ended.py`): `gpt-6-astra` judges **every** open-ended answer of every system,
  right or wrong by the strict rules, for meaning equivalence with the ground truth, blinded to the system (answers
  shuffled and anonymised). Its verdict is the "judged" score for open-ended answers; yes/no answers stay rule-based.
  Astra neither answers (`gpt-6-luna`) nor verifies (`gpt-6.1-sol`) in any system, so it favours none. A 30-verdict
  spot check agreed with 29 credit decisions; the other was a lenient credit for a workflow-baseline answer
  (`judge_review_sample.md` in the run folder; AI-assisted review, not human annotation).
- **Statistics:** 95% intervals are about ±5 points at n = 300; paired comparisons use the exact McNemar test.
- **Variance:** the baseline and agentic loop are run again on 100 of the evaluated questions.
<!-- generated:cost -->
- **Cost:** every OpenAI call is recorded (`backend/utils/usage.py`): evaluation $6.25, repeats
  $1.28, judge $1.09 ($8.62 in total).

### Run timings

Wall-clock time of each stage of run `2026-10-15_1308`, recorded by `scripts/run_all.py`
(`experiments/runs/2026-10-15_1308/run.json`):

| Stage | Started | Finished | Wall clock | Work | Cost |
|---|---|---|---|---|---|
| Main evaluation (4 systems) | 2026-10-15 13:08:44 | 14:19:23 | 70 min 38 s | 300 cases x 4 systems, 4 at a time | $6.25 |
| LLM judge (`gpt-6-astra`) | 2026-10-15 14:19:23 | 14:29:48 | 10 min 24 s | 131 open-ended cases | $1.09 |
| Variance repeats (baseline + agentic) | 2026-10-15 14:29:48 | 15:52:46 | 82 min 57 s | 100 cases, one at a time | $1.28 |

Per-case times in the results tables are each system's own latency inside a case; a full case (all four systems in
turn) took 55.8 s on average.
<!-- /generated:cost -->

---

## Results

<!-- generated:results -->
### Overall

| System | Strict | Judged | Close-ended (169) | Open-ended (131), strict → judged |
|---|---|---|---|---|
| Single-shot baseline | 68.3% | 70.7% | 81.7% | 51.1% → 56.5% |
| Workflow baseline | 32.7% | 34.0% | 46.2% | 15.3% → 18.3% |
| **Agentic loop** | 66.3% | 70.3% | 78.7% | 50.4% → 59.5% |
| Fixed pipeline (ablation) | 67.0% | 68.7% | 77.5% | 53.4% → 57.3% |

Paired comparisons (cases where exactly one of the two systems is right):

| Comparison | Strict | Judged |
|---|---|---|
| Agentic vs single-shot | 27 vs 33, p = 0.52 | 30 vs 31, p = 1.00 |
| Agentic vs fixed pipeline | 26 vs 28, p = 0.89 | 26 vs 21, p = 0.56 |

### By controller path

| Path | Cases | Agentic | Single-shot | Fixed pipeline |
|---|---|---|---|---|
| direct | 186 | 82% | 81% | 81% |
| relook | 57 | 39% | 39% | 35% |
| reanswer | 35 | 69% | 54% | 57% |
| abstained | 22 | — (abstained) | 59% | 50% |

![Accuracy by controller path](eval-dashboard-screenshots/03-accuracy-by-controller-path.png)

### Uncertainty signals predict errors

| Signal | Levels | Agentic accuracy |
|---|---|---|
| Verification | verified (225) / partially verified (66) / unsupported or low confidence (9) | 79% / 32% / 11% |
| Self-consistency | high (238) / medium or low (40) | 75% / 40–57% |
| Abstention | 22 cases, on which the single-shot baseline is wrong 41% of the time | — |

On the 278 cases it answered, the agentic loop is 72% correct (strict).

### Stability across runs (same 100 cases)

| System | Run 1 | Run 2 | Answers that flipped right ↔ wrong |
|---|---|---|---|
| Single-shot baseline | 71/100 | 76/100 | 9 (9%) |
| Agentic loop | 74/100 | 75/100 | 15 (15%) |

### What the loop's mechanisms did

| Mechanism | Result in this run |
|---|---|
| Revision (re-look or re-answer) | 92 questions: 23 wrong first answers fixed, 5 right ones broken |
| Blind check | Objected to 85 first answers; 64 of them (75%) were wrong. The meaning comparison decided 44 agreements |
| Zoom | Used on 66 questions, the hardest: agentic 33.3% vs single-shot 40.9% there |
| Abstention | 22 questions; the loop's own last answer was right in 11, the single prompt in 13 |
| Answered questions only | Agentic 71.6% vs single-shot 69.1% on the same 278 questions |

The blind reading is the verifier model (`gpt-6.1-sol`) answering from the image alone. Scored on its own it reaches
72.0% strict (agentic vs blind reading 17 vs 34, p = 0.024; blind reading vs single-shot
37 vs 26, p = 0.21).
<!-- /generated:results -->

---

## Failure analysis

The four analyses Template 1 asks for are built into every agentic run and summarised on the dashboard.

<!-- generated:failure -->
- **Self-consistency checks:** 2 independent re-answers per case; low agreement flags the case for review. Low
  consistency marks uncertainty, not certain error: 57% of low-consistency answers were still correct.
- **Cross-modal agreement:** the verifier sees the image, so "explanation grounded in evidence" and "no
  contradictions" are checked against the image itself, not only a text description.
- **Verifier model:** a different, larger model (`gpt-6.1-sol`) verifies `gpt-6-luna`'s answers and steers
  114 of 300 episodes away from the direct path.
- **Disagreement analysis:** the three systems disagree on 68% of cases under the same matching rules;
  wins (only one system right): agentic 23, single-shot 24, workflow 13.

Representative cases, chosen automatically by `scripts/final_metrics.py` (screenshots in `eval-dashboard-screenshots/`):

| Case | What happened |
|---|---|
| 1895 "is the lesion on the left or right side of the br…" | A focused re-look got it right; the single prompt did not (agentic: "The lesion is on the patient's left sid…"; single prompt: "The lesion is on the right side of the …"; reference: "left") |
| 1794 "is there airspace consolidation on the left side?" | The verifier's guidance fixed the answer; the single prompt was wrong (agentic: "Yes, there is left perihilar airspace c…"; single prompt: "No, there is no definite airspace conso…"; reference: "yes") |
| 1806 "is there evidence of any abnormalities of the lun…" | The loop re-examined or revised and still ended wrong, while the single prompt was right (agentic: "yes"; single prompt: "No, there is no focal lung parenchymal …"; reference: "no") |
| 1851 "is there fluid in the lung?" | Low self-consistency flagged the case for review, though the final answer was right (agentic: "yes"; single prompt: "Yes, there is fluid at the left lung ba…"; reference: "yes") |
<!-- /generated:failure -->

**Systematic failure patterns**

1. **Shared perception blind spot:** when the looker and the verifier misread the image the same way, no check catches
   it; self-consistency flags some of these cases.
2. **Abstention withholds right answers:** when the blind reading disagrees, or the review judges one image
   insufficient, the loop abstains, even where its own last answer was right (see the mechanisms table above).
3. **The hardest questions resist re-looking:** zoomed re-looks concentrate on the hardest questions and do not lift
   them above the single prompt; occasionally a revision breaks a right answer, which keeping the best-verified answer
   limits.
4. **Benchmark quirks:** some questions cannot be answered from their image (e.g. a diagnosis that needs another
   modality); every system fails them.
5. **Workflow rigidity:** fixed rules answer "Brain" or "X-ray" where a specific finding is needed.

---

## Reflection and limitations

**When is agentic behaviour beneficial?** In this run, not for accuracy. The agentic loop is not more accurate than
a single prompt: the differences are small and not significant on either score, and its larger verifier, answering
alone, is significantly more accurate. Its mechanisms work individually: revising fixes far more answers than it
breaks, and when the blind check objects, the first answer is usually wrong. But abstention withholds about as many
correct answers as revision gains, and the loop's answers are not more stable between runs. What it adds is
*calibrated confidence*: answers the check verifies are far more often right than the rest, so "verified" is a usable
signal for deciding which answers need review, and every decision is auditable. That costs roughly 60 times as much as
a single prompt. Answering with a low-confidence flag instead of abstaining, and a cheaper verifier, are the next
steps this run points to.

**Limitations**

<!-- generated:limit-cost -->
- About 59× the cost and 10× the (median) latency of a single prompt.
<!-- /generated:limit-cost -->
- The verifier shares the perception weaknesses of vision models; agreement is not correctness.
- The loop's accuracy cannot be fully separated from its verifier's, because the verifier's guidance may suggest
  answers. A critique-only verifier (issues, no suggested answer) would isolate the loop's own contribution.
- Abstentions are scored as wrong, so the abstention policy lowers accuracy by design.
- The loop's answers are not more stable between runs than a single prompt's.
- 300 questions give about ±5-point intervals; per-type results for small categories are noisy.
- The strict scorer under-credits paraphrases and over-credits vague answers that contain the reference word; the
  LLM judge corrects both, but it is a model, spot-checked rather than human-annotated.
- For answers that are not abstentions, the yes/no normaliser maps "cannot determine" to "no".
- The workflow baseline uses a question-agnostic image description by design.
- The system uses no retrieval, so it cannot draw on knowledge that is not visible in the image.

---

## User interface

- **Dashboard:** upload an image, ask a question, and compare the agentic loop with both baselines. The agentic view
  shows the answer, explanation, verification checks, failure categories and the full trace.
- **Case Explorer:** browse the evaluation questions; ones in the current run open instantly, others run live.
- **Evaluation:** strict and LLM-judged accuracy for all four systems, close- and open-ended accuracy, accuracy by
  question type and by controller path, disagreement, and a filterable table of every case.

---

## Quick start

**Prerequisites:** Python 3.12, Node.js 18+, and an OpenAI API key with access to the models in `.env`.

```bash
git clone https://github.com/clarence-ck/med-vqa-rad-agentic.git
cd med-vqa-rad-agentic
cp .env.example .env            # set OPENAI_API_KEY; per-role models are documented in the file

conda env create -f environment.yml   # recommended (native deps on Windows)
conda activate medvisionquery
# or: pip install -r backend/requirements.txt

python scripts/download_dataset.py    # VQA-RAD → data/vqa-rad/

cd backend && uvicorn main:app --port 8000      # terminal 1
cd frontend && npm install && npm run dev       # terminal 2
```

<!-- generated:question-cost -->
Open http://localhost:5173. Each new question runs all three systems and costs about $0.01 with the default models.
<!-- /generated:question-cost -->

Run the tests from `backend/`: `python -m unittest discover tests` (no API calls). To rebuild every result file,
figure, screenshot and the report from the cached results without spending anything:
`python scripts/run_all.py --stages metrics,docs,diagram,screenshots,report`.

---

## Reproducing the evaluation

<!-- generated:reproduce -->
```bash
python scripts/run_all.py --dry-run         # show the stages and the estimated cost; spends nothing
python scripts/run_all.py                   # everything: evaluation, repeats, judge, metrics, docs, figures, report
python scripts/run_all.py --stages metrics,docs,diagram,report   # rebuild outputs from cached results (free)
```

| Stage | What it runs | Last cost | Last wall clock |
|---|---|---|---|
| `eval` | `run_case_explorer_cache.py --ablation --concurrency 4` (300 questions × 4 systems) | $6.25 | ~71 min |
| `variance` | `repeat_cases.py --from-eval-sample 300 --n 100` (baseline + agentic loop again) | $1.28 | ~83 min |
| `judge` | `judge_open_ended.py` | $1.09 | ~10 min |
| `metrics` | `final_metrics.py` → `final_metrics.json` and `cases.csv` in the run folder | free | seconds |
| `docs` | `update_docs.py` → README blocks and `docs/report/results.tex` | free | seconds |
| `diagram` | `system_diagram.py` → `docs/report/system_diagram.png` (README and report) | free | seconds |
| `screenshots` | starts backend + frontend, `capture-screenshots.mjs` for the showcase cases | free | ~1 min |
| `report` | `tectonic main.tex` → `docs/report/MGAI_Final_Report_Group9_revised.pdf` | free | ~1 min |
<!-- /generated:reproduce -->

- Paid stages ask for confirmation (or `--yes`) and share one spend cap per run, `--budget` (default $10). The judge
  runs before the variance repeats, so if money runs short only the repeats are cut.
- An interrupted run (crash, closed laptop, budget reached) continues with `python scripts/run_all.py --resume`:
  finished cases are kept, and the README and report are only rebuilt once the run is complete.
- `screenshots` needs Node.js and a local Chrome or Edge; `report` needs `tectonic` (included in `environment.yml`).
- Two things stay manual after a full re-run: re-checking the judge spot check
  (`judge_review_sample.md` in the run folder) and re-reading the narrative in *Systematic failure patterns* and the
  report.
- Every cached result records the git commit, models, reasoning efforts and token usage. Failed API calls are never
  cached; the runner retries them and counts their cost toward the cap.

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
  routes/                      /api/analyze*, /api/dataset/* (samples, analysis, evaluation)
  utils/                       scoring, answer normalisation, usage tracking, data loading
  tests/                       unit tests (no API calls)
  utils/runs.py                where evaluation runs live; which run is current
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
| POST | `/api/analyze` | Agentic system (`AGENT_MODE=loop` by default, `pipeline` for the ablation); multipart `image`, `question` |
| POST | `/api/analyze/baseline` | Single-shot baseline |
| POST | `/api/analyze/workflow` | Workflow baseline |
| GET | `/api/dataset/samples` | The evaluation questions (`experiments/eval_set.json`) |
| GET | `/api/dataset/sample/{id}` | One case with its image |
| GET | `/api/dataset/sample/{id}/analysis?refresh=&ablation=` | Run (or load cached) all systems on a case |
| GET | `/api/dataset/evaluation?run=` | Aggregate metrics and per-case rows of a run (default: the current run) |

## License

MIT
