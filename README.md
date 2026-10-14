# MedVisionQuery — an agentic multimodal system for medical visual question answering

MedVisionQuery answers natural-language questions about radiology images (X-ray, CT, MRI). Its agentic system is a
**Haystack Agent controller** that decides, case by case, how to look at the image, answer, check the answer *against
the image*, and whether to re-examine a region, revise, abstain or finish. It is evaluated against a single-prompt
baseline, a hand-designed workflow and a fixed-pipeline ablation on 200 random VQA-RAD test questions.

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
- [What changed since the April 2026 version](#what-changed-since-the-april-2026-version)
- [User interface](#user-interface)
- [Quick start](#quick-start)
- [Reproducing the evaluation](#reproducing-the-evaluation)
- [Project structure](#project-structure)
- [API reference](#api-reference)

---

## Results at a glance

<!-- generated:glance -->
Run `2026-10-14_1330`: 200 seeded-random VQA-RAD test questions (111 close-ended, 89 open-ended), all systems on the same answering model
(`gpt-6-luna`; the agentic loop and the fixed pipeline also use `gpt-6.1-sol` as verifier). Per-case times were
measured with four cases running at once; the mean is pulled up by a few long re-look cases (actual run times are in
[Run timings](#run-timings)). "Strict" uses the rule-based scorer; "judged" also accepts open-ended paraphrases
confirmed by a blinded LLM judge (see [Evaluation design](#evaluation-design)).

| System | Strict accuracy | Judged accuracy | Time / case (mean · median) | Cost / case |
|---|---|---|---|---|
| Single-shot baseline | 63.5% | 70.0% | 4.4 s · 3.6 s | $0.0002 |
| Workflow baseline (hand-designed) | 28.0% | 31.0% | 9.0 s · 8.7 s | $0.0005 |
| **Agentic controller loop** | **64.5%** | **72.0%** | 37.2 s · 29.3 s | $0.0094 |
| Fixed pipeline (ablation) | 57.0% | 61.5% | 38.1 s · 29.4 s | $0.0070 |

- **Agentic vs single-shot: a tie** (exact McNemar p = 0.86 strict, 0.61 judged).
- **Agentic vs fixed pipeline:** +7.5 points strict (p = 0.02), +10.5 judged (p = 0.002).
- **Reliability:** verification and self-consistency signals predict errors, the loop can abstain, and
  2% of its answers flip between runs vs 12% for the single prompt.
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

![System diagram](system_diagram.png)

### Controller loop

A fresh Haystack `Agent` is created per request (`backend/agents/agentic_loop.py`). Its tools are bound to a
per-request working memory that holds the image, the question analysis, every look, every answer and every check.
The controller sees only short text results from each tool and chooses the next call. There is one agent (the
controller); the five tools are functions it calls, not agents. `check` only *recommends* the next step; the
controller decides whether to follow it, which is what separates the loop from the fixed-order ablation.

| Tool | Role | Model (default) | Notes |
|---|---|---|---|
| `analyze_question()` | Task understanding | gpt-6-luna | Question type and expected answer format; keyword rules use whole-word matching |
| `look(focus)` | Perception (executor) | gpt-6-luna, vision | Looks *with the question in mind*; with `focus`, re-examines a named region. The first look also rejects non-radiology images |
| `answer(guidance)` | Synthesis (executor) | gpt-6-luna, vision | Receives the image plus all observations; `guidance` carries what the verifier asked to fix |
| `check()` | Evaluator | gpt-6.1-sol, vision | Verifies against the image and recommends `accept`, `relook` (with a focus), `reanswer` (with guidance) or `abstain` |
| `finish(abstain)` | Decision | — | Ends the episode; refused while the latest answer is unchecked |

Budgets: at most 2 focused re-looks, 3 answers and 16 agent steps. If the controller stalls, a safety net answers
and/or checks before finishing. Paths observed on the 200-case evaluation:

<!-- generated:paths -->
| Path | Cases | Tool sequence |
|---|---|---|
| direct | 141 | analyze → look → answer → check → finish |
| relook | 19 | … → check → look(focus) → answer(guided) → check → finish |
| reanswer | 29 | … → check → answer(guided) → check → finish |
| abstained | 11 | … → check → finish(abstain) |
<!-- /generated:paths -->

### Why these models and tools

- **Haystack `Agent` + `Tool`** gives a tool-calling loop with typed tools and exit conditions. Keeping the image in
  per-request memory (not in the controller's context) keeps the controller cheap.
- **One model family (OpenAI), roles sized by need.** The controller and question analysis are simple decisions, so
  they use `gpt-6-luna` with low or no reasoning effort. Answering uses `gpt-6-luna` with medium effort, and the answerer always sees
  the image: in a 20-case pilot that alone raised accuracy from 50% to 70%.
- **A different model verifies.** `check` (and the evaluation judge) use `gpt-6.1-sol`, so the model that answered
  does not verify its own answer.
- All roles are configurable (`MODEL_<ROLE>`, `REASONING_EFFORT_<ROLE>` in `.env`); every call's tokens and cost are
  recorded.

### Baselines and ablation

| System | File | What it does |
|---|---|---|
| Single-shot baseline | `backend/agents/baseline.py` | One vision prompt; asked for a concise answer that starts with yes/no where applicable |
| Workflow baseline | `backend/agents/workflow_baseline.py` | Fixed chain: question-agnostic image description → question analysis → hand-written rules per question type |
| Fixed pipeline (ablation) | `backend/agents/orchestrator.py` | The original six stages (interpret → analyze → filter → route → reason → verify, with retry) in a fixed order, with the reasoner given the image; isolates what the controller's decisions add |

---

## Multimodal grounding

The two modalities are the image and the text question. They interact at every step that matters:

| Step | Image | Text |
|---|---|---|
| `look(focus)` | ✓ sent at high detail | the question (and focus) decide what is examined |
| `answer(guidance)` | ✓ sent directly | question analysis, all observations, verifier guidance |
| `check()` | ✓ sent directly | answer, explanation, observations |
| Controller | ✗ | short tool results only |

The April version described the image once, *without the question*, and every later stage worked from that text. In a
20-case pilot, giving the reasoner the image raised accuracy from 10/20 to 14/20; typical text-only failures were
"unclear" or "cannot be determined from the provided image description".

---

## Control and failure handling

| Mechanism | What it does |
|---|---|
| Image-grounded verification | `check` judges the answer against the image: 5 checks, a status (`verified`, `partially_verified`, `low_confidence`, `unsupported`) and a recommended next action |
| Targeted recovery | The controller re-examines a region (`look(focus)`) or revises (`answer(guidance)`) instead of re-running everything |
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

- **Data:** the VQA-RAD test split (451 questions); a seeded random sample of 200 (`EVAL_SAMPLE_SEED=42`), excluding
  one known-bad item. The sample follows the natural mix (56% yes/no), so per-type numbers for small categories are
  noisy.
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
- **LLM judge** (`scripts/judge_open_ended.py`): for open-ended answers the strict scorer rejected, `gpt-6.1-sol`
  judges meaning equivalence, blinded to the system (answers shuffled and anonymised). A 30-verdict spot check agreed
  on 29/30 credit decisions (AI-assisted review, `judge_review_sample.md` in the run folder).
- **Statistics:** 95% intervals are about ±6.7 points at n = 200; paired comparisons use the exact McNemar test.
- **Variance:** the baseline and agentic loop were re-run on 50 of the evaluated cases.
<!-- generated:cost -->
- **Cost:** every OpenAI call is recorded (`backend/utils/usage.py`): evaluation $3.43, repeats
  $0.54, judge $0.11 ($4.08 in total).

### Run timings

Wall-clock time of each stage of run `2026-10-14_1330`, recorded by `scripts/run_all.py`
(`experiments/runs/2026-10-14_1330/run.json`):

| Stage | Started | Finished | Wall clock | Work | Cost |
|---|---|---|---|---|---|
| Main evaluation (4 systems) | 2026-10-14 13:30:51 | 14:10:55 | 40 min 4 s | 199 cases (1 cached earlier), 4 at a time | $3.43 |
| Variance repeats (baseline + agentic) | 2026-10-14 14:10:55 | 14:45:36 | 34 min 41 s | 50 cases, one at a time | $0.54 |
| LLM judge (`gpt-6.1-sol`) | 2026-10-14 15:20:09 | 15:26:28 | 6 min 19 s | 82 open-ended cases | $0.11 |

Per-case times in the results tables are each system's own latency inside a case; a full case (all four systems in
turn) took 45.9 s on average.
<!-- /generated:cost -->

---

## Results

<!-- generated:results -->
### Overall

| System | Strict | Judged | Close-ended (111) | Open-ended (89), strict → judged |
|---|---|---|---|---|
| Single-shot baseline | 63.5% | 70.0% | 76.6% | 47.2% → 61.8% |
| Workflow baseline | 28.0% | 31.0% | 42.3% | 10.1% → 16.9% |
| **Agentic loop** | **64.5%** | **72.0%** | 76.6% | 49.4% → 66.3% |
| Fixed pipeline (ablation) | 57.0% | 61.5% | 68.5% | 42.7% → 52.8% |

Paired comparisons (cases where exactly one of the two systems is right):

| Comparison | Strict | Judged |
|---|---|---|
| Agentic vs single-shot | 17 vs 15, p = 0.86 | 19 vs 15, p = 0.61 |
| Agentic vs fixed pipeline | 26 vs 11, p = 0.02 | 32 vs 11, p = 0.002 |

### By controller path

| Path | Cases | Agentic | Single-shot | Fixed pipeline |
|---|---|---|---|---|
| direct | 141 | 74% | 72% | 67% |
| relook | 19 | 47% | 37% | 37% |
| reanswer | 29 | 55% | 45% | 31% |
| abstained | 11 | — (abstained) | 45% | 36% |

![Accuracy by controller path](eval-dashboard-screenshots/03-accuracy-by-controller-path.png)

### Uncertainty signals predict errors

| Signal | Levels | Agentic accuracy |
|---|---|---|
| Verification | verified (169) / partially verified (25) / unsupported or low confidence (6) | 70% / 40% / 0% |
| Self-consistency | high (162) / medium or low (27) | 72% / 40–53% |
| Abstention | 11 cases, on which the single-shot baseline is wrong 55% of the time | — |

On the 189 cases it answered, the agentic loop is 68% correct (strict).

### Stability across runs (same 50 cases)

| System | Run 1 | Run 2 | Answers that flipped right ↔ wrong |
|---|---|---|---|
| Single-shot baseline | 33/50 | 33/50 | 6 (12%) |
| Agentic loop | 35/50 | 34/50 | 1 (2%) |
<!-- /generated:results -->

### Development pilots (20 cases, seed 42)

Small runs that guided the design, in order (`experiments/pilots/`, described in `experiments/README.md`):

| Pilot folder | What it tested | Result | Single-shot baseline, same run |
|---|---|---|---|
| `1-reasoner-sees-image` | Fixed pipeline: text-only reasoner (April design) vs reasoner that sees the image | 10/20 → 14/20 | 14/20 |
| `2-loop-initial` | First agentic controller loop | 14/20 | 15/20 |
| `3-loop-best-verified` | Loop after keeping the best-verified answer | 14/20 | 14/20 |
| `4-workflow-baseline` | Rewritten hand-designed workflow baseline | 8/20 | – |

---

## Failure analysis

The four analyses Template 1 asks for are built into every agentic run and summarised on the dashboard.

<!-- generated:failure -->
- **Self-consistency checks:** 2 independent re-answers per case; low agreement flags the case for review. Low
  consistency marks uncertainty, not certain error: 53% of low-consistency answers were still correct.
- **Cross-modal agreement:** the verifier sees the image, so "explanation grounded in evidence" and "no
  contradictions" are checked against the image itself, not only a text description.
- **Verifier model:** a different, larger model (`gpt-6.1-sol`) verifies `gpt-6-luna`'s answers and steers
  59 of 200 episodes away from the direct path.
- **Disagreement analysis:** the three systems disagree on 65% of cases under the same matching rules;
  wins (only one system right): agentic 16, single-shot 12, workflow 7.

Representative cases, chosen automatically by `scripts/final_metrics.py` (screenshots in `eval-dashboard-screenshots/`):

| Case | What happened |
|---|---|
| 1900 "is there an acute infarction?" | A focused re-look got it right; the single prompt did not (agentic: "yes"; single prompt: "No, there is no definite acute infarcti…"; reference: "yes") |
| 1794 "is there airspace consolidation on the left side?" | The verifier's guidance fixed the answer; the single prompt was wrong (agentic: "Yes—there is airspace consolidation in …"; single prompt: "No, there is no focal airspace consolid…"; reference: "yes") |
| 1825 "is this supratentorial or infratentorial?" | The loop re-examined or revised and still ended wrong, while the single prompt was right (agentic: "yes"; single prompt: "Supratentorial."; reference: "supratentorial") |
| 1866 "are there increased vascular markings" | Low self-consistency flagged the case for review, though the final answer was right (agentic: "yes"; single prompt: "Yes, the central/perihilar vascular mar…"; reference: "yes") |
<!-- /generated:failure -->

**Systematic failure patterns**

1. **Shared perception blind spot:** when the looker and the verifier misread the image the same way, no check catches
   it; self-consistency flags some of these cases.
2. **Over-reading on re-looks:** a focused look can talk the system into a finding; keeping the best-verified
   answer limits the damage.
3. **Benchmark quirks:** some questions don't match their image (e.g. a "right upper quadrant mass" asked of a chest
   film); every system fails them.
4. **Workflow rigidity:** fixed rules answer "Brain" or "X-ray" where a specific finding is needed (28%).

---

## Reflection and limitations

**When is agentic behaviour beneficial?** Not for raw accuracy on this benchmark: a strong single prompt on the same
model is equally accurate. The agentic system earns its cost when answers must be *trustworthy*. It knows when it is
unsure (verification and consistency signals predict errors), it can abstain, it recovers on hard cases by
re-examining specific regions, it is far more stable across runs, and every decision is auditable. Against a fixed
multi-stage pipeline with the same models, letting a controller decide is clearly better.

**Limitations**

<!-- generated:limit-cost -->
- About 41× the cost and 8× the (median) latency of a single prompt for similar accuracy.
<!-- /generated:limit-cost -->
- The verifier shares the perception weaknesses of vision models; agreement is not correctness.
- 200 cases give ±6.7-point intervals; per-type results for small categories are noisy.
- The strict scorer under-credits paraphrases; the LLM judge is an upper bound, spot-checked but not human-annotated.
- For non-abstained answers, the yes/no normaliser maps "cannot determine" to "no" (no measurable effect in this run).
- The workflow baseline uses a question-agnostic image description by design.
- Retrieval over SLAKE was removed: it returned other patients' question–answer pairs, which did not help
  perception-bound questions (see below).

---

## What changed since the April 2026 version

The April report's numbers (80% / 65% / 75%; its screenshots, logs and cached answers were removed and remain in git
history at commit `b804160`) came from
a pipeline and a scorer with defects found in a later review. Re-scoring the cached April answers with the fixed
scorer gives roughly 62% / 30% / 50%. The main changes:

| Area | April 2026 | Now |
|---|---|---|
| Scorer | Embedding similarity ≥ 0.85 accepted "left lung" for "right lung" | Rule-based strict scorer + blinded LLM judge |
| Question classifier | Substring `"ct"` matched "fra**ct**ure", forcing 72/451 questions to *modality* | Whole-word matching |
| Agent | Fixed six-stage sequence; the controller never decided anything | Controller loop with look / answer / check / finish |
| Image use | Image described once, without the question | Question-aware looks; answerer and verifier see the image |
| Verifier | Same model, text only; crashed on counting questions | Different model, sees the image, recommends the next action |
| Retrieval | SLAKE hybrid RAG | Removed (no measurable benefit; contradictory look-alike cases) |
| Self-consistency | Ratio of verifier checks passed | Independent re-answers |
| Evaluation | 200 hand-ordered cases, grouped by the agent's own labels | 200 seeded-random cases, dataset labels, ablation, variance repeats |

---

## User interface

- **Dashboard:** upload an image, ask a question, and compare the agentic loop with both baselines. The agentic view
  shows the answer, explanation, verification checks, failure categories and the full trace.
- **Case Explorer:** browse the 200 evaluation cases; cached cases open instantly, others run live.
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
| `eval` | `run_case_explorer_cache.py --ablation --concurrency 4` (200 cases × 4 systems) | $3.43 | ~40 min |
| `variance` | `pilot.py --from-eval-sample 200 --n 50` (baseline + agentic loop again) | $0.54 | ~35 min |
| `judge` | `judge_open_ended.py` | $0.11 | ~6 min |
| `metrics` | `final_metrics.py` → `final_metrics.json` and `cases.csv` in the run folder | free | seconds |
| `docs` | `update_docs.py` → README blocks and `docs/report/results.tex` | free | seconds |
| `diagram` | `system_diagram.py` → `system_diagram.png` (+ copy for the report) | free | seconds |
| `screenshots` | starts backend + frontend, `capture-screenshots.mjs` for the showcase cases | free | ~1 min |
| `report` | `tectonic main.tex` → `docs/report/MGAI_Final_Report_Group9_revised.pdf` | free | ~1 min |
<!-- /generated:reproduce -->

- Paid stages ask for confirmation (or `--yes`) and share one spend cap per run, `--budget` (default $5). The judge
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
experiments/pilots/            20-case development pilots
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
| GET | `/api/dataset/samples` | The 200 evaluation cases |
| GET | `/api/dataset/sample/{id}` | One case with its image |
| GET | `/api/dataset/sample/{id}/analysis?refresh=&ablation=` | Run (or load cached) all systems on a case |
| GET | `/api/dataset/evaluation?run=` | Aggregate metrics and per-case rows of a run (default: the current run) |

## License

MIT
