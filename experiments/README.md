# Experiments

| Path | What it is |
|---|---|
| `runs/` | One folder per evaluation run, written by `scripts/run_all.py`; `runs/README.md` lists every run |
| `pilots/` | 20-case development pilots (seed 42) that guided the design, in the order they were run |

## A run folder (`runs/<run_id>/`)

| File | What it is | Written by |
|---|---|---|
| `run.json` | Status (running / completed), start and finish, code version, models, budget, and per-stage wall-clock time and cost | `run_all.py` |
| `cases/` | One JSON per case: every system's answer, trace, timings and token usage | `run_case_explorer_cache.py` |
| `judgments.json` | Blinded `gpt-6.1-sol` verdicts on strict-wrong open-ended answers | `judge_open_ended.py` |
| `variance.json` | Baseline and agentic loop answered again on 50 of the cases | `pilot.py` |
| `final_metrics.json` | Every headline number (accuracy, McNemar tests, signals, variance, costs, timings, showcase cases) | `final_metrics.py` |
| `cases.csv` | Per-case answers and correctness for all systems, readable without the app | `final_metrics.py` |
| `judge_review_sample.md` | Spot check of 30 judge verdicts (manual review) | by hand |

A run is never changed by a later run. An unfinished run is continued with `python scripts/run_all.py --resume`.

## Pilots

| Folder | Question | Result (of 20) | Decision |
|---|---|---|---|
| `1-reasoner-sees-image` | Does the old fixed pipeline fail because its reasoner never sees the image? | text-only 10, with image 14, baseline 14 | Give the answerer the image |
| `2-loop-initial` | First agentic controller loop vs single-shot | loop 14, baseline 15 | Later answers sometimes replaced better ones |
| `3-loop-best-verified` | Loop that keeps the best-verified answer | loop 14, baseline 14 | Kept for the final evaluation |
| `4-workflow-baseline` | Rewritten hand-designed workflow baseline | 8 | Kept as the non-agentic baseline |

Each pilot's `results.json` holds the seed, the total cost, and per-case answers, correctness, time and cost for each system.
