# Experiments

| Path | What it is |
|---|---|
| `runs/` | One folder per evaluation run, written by `scripts/run_all.py`; `runs/README.md` lists every run |
| `eval_set.json` | The evaluation questions (written by `scripts/make_eval_set.py`) |

## A run folder (`runs/<run_id>/`)

| File | What it is | Written by |
|---|---|---|
| `run.json` | Status (running / completed), start and finish, code version, models, budget, and per-stage wall-clock time and cost | `run_all.py` |
| `cases/` | One JSON per case: every system's answer, trace, timings and token usage | `run_case_explorer_cache.py` |
| `judgments.json` | Blinded `gpt-6.1-sol` verdicts on strict-wrong open-ended answers | `judge_open_ended.py` |
| `variance.json` | Baseline and agentic loop answered again on 50 of the cases | `repeat_cases.py` |
| `final_metrics.json` | Every headline number (accuracy, McNemar tests, signals, variance, costs, timings, showcase cases) | `final_metrics.py` |
| `cases.csv` | Per-case answers and correctness for all systems, readable without the app | `final_metrics.py` |
| `judge_review_sample.md` | Spot check of 30 judge verdicts (manual review) | by hand |

A run is never changed by a later run. An unfinished run is continued with `python scripts/run_all.py --resume`.

