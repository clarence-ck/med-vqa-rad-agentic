# Experiments

| Path | What it is |
|---|---|
| `runs/` | Stores one folder for each evaluation. `scripts/run_all.py` creates the folders, and `runs/README.md` lists them. |
| `eval_set.json` | Stores the evaluation questions selected by `scripts/make_eval_set.py`. |

## A run folder (`runs/<run_id>/`)

| File | What it is | Written by |
|---|---|---|
| `run.json` | Records whether the run is active or complete, its start and finish times, code version, models, budget, and the time and cost of each stage. | `run_all.py` |
| `cases/` | Contains one JSON file per case, including each system's answer, decision trace, timing, and token use. | `run_case_explorer_cache.py` |
| `judgments.json` | Stores blinded `gpt-6-astra` judgments for every open-ended answer. | `judge_open_ended.py` |
| `variance.json` | Stores an independent repeat of the single-shot baseline and agentic loop on 100 cases from the main evaluation. | `repeat_cases.py` |
| `final_metrics.json` | Stores the main results, including accuracy, McNemar tests, confidence signals, repeat results, costs, timings, and example cases. | `final_metrics.py` |
| `cases.csv` | Lists every system's answer and score for each case in a format that can be read without the app. | `final_metrics.py` |
| `judge_review_sample.md` | Compares 30 randomly selected judge decisions with the reference answers. | AI-assisted manual review; not created by the pipeline |

Starting a new evaluation does not change an older run. To continue an unfinished run, use
`python scripts/run_all.py --resume`.
