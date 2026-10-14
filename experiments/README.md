# Experiments

Raw results behind the numbers in the [README](../README.md) and the report. The 200-case evaluation itself lives in
`backend/.case_explorer_cache/` (one JSON per case); everything here is computed from or alongside it.

| Path | What it is | Produced by |
|---|---|---|
| `final_metrics.json` | Every headline number (accuracy, McNemar tests, signals, variance, costs, run timings) | `scripts/final_metrics.py` |
| `judge/judgments.json` | Blinded `gpt-6.1-sol` verdicts on the 82 strict-wrong open-ended cases | `scripts/judge_open_ended.py` |
| `judge/review_sample.md` | 30-verdict spot check of the judge (29/30 agreed) | manual review |
| `variance_repeat/results.json` | Baseline and agentic loop re-run on 50 evaluated cases | `scripts/pilot.py --from-eval-sample 200 --n 50 --name variance_repeat` |
| `pilots/` | 20-case development pilots (seed 42), in the order they were run | `scripts/pilot.py` |

## Pilots

| Folder | Question | Result (of 20) | Decision |
|---|---|---|---|
| `1-reasoner-sees-image` | Does the old fixed pipeline fail because its reasoner never sees the image? | text-only 10, with image 14, baseline 14 | Give the answerer the image |
| `2-loop-initial` | First agentic controller loop vs single-shot | loop 14, baseline 15 | Later answers sometimes replaced better ones |
| `3-loop-best-verified` | Loop that keeps the best-verified answer | loop 14, baseline 14 | Kept for the final evaluation |
| `4-workflow-baseline` | Rewritten hand-designed workflow baseline | 8 | Kept as the non-agentic baseline |

Each pilot's `results.json` holds the seed, the total cost, and per-case answers, correctness, time and cost for each system.
