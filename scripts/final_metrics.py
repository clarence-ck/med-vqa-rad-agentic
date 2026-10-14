"""Compute every headline number quoted in the README and the report, from the cached results.

Uses the same evaluation code as the dashboard (routes/dataset.get_evaluation_summary), plus
the variance repeats and the LLM-judge file. Writes experiments/final_metrics.json.

Usage (run from repo root): python scripts/final_metrics.py
"""
from __future__ import annotations

import asyncio
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "backend"))

from scipy.stats import binomtest  # noqa: E402

from routes import dataset as ds  # noqa: E402

SYSTEMS = ("baseline", "workflow", "agentic", "pipeline")
OUTPUT = ROOT_DIR / "experiments" / "final_metrics.json"


def pct(x):
    return None if x is None else round(100 * x, 1)


def mcnemar(rows, a, b):
    only_a = sum(1 for r in rows if r[a] and r[b] is False)
    only_b = sum(1 for r in rows if r[b] and r[a] is False)
    return {"only_first": only_a, "only_second": only_b,
            "p": round(binomtest(only_a, only_a + only_b, 0.5).pvalue, 3)}


def main() -> None:
    result = asyncio.run(ds.get_evaluation_summary())
    s, rows = result["summary"], result["rows"]
    cache = [json.load(open(f, encoding="utf-8")) for f in glob.glob(str(ROOT_DIR / "backend" / ".case_explorer_cache" / "*.json"))]

    metrics = {"cases": s["total_cases"], "close_ended_cases": s["close_ended_cases"], "open_ended_cases": s["open_ended_cases"],
               "systems": {}}
    for system in SYSTEMS:
        durations = [r[f"{system}_duration_ms"] for r in rows if r.get(f"{system}_duration_ms") is not None]
        costs = [p["usage"][system]["cost_usd"] for p in cache if system in (p.get("usage") or {})]
        metrics["systems"][system] = {
            "strict": pct(s[f"{system}_accuracy"]),
            "judged": pct(s.get(f"{system}_judged_accuracy")),
            "close_ended": pct(s[f"{system}_close_ended_accuracy"]),
            "open_ended": pct(s[f"{system}_open_ended_accuracy"]),
            "open_ended_judged": pct(s.get(f"{system}_judged_open_ended_accuracy")),
            "mean_time_s": round(statistics.mean(durations) / 1000, 1),
            "median_time_s": round(statistics.median(durations) / 1000, 1),
            "mean_cost_usd": round(statistics.mean(costs), 4),
        }

    metrics["paired"] = {
        "agentic_vs_baseline_strict": mcnemar(rows, "agentic_correct", "baseline_correct"),
        "agentic_vs_baseline_judged": mcnemar(rows, "agentic_judged_correct", "baseline_judged_correct"),
        "agentic_vs_pipeline_strict": mcnemar(rows, "agentic_correct", "pipeline_correct"),
        "agentic_vs_pipeline_judged": mcnemar(rows, "agentic_judged_correct", "pipeline_judged_correct"),
    }
    metrics["wins"] = {"baseline": s["baseline_wins"], "workflow": s["workflow_wins"], "agentic": s["agentic_wins"], "ties": s["ties"]}
    metrics["disagreement_pct"] = pct(s["disagreement_rate"])
    metrics["controller_paths"] = {
        b["label"]: {"cases": b["count"], "agentic": pct(b["agentic_accuracy"]), "baseline": pct(b["baseline_accuracy"]),
                     "workflow": pct(b["workflow_accuracy"]), "pipeline": pct(b["pipeline_accuracy"])}
        for b in s["by_controller_path"]
    }

    def signal(key, groups):
        out = {}
        for label, values in groups.items():
            subset = [r for r in rows if r[key] in values]
            out[label] = {"cases": len(subset), "agentic": pct(sum(r["agentic_correct"] for r in subset) / len(subset)) if subset else None}
        return out
    metrics["verification_signal"] = signal("verification_status", {
        "verified": {"verified"}, "partially_verified": {"partially_verified"}, "unsupported_or_low": {"unsupported", "low_confidence"}})
    metrics["self_consistency_signal"] = signal("self_consistency_level", {
        "high": {"high"}, "medium": {"medium"}, "low": {"low"}, "not_run": {"not_run"}})
    answered = [r for r in rows if r["controller_path"] != "abstained"]
    metrics["agentic_answered"] = {"cases": len(answered), "strict": pct(sum(r["agentic_correct"] for r in answered) / len(answered))}

    variance_file = ROOT_DIR / "experiments" / "variance_repeat" / "results.json"
    if variance_file.exists():
        rep = {r["sample_id"]: r for r in json.load(open(variance_file, encoding="utf-8"))["rows"]}
        main_rows = {r["sample_id"]: r for r in rows}
        variance = {}
        for main_key, rep_key in (("baseline", "baseline"), ("agentic", "agentic_loop")):
            first = [main_rows[i][f"{main_key}_correct"] for i in rep]
            second = [rep[i][rep_key]["correct"] for i in rep]
            variance[main_key] = {"run1": sum(first), "run2": sum(second), "cases": len(rep),
                                  "flips": sum(a != b for a, b in zip(first, second))}
        metrics["variance"] = variance

    run_cost = sum(sum(v["cost_usd"] for v in (p.get("usage") or {}).values()) for p in cache)
    judge_file = ROOT_DIR / "experiments" / "judge" / "judgments.json"
    metrics["costs_usd"] = {
        "evaluation_run": round(run_cost, 2),
        "variance_repeats": round(json.load(open(variance_file, encoding="utf-8")).get("total_cost_usd", 0), 2) if variance_file.exists() else None,
        "judge": round(json.load(open(judge_file, encoding="utf-8")).get("cost_usd", 0), 2) if judge_file.exists() else None,
    }
    # Wall-clock times of the final runs (14 Oct 2026), copied from the runner logs; not stored in the result files.
    metrics["run_timings"] = {
        "main_evaluation": {"start": "13:30:51", "end": "14:10:55", "wall_clock_s": 2404, "cases": 199, "concurrency": 4},
        "variance_repeats": {"start": "14:10:55", "end": "14:45:36", "wall_clock_s": 2081, "cases": 50, "concurrency": 1},
        "judge": {"start": "15:20:09", "end": "15:26:28", "wall_clock_s": 379, "cases": 82},
    }
    agentic, baseline = metrics["systems"]["agentic"], metrics["systems"]["baseline"]
    baseline_cost = statistics.mean(p["usage"]["baseline"]["cost_usd"] for p in cache)
    agentic_cost = statistics.mean(p["usage"]["agentic"]["cost_usd"] for p in cache)
    metrics["agentic_vs_baseline"] = {
        "cost_ratio": round(agentic_cost / baseline_cost, 1),
        "mean_latency_ratio": round(agentic["mean_time_s"] / baseline["mean_time_s"], 1),
        "median_latency_ratio": round(agentic["median_time_s"] / baseline["median_time_s"], 1),
    }
    metrics["case_wall_time_mean_s"] = round(statistics.mean(p["duration_ms"] for p in cache) / 1000, 1)

    OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
