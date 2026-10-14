"""Compute every headline number quoted in the README and the report, for one evaluation run.

Uses the same evaluation code as the dashboard (routes/dataset.get_evaluation_summary), plus the run's
variance repeats and LLM-judge verdicts. Writes final_metrics.json and cases.csv into the run folder
(experiments/runs/<run_id>/).

Usage (run from repo root):
    python scripts/final_metrics.py                    # the current run
    python scripts/final_metrics.py --run-id 2026-10-16_0930
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "backend"))

from scipy.stats import binomtest  # noqa: E402

from routes import dataset as ds  # noqa: E402
from utils import runs  # noqa: E402

SYSTEMS = ("baseline", "workflow", "agentic", "pipeline")
CSV_COLUMNS = ["sample_id", "question", "ground_truth_answer", "answer_style", "controller_path",
               "verification_status", "self_consistency_level"] + [
    f"{system}_{field}" for system in SYSTEMS for field in ("answer", "correct", "judged_correct")]


def pct(x, digits=1):
    return None if x is None else round(100 * x, digits)


def pct2(x):
    """Two decimals, for values the docs round again to whole percents (avoids double rounding)."""
    return pct(x, 2)


def mcnemar(rows, a, b):
    only_a = sum(1 for r in rows if r[a] and r[b] is False)
    only_b = sum(1 for r in rows if r[b] and r[a] is False)
    return {"only_first": only_a, "only_second": only_b,
            "p": round(binomtest(only_a, only_a + only_b, 0.5).pvalue, 3)}


def showcase_cases(rows):
    """Four representative cases for the README trace table and the screenshots, chosen by fixed rules."""
    used = set()

    def first(predicate):
        row = next((r for r in sorted(rows, key=lambda r: r["sample_id"]) if r["sample_id"] not in used and predicate(r)), None)
        if row:
            used.add(row["sample_id"])
        return row

    picks = [
        ("relook_win", "A focused re-look got it right; the single prompt did not",
         lambda r: r["controller_path"] == "relook" and r["agentic_correct"] and not r["baseline_correct"]),
        ("reanswer_win", "The verifier's guidance fixed the answer; the single prompt was wrong",
         lambda r: r["controller_path"] == "reanswer" and r["agentic_correct"] and not r["baseline_correct"]),
        ("recovery_failure", "The loop re-examined or revised and still ended wrong, while the single prompt was right",
         lambda r: r["controller_path"] in ("relook", "reanswer") and not r["agentic_correct"] and r["baseline_correct"]),
        ("flagged_correct", "Low self-consistency flagged the case for review, though the final answer was right",
         lambda r: r["self_consistency_level"] == "low" and r["agentic_correct"]),
    ]
    out = []
    for kind, story, predicate in picks:
        row = first(predicate)
        if row:
            out.append({"kind": kind, "story": story, "sample_id": row["sample_id"], "question": row["question"],
                        "ground_truth": row["ground_truth_answer"], "agentic_answer": row["agentic_answer"],
                        "baseline_answer": row["baseline_answer"], "controller_path": row["controller_path"],
                        "verification_status": row["verification_status"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default=None, help="Run under experiments/runs/ (default: the current run)")
    run_id = parser.parse_args().run_id or runs.current_run_id()
    if not run_id:
        sys.exit("No run yet: start one with python scripts/run_all.py")
    run_dir = runs.run_dir(run_id)
    result = asyncio.run(ds.get_evaluation_summary(run=run_id))
    s, rows = result["summary"], result["rows"]
    cache = [json.load(open(f, encoding="utf-8")) for f in glob.glob(str(runs.cases_dir(run_id) / "*.json"))]

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
    metrics["disagreement_pct"] = pct2(s["disagreement_rate"])
    metrics["controller_paths"] = {
        b["label"]: {"cases": b["count"], "agentic": pct2(b["agentic_accuracy"]), "baseline": pct2(b["baseline_accuracy"]),
                     "workflow": pct2(b["workflow_accuracy"]), "pipeline": pct2(b["pipeline_accuracy"])}
        for b in s["by_controller_path"]
    }

    def signal(key, groups):
        out = {}
        for label, values in groups.items():
            subset = [r for r in rows if r[key] in values]
            out[label] = {"cases": len(subset), "agentic": pct2(sum(r["agentic_correct"] for r in subset) / len(subset)) if subset else None}
        return out
    metrics["verification_signal"] = signal("verification_status", {
        "verified": {"verified"}, "partially_verified": {"partially_verified"}, "unsupported_or_low": {"unsupported", "low_confidence"}})
    metrics["self_consistency_signal"] = signal("self_consistency_level", {
        "high": {"high"}, "medium": {"medium"}, "low": {"low"}, "not_run": {"not_run"}})
    answered = [r for r in rows if r["controller_path"] != "abstained"]
    metrics["agentic_answered"] = {"cases": len(answered), "strict": pct2(sum(r["agentic_correct"] for r in answered) / len(answered))}

    variance_file = run_dir / "variance.json"
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
    judge_file = run_dir / "judgments.json"
    metrics["costs_usd"] = {
        "evaluation_run": round(run_cost, 2),
        "variance_repeats": round(json.load(open(variance_file, encoding="utf-8")).get("total_cost_usd", 0), 2) if variance_file.exists() else None,
        "judge": round(json.load(open(judge_file, encoding="utf-8")).get("cost_usd", 0), 2) if judge_file.exists() else None,
    }
    # Wall-clock times of each stage, recorded by scripts/run_all.py.
    run_info = runs.read_run(run_id)
    metrics["run"] = {k: run_info.get(k) for k in ("run_id", "status", "started", "finished", "code_version")}
    metrics["run_timings"] = {k: v for k, v in run_info.get("stages", {}).items() if k in ("eval", "variance", "judge")}
    metrics["showcase"] = showcase_cases(rows)
    agentic, baseline = metrics["systems"]["agentic"], metrics["systems"]["baseline"]
    baseline_cost = statistics.mean(p["usage"]["baseline"]["cost_usd"] for p in cache)
    agentic_cost = statistics.mean(p["usage"]["agentic"]["cost_usd"] for p in cache)
    metrics["agentic_vs_baseline"] = {
        "cost_ratio": round(agentic_cost / baseline_cost, 1),
        "mean_latency_ratio": round(agentic["mean_time_s"] / baseline["mean_time_s"], 1),
        "median_latency_ratio": round(agentic["median_time_s"] / baseline["median_time_s"], 1),
    }
    metrics["case_wall_time_mean_s"] = round(statistics.mean(p["duration_ms"] for p in cache) / 1000, 1)

    output = run_dir / "final_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(run_dir / "cases.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["sample_id"]))
    print(f"Wrote {output.relative_to(ROOT_DIR)} and cases.csv: " + ", ".join(
        f"{k} {v['strict']}% strict / {v['judged']}% judged" for k, v in metrics["systems"].items()))


if __name__ == "__main__":
    main()
