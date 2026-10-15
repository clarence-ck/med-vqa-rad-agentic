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
from utils import runs, scoring  # noqa: E402

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


def _similarity(first: str, second: str) -> float:
    a, b = ds._get_cached_embedding(first), ds._get_cached_embedding(second)
    if a is None or b is None:
        return 0.0
    return float(ds.cosine_similarity([a], [b])[0][0])


def strict_right(payload: dict, answer) -> bool:
    """Score any answer of a case (also intermediate ones) with the evaluation's strict scorer."""
    gt = ds._normalize_answer(payload.get("ground_truth_answer", ""))
    return scoring.answers_match(ds._normalize_answer(str(answer)), gt, scoring.is_close_ended(gt),
                                 payload["sample"].get("question_type"), _similarity)


def loop_answers(payload: dict) -> list[str]:
    return [(s.get("output") or {}).get("answer", "") for s in payload["agentic_result"].get("agent_trace", [])
            if s.get("agent") == "Reasoning & Answer"]


def showcase_cases(rows, cache=()):
    """Four representative cases for the README trace table and the screenshots, chosen by fixed rules."""
    by_id = {p["sample"]["id"]: p for p in cache}
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
        if row and kind == "recovery_failure" and row["sample_id"] in by_id:
            # Say so when a right intermediate answer was later replaced (the check may even verify the wrong one).
            payload = by_id[row["sample_id"]]
            answers = loop_answers(payload)
            if any(strict_right(payload, a) for a in answers[:-1]):
                verified = row["verification_status"] == "verified"
                story = ("An intermediate answer was right, but a later revision replaced it with a wrong one"
                         + (" that the check verified" if verified else "") + "; the single prompt was right")
        if row:
            out.append({"kind": kind, "story": story, "sample_id": row["sample_id"], "question": row["question"],
                        "ground_truth": row["ground_truth_answer"], "agentic_answer": row["agentic_answer"],
                        "baseline_answer": row["baseline_answer"], "controller_path": row["controller_path"],
                        "verification_status": row["verification_status"]})
    return out


def mechanisms(rows: list[dict], cache: list[dict]) -> dict:
    """What the loop's own mechanisms did: revisions, the blind check, abstention.

    Intermediate answers are scored with the same strict scorer as the final ones (word rules plus the
    S-PubMedBert fallback), so "fixed" and "broken" compare like with like.
    """
    by_id = {p["sample"]["id"]: p for p in cache}
    right = strict_right

    def trace(payload, agent):
        return [s.get("output") or {} for s in payload["agentic_result"].get("agent_trace", []) if s.get("agent") == agent]

    revised = fixed = broken = 0
    flagged = flagged_wrong = 0
    abstained = abstained_own_right = abstained_baseline_right = 0
    for r in rows:
        payload = by_id.get(r["sample_id"])
        if not payload:
            continue
        answers = [o.get("answer", "") for o in trace(payload, "Reasoning & Answer")]
        checks = trace(payload, "Verification")
        if r["controller_path"] in ("relook", "reanswer") and answers:
            revised += 1
            first = right(payload, answers[0])
            fixed += (not first) and bool(r["agentic_correct"])
            broken += first and not r["agentic_correct"]
        if checks and answers and "independent blind reading" in " ".join(checks[0].get("issues_found") or []):
            flagged += 1
            flagged_wrong += not right(payload, answers[0])
        if r["controller_path"] == "abstained":
            abstained += 1
            abstained_own_right += bool(answers) and right(payload, answers[-1])
            abstained_baseline_right += bool(r["baseline_correct"])
    answered = [r for r in rows if r["controller_path"] != "abstained"]
    meaning = sum(1 for p in cache if (p["agentic_result"].get("blind_reading") or {}).get("agreement_method") == "meaning")
    # Post-hoc (rule chosen after seeing the results): take the blind reading whenever the loop abstained or its
    # final answer disagreed with the blind reading.
    deferred = [r for r in rows if r.get("blind_correct") is not None and r["sample_id"] in by_id]
    defer = lambda r: r["controller_path"] == "abstained" or (
        (by_id[r["sample_id"]]["agentic_result"].get("blind_reading") or {}).get("agrees_with_final") is False)
    posthoc = pct(sum(bool(r["blind_correct"]) if defer(r) else bool(r["agentic_correct"]) for r in deferred)
                  / len(deferred)) if deferred else None
    return {
        "posthoc_defer_to_blind_strict": posthoc,
        "revised": revised, "revised_fixed": fixed, "revised_broken": broken,
        "blind_flagged_first_answer": flagged, "blind_flagged_and_wrong": flagged_wrong,
        "blind_decided_by_meaning": meaning,
        "abstained": abstained, "abstained_own_answer_right": abstained_own_right,
        "abstained_baseline_right": abstained_baseline_right,
        "answered": len(answered),
        "answered_agentic_strict": pct2(sum(bool(r["agentic_correct"]) for r in answered) / len(answered)) if answered else None,
        "answered_baseline_strict": pct2(sum(bool(r["baseline_correct"]) for r in answered) / len(answered)) if answered else None,
    }


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

    answered_rows = [r for r in rows if r["controller_path"] != "abstained"]

    def signal(key, groups):
        # Answered questions only: an abstention is never verified and always scored wrong.
        out = {}
        for label, values in groups.items():
            subset = [r for r in answered_rows if r[key] in values]
            out[label] = {"cases": len(subset), "agentic": pct2(sum(r["agentic_correct"] for r in subset) / len(subset)) if subset else None}
        return out
    metrics["verification_signal"] = signal("verification_status", {
        "verified": {"verified"}, "partially_verified": {"partially_verified"}, "unsupported_or_low": {"unsupported", "low_confidence"}})
    metrics["self_consistency_signal"] = signal("self_consistency_level", {
        "high": {"high"}, "medium": {"medium"}, "low": {"low"}, "not_run": {"not_run"}})
    blind_rows = [r for r in rows if r.get("blind_correct") is not None]
    if blind_rows:  # runs with the blind check
        metrics["blind_reading"] = {
            "cases": len(blind_rows),
            "strict": pct(sum(r["blind_correct"] for r in blind_rows) / len(blind_rows)),
            "agentic_vs_blind": mcnemar(blind_rows, "agentic_correct", "blind_correct"),
            "blind_vs_baseline": mcnemar(blind_rows, "blind_correct", "baseline_correct"),
        }
    zoomed_all = [r for r in rows if r.get("zoomed_looks")]
    zoomed = [r for r in zoomed_all if r["controller_path"] != "abstained"]
    metrics["zoom"] = {"episodes": len(zoomed_all), "cases": len(zoomed),
                       "agentic": pct(sum(r["agentic_correct"] for r in zoomed) / len(zoomed)) if zoomed else None,
                       "baseline": pct(sum(r["baseline_correct"] for r in zoomed) / len(zoomed)) if zoomed else None}
    answered = [r for r in rows if r["controller_path"] != "abstained"]
    metrics["agentic_answered"] = {"cases": len(answered), "strict": pct2(sum(r["agentic_correct"] for r in answered) / len(answered))}
    metrics["mechanisms"] = mechanisms(rows, cache)

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
    metrics["showcase"] = showcase_cases(rows, cache)
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
