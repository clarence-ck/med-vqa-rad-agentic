"""Run chosen systems again on a seeded subset of the evaluation questions (the variance repeats).

Answers are scored with backend/utils/scoring.py and saved after every case to --out, so an interrupted run resumes
where it stopped. scripts/run_all.py uses it to measure how often answers change between runs.

Systems: baseline, workflow, agentic_loop, pipeline.

Usage (run from repo root):
    python scripts/repeat_cases.py --from-eval-sample 300 --n 100 --out experiments/runs/<run_id>/variance.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "backend"))

from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from agents.baseline import baseline_answer  # noqa: E402
from agents.workflow_baseline import workflow_answer  # noqa: E402
from agents.agentic_loop import run_agentic_loop  # noqa: E402
from agents.orchestrator import run_pipeline  # noqa: E402
from config import model_manifest  # noqa: E402
from routes.dataset import _get_cached_embedding, _normalize_answer  # noqa: E402
from utils import scoring  # noqa: E402
from utils.data_loader import EXCLUDED_SAMPLE_IDS, get_image_bytes, get_sample_by_id, get_sample_cases, load_dataset  # noqa: E402
from utils.usage import summarize, track_usage  # noqa: E402

ALL_SYSTEMS = ("baseline", "workflow", "agentic_loop", "pipeline")


def similarity(first: str, second: str) -> float:
    a, b = _get_cached_embedding(first), _get_cached_embedding(second)
    if a is None or b is None:
        return 0.0
    return float(cosine_similarity([a], [b])[0][0])


def run_system(name: str, image_bytes: bytes, question: str):
    started = time.time()
    with track_usage() as records:
        if name == "baseline":
            result = baseline_answer(image_bytes, question, "image/png")
        elif name == "workflow":
            result = workflow_answer(image_bytes, question, "image/png")
        elif name == "agentic_loop":
            result = run_agentic_loop(image_bytes, question, "image/png")
        else:
            result = run_pipeline(image_bytes, question, "image/png")
    return result, int((time.time() - started) * 1000), summarize(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cost", type=float, default=1.0, help="Stop before starting a case once spend reaches this (USD).")
    parser.add_argument("--systems", default="baseline,agentic_loop",
                        help=f"Comma-separated subset of {', '.join(ALL_SYSTEMS)}")
    parser.add_argument("--out", required=True,
                        help="Results file; an existing one is resumed, skipping finished cases")
    parser.add_argument("--from-eval-sample", type=int, default=0, metavar="N",
                        help="Draw the --n cases from the first N evaluation questions (experiments/eval_set.json), "
                             "so repeats cover questions from the main run")
    args = parser.parse_args()
    systems = tuple(name.strip() for name in args.systems.split(",") if name.strip())
    unknown = set(systems) - set(ALL_SYSTEMS)
    if unknown:
        parser.error(f"unknown systems: {sorted(unknown)}")
    output = Path(args.out).resolve()

    if args.from_eval_sample:
        pool_ids = sorted(sample["id"] for sample in get_sample_cases(limit=args.from_eval_sample))
    else:
        pool_ids = [item["id"] for item in load_dataset()
                    if item.get("split") == "test" and item["id"] not in EXCLUDED_SAMPLE_IDS]
    sample_ids = random.Random(args.seed).sample(pool_ids, args.n)

    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    rows = previous.get("rows", [])
    total_cost, new_cost = float(previous.get("total_cost_usd", 0.0)), 0.0
    done_ids = {row["sample_id"] for row in rows}
    if done_ids:
        print(f"Resuming: {len(done_ids)} of {len(sample_ids)} cases already done")

    def save() -> dict:
        summary = {}
        for name in systems:
            done = [row[name] for row in rows]
            summary[name] = {
                "accuracy": sum(r["correct"] for r in done) / len(done) if done else 0.0,
                "correct": sum(r["correct"] for r in done),
                "cases": len(done),
                "avg_duration_ms": int(sum(r["duration_ms"] for r in done) / len(done)) if done else 0,
                "avg_calls": round(sum(r["usage"]["calls"] for r in done) / len(done), 1) if done else 0,
                "cost_usd": round(sum(r["usage"]["cost_usd"] for r in done), 4),
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"seed": args.seed, "models": model_manifest(), "summary": summary,
                                      "total_cost_usd": round(total_cost, 4), "rows": rows}, indent=2), encoding="utf-8")
        return summary

    for index, sample_id in enumerate(sample_ids, start=1):
        if sample_id in done_ids:
            continue
        if new_cost >= args.max_cost:
            print(f"Stopping: spend ${new_cost:.4f} reached --max-cost ${args.max_cost:.2f}")
            break
        sample = get_sample_by_id(sample_id)
        image_bytes = get_image_bytes(sample["image_name"])
        question, ground_truth = sample["question"], sample["answer"]
        gt_norm = _normalize_answer(ground_truth)
        close_ended = scoring.is_close_ended(gt_norm)

        with ThreadPoolExecutor(max_workers=len(systems)) as pool:
            futures = {name: pool.submit(run_system, name, image_bytes, question) for name in systems}
            outputs = {name: future.result() for name, future in futures.items()}

        row = {"sample_id": sample_id, "question": question, "ground_truth": ground_truth,
               "question_type": sample["question_type"], "close_ended": close_ended}
        case_cost = 0.0
        for name, (result, duration_ms, usage) in outputs.items():
            answer = str(result.get("answer", ""))
            row[name] = {
                "answer": answer,
                "correct": scoring.answers_match(_normalize_answer(answer), gt_norm, close_ended,
                                                 sample["question_type"], similarity),
                "duration_ms": duration_ms,
                "usage": usage,
                "verification_status": result.get("verification_status"),
                "retry_attempts": result.get("retry_attempts"),
                "controller_actions": result.get("controller_actions"),
            }
            case_cost += usage["cost_usd"]
        total_cost += case_cost
        new_cost += case_cost
        rows.append(row)
        save()  # after every case, so an interrupted run can resume
        marks = "  ".join(f"{name}={'Y' if row[name]['correct'] else 'N'}" for name in systems)
        print(f"[{index}/{len(sample_ids)}] {sample_id} {question[:50]!r:54} {marks}  ${case_cost:.4f} (total ${total_cost:.4f})", flush=True)

    summary = save()

    print("\nsystem               accuracy   avg time  avg calls  cost")
    for name, s in summary.items():
        print(f"{name:20} {s['correct']:>3}/{s['cases']:<3} {s['accuracy']:6.0%}  {s['avg_duration_ms']/1000:6.1f}s  {s['avg_calls']:9}  ${s['cost_usd']:.4f}")
    print(f"total spend ${total_cost:.4f}  ->  {output.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
