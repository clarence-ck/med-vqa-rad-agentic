"""Warm the Case Explorer cache for the curated UI test cases.

This script runs the same 200 test cases shown in the Case Explorer dropdown
and stores each analysis payload in ``backend/.case_explorer_cache`` via the
existing ``/api/dataset/sample/{id}/analysis`` logic.

Usage (run from repo root):
    python scripts/run_case_explorer_cache.py             # Resume:  process only missing cases, picks up from interruption
    python scripts/run_case_explorer_cache.py --refresh   # Restart: recompute all 200 cases
    python scripts/run_case_explorer_cache.py --dry-run   # Preview: show what would run
    python scripts/run_case_explorer_cache.py --status    # Stats: show cache coverage only
    python scripts/run_case_explorer_cache.py --limit 20 --max-cost 1.0   # Pilot with a spend cap

Note: Only the base command (no flags) resumes from interruption. Already-cached cases are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routes.dataset import CACHE_DIR, analyze_sample_case  # noqa: E402
from utils.data_loader import get_sample_cases  # noqa: E402


def _cache_path_for(sample_id: int, question: str) -> Path:
    from hashlib import md5

    question_hash = md5(question.encode("utf-8")).hexdigest()[:10]
    return Path(CACHE_DIR) / f"sample_{sample_id}_{question_hash}.json"


def _show_cache_status(selected_samples: list) -> None:
    """Display cache coverage statistics."""
    total = len(selected_samples)
    cached = 0
    missing = 0

    for sample in selected_samples:
        sample_id = int(sample["id"])
        question = str(sample.get("question", ""))
        cache_path = _cache_path_for(sample_id, question)
        if cache_path.exists():
            cached += 1
        else:
            missing += 1

    coverage_pct = (cached / total * 100) if total else 0
    print("=" * 50)
    print("Case Explorer Cache Status")
    print("=" * 50)
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Total curated samples: {total}")
    print(f"Cached: {cached} ({coverage_pct:.1f}%)")
    print(f"Missing: {missing} ({100 - coverage_pct:.1f}%)")
    print("=" * 50)


async def warm_cache(skip: int, limit: int, refresh: bool, dry_run: bool = False, status_only: bool = False,
                     max_cost: float | None = None, concurrency: int = 1, ablation: bool = False,
                     retries: int = 1) -> None:
    if skip < 0:
        raise ValueError("skip must be >= 0")
    if limit < 0:
        raise ValueError("limit must be >= 0")

    selected_samples = get_sample_cases(limit=skip + limit)
    if not selected_samples:
        print("No curated test cases were found.")
        return

    samples = selected_samples[skip:skip + limit]
    if not samples:
        print(f"No curated test cases found for skip={skip}, limit={limit}.")
        return

    # Status-only mode
    if status_only:
        _show_cache_status(selected_samples)
        return

    # Show header
    print(f"Loaded {len(samples)} curated Case Explorer samples (skip={skip}, limit={limit})")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Refresh mode: {'on' if refresh else 'off'}")
    print(f"Dry-run mode: {'on' if dry_run else 'off'}")
    print()

    # Preview what would be processed
    to_process = []
    already_cached = []

    for sample in samples:
        sample_id = int(sample["id"])
        question = str(sample.get("question", ""))
        cache_path = _cache_path_for(sample_id, question)

        if cache_path.exists() and not refresh:
            already_cached.append(sample)
        else:
            to_process.append(sample)

    print(f"Preview: {len(to_process)} to process, {len(already_cached)} already cached")
    print()

    if dry_run:
        print("Dry-run mode - no analysis will be performed.")
        for index, sample in enumerate(to_process, start=1):
            sample_id = sample["id"]
            q_type = sample.get("question_type", "unknown")
            print(f"  [{index}] sample_id={sample_id} question_type={q_type}")
        return

    # Process samples, up to `concurrency` cases at a time. The spend cap is checked
    # before each case starts, so the final spend can exceed it by the cases in flight.
    stats = {"completed": 0, "errors": 0, "cost": 0.0, "stopped": False}
    semaphore = asyncio.Semaphore(max(1, concurrency))
    def needs_run(sample: dict) -> bool:
        path = _cache_path_for(int(sample["id"]), str(sample.get("question", "")))
        if refresh or not path.exists():
            return True
        if ablation:
            try:
                return not json.loads(path.read_text(encoding="utf-8")).get("pipeline_result")
            except (OSError, ValueError):
                return True
        return False

    pending = [(index, sample) for index, sample in enumerate(samples, start=1) if needs_run(sample)]

    async def process(index: int, sample: dict) -> None:
        async with semaphore:
            if max_cost is not None and stats["cost"] >= max_cost:
                if not stats["stopped"]:
                    print(f"Stopping: estimated spend ${stats['cost']:.4f} reached --max-cost ${max_cost:.2f}")
                    stats["stopped"] = True
                return
            sample_id = int(sample["id"])
            question_type = sample.get("question_type", "unknown")
            payload = await analyze_sample_case(sample_id, refresh=refresh, ablation=ablation)
            if not isinstance(payload, dict) or payload.get("error"):
                message = payload.get("error") if isinstance(payload, dict) else getattr(payload, "status_code", payload)
                body = {}
                if hasattr(payload, "body"):
                    try:
                        body = json.loads(payload.body)
                    except ValueError:
                        body = {}
                stats["cost"] += float(body.get("cost_usd", 0.0) or 0.0)   # failed attempts still cost money
                print(f"[{index}/{len(samples)}] sample_id={sample_id}  -> ERROR: {message} "
                      f"{body.get('failed_systems', '')}  (spent ${float(body.get('cost_usd', 0.0) or 0.0):.4f})", flush=True)
                stats["errors"] += 1
                return
            usage = payload.get("usage") or {}
            costs = {system: (usage.get(system) or {}).get("cost_usd", 0.0)
                     for system in ("baseline", "workflow", "agentic", "pipeline") if usage.get(system)}
            case_cost = sum(costs.values())
            stats["cost"] += case_cost
            stats["completed"] += 1
            cost_note = "  ".join(f"{system}=${cost:.4f}" for system, cost in costs.items())
            print(f"[{index}/{len(samples)}] sample_id={sample_id}  {question_type}  case=${case_cost:.4f}  "
                  f"({cost_note})  total=${stats['cost']:.4f}", flush=True)

    await asyncio.gather(*(process(index, sample) for index, sample in pending))

    # One automatic retry for cases that failed (API errors are not cached, so they are still pending).
    for attempt in range(retries):
        failed = [(index, sample) for index, sample in pending if needs_run(sample)]
        if not failed or stats["stopped"]:
            break
        print(f"Retrying {len(failed)} failed case(s) (attempt {attempt + 1}/{retries})", flush=True)
        stats["errors"] = 0
        await asyncio.gather(*(process(index, sample) for index, sample in failed))

    completed, errors, total_cost = stats["completed"], stats["errors"], stats["cost"]

    print()
    print(f"Done. Successfully cached: {completed} | Errors: {errors}")
    if completed:
        print(f"Estimated spend: ${total_cost:.4f} total, ${total_cost / completed:.4f} per case")
    print(f"Cache files are written under backend/.case_explorer_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the curated Case Explorer test cases and warm the cache.")
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of curated UI cases to skip before processing (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of curated UI test cases to run after skipping (default: 200).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force recomputation even if a cache file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which cases would be processed without running analysis.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Automatic retries for cases that failed with API errors (default: 1).",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Also run the fixed-pipeline ablation (added to existing cache files if missing).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of cases to run at the same time (default: 1).",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Stop starting new cases once estimated OpenAI spend reaches this many USD.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache coverage statistics and exit.",
    )
    return parser.parse_args()


def _keep_awake() -> None:
    """Stop Windows from sleeping while a long run is in progress (no-op elsewhere)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main() -> None:
    args = parse_args()
    if not (args.dry_run or args.status):
        _keep_awake()

    asyncio.run(warm_cache(
        skip=args.skip,
        limit=args.limit,
        refresh=args.refresh,
        dry_run=args.dry_run,
        status_only=args.status,
        max_cost=args.max_cost,
        concurrency=args.concurrency,
        ablation=args.ablation,
        retries=args.retries,
    ))


if __name__ == "__main__":
    main()
