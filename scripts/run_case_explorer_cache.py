"""Warm the Case Explorer cache for the curated UI test cases.

This script runs the same 200 test cases shown in the Case Explorer dropdown
and stores each analysis payload in ``backend/.case_explorer_cache`` via the
existing ``/api/dataset/sample/{id}/analysis`` logic.

Purpose of each command:
    (no flags)            # Resume cache warming by processing only missing cases.

    --refresh             # Recompute every case and overwrite existing cache results.

    --dry-run             # Show what would run without generating cache files.

    --status              # Show cache coverage and exit.
"""
from __future__ import annotations

import argparse
import asyncio
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


async def warm_cache(skip: int, limit: int, refresh: bool, dry_run: bool = False, status_only: bool = False) -> None:
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

    # Process samples
    completed = 0
    errors = 0
    for index, sample in enumerate(samples, start=1):
        sample_id = int(sample["id"])
        question = str(sample.get("question", ""))
        question_type = sample.get("question_type", "unknown")
        cache_path = _cache_path_for(sample_id, question)

        if cache_path.exists() and not refresh:
            continue  # Skip silently (already counted in preview)

        print(f"[{index}/{len(samples)}] sample_id={sample_id}  question_type={question_type}")
        payload = await analyze_sample_case(sample_id, refresh=refresh)
        if isinstance(payload, dict) and payload.get("error"):
            print(f"  -> ERROR: {payload['error']}")
            errors += 1
        else:
            cached_flag = payload.get("cached", False) if isinstance(payload, dict) else False
            print(f"  -> stored in cache (cached={cached_flag})")
            completed += 1

    print()
    print(f"Done. Successfully cached: {completed} | Errors: {errors}")
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
        "--status",
        action="store_true",
        help="Show cache coverage statistics and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    asyncio.run(warm_cache(
        skip=args.skip,
        limit=args.limit,
        refresh=args.refresh,
        dry_run=args.dry_run,
        status_only=args.status
    ))


if __name__ == "__main__":
    main()
