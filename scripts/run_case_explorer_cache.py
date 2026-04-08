"""Warm the Case Explorer cache for the curated UI test cases.

This script runs the same 200 test cases shown in the Case Explorer dropdown
and stores each analysis payload in ``backend/.case_explorer_cache`` via the
existing ``/api/dataset/sample/{id}/analysis`` logic.

Usage:
    python scripts/run_case_explorer_cache.py
    python scripts/run_case_explorer_cache.py --refresh
    python scripts/run_case_explorer_cache.py --skip 30 --limit 170
    python scripts/run_case_explorer_cache.py --skip 0 --limit 200
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


async def warm_cache(skip: int, limit: int, refresh: bool) -> None:
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

    print(f"Loaded {len(samples)} curated Case Explorer samples (skip={skip}, limit={limit})")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Refresh mode: {'on' if refresh else 'off'}")
    print()

    completed = 0
    skipped = 0
    for index, sample in enumerate(samples, start=1):
        sample_id = int(sample["id"])
        question = str(sample.get("question", ""))
        question_type = sample.get("question_type", "unknown")
        cache_path = _cache_path_for(sample_id, question)
        if cache_path.exists() and not refresh:
            try:
                cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached_payload, dict):
                    print(f"[{index}/{len(samples)}] sample_id={sample_id}  question_type={question_type}")
                    print("  -> skipped (already cached)")
                    skipped += 1
                    continue
            except Exception:
                pass

        print(f"[{index}/{len(samples)}] sample_id={sample_id}  question_type={question_type}")
        payload = await analyze_sample_case(sample_id, refresh=refresh)
        if isinstance(payload, dict) and payload.get("error"):
            print(f"  -> ERROR: {payload['error']}")
        else:
            cached_flag = payload.get("cached", False) if isinstance(payload, dict) else False
            print(f"  -> stored in cache (cached={cached_flag})")
        completed += 1

    print()
    print(f"Done. Processed {completed} curated Case Explorer samples.")
    print(f"Skipped {skipped} already-cached samples.")
    total = len(samples)
    if total:
        processed_pct = (completed / total) * 100
        skipped_pct = (skipped / total) * 100
        print(f"Processed: {processed_pct:.1f}% | Skipped: {skipped_pct:.1f}% | Total: {total}")
    print("Cache files are written under backend/.case_explorer_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the curated Case Explorer test cases and warm the cache.")
    parser.add_argument(
        "--skip",
        type=int,
        default=30,
        help="Number of curated UI cases to skip before processing (default: 30).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=170,
        help="Number of curated UI test cases to run after skipping (default: 170).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force recomputation even if a cache file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(warm_cache(skip=args.skip, limit=args.limit, refresh=args.refresh))


if __name__ == "__main__":
    main()
