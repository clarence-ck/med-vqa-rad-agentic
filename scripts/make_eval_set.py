"""Choose the evaluation questions and write them to experiments/eval_set.json.

A seeded random sample that keeps the pool's mix of close-ended (yes/no) and open-ended questions. The pool is the
VQA-RAD test split by default, or the whole dataset with --split all. With --exclude-run, no question an earlier run
used is chosen. The list is shuffled; the same command always gives the same set.

Usage (run from repo root):
    python scripts/make_eval_set.py --size 300
    python scripts/make_eval_set.py --size 300 --split all --exclude-run <run_id> --seed 51511
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "backend"))

from routes.dataset import _normalize_answer  # noqa: E402
from utils import runs, scoring  # noqa: E402
from utils.data_loader import EVAL_SET_PATH, EXCLUDED_SAMPLE_IDS, load_dataset  # noqa: E402


def stratified(pool: list[dict], n: int, close_share: float, rng: random.Random) -> list[dict]:
    """n questions from pool with the given share of close-ended ones."""
    close = [q for q in pool if q["close_ended"]]
    open_ = [q for q in pool if not q["close_ended"]]
    n_close = round(n * close_share)
    if n_close > len(close) or n - n_close > len(open_):
        sys.exit(f"Not enough questions: need {n_close} close / {n - n_close} open, have {len(close)} / {len(open_)}")
    return rng.sample(close, n_close) + rng.sample(open_, n - n_close)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=300, help="Number of questions")
    parser.add_argument("--exclude-run", default=None, help="Optional earlier run whose questions are not chosen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["test", "all"], default="test",
                        help="Draw from the test split or from the whole dataset (all)")
    args = parser.parse_args()

    pool = []
    for index, item in enumerate(load_dataset()):
        if args.split in ("all", item.get("split")) and index not in EXCLUDED_SAMPLE_IDS:
            pool.append({"id": index, "close_ended": scoring.is_close_ended(_normalize_answer(item.get("answer", "")))})
    if not pool:
        sys.exit("VQA-RAD not found: run python scripts/download_dataset.py first")
    extra = {}
    if args.exclude_run:
        excluded = {int(p.name.split("_")[1]) for p in runs.cases_dir(args.exclude_run).glob("sample_*.json")}
        if not excluded:
            sys.exit(f"Run {args.exclude_run} has no cases")
        pool = [q for q in pool if q["id"] not in excluded]
        extra = {"excluded_run": args.exclude_run}
    pool.sort(key=lambda q: q["id"])
    close_share = sum(q["close_ended"] for q in pool) / len(pool)
    rng = random.Random(args.seed)

    chosen = stratified(pool, args.size, close_share, rng)
    pool_name = "VQA-RAD questions (full dataset)" if args.split == "all" else "VQA-RAD test questions"
    description = (f"{args.size} of the {len(pool)} {pool_name}, drawn at random while keeping their "
                   f"close/open-ended mix ({close_share:.0%} close-ended).")
    if args.exclude_run:
        description += f" None of the {len(excluded)} questions run {args.exclude_run} used is included."
    rng.shuffle(chosen)

    eval_set = {
        "created": dt.date.today().isoformat(),
        "seed": args.seed,
        "split": args.split,
        "description": description,
        "ids": [q["id"] for q in chosen],
        "close_ended": sum(q["close_ended"] for q in chosen),
        "open_ended": sum(not q["close_ended"] for q in chosen),
        **extra,
    }
    EVAL_SET_PATH.write_text(json.dumps(eval_set, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {EVAL_SET_PATH.relative_to(ROOT_DIR)}: {len(chosen)} questions "
          f"({eval_set['close_ended']} close-ended / {eval_set['open_ended']} open-ended)")


if __name__ == "__main__":
    main()
