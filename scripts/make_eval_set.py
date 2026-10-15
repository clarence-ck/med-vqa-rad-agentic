"""Choose the evaluation questions and write them to experiments/eval_set.json.

By default: a seeded random sample of the VQA-RAD test split that keeps the split's mix of close-ended (yes/no) and
open-ended questions. With --reuse-from, half the questions come from an earlier run (for a before/after
comparison on identical questions) and half are questions that run did not use. The list is shuffled; the same
command always gives the same set.

Usage (run from repo root):
    python scripts/make_eval_set.py --size 300
    python scripts/make_eval_set.py --size 250 --reuse-from <run_id>
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
    parser.add_argument("--reuse-from", default=None,
                        help="Optional earlier run: half the questions are reused from it, half are new")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test = []
    for index, item in enumerate(load_dataset()):
        if item.get("split") == "test" and index not in EXCLUDED_SAMPLE_IDS:
            test.append({"id": index, "close_ended": scoring.is_close_ended(_normalize_answer(item.get("answer", "")))})
    if not test:
        sys.exit("VQA-RAD not found: run python scripts/download_dataset.py first")
    test.sort(key=lambda q: q["id"])
    close_share = sum(q["close_ended"] for q in test) / len(test)
    rng = random.Random(args.seed)

    extra = {}
    if args.reuse_from:
        previous = {int(p.name.split("_")[1]) for p in runs.cases_dir(args.reuse_from).glob("sample_*.json")}
        if not previous:
            sys.exit(f"Run {args.reuse_from} has no cases")
        half = args.size // 2
        reused = stratified([q for q in test if q["id"] in previous], half, close_share, rng)
        new = stratified([q for q in test if q["id"] not in previous], args.size - half, close_share, rng)
        chosen = reused + new
        extra = {"reused_from": args.reuse_from, "reused_ids": sorted(q["id"] for q in reused),
                 "new_ids": sorted(q["id"] for q in new)}
        description = (f"{len(reused)} questions reused from run {args.reuse_from} and {len(new)} it did not use; "
                       f"each half keeps the test split's close/open-ended mix ({close_share:.0%} close-ended).")
    else:
        chosen = stratified(test, args.size, close_share, rng)
        description = (f"{args.size} of the {len(test)} VQA-RAD test questions, drawn at random while keeping the "
                       f"split's close/open-ended mix ({close_share:.0%} close-ended).")
    rng.shuffle(chosen)

    eval_set = {
        "created": dt.date.today().isoformat(),
        "seed": args.seed,
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
