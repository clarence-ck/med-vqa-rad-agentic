"""LLM judge for open-ended answers the strict scorer marked wrong.

The strict scorer (backend/utils/scoring.py) is a lower bound: it rejects correct
paraphrases ("frontal radiograph" for "posterior-anterior"). This script asks a judge
model whether each strict-wrong open-ended answer means the same as the ground truth.

- Only open-ended cases; only answers the strict scorer marked wrong.
- One call per case; candidate answers are shuffled and labelled A, B, C... so the
  judge does not know which system produced which answer.
- Verdicts: correct / partially_correct / incorrect. Only "correct" adds credit.

Output: experiments/runs/<run_id>/judgments.json (read by the Evaluation dashboard). A case is re-judged
whenever its strict-wrong answers differ from the ones already judged, so re-runs never reuse a
verdict for a different answer; --fresh discards all earlier verdicts.

Usage (run from repo root):
    python scripts/judge_open_ended.py --max-cost 0.6              # the current run
    python scripts/judge_open_ended.py --run-id 2026-10-16_0930 --fresh
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "backend"))

from haystack.components.generators.chat import OpenAIChatGenerator  # noqa: E402
from haystack.dataclasses import ChatMessage  # noqa: E402
from haystack.utils import Secret  # noqa: E402

from config import MODELS, OPENAI_API_KEY, generation_kwargs  # noqa: E402
from routes import dataset as ds  # noqa: E402
from utils import runs  # noqa: E402
from utils.json_utils import strip_json_fences  # noqa: E402
from utils.usage import record_usage, summarize, track_usage  # noqa: E402

SYSTEMS = ("baseline", "workflow", "agentic", "pipeline")

PROMPT = """You are grading answers to a radiology visual question answering benchmark.
Decide, for each candidate, whether it gives the same answer as the reference answer.

Question: {question}
Reference answer: {reference}

Candidates:
{candidates}

Rules:
- "correct": the candidate states the same finding/location/value as the reference, possibly in different words,
  more specific, or with extra detail that does not contradict it.
- "partially_correct": the candidate overlaps with the reference but misses or changes an essential part
  (e.g. right organ but wrong side, one of two findings, a much vaguer location).
- "incorrect": different answer, contradicts the reference, or does not answer.
- Judge meaning only. Do not reward hedging; do not penalise extra correct detail.

Return ONLY a JSON object mapping each candidate label to {{"verdict": "...", "reason": "<one short sentence>"}}."""


def strict_wrong_candidates(row: dict) -> dict[str, str]:
    candidates = {}
    for system in SYSTEMS:
        correct = row.get(f"{system}_correct")
        answer = row.get(f"{system}_answer")
        if correct is False and answer:
            candidates[system] = str(answer)
    return candidates


def judge_case(generator: OpenAIChatGenerator, row: dict, seed: int) -> dict:
    candidates = strict_wrong_candidates(row)
    systems = list(candidates)
    random.Random(f"{seed}-{row['sample_id']}").shuffle(systems)
    labels = {chr(ord("A") + i): system for i, system in enumerate(systems)}
    listing = "\n".join(f"{label}: {candidates[system]}" for label, system in labels.items())
    prompt = PROMPT.format(question=row["question"], reference=row["ground_truth_answer"], candidates=listing)
    response = generator.run(messages=[ChatMessage.from_user(prompt)], generation_kwargs=generation_kwargs("judge"))
    record_usage("judge", MODELS["judge"], response["replies"][0].meta.get("usage"))
    verdicts = json.loads(strip_json_fences(response["replies"][0].text or ""))
    return {
        system: {
            "answer": candidates[system],
            "verdict": str((verdicts.get(label) or {}).get("verdict", "incorrect")).strip().lower(),
            "reason": (verdicts.get(label) or {}).get("reason", ""),
        }
        for label, system in labels.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-cost", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7, help="Seed for shuffling candidate labels")
    parser.add_argument("--fresh", action="store_true", help="Discard earlier verdicts and judge every case again")
    parser.add_argument("--run-id", default=None, help="Run under experiments/runs/ (default: the current run)")
    args = parser.parse_args()

    run_id = args.run_id or runs.current_run_id()
    if not run_id:
        sys.exit("No run yet: start one with python scripts/run_all.py")
    OUTPUT = runs.run_dir(run_id) / "judgments.json"
    rows = asyncio.run(ds.get_evaluation_summary(run=run_id))["rows"]
    todo = [row for row in rows if row["answer_style"] == "open_ended" and strict_wrong_candidates(row)]
    existing = {} if args.fresh or not OUTPUT.exists() else json.loads(OUTPUT.read_text(encoding="utf-8"))
    # Keep only verdicts that judged exactly the answers in the current results.
    judgments = {
        str(row["sample_id"]): existing["judgments"][str(row["sample_id"])]
        for row in todo
        if str(row["sample_id"]) in existing.get("judgments", {})
        and {s: v["answer"] for s, v in existing["judgments"][str(row["sample_id"])]["verdicts"].items()}
        == strict_wrong_candidates(row)
    }
    print(f"{len(todo)} open-ended cases with strict-wrong answers; {len(judgments)} already judged; judge={MODELS['judge']}")

    generator = OpenAIChatGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model=MODELS["judge"])

    def save(cost: float) -> None:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps({"judge_model": MODELS["judge"], "seed": args.seed, "cost_usd": round(cost, 4),
                                      "judgments": judgments}, indent=2), encoding="utf-8")
    spent = float(existing.get("cost_usd", 0.0))
    with track_usage() as records:
        for index, row in enumerate(todo, start=1):
            key = str(row["sample_id"])
            if key in judgments:
                continue
            if spent + summarize(records)["cost_usd"] >= args.max_cost:
                print(f"Stopping at --max-cost ${args.max_cost:.2f}")
                break
            try:
                judgments[key] = {"question": row["question"], "ground_truth": row["ground_truth_answer"],
                                  "verdicts": judge_case(generator, row, args.seed)}
            except Exception as error:  # parse/API failure: leave for the next run
                print(f"[{index}/{len(todo)}] {key} failed: {error}")
                continue
            marks = " ".join(f"{s}={v['verdict'][:4]}" for s, v in judgments[key]["verdicts"].items())
            print(f"[{index}/{len(todo)}] {key} {marks}", flush=True)
            save(spent + summarize(records)["cost_usd"])  # after every case, so an interrupted run can resume
        spent += summarize(records)["cost_usd"]

    save(spent)
    print(f"Saved {len(judgments)} judged cases -> {OUTPUT.relative_to(ROOT_DIR)}  (total judge spend ${spent:.4f})")


if __name__ == "__main__":
    main()
