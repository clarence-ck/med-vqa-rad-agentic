import asyncio
import base64
import json
import os
import time
from hashlib import md5
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from agents.baseline import baseline_answer
from agents.orchestrator import run_pipeline
from agents.workflow_baseline import workflow_answer
from utils.data_loader import get_sample_cases, get_sample_by_id, get_image_bytes

router = APIRouter(prefix="/api/dataset")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".case_explorer_cache")


@router.get("/samples")
async def list_samples(limit: int = 20, split: str | None = Query(default=None)):
    samples = get_sample_cases(limit, split=split)
    return {"samples": samples, "count": len(samples)}


@router.get("/sample/{sample_id}")
async def get_sample(sample_id: int):
    """Return a specific sample case including base64-encoded image."""
    sample = get_sample_by_id(sample_id)
    if sample is None:
        return JSONResponse(status_code=404, content={"error": "Sample not found"})

    # Include base64 image if available
    if sample.get("image_name"):
        image_bytes = get_image_bytes(sample["image_name"])
        if image_bytes:
            ext = os.path.splitext(sample["image_name"])[1].lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".gif": "image/gif"}
            mime = mime_map.get(ext, "image/png")
            sample["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
            sample["image_mime"] = mime
        else:
            sample["image_base64"] = None
            sample["image_mime"] = None
    else:
        sample["image_base64"] = None
        sample["image_mime"] = None

    # Remove file path from response (security)
    sample.pop("image_path", None)

    return sample


def _get_sample_with_image(sample_id: int):
    sample = get_sample_by_id(sample_id)
    if sample is None:
        return None

    if sample.get("image_name"):
        image_bytes = get_image_bytes(sample["image_name"])
        if image_bytes:
            ext = os.path.splitext(sample["image_name"])[1].lower()
            mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".gif": "image/gif"}
            mime = mime_map.get(ext, "image/png")
            sample["image_base64"] = base64.b64encode(image_bytes).decode("utf-8")
            sample["image_mime"] = mime
        else:
            image_bytes = None
            sample["image_base64"] = None
            sample["image_mime"] = None
    else:
        image_bytes = None
        sample["image_base64"] = None
        sample["image_mime"] = None

    sample.pop("image_path", None)
    return sample, image_bytes


def _get_cache_path(sample_id: int, question: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    question_hash = md5(question.encode("utf-8")).hexdigest()[:10]
    return os.path.join(CACHE_DIR, f"sample_{sample_id}_{question_hash}.json")


def _normalize_answer(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().replace("\n", " ").split())


def _build_evaluation_row(payload: dict) -> dict:
    sample = payload.get("sample", {})
    baseline_result = payload.get("baseline_result", {})
    workflow_result = payload.get("workflow_result", {})
    agentic_result = payload.get("agentic_result", {})
    ground_truth = payload.get("ground_truth_answer", sample.get("answer", ""))
    baseline_answer_text = baseline_result.get("answer", "")
    workflow_answer_text = workflow_result.get("answer", "")
    agentic_answer_text = agentic_result.get("answer", "")
    baseline_normalized = _normalize_answer(baseline_answer_text)
    workflow_normalized = _normalize_answer(workflow_answer_text)
    agentic_normalized = _normalize_answer(agentic_answer_text)
    ground_truth_normalized = _normalize_answer(ground_truth)
    baseline_correct = baseline_normalized == ground_truth_normalized
    workflow_correct = workflow_normalized == ground_truth_normalized
    agentic_correct = agentic_normalized == ground_truth_normalized
    candidates = {
        "baseline": baseline_correct,
        "workflow": workflow_correct,
        "agentic": agentic_correct,
    }
    correct_systems = [name for name, is_correct in candidates.items() if is_correct]
    if len(correct_systems) == 1:
        winner = correct_systems[0]
    elif len(correct_systems) > 1:
        winner = "tie_multi_correct"
    else:
        winner = "tie_all_wrong"

    return {
        "sample_id": sample.get("id"),
        "question": sample.get("question", ""),
        "question_type": sample.get("question_type", "unknown"),
        "split": sample.get("split", "unknown"),
        "ground_truth_answer": ground_truth,
        "baseline_answer": baseline_answer_text,
        "workflow_answer": workflow_answer_text,
        "agentic_answer": agentic_answer_text,
        "baseline_correct": baseline_correct,
        "workflow_correct": workflow_correct,
        "agentic_correct": agentic_correct,
        "winner": winner,
        "disagreement": len({baseline_normalized, workflow_normalized, agentic_normalized}) > 1,
        "verification_status": agentic_result.get("verification_status", "unknown"),
        "confidence": agentic_result.get("confidence", "Low"),
        "issues_found": agentic_result.get("issues_found", []),
        "self_consistency_level": agentic_result.get("self_consistency_signal", {}).get("level", "low"),
        "cross_modal_agreement_level": agentic_result.get("cross_modal_agreement", {}).get("level", "medium"),
        "duration_ms": payload.get("duration_ms", 0),
        "cached_at": payload.get("cached_at"),
    }


def _build_evaluation_summary(rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {
            "total_cases": 0,
            "baseline_accuracy": 0.0,
            "workflow_accuracy": 0.0,
            "agentic_accuracy": 0.0,
            "disagreement_rate": 0.0,
            "baseline_wins": 0,
            "workflow_wins": 0,
            "agentic_wins": 0,
            "ties": 0,
            "avg_duration_ms": 0,
            "split_counts": {},
            "by_question_type": [],
        }

    baseline_correct = sum(1 for row in rows if row["baseline_correct"])
    workflow_correct = sum(1 for row in rows if row["workflow_correct"])
    agentic_correct = sum(1 for row in rows if row["agentic_correct"])
    disagreement_count = sum(1 for row in rows if row["disagreement"])
    baseline_wins = sum(1 for row in rows if row["winner"] == "baseline")
    workflow_wins = sum(1 for row in rows if row["winner"] == "workflow")
    agentic_wins = sum(1 for row in rows if row["winner"] == "agentic")
    ties = sum(1 for row in rows if row["winner"].startswith("tie"))
    avg_duration_ms = int(sum(row.get("duration_ms", 0) for row in rows) / total)
    split_counts = {}

    by_question_type = {}
    for row in rows:
        question_type = row.get("question_type", "unknown")
        split_name = row.get("split", "unknown")
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
        bucket = by_question_type.setdefault(question_type, {
            "question_type": question_type,
            "count": 0,
            "baseline_correct": 0,
            "workflow_correct": 0,
            "agentic_correct": 0,
            "disagreement_count": 0,
        })
        bucket["count"] += 1
        bucket["baseline_correct"] += 1 if row["baseline_correct"] else 0
        bucket["workflow_correct"] += 1 if row["workflow_correct"] else 0
        bucket["agentic_correct"] += 1 if row["agentic_correct"] else 0
        bucket["disagreement_count"] += 1 if row["disagreement"] else 0

    return {
        "total_cases": total,
        "baseline_accuracy": baseline_correct / total,
        "workflow_accuracy": workflow_correct / total,
        "agentic_accuracy": agentic_correct / total,
        "disagreement_rate": disagreement_count / total,
        "baseline_wins": baseline_wins,
        "workflow_wins": workflow_wins,
        "agentic_wins": agentic_wins,
        "ties": ties,
        "avg_duration_ms": avg_duration_ms,
        "split_counts": split_counts,
        "by_question_type": [
            {
                "question_type": bucket["question_type"],
                "count": bucket["count"],
                "baseline_accuracy": bucket["baseline_correct"] / bucket["count"] if bucket["count"] else 0.0,
                "workflow_accuracy": bucket["workflow_correct"] / bucket["count"] if bucket["count"] else 0.0,
                "agentic_accuracy": bucket["agentic_correct"] / bucket["count"] if bucket["count"] else 0.0,
                "disagreement_rate": bucket["disagreement_count"] / bucket["count"] if bucket["count"] else 0.0,
            }
            for bucket in sorted(by_question_type.values(), key=lambda item: item["question_type"])
        ],
    }


@router.get("/sample/{sample_id}/analysis")
async def analyze_sample_case(sample_id: int, refresh: bool = Query(default=False)):
    sample_and_image = _get_sample_with_image(sample_id)
    if sample_and_image is None:
        return JSONResponse(status_code=404, content={"error": "Sample not found"})

    sample, image_bytes = sample_and_image
    if not image_bytes:
        return JSONResponse(status_code=404, content={"error": "Sample image not found"})

    question = sample.get("question", "")
    mime_type = sample.get("image_mime", "image/png")
    cache_path = _get_cache_path(sample_id, question)

    if not refresh and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_payload = json.load(f)
            cached_payload["cached"] = True
            return cached_payload

    started_at = time.time()
    baseline_result, workflow_result, agentic_result = await asyncio.gather(
        asyncio.to_thread(baseline_answer, image_bytes, question, mime_type),
        asyncio.to_thread(workflow_answer, image_bytes, question, mime_type),
        asyncio.to_thread(run_pipeline, image_bytes, question, mime_type),
    )
    payload = {
        "sample": sample,
        "baseline_result": baseline_result,
        "workflow_result": workflow_result,
        "agentic_result": agentic_result,
        "ground_truth_answer": sample.get("answer", ""),
        "cached": False,
        "cached_at": int(time.time()),
        "duration_ms": int((time.time() - started_at) * 1000),
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return payload


@router.get("/evaluation")
async def get_evaluation_summary(split: str | None = Query(default=None)):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_files = [
        os.path.join(CACHE_DIR, filename)
        for filename in os.listdir(CACHE_DIR)
        if filename.endswith(".json")
    ]

    rows = []
    for cache_file in cache_files:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            row = _build_evaluation_row(payload)
            if split and row.get("split") != split:
                continue
            rows.append(row)
        except Exception:
            continue

    rows.sort(key=lambda row: (
        row.get("cached_at") is None,
        -(row.get("cached_at") or 0),
        row.get("sample_id") is None,
        row.get("sample_id") or 0,
    ))

    return {
        "summary": _build_evaluation_summary(rows),
        "rows": rows,
    }
