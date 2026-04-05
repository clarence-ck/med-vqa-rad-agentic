import asyncio
import base64
import json
import os
import time
from hashlib import md5

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from agents.baseline import baseline_answer
from agents.orchestrator import run_pipeline
from agents.workflow_baseline import workflow_answer
from utils.data_loader import get_sample_cases, get_sample_by_id, get_image_bytes
from logger import get_logger, sample_id_var

router = APIRouter(prefix="/api/dataset")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".case_explorer_cache")
_log = get_logger("routes.dataset")


_EMBED_MODEL = None
_SEMANTIC_THRESHOLD = 0.65  # Threshold for semantic similarity match (lowered for medical synonyms)


def _get_embedding_model():
    """Lazy-load BioBERT embedding model for medical semantic similarity."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        # BioBERT fine-tuned for semantic similarity (NLI + STS + Medical NLI)
        _EMBED_MODEL = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    return _EMBED_MODEL


def _semantic_similarity(prediction: str, ground_truth: str) -> float:
    """Compute semantic similarity using embeddings (cosine similarity).
    
    Returns float between 0 and 1, where 1 = identical meaning.
    """
    if not prediction or not ground_truth:
        return 1.0 if prediction == ground_truth else 0.0
    
    model = _get_embedding_model()
    embeddings = model.encode([prediction, ground_truth])
    similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
    return float(similarity)


@router.get("/samples")
async def list_samples():
    samples = get_sample_cases()
    _log.info("Listing samples  returned=%d", len(samples))
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


def _get_sample_with_image(sample_id: int) -> tuple[dict | None, bytes | None]:
    """Always returns a (sample, image_bytes) 2-tuple; (None, None) if not found."""
    sample = get_sample_by_id(sample_id)
    if sample is None:
        return None, None

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


_YES_TOKENS = {"yes", "true", "present", "positive", "abnormal", "found", "visible", "seen"}
_NO_TOKENS = {"no", "false", "absent", "negative", "normal"}
_NO_PHRASES = {"not found", "not visible", "not seen"}  # multi-word; must be checked before token split


def _normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison."""
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().replace("\n", " ").split())


def _normalize_yes_no(text: str) -> str:
    """Collapse common yes/no synonyms to a canonical 'yes' or 'no' token."""
    normalized = _normalize_answer(text)
    # Multi-word NO phrases must be checked before splitting ("not found" contains "found" which is a YES token)
    if any(phrase in normalized for phrase in _NO_PHRASES):
        return "no"
    tokens = set(normalized.split())
    if tokens & _YES_TOKENS:
        return "yes"
    if tokens & _NO_TOKENS:
        return "no"
    return normalized


def _token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between two normalized answer strings."""
    pred_tokens = set(prediction.split())
    gt_tokens = set(ground_truth.split())
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    common = pred_tokens & gt_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _answers_match(predicted: str, ground_truth: str, is_close_ended: bool = False) -> bool:
    """Return True if the predicted answer is correct based on question type.
    
    For close-ended (yes/no) questions: uses synonym matching only.
    For open-ended questions: uses semantic similarity >= 0.65 threshold only.
    """
    if is_close_ended:
        # Close-ended: strict yes/no synonym matching
        return _normalize_yes_no(predicted) == _normalize_yes_no(ground_truth)
    else:
        # Open-ended: Semantic similarity >= threshold
        return _semantic_similarity(predicted, ground_truth) >= _SEMANTIC_THRESHOLD


def _is_close_ended_answer(text: str) -> bool:
    normalized = _normalize_yes_no(text)
    return normalized in {"yes", "no"}


def _subset_accuracy(subset_rows: list[dict], key: str) -> float:
    total = len(subset_rows)
    if total == 0:
        return 0.0
    return sum(1 for row in subset_rows if row.get(key)) / total


def _has_semantic_disagreement(baseline: str, workflow: str, agentic: str) -> bool:
    """Check if any two systems disagree using BioBERT semantic similarity.
    
    Disagreement exists if not all pairs are semantically equivalent (similarity >= 0.75).
    """
    baseline_workflow_same = _semantic_similarity(baseline, workflow) >= _SEMANTIC_THRESHOLD
    baseline_agentic_same = _semantic_similarity(baseline, agentic) >= _SEMANTIC_THRESHOLD
    workflow_agentic_same = _semantic_similarity(workflow, agentic) >= _SEMANTIC_THRESHOLD
    
    # Disagreement if any pair doesn't match semantically
    return not (baseline_workflow_same and baseline_agentic_same and workflow_agentic_same)


def _extract_knowledge_retriever_details(agentic_result: dict) -> dict:
    trace = agentic_result.get("agent_trace", []) or []
    knowledge_step = next(
        (
            step for step in trace
            if str(step.get("agent", "")).strip().lower() == "knowledge support"
        ),
        {},
    )
    status = str(knowledge_step.get("status", "unknown")).strip().lower()
    key_facts = knowledge_step.get("output", {}).get("key_facts", []) or []
    summary = knowledge_step.get("summary", "")
    triggered = status not in {"", "unknown", "skipped"}

    return {
        "knowledge_retriever_triggered": triggered,
        "knowledge_retriever_status": status or "unknown",
        "knowledge_key_facts_count": len(key_facts),
        "knowledge_summary": summary,
    }


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
    is_close_ended = _is_close_ended_answer(ground_truth_normalized)
    baseline_correct = _answers_match(baseline_normalized, ground_truth_normalized, is_close_ended)
    workflow_correct = _answers_match(workflow_normalized, ground_truth_normalized, is_close_ended)
    agentic_correct = _answers_match(agentic_normalized, ground_truth_normalized, is_close_ended)
    answer_style = "close_ended" if is_close_ended else "open_ended"
    candidates = {
        "baseline": baseline_correct,
        "workflow": workflow_correct,
        "agentic": agentic_correct,
    }
    correct_systems = [name for name, is_correct in candidates.items() if is_correct]
    if len(correct_systems) == 1:
        winner = correct_systems[0]
    elif len(correct_systems) > 1:
        winner = f"tie: {' & '.join(sys.capitalize() for sys in correct_systems)}"
    else:
        winner = "tie_all_wrong"

    original_question_type = sample.get("question_type", "unknown")
    agentic_question_type = agentic_result.get("question_analysis", {}).get("question_type")
    question_type = agentic_question_type or original_question_type
    question_type_source = "agentic" if agentic_question_type else "dataset"
    knowledge_details = _extract_knowledge_retriever_details(agentic_result)

    return {
        "sample_id": sample.get("id"),
        "question": sample.get("question", ""),
        "question_type": question_type,
        "original_question_type": original_question_type,
        "agentic_question_type": agentic_question_type,
        "question_type_source": question_type_source,
        "split": sample.get("split", "unknown"),
        "answer_style": answer_style,
        "ground_truth_answer": ground_truth,
        "baseline_answer": baseline_answer_text,
        "workflow_answer": workflow_answer_text,
        "agentic_answer": agentic_answer_text,
        "baseline_correct": baseline_correct,
        "workflow_correct": workflow_correct,
        "agentic_correct": agentic_correct,
        "baseline_token_f1": round(_token_f1(baseline_normalized, ground_truth_normalized), 4),
        "workflow_token_f1": round(_token_f1(workflow_normalized, ground_truth_normalized), 4),
        "agentic_token_f1": round(_token_f1(agentic_normalized, ground_truth_normalized), 4),
        "baseline_semantic": round(_semantic_similarity(baseline_normalized, ground_truth_normalized), 4),
        "workflow_semantic": round(_semantic_similarity(workflow_normalized, ground_truth_normalized), 4),
        "agentic_semantic": round(_semantic_similarity(agentic_normalized, ground_truth_normalized), 4),
        "winner": winner,
        "disagreement": _has_semantic_disagreement(baseline_normalized, workflow_normalized, agentic_normalized),
        "verification_status": agentic_result.get("verification_status", "unknown"),
        "confidence": agentic_result.get("confidence", "Low"),
        "issues_found": agentic_result.get("issues_found", []),
        "self_consistency_level": agentic_result.get("self_consistency_signal", {}).get("level", "low"),
        "cross_modal_agreement_level": agentic_result.get("cross_modal_agreement", {}).get("level", "medium"),
        "knowledge_retriever_triggered": knowledge_details["knowledge_retriever_triggered"],
        "knowledge_retriever_status": knowledge_details["knowledge_retriever_status"],
        "knowledge_key_facts_count": knowledge_details["knowledge_key_facts_count"],
        "knowledge_summary": knowledge_details["knowledge_summary"],
        "duration_ms": payload.get("duration_ms", 0),
        "baseline_duration_ms": payload.get("baseline_duration_ms"),
        "workflow_duration_ms": payload.get("workflow_duration_ms"),
        "agentic_duration_ms": payload.get("agentic_duration_ms"),
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
            "close_ended_cases": 0,
            "open_ended_cases": 0,
            "baseline_close_ended_accuracy": 0.0,
            "workflow_close_ended_accuracy": 0.0,
            "agentic_close_ended_accuracy": 0.0,
            "baseline_open_ended_accuracy": 0.0,
            "workflow_open_ended_accuracy": 0.0,
            "agentic_open_ended_accuracy": 0.0,
            "disagreement_rate": 0.0,
            "baseline_wins": 0,
            "workflow_wins": 0,
            "agentic_wins": 0,
            "ties": 0,
            "avg_duration_ms": 0,
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
    rows_with_baseline_ms = [row for row in rows if row.get("baseline_duration_ms") is not None]
    rows_with_workflow_ms = [row for row in rows if row.get("workflow_duration_ms") is not None]
    rows_with_agentic_ms = [row for row in rows if row.get("agentic_duration_ms") is not None]
    total_baseline_duration_ms = sum(r["baseline_duration_ms"] for r in rows_with_baseline_ms) if rows_with_baseline_ms else None
    total_workflow_duration_ms = sum(r["workflow_duration_ms"] for r in rows_with_workflow_ms) if rows_with_workflow_ms else None
    total_agentic_duration_ms = sum(r["agentic_duration_ms"] for r in rows_with_agentic_ms) if rows_with_agentic_ms else None
    avg_baseline_duration_ms = int(total_baseline_duration_ms / len(rows_with_baseline_ms)) if total_baseline_duration_ms is not None else None
    avg_workflow_duration_ms = int(total_workflow_duration_ms / len(rows_with_workflow_ms)) if total_workflow_duration_ms is not None else None
    avg_agentic_duration_ms = int(total_agentic_duration_ms / len(rows_with_agentic_ms)) if total_agentic_duration_ms is not None else None
    close_ended_rows = [row for row in rows if row.get("answer_style") == "close_ended"]
    open_ended_rows = [row for row in rows if row.get("answer_style") == "open_ended"]

    by_question_type = {}
    for row in rows:
        question_type = row.get("question_type", "unknown")
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
        "close_ended_cases": len(close_ended_rows),
        "open_ended_cases": len(open_ended_rows),
        "baseline_close_ended_accuracy": _subset_accuracy(close_ended_rows, "baseline_correct"),
        "workflow_close_ended_accuracy": _subset_accuracy(close_ended_rows, "workflow_correct"),
        "agentic_close_ended_accuracy": _subset_accuracy(close_ended_rows, "agentic_correct"),
        "baseline_open_ended_accuracy": _subset_accuracy(open_ended_rows, "baseline_correct"),
        "workflow_open_ended_accuracy": _subset_accuracy(open_ended_rows, "workflow_correct"),
        "agentic_open_ended_accuracy": _subset_accuracy(open_ended_rows, "agentic_correct"),
        "disagreement_rate": disagreement_count / total,
        "baseline_wins": baseline_wins,
        "workflow_wins": workflow_wins,
        "agentic_wins": agentic_wins,
        "ties": ties,
        "avg_duration_ms": avg_duration_ms,
        "avg_baseline_duration_ms": avg_baseline_duration_ms,
        "avg_workflow_duration_ms": avg_workflow_duration_ms,
        "avg_agentic_duration_ms": avg_agentic_duration_ms,
        "total_baseline_duration_ms": total_baseline_duration_ms,
        "total_workflow_duration_ms": total_workflow_duration_ms,
        "total_agentic_duration_ms": total_agentic_duration_ms,
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
    token = sample_id_var.set(str(sample_id))
    try:
        sample, image_bytes = _get_sample_with_image(sample_id)

        if sample is None:
            _log.warning("Sample %d not found", sample_id)
            return JSONResponse(status_code=404, content={"error": "Sample not found"})

        if not image_bytes:
            _log.warning("Image bytes missing for sample %d", sample_id)
            return JSONResponse(status_code=404, content={"error": "Sample image not found"})

        question = sample.get("question", "")
        mime_type = sample.get("image_mime", "image/png")
        cache_path = _get_cache_path(sample_id, question)

        if not refresh and os.path.exists(cache_path):
            _log.info("Cache HIT for sample %d (question=%r)", sample_id, question[:80])
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_payload = json.load(f)
                cached_payload["cached"] = True
                return cached_payload

        _log.info(
            "Cache MISS for sample %d  question=%r  refresh=%s  running all 3 pipelines in parallel",
            sample_id, question[:80], refresh,
        )
        started_at = time.time()

        async def timed(fn, *args):
            t0 = time.time()
            result = await asyncio.to_thread(fn, *args)
            return result, int((time.time() - t0) * 1000)

        (baseline_result, baseline_ms), (workflow_result, workflow_ms), (agentic_result, agentic_ms) = \
            await asyncio.gather(
                timed(baseline_answer, image_bytes, question, mime_type),
                timed(workflow_answer, image_bytes, question, mime_type),
                timed(run_pipeline, image_bytes, question, mime_type),
            )

        total_ms = int((time.time() - started_at) * 1000)
        _log.info(
            "Sample %d analysis complete  baseline=%dms  workflow=%dms  agentic=%dms  total=%dms",
            sample_id, baseline_ms, workflow_ms, agentic_ms, total_ms,
        )
        _log.info(
            "Sample %d answers  baseline=%r  workflow=%r  agentic=%r  ground_truth=%r",
            sample_id,
            str(baseline_result.get("answer", ""))[:60],
            str(workflow_result.get("answer", ""))[:60],
            str(agentic_result.get("answer", ""))[:60],
            str(sample.get("answer", ""))[:60],
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
            "baseline_duration_ms": baseline_ms,
            "workflow_duration_ms": workflow_ms,
            "agentic_duration_ms": agentic_ms,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        return payload
    finally:
        sample_id_var.reset(token)


@router.get("/evaluation")
async def get_evaluation_summary():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_files = [
        os.path.join(CACHE_DIR, filename)
        for filename in os.listdir(CACHE_DIR)
        if filename.endswith(".json")
    ]
    _log.info("Evaluation summary requested  cache_files=%d", len(cache_files))

    rows = []
    skipped = 0
    for cache_file in cache_files:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            row = _build_evaluation_row(payload)
            rows.append(row)
        except Exception as exc:
            _log.warning("Skipping cache file %s: %s", os.path.basename(cache_file), exc)
            skipped += 1
            continue

    rows.sort(key=lambda row: (
        row.get("cached_at") is None,
        -(row.get("cached_at") or 0),
        row.get("sample_id") is None,
        row.get("sample_id") or 0,
    ))

    _log.info(
        "Evaluation summary built  total_rows=%d  skipped=%d",
        len(rows), skipped,
    )
    return {
        "summary": _build_evaluation_summary(rows),
        "rows": rows,
        "skipped_files": skipped,
    }
