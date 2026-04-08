import asyncio
import base64
import json
import os
import time
import threading
from collections import OrderedDict
from hashlib import md5

import torch
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
_EMBED_MODEL_LOCK = threading.Lock()
_SEMANTIC_THRESHOLD = 0.75  # Threshold for semantic similarity match for open-ended evaluation

_MEDICAL_SBERT_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

# LRU cache for NeuML/pubmedbert-base-embeddings encodings across evaluation rows
_EMBEDDING_CACHE = OrderedDict()
_EMBEDDING_CACHE_LOCK = threading.Lock()
_MAX_EMBEDDING_CACHE_SIZE = 1000  # Limit cache size to prevent memory bloat


def _resolve_device() -> str:
    """Pick a safe runtime device for sentence-transformers models."""
    requested = os.getenv("EVALUATION_DEVICE", "auto").strip().lower()

    if requested == "cpu":
        return "cpu"

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        _log.warning("EVALUATION_DEVICE requests CUDA but CUDA is unavailable, falling back to CPU")
        return "cpu"

    if requested != "auto":
        _log.warning("Unrecognized EVALUATION_DEVICE='%s', defaulting to auto", requested)

    if not torch.cuda.is_available():
        return "cpu"

    try:
        major, minor = torch.cuda.get_device_capability(0)
        if major < 7:
            _log.warning(
                "GPU compute capability %d.%d is below recommended threshold; using CPU for NeuML/pubmedbert-base-embeddings",
                major,
                minor,
            )
            return "cpu"

        _ = torch.zeros(1, device="cuda")
        return "cuda"
    except Exception as exc:
        _log.warning("CUDA probe failed (%s); using CPU for NeuML/pubmedbert-base-embeddings", exc)
        return "cpu"


def _get_embedding_model():
    """Lazy-load NeuML/pubmedbert-base-embeddings for semantic similarity."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _EMBED_MODEL_LOCK:
            if _EMBED_MODEL is None:
                device = _resolve_device()
                _log.info("Loading NeuML/pubmedbert-base-embeddings on device: %s", device)
                # PubMedBERT-based embedding model for biomedical semantic similarity
                _EMBED_MODEL = SentenceTransformer(
                    _MEDICAL_SBERT_MODEL_NAME,
                    device=device
                )
    return _EMBED_MODEL


def _get_cached_embedding(text: str) -> any:
    """Get embedding from cache or compute and cache it."""
    if not text:
        return None
    
    with _EMBEDDING_CACHE_LOCK:
        if text in _EMBEDDING_CACHE:
            # Move to end to mark as recently used
            _EMBEDDING_CACHE.move_to_end(text)
            return _EMBEDDING_CACHE[text]
    
    # Cache miss - compute embedding
    try:
        model = _get_embedding_model()
        embedding = model.encode([text])[0]
        
        with _EMBEDDING_CACHE_LOCK:
            # Remove oldest if cache is full
            if len(_EMBEDDING_CACHE) >= _MAX_EMBEDDING_CACHE_SIZE:
                _EMBEDDING_CACHE.popitem(last=False)
            _EMBEDDING_CACHE[text] = embedding
        
        return embedding
    except Exception as exc:
        _log.error("Failed to compute embedding for text: %s", exc)
        return None


def _clear_embedding_cache():
    """Clear the embedding cache to free memory."""
    with _EMBEDDING_CACHE_LOCK:
        _EMBEDDING_CACHE.clear()
        _log.info("Embedding cache cleared")


def _semantic_similarity(prediction: str, ground_truth: str) -> float:
    """Compute semantic similarity using embeddings (cosine similarity).
    
    Returns float between 0 and 1, where 1 = identical meaning.
    """
    if not prediction or not ground_truth:
        return 1.0 if prediction == ground_truth else 0.0
    
    try:
        model = _get_embedding_model()
        embeddings = model.encode([prediction, ground_truth])
        similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
        return float(similarity)
    except Exception as exc:
        _log.error("Failed to compute semantic similarity: %s", exc)
        return 0.0


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


def _normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison."""
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().replace("\n", " ").split())


def _normalize_yes_no(text: str) -> str:
    """Normalize answer text for exact yes/no comparison only."""
    return _normalize_answer(text)


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
    
    For close-ended (yes/no) questions: uses exact yes/no only.
    For open-ended questions: uses semantic similarity >= 0.75 threshold only.
    """
    if is_close_ended:
        # Close-ended: strict exact yes/no matching
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
    """Check if any two systems disagree using NeuML/pubmedbert-base-embeddings semantic similarity with cached embeddings.
    
    Disagreement exists if not all pairs are semantically equivalent (similarity >= 0.75).
    """
    try:
        baseline_emb = _get_cached_embedding(baseline)
        workflow_emb = _get_cached_embedding(workflow)
        agentic_emb = _get_cached_embedding(agentic)
        
        if baseline_emb is None or workflow_emb is None or agentic_emb is None:
            return False
        
        baseline_workflow_same = cosine_similarity([baseline_emb], [workflow_emb])[0][0] >= _SEMANTIC_THRESHOLD
        baseline_agentic_same = cosine_similarity([baseline_emb], [agentic_emb])[0][0] >= _SEMANTIC_THRESHOLD
        workflow_agentic_same = cosine_similarity([workflow_emb], [agentic_emb])[0][0] >= _SEMANTIC_THRESHOLD
        return not (baseline_workflow_same and baseline_agentic_same and workflow_agentic_same)
    except Exception as exc:
        _log.error("Failed to compute semantic disagreement: %s", exc)
        return False


def _compute_semantic_scores_batch(baseline: str, workflow: str, agentic: str, ground_truth: str) -> dict:
    """Compute semantic similarity scores for all systems vs ground truth using cached embeddings.
    
    Returns dict with baseline_semantic, workflow_semantic, agentic_semantic.
    """
    if not baseline or not workflow or not agentic or not ground_truth:
        return {
            "baseline_semantic": 1.0 if baseline == ground_truth else 0.0,
            "workflow_semantic": 1.0 if workflow == ground_truth else 0.0,
            "agentic_semantic": 1.0 if agentic == ground_truth else 0.0,
        }
    
    try:
        baseline_emb = _get_cached_embedding(baseline)
        workflow_emb = _get_cached_embedding(workflow)
        agentic_emb = _get_cached_embedding(agentic)
        gt_emb = _get_cached_embedding(ground_truth)
        
        if baseline_emb is None or workflow_emb is None or agentic_emb is None or gt_emb is None:
            return {"baseline_semantic": 0.0, "workflow_semantic": 0.0, "agentic_semantic": 0.0}
        
        return {
            "baseline_semantic": float(cosine_similarity([baseline_emb], [gt_emb])[0][0]),
            "workflow_semantic": float(cosine_similarity([workflow_emb], [gt_emb])[0][0]),
            "agentic_semantic": float(cosine_similarity([agentic_emb], [gt_emb])[0][0]),
        }
    except Exception as exc:
        _log.error("Failed to compute semantic scores batch: %s", exc)
        return {"baseline_semantic": 0.0, "workflow_semantic": 0.0, "agentic_semantic": 0.0}


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


def _build_accuracy_breakdown(rows: list[dict], group_key: str, preferred_order: list[str] | None = None) -> list[dict]:
    grouped_rows = {}
    for row in rows:
        group_value = str(row.get(group_key, "unknown") or "unknown").strip().lower() or "unknown"
        bucket = grouped_rows.setdefault(group_value, {
            "label": group_value,
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

    order_map = {value: index for index, value in enumerate(preferred_order or [])}

    def sort_key(bucket: dict) -> tuple[int, int | str]:
        label = bucket["label"]
        if label in order_map:
            return (0, order_map[label])
        return (1, label)

    return [
        {
            "label": bucket["label"],
            "count": bucket["count"],
            "baseline_accuracy": bucket["baseline_correct"] / bucket["count"] if bucket["count"] else 0.0,
            "workflow_accuracy": bucket["workflow_correct"] / bucket["count"] if bucket["count"] else 0.0,
            "agentic_accuracy": bucket["agentic_correct"] / bucket["count"] if bucket["count"] else 0.0,
            "disagreement_rate": bucket["disagreement_count"] / bucket["count"] if bucket["count"] else 0.0,
        }
        for bucket in sorted(grouped_rows.values(), key=sort_key)
    ]


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
    routing_decision = agentic_result.get("routing_decision", {}) or {}
    routing_difficulty = routing_decision.get("difficulty", "unknown")
    routing_needs_retrieval = bool(routing_decision.get("needs_retrieval", False))
    routing_verification_policy = routing_decision.get("verification_policy", "standard")
    routing_fallback_policy = routing_decision.get("fallback_policy", "none")
    routing_rationale = routing_decision.get("routing_rationale", "")
    routing_retrieval_rationale = routing_decision.get("retrieval_rationale", "")

    semantic_scores = _compute_semantic_scores_batch(baseline_normalized, workflow_normalized, agentic_normalized, ground_truth_normalized)

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
        "baseline_semantic": round(semantic_scores["baseline_semantic"], 4),
        "workflow_semantic": round(semantic_scores["workflow_semantic"], 4),
        "agentic_semantic": round(semantic_scores["agentic_semantic"], 4),
        "winner": winner,
        "disagreement": _has_semantic_disagreement(baseline_normalized, workflow_normalized, agentic_normalized),
        "verification_status": agentic_result.get("verification_status", "unknown"),
        "confidence": agentic_result.get("confidence", "Low"),
        "issues_found": agentic_result.get("issues_found", []),
        "self_consistency_level": agentic_result.get("self_consistency_signal", {}).get("level", "low"),
        "cross_modal_agreement_level": agentic_result.get("cross_modal_agreement", {}).get("level", "medium"),
        "routing_difficulty": routing_difficulty,
        "routing_needs_retrieval": routing_needs_retrieval,
        "routing_verification_policy": routing_verification_policy,
        "routing_fallback_policy": routing_fallback_policy,
        "routing_rationale": routing_rationale,
        "routing_retrieval_rationale": routing_retrieval_rationale,
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
            "by_difficulty_type": [],
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
        "by_question_type": _build_accuracy_breakdown(rows, "question_type"),
        "by_difficulty_type": _build_accuracy_breakdown(rows, "routing_difficulty", ["easy", "medium", "hard", "unknown"]),
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
    
    summary = {
        "summary": _build_evaluation_summary(rows),
        "rows": rows,
        "skipped_files": skipped,
    }
    
    # Clear embedding cache to free memory after evaluation is complete
    _clear_embedding_cache()
    
    return summary
