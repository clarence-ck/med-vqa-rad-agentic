import asyncio
import base64
import json
import os
import re
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
from utils.data_loader import get_sample_cases, get_sample_by_id, get_image_bytes, EXCLUDED_SAMPLE_IDS
from utils import scoring
from config import model_manifest
from utils.usage import track_usage, summarize as summarize_usage
from logger import get_logger, sample_id_var

router = APIRouter(prefix="/api/dataset")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".case_explorer_cache")
_log = get_logger("routes.dataset")


_EMBED_MODEL = None
_EMBED_MODEL_LOCK = threading.Lock()

_MEDICAL_SBERT_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
_MEDICAL_SBERT_MODEL_LABEL = "S-PubMedBert-MS-MARCO"

# LRU cache for S-PubMedBert-MS-MARCO encodings across evaluation rows
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
                "GPU compute capability %d.%d is below recommended threshold; using CPU for pritamdeka/S-PubMedBert-MS-MARCO",
                major,
                minor,
            )
            return "cpu"

        _ = torch.zeros(1, device="cuda")
        return "cuda"
    except Exception as exc:
        _log.warning("CUDA probe failed (%s); using CPU for pritamdeka/S-PubMedBert-MS-MARCO", exc)
        return "cpu"


def _get_embedding_model():
    """Lazy-load S-PubMedBert-MS-MARCO for semantic similarity."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _EMBED_MODEL_LOCK:
            if _EMBED_MODEL is None:
                device = _resolve_device()
                _log.info("Loading %s (%s) on device: %s", _MEDICAL_SBERT_MODEL_LABEL, _MEDICAL_SBERT_MODEL_NAME, device)
                # Biomedical sentence embedding model for semantic similarity evaluation
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
    # Lowercase, normalize whitespace, strip punctuation
    cleaned = str(text).strip().lower().replace("\n", " ")
    # Remove common punctuation for exact matching
    cleaned = re.sub(r'[.,;:!?]+$', '', cleaned)
    return " ".join(cleaned.split())


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


def _subset_accuracy(subset_rows: list[dict], key: str) -> float:
    total = len(subset_rows)
    if total == 0:
        return 0.0
    return sum(1 for row in subset_rows if row.get(key)) / total


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

        def run_with_usage(fn, *args):
            with track_usage() as records:
                result = fn(*args)
            return result, summarize_usage(records)

        async def timed(fn, *args):
            t0 = time.time()
            result, usage = await asyncio.to_thread(run_with_usage, fn, *args)
            return result, int((time.time() - t0) * 1000), usage

        (baseline_result, baseline_ms, baseline_usage), (workflow_result, workflow_ms, workflow_usage), \
            (agentic_result, agentic_ms, agentic_usage) = await asyncio.gather(
                timed(baseline_answer, image_bytes, question, mime_type),
                timed(workflow_answer, image_bytes, question, mime_type),
                timed(run_pipeline, image_bytes, question, mime_type),
            )
        _log.info(
            "Sample %d usage  baseline=$%.4f (%d calls)  workflow=$%.4f (%d calls)  agentic=$%.4f (%d calls)",
            sample_id,
            baseline_usage["cost_usd"], baseline_usage["calls"],
            workflow_usage["cost_usd"], workflow_usage["calls"],
            agentic_usage["cost_usd"], agentic_usage["calls"],
        )

        total_ms = int((time.time() - started_at) * 1000)
        question_subtype = str(
            agentic_result.get("question_analysis", {}).get("question_subtype", "unknown")
            if isinstance(agentic_result, dict)
            else "unknown"
        )
        question_type = str(
            agentic_result.get("question_analysis", {}).get("question_type", "unknown")
            if isinstance(agentic_result, dict)
            else "unknown"
        )
        _log.info(
            "Sample %d analysis complete  baseline=%dms  workflow=%dms  agentic=%dms  total=%dms",
            sample_id, baseline_ms, workflow_ms, agentic_ms, total_ms,
        )
        _log.info(
            "Sample %d answers  type=%s  subtype=%s  baseline=%r  workflow=%r  agentic=%r  ground_truth=%r",
            sample_id,
            question_type,
            question_subtype,
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
            "models": model_manifest(),
            "usage": {
                "baseline": baseline_usage,
                "workflow": workflow_usage,
                "agentic": agentic_usage,
            },
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        return payload
    finally:
        sample_id_var.reset(token)


async def _load_cache_file(cache_file: str) -> dict | None:
    """Load a single cache file asynchronously."""
    try:
        loop = asyncio.get_event_loop()
        with open(cache_file, "r", encoding="utf-8") as f:
            payload = await loop.run_in_executor(None, json.load, f)
        return payload
    except Exception as exc:
        _log.warning("Skipping cache file %s: %s", os.path.basename(cache_file), exc)
        return None


def _compute_embeddings_batch(texts: list[str]) -> dict[str, any]:
    """Compute embeddings for a batch of texts efficiently.
    
    Returns a dict mapping text to embedding vector.
    """
    if not texts:
        return {}
    
    # Deduplicate texts while preserving order
    unique_texts = list(dict.fromkeys(t for t in texts if t))
    if not unique_texts:
        return {}
    
    model = _get_embedding_model()
    batch_size = 64  # Process embeddings in batches for efficiency
    embeddings_map = {}
    
    for i in range(0, len(unique_texts), batch_size):
        batch = unique_texts[i:i + batch_size]
        try:
            embeddings = model.encode(batch, show_progress_bar=False)
            for text, emb in zip(batch, embeddings):
                embeddings_map[text] = emb
        except Exception as exc:
            _log.error("Failed to encode batch %d-%d: %s", i, i + len(batch), exc)
    
    return embeddings_map


def _compute_semantic_similarity_batch(embeddings_map: dict, text1: str, text2: str) -> float:
    """Compute cosine similarity using pre-computed embeddings."""
    if not text1 or not text2:
        return 1.0 if text1 == text2 else 0.0
    
    emb1 = embeddings_map.get(text1)
    emb2 = embeddings_map.get(text2)
    
    if emb1 is None or emb2 is None:
        # Fallback to cached embedding computation
        emb1 = _get_cached_embedding(text1)
        emb2 = _get_cached_embedding(text2)
        if emb1 is None or emb2 is None:
            return 0.0
    
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return float(similarity)


def _build_evaluation_row_batch(payload: dict, embeddings_map: dict) -> dict:
    """Build evaluation row using pre-computed batch embeddings."""
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
    is_close_ended = scoring.is_close_ended(ground_truth_normalized)
    original_question_type = sample.get("question_type", "unknown")

    def similarity(text1: str, text2: str) -> float:
        return _compute_semantic_similarity_batch(embeddings_map, text1, text2)

    baseline_correct = scoring.answers_match(baseline_normalized, ground_truth_normalized, is_close_ended, original_question_type, similarity)
    workflow_correct = scoring.answers_match(workflow_normalized, ground_truth_normalized, is_close_ended, original_question_type, similarity)
    agentic_correct = scoring.answers_match(agentic_normalized, ground_truth_normalized, is_close_ended, original_question_type, similarity)

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

    # Group by the dataset-side label, which is independent of the system under test.
    agentic_question_type = agentic_result.get("question_analysis", {}).get("question_type")
    question_type = original_question_type
    question_type_source = "dataset"
    routing_decision = agentic_result.get("routing_decision", {}) or {}
    routing_difficulty = routing_decision.get("difficulty", "unknown")
    routing_verification_policy = routing_decision.get("verification_policy", "standard")
    routing_fallback_policy = routing_decision.get("fallback_policy", "none")
    routing_rationale = routing_decision.get("routing_rationale", "")

    # Use batch embeddings for semantic scores
    baseline_semantic = _compute_semantic_similarity_batch(embeddings_map, baseline_normalized, ground_truth_normalized)
    workflow_semantic = _compute_semantic_similarity_batch(embeddings_map, workflow_normalized, ground_truth_normalized)
    agentic_semantic = _compute_semantic_similarity_batch(embeddings_map, agentic_normalized, ground_truth_normalized)

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
        "baseline_semantic": round(baseline_semantic, 4),
        "workflow_semantic": round(workflow_semantic, 4),
        "agentic_semantic": round(agentic_semantic, 4),
        "winner": winner,
        "disagreement": not all(
            scoring.answers_agree(first, second, is_close_ended, original_question_type, similarity)
            for first, second in [
                (baseline_normalized, workflow_normalized),
                (baseline_normalized, agentic_normalized),
                (workflow_normalized, agentic_normalized),
            ]
        ),
        "verification_status": agentic_result.get("verification_status", "unknown"),
        "confidence": agentic_result.get("confidence", "Low"),
        "issues_found": agentic_result.get("issues_found", []),
        "self_consistency_level": agentic_result.get("self_consistency_signal", {}).get("level", "low"),
        "cross_modal_agreement_level": agentic_result.get("cross_modal_agreement", {}).get("level", "medium"),
        "routing_difficulty": routing_difficulty,
        "routing_verification_policy": routing_verification_policy,
        "routing_fallback_policy": routing_fallback_policy,
        "routing_rationale": routing_rationale,
        "duration_ms": payload.get("duration_ms", 0),
        "baseline_duration_ms": payload.get("baseline_duration_ms"),
        "workflow_duration_ms": payload.get("workflow_duration_ms"),
        "agentic_duration_ms": payload.get("agentic_duration_ms"),
        "cached_at": payload.get("cached_at"),
    }


@router.get("/evaluation")
async def get_evaluation_summary():
    import time
    start_time = time.time()
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_files = [
        os.path.join(CACHE_DIR, filename)
        for filename in os.listdir(CACHE_DIR)
        if filename.endswith(".json")
    ]
    _log.info("Evaluation summary requested  cache_files=%d", len(cache_files))

    # Load all cache files concurrently
    load_start = time.time()
    payload_tasks = [_load_cache_file(f) for f in cache_files]
    payloads = await asyncio.gather(*payload_tasks)
    payloads = [p for p in payloads if p is not None]
    skipped = len(cache_files) - len(payloads)
    _log.info("Loaded %d cache files in %.2fs", len(payloads), time.time() - load_start)

    # Collect all unique texts that need embeddings
    # This avoids recomputing embeddings for the same text multiple times
    all_texts = set()
    for payload in payloads:
        sample = payload.get("sample", {})
        baseline_result = payload.get("baseline_result", {})
        workflow_result = payload.get("workflow_result", {})
        agentic_result = payload.get("agentic_result", {})
        ground_truth = payload.get("ground_truth_answer", sample.get("answer", ""))
        
        texts = [
            _normalize_answer(baseline_result.get("answer", "")),
            _normalize_answer(workflow_result.get("answer", "")),
            _normalize_answer(agentic_result.get("answer", "")),
            _normalize_answer(ground_truth),
        ]
        all_texts.update(t for t in texts if t)
    
    # Compute all embeddings in batches (much faster than individual calls)
    embed_start = time.time()
    embeddings_map = await asyncio.to_thread(_compute_embeddings_batch, list(all_texts))
    _log.info("Computed %d unique embeddings in %.2fs", len(embeddings_map), time.time() - embed_start)

    # Build evaluation rows using pre-computed embeddings
    rows = []
    for payload in payloads:
        try:
            row = _build_evaluation_row_batch(payload, embeddings_map)
            if row.get("sample_id") in EXCLUDED_SAMPLE_IDS:
                continue
            rows.append(row)
        except Exception as exc:
            _log.warning("Error building evaluation row: %s", exc)
            skipped += 1
            continue

    rows.sort(key=lambda row: (
        row.get("cached_at") is None,
        -(row.get("cached_at") or 0),
        row.get("sample_id") is None,
        row.get("sample_id") or 0,
    ))

    total_time = time.time() - start_time
    _log.info(
        "Evaluation summary built  total_rows=%d  skipped=%d  total_time=%.2fs",
        len(rows), skipped, total_time,
    )
    
    summary = {
        "summary": _build_evaluation_summary(rows),
        "rows": rows,
        "skipped_files": skipped,
    }
    
    # Clear embedding cache to free memory after evaluation is complete
    _clear_embedding_cache()
    
    return summary
