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
from agents.agentic_loop import ABSTAIN_ANSWER, controller_path, run_agentic
from agents.orchestrator import run_pipeline
from agents.workflow_baseline import workflow_answer
from utils.data_loader import get_sample_cases, get_sample_by_id, get_image_bytes, EXCLUDED_SAMPLE_IDS
from utils import scoring
from config import AGENT_MODE, CODE_VERSION, model_manifest
from utils.usage import track_usage, summarize as summarize_usage
from logger import get_logger, sample_id_var

router = APIRouter(prefix="/api/dataset")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".case_explorer_cache")
JUDGMENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "judge", "judgments.json")
_JUDGED_SYSTEMS = ("baseline", "workflow", "agentic", "pipeline")


def _load_judgments() -> dict:
    """LLM-judge verdicts for open-ended answers the strict scorer marked wrong (scripts/judge_open_ended.py)."""
    try:
        with open(JUDGMENTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("judgments", {})
    except (OSError, ValueError):
        return {}


def _apply_judgments(row: dict, judgments: dict) -> None:
    """Judged correctness = strict-correct, or judged 'correct' by the LLM judge (open-ended only)."""
    verdicts = (judgments.get(str(row.get("sample_id"))) or {}).get("verdicts", {})
    row["judged"] = bool(verdicts)
    for system in _JUDGED_SYSTEMS:
        strict = row.get(f"{system}_correct")
        if strict is None:
            row[f"{system}_judged_correct"] = None
            continue
        verdict = (verdicts.get(system) or {}).get("verdict")
        row[f"{system}_judged_correct"] = bool(strict) or verdict == "correct"
        row[f"{system}_judge_verdict"] = verdict
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
            "pipeline_correct": 0,
            "pipeline_count": 0,
            "disagreement_count": 0,
        })
        bucket["count"] += 1
        bucket["baseline_correct"] += 1 if row["baseline_correct"] else 0
        bucket["workflow_correct"] += 1 if row["workflow_correct"] else 0
        bucket["agentic_correct"] += 1 if row["agentic_correct"] else 0
        if row.get("pipeline_correct") is not None:
            bucket["pipeline_count"] += 1
            bucket["pipeline_correct"] += 1 if row["pipeline_correct"] else 0
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
            "pipeline_accuracy": bucket["pipeline_correct"] / bucket["pipeline_count"] if bucket["pipeline_count"] else None,
            "disagreement_rate": bucket["disagreement_count"] / bucket["count"] if bucket["count"] else 0.0,
        }
        for bucket in sorted(grouped_rows.values(), key=sort_key)
    ]


def _judged_summary(rows: list[dict]) -> dict:
    """Accuracy when LLM-judged paraphrases count as correct (upper bound to the strict scorer)."""
    if not any(row.get("judged") for row in rows):
        return {"judge_available": False}
    summary = {"judge_available": True, "judged_cases": sum(1 for row in rows if row.get("judged"))}
    open_rows = [row for row in rows if row.get("answer_style") == "open_ended"]
    for system in _JUDGED_SYSTEMS:
        key = f"{system}_judged_correct"
        scored = [row for row in rows if row.get(key) is not None]
        scored_open = [row for row in open_rows if row.get(key) is not None]
        summary[f"{system}_judged_accuracy"] = _subset_accuracy(scored, key) if scored else None
        summary[f"{system}_judged_open_ended_accuracy"] = _subset_accuracy(scored_open, key) if scored_open else None
    return summary


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
            "by_controller_path": [],
            "pipeline_cases": 0,
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

    # Fixed-pipeline ablation: only for rows that have it.
    pipeline_rows = [row for row in rows if row.get("pipeline_correct") is not None]
    pipeline_ms = [row["pipeline_duration_ms"] for row in pipeline_rows if row.get("pipeline_duration_ms") is not None]

    def pipeline_subset(subset: list[dict]) -> float | None:
        with_pipeline = [row for row in subset if row.get("pipeline_correct") is not None]
        return _subset_accuracy(with_pipeline, "pipeline_correct") if with_pipeline else None

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
        "by_controller_path": _build_accuracy_breakdown(rows, "controller_path", ["direct", "relook", "reanswer", "abstained", "pipeline"]),
        "pipeline_cases": len(pipeline_rows),
        "pipeline_accuracy": pipeline_subset(rows),
        **_judged_summary(rows),
        "pipeline_close_ended_accuracy": pipeline_subset(close_ended_rows),
        "pipeline_open_ended_accuracy": pipeline_subset(open_ended_rows),
        "avg_pipeline_duration_ms": int(sum(pipeline_ms) / len(pipeline_ms)) if pipeline_ms else None,
        "total_pipeline_duration_ms": sum(pipeline_ms) if pipeline_ms else None,
    }


_FAILURE_ANSWERS = {
    "baseline failed to produce an answer.",
    "baseline error.",
    "workflow failed: image interpretation error.",
    "workflow failed: question analysis error.",
    "unable to generate an answer.",
}
_FAILURE_NOTES = {"verification failed due to an error.", "verification could not be completed."}


def _api_failures(results: dict) -> list[str]:
    """Systems whose result shows an API/parse failure rather than a genuine answer.

    Such results must not be cached: they would be scored as wrong answers.
    """
    failed = []
    for name, result in results.items():
        if not isinstance(result, dict):
            failed.append(name)
            continue
        answer = str(result.get("answer", "")).strip().lower()
        notes = str(result.get("verification_notes", "")).strip().lower()
        if (
            (name in {"baseline", "workflow"} and result.get("status") == "error")
            or answer in _FAILURE_ANSWERS
            or notes in _FAILURE_NOTES
            or (result.get("api_errors") or 0) > 0
        ):
            failed.append(name)
    return failed


def _write_cache(cache_path: str, payload: dict) -> None:
    """Write atomically, without the base64 image (it is re-read from disk on a cache hit)."""
    stored = dict(payload)
    stored["sample"] = {k: v for k, v in (payload.get("sample") or {}).items() if k != "image_base64"}
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(stored, f)
    os.replace(tmp_path, cache_path)


def _with_sample_image(payload: dict, sample: dict) -> dict:
    result = dict(payload)
    result["sample"] = {**(payload.get("sample") or {}), **sample}
    result["cached"] = True
    return result


@router.get("/sample/{sample_id}/analysis")
async def analyze_sample_case(sample_id: int, refresh: bool = Query(default=False),
                              ablation: bool = Query(default=False)):
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

        cached_payload = None
        if not refresh and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_payload = json.load(f)
            if not ablation or cached_payload.get("pipeline_result"):
                _log.info("Cache HIT for sample %d (question=%r)", sample_id, question[:80])
                return _with_sample_image(cached_payload, sample)

        def run_with_usage(fn, *args, **kwargs):
            with track_usage() as records:
                result = fn(*args, **kwargs)
            return result, summarize_usage(records)

        async def timed(fn, *args, **kwargs):
            t0 = time.time()
            result, usage = await asyncio.to_thread(run_with_usage, fn, *args, **kwargs)
            return result, int((time.time() - t0) * 1000), usage

        def run_ablation(image: bytes, text: str, mime: str) -> dict:
            # Fixed six-stage pipeline with the image passed to the reasoner: isolates what
            # the controller loop adds over a fixed chain with the same image access.
            return run_pipeline(image, text, mime, reasoner_sees_image=True)

        if cached_payload is not None:
            # Cached three-system result exists; only the ablation is missing.
            _log.info("Cache HIT for sample %d without ablation; running fixed pipeline only", sample_id)
            pipeline_result, pipeline_ms, pipeline_usage = await timed(run_ablation, image_bytes, question, mime_type)
            failed = _api_failures({"pipeline": pipeline_result})
            if failed:
                _log.warning("Sample %d ablation had API failures (%s); not caching", sample_id, failed)
                return JSONResponse(status_code=503, content={"error": "API failure", "failed_systems": failed,
                                                              "cost_usd": pipeline_usage["cost_usd"]})
            cached_payload.update({
                "pipeline_result": pipeline_result,
                "pipeline_duration_ms": pipeline_ms,
            })
            cached_payload.setdefault("usage", {})["pipeline"] = pipeline_usage
            _write_cache(cache_path, cached_payload)
            return _with_sample_image(cached_payload, sample)

        _log.info(
            "Cache MISS for sample %d  question=%r  refresh=%s  ablation=%s  running systems in parallel",
            sample_id, question[:80], refresh, ablation,
        )
        started_at = time.time()

        jobs = [
            timed(baseline_answer, image_bytes, question, mime_type),
            timed(workflow_answer, image_bytes, question, mime_type),
            timed(run_agentic, image_bytes, question, mime_type),
        ]
        if ablation:
            jobs.append(timed(run_ablation, image_bytes, question, mime_type))
        outcomes = await asyncio.gather(*jobs)
        (baseline_result, baseline_ms, baseline_usage), (workflow_result, workflow_ms, workflow_usage), \
            (agentic_result, agentic_ms, agentic_usage) = outcomes[:3]
        pipeline_outcome = outcomes[3] if ablation else None

        failed = _api_failures({
            "baseline": baseline_result,
            "workflow": workflow_result,
            "agentic": agentic_result,
            **({"pipeline": pipeline_outcome[0]} if pipeline_outcome else {}),
        })
        if failed:
            _log.warning("Sample %d had API failures (%s); not caching so it can be re-run", sample_id, failed)
            spent = sum(outcome[2]["cost_usd"] for outcome in outcomes)  # failed attempts still cost money
            return JSONResponse(status_code=503, content={"error": "API failure", "failed_systems": failed,
                                                          "cost_usd": spent})

        _log.info(
            "Sample %d usage  baseline=$%.4f (%d calls)  workflow=$%.4f (%d calls)  agentic=$%.4f (%d calls)",
            sample_id,
            baseline_usage["cost_usd"], baseline_usage["calls"],
            workflow_usage["cost_usd"], workflow_usage["calls"],
            agentic_usage["cost_usd"], agentic_usage["calls"],
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
            "models": model_manifest(),
            "code_version": CODE_VERSION,
            "agent_mode": AGENT_MODE,
            "usage": {
                "baseline": baseline_usage,
                "workflow": workflow_usage,
                "agentic": agentic_usage,
            },
        }
        if pipeline_outcome:
            payload["pipeline_result"] = pipeline_outcome[0]
            payload["pipeline_duration_ms"] = pipeline_outcome[1]
            payload["usage"]["pipeline"] = pipeline_outcome[2]

        _write_cache(cache_path, payload)
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
    if agentic_result.get("abstained"):
        # Results cached before the abstention fix stored a normalised "no" for abstained yes/no questions.
        agentic_answer_text = ABSTAIN_ANSWER
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
    pipeline_result = payload.get("pipeline_result")
    pipeline_answer_text = pipeline_result.get("answer", "") if isinstance(pipeline_result, dict) else None
    pipeline_correct = (
        scoring.answers_match(_normalize_answer(pipeline_answer_text), ground_truth_normalized, is_close_ended,
                              original_question_type, similarity)
        if pipeline_answer_text is not None else None
    )

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
        "pipeline_answer": pipeline_answer_text,
        "pipeline_correct": pipeline_correct,
        "pipeline_duration_ms": payload.get("pipeline_duration_ms"),
        "controller_path": (
            controller_path(agentic_result.get("controller_actions") or [], bool(agentic_result.get("abstained")))
            if agentic_result.get("agent_mode") == "loop"
            else ("pipeline" if agentic_result.get("agent_mode") == "pipeline" else "unknown")
        ),
        "controller_actions": agentic_result.get("controller_actions", []),
        "agentic_attempts": agentic_result.get("attempts"),
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
            _normalize_answer((payload.get("pipeline_result") or {}).get("answer", "")),
            _normalize_answer(ground_truth),
        ]
        all_texts.update(t for t in texts if t)
    
    # Compute all embeddings in batches (much faster than individual calls)
    embed_start = time.time()
    embeddings_map = await asyncio.to_thread(_compute_embeddings_batch, list(all_texts))
    _log.info("Computed %d unique embeddings in %.2fs", len(embeddings_map), time.time() - embed_start)

    # Build evaluation rows using pre-computed embeddings
    judgments = _load_judgments()
    rows = []
    for payload in payloads:
        try:
            row = _build_evaluation_row_batch(payload, embeddings_map)
            if row.get("sample_id") in EXCLUDED_SAMPLE_IDS:
                continue
            _apply_judgments(row, judgments)
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
