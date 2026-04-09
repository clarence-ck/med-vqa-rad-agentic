import asyncio
import time
from fastapi import APIRouter, UploadFile, File, Form
from agents.orchestrator import run_pipeline
from agents.baseline import baseline_answer
from agents.workflow_baseline import workflow_answer
from logger import get_logger

router = APIRouter(prefix="/api")
_log = get_logger("routes.analyze")


@router.post("/analyze")
async def analyze(image: UploadFile = File(...), question: str = Form(...)):
    """
    Run the full 6-stage agentic pipeline.
    Accepts a medical image and a question, returns structured analysis.
    """
    image_bytes = await image.read()
    mime_type = image.content_type or "image/png"
    _log.info(
        "[agentic-pipeline] request  image=%s  size=%d bytes  question=%r",
        image.filename,
        len(image_bytes),
        question,
    )
    started = time.perf_counter()
    result = await asyncio.to_thread(run_pipeline, image_bytes, question, mime_type)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    _log.info(
        "[agentic-pipeline] done  answer=%r  verification=%s  confidence=%s  duration=%dms",
        result.get("answer", ""),
        result.get("verification_status", "unknown"),
        result.get("confidence", "unknown"),
        elapsed_ms,
    )
    return result


@router.post("/analyze/baseline")
async def analyze_baseline(image: UploadFile = File(...), question: str = Form(...)):
    """
    Run the single-shot baseline for comparison.
    """
    image_bytes = await image.read()
    mime_type = image.content_type or "image/png"
    _log.info(
        "[baseline] request  image=%s  size=%d bytes  question=%r",
        image.filename,
        len(image_bytes),
        question,
    )
    started = time.perf_counter()
    result = await asyncio.to_thread(baseline_answer, image_bytes, question, mime_type)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    _log.info(
        "[baseline] done  answer=%r  confidence=%s  duration=%dms",
        result.get("answer", ""),
        result.get("confidence", "unknown"),
        elapsed_ms,
    )
    return result


@router.post("/analyze/workflow")
async def analyze_workflow(image: UploadFile = File(...), question: str = Form(...)):
    image_bytes = await image.read()
    mime_type = image.content_type or "image/png"
    _log.info(
        "[workflow] request  image=%s  size=%d bytes  question=%r",
        image.filename,
        len(image_bytes),
        question,
    )
    started = time.perf_counter()
    result = await asyncio.to_thread(workflow_answer, image_bytes, question, mime_type)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    _log.info(
        "[workflow] done  answer=%r  confidence=%s  duration=%dms",
        result.get("answer", ""),
        result.get("confidence", "unknown"),
        elapsed_ms,
    )
    return result
