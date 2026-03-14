from fastapi import APIRouter, UploadFile, File, Form
from agents.orchestrator import run_pipeline
from agents.baseline import baseline_answer
from agents.workflow_baseline import workflow_answer

router = APIRouter(prefix="/api")


@router.post("/analyze")
async def analyze(image: UploadFile = File(...), question: str = Form(...)):
    """
    Run the full 5-agent agentic pipeline.
    Accepts a medical image and a question, returns structured analysis.
    """
    image_bytes = await image.read()
    mime_type = image.content_type or "image/png"

    result = run_pipeline(image_bytes, question, mime_type)
    return result


@router.post("/analyze/baseline")
async def analyze_baseline(image: UploadFile = File(...), question: str = Form(...)):
    """
    Run the single-shot baseline for comparison.
    """
    image_bytes = await image.read()
    mime_type = image.content_type or "image/png"

    result = baseline_answer(image_bytes, question, mime_type)
    return result


@router.post("/analyze/workflow")
async def analyze_workflow(image: UploadFile = File(...), question: str = Form(...)):
    image_bytes = await image.read()
    mime_type = image.content_type or "image/png"
    result = workflow_answer(image_bytes, question, mime_type)
    return result
