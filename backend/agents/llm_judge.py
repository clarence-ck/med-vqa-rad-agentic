import json
import time
from typing import Dict, Any
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
from utils.json_utils import strip_json_fences
from logger import get_logger

_log = get_logger("agents.llm_judge")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazily initialize the OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def llm_judge_evaluate(
    question: str,
    ground_truth: str,
    predicted_answer: str,
    image_context: dict = None,
) -> dict:
    """
    LLM-as-Judge evaluation for medical VQA answers.
    
    Returns a structured evaluation with:
    - correctness_score: 0-5 scale (0=completely wrong, 5=perfect)
    - clinical_accuracy: whether medically accurate
    - completeness: whether fully answers the question
    - reasoning_quality: quality of explanation/evidence
    - overall_assessment: brief explanation
    - is_correct: binary pass/fail (score >= 4)
    """
    image_ctx = image_context or {}
    
    prompt = f"""You are an expert medical evaluator assessing the quality of a medical visual question answering (VQA) system.

TASK: Evaluate the predicted answer against the ground truth for a medical image question.

QUESTION: "{question}"

GROUND TRUTH ANSWER: "{ground_truth}"

PREDICTED ANSWER: "{predicted_answer}"

IMAGE CONTEXT (if available):
- Modality: {image_ctx.get('modality', 'unknown')}
- Body Region: {image_ctx.get('body_region', 'unknown')}
- Visual Summary: {image_ctx.get('visual_summary', 'Not available')}

EVALUATION CRITERIA:
1. CORRECTNESS (0-5): How close is the predicted answer to the ground truth?
   - 5: Exact match or semantically equivalent
   - 4: Mostly correct, minor details missing
   - 3: Partially correct, significant details missing
   - 2: Mostly incorrect, some relevant elements
   - 1: Very incorrect, minimal relevance
   - 0: Completely wrong or unrelated

2. CLINICAL ACCURACY: Is the answer medically accurate given typical findings?

3. COMPLETENESS: Does the answer fully address what was asked?

4. REASONING QUALITY: If explanation provided, is it well-reasoned and grounded?

Return a JSON object:
{{
  "correctness_score": 0-5,
  "clinical_accuracy": "high" | "medium" | "low" | "unclear",
  "completeness": "complete" | "partial" | "incomplete",
  "reasoning_quality": "high" | "medium" | "low" | "not_applicable",
  "overall_assessment": "Brief 1-2 sentence explanation of the evaluation",
  "is_correct": true if correctness_score >= 4, else false
}}

Return ONLY the JSON object."""

    _log.info("LLM Judge evaluating: question=%r predicted=%r", question[:60], predicted_answer[:60])
    started = time.perf_counter()
    
    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise medical evaluator. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=1000,
            temperature=0.1,
        )
        
        raw = strip_json_fences(response.choices[0].message.content or "")
        result = json.loads(raw)
        
        # Ensure is_correct is computed from score
        score = result.get("correctness_score", 0)
        result["is_correct"] = score >= 4
        result["status"] = "success"
        
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _log.info(
            "LLM Judge done: score=%d/5 is_correct=%s clinical=%s duration=%dms",
            score,
            result["is_correct"],
            result.get("clinical_accuracy", "unknown"),
            elapsed_ms
        )
        
        return result
        
    except json.JSONDecodeError as e:
        _log.error("LLM Judge JSON parse error: %s", e)
        return {
            "status": "error",
            "correctness_score": 0,
            "clinical_accuracy": "unclear",
            "completeness": "incomplete",
            "reasoning_quality": "low",
            "overall_assessment": "Failed to parse LLM judge response",
            "is_correct": False,
            "error": str(e)
        }
    except Exception as e:
        _log.error("LLM Judge unexpected error: %s", e)
        return {
            "status": "error",
            "correctness_score": 0,
            "clinical_accuracy": "unclear",
            "completeness": "incomplete",
            "reasoning_quality": "low",
            "overall_assessment": f"Error: {str(e)}",
            "is_correct": False,
            "error": str(e)
        }
