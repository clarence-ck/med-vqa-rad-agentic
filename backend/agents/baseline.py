import base64
import json
import time
from openai import OpenAI
from config import OPENAI_API_KEY, VISION_MODEL
from utils.json_utils import strip_json_fences
from logger import get_logger

_log = get_logger("agents.baseline")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazily initialize the OpenAI client to avoid failures at import time."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def baseline_answer(image_bytes: bytes, question: str,
                    mime_type: str = "image/png") -> dict:
    """
    Baseline: Single-shot answer
    Sends image + question in one prompt to get a direct answer.
    Used for comparison against the agentic pipeline.
    """
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    _log.info(
        "Baseline (single-shot) request  mime_type=%s  size=%d bytes  question=%r",
        mime_type, len(image_bytes), question[:100],
    )
    started = time.perf_counter()

    prompt = f"""You are a medical visual question answering system. Look at this medical image and answer the following question.

Question: "{question}"

Return a JSON object with:
{{
  "answer": "Direct answer to the question",
  "explanation": "Brief explanation (1-2 sentences)",
  "confidence": "high / medium / low"
}}

Return ONLY the JSON object, no additional text."""

    try:
        response = _get_client().chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=3000,

        )

        raw = strip_json_fences(response.choices[0].message.content or "")
        _log.debug("Raw baseline response: %s", raw[:500])
        result = json.loads(raw)
        result["status"] = "success"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _log.info(
            "Baseline done  answer=%r  confidence=%s  duration=%dms",
            str(result.get("answer", ""))[:80],
            result.get("confidence", "unknown"),
            elapsed_ms,
        )
        return result

    except json.JSONDecodeError:
        _log.error("Failed to parse baseline JSON response")
        return {
            "status": "error",
            "answer": "Baseline failed to produce an answer.",
            "explanation": "",
            "confidence": "low"
        }
    except Exception as e:
        _log.error("Unexpected error in baseline: %s", e)
        return {
            "status": "error",
            "answer": "Baseline error.",
            "explanation": str(e),
            "confidence": "low"
        }
