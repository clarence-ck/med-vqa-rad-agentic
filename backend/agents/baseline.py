import base64
import json
import time
from openai import OpenAI
from config import OPENAI_API_KEY, MODELS, generation_kwargs
from utils.json_utils import strip_json_fences
from utils.usage import record_usage
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
IMPORTANT INSTRUCTION: You MUST evaluate if the image is a radiology image (e.g. X-ray, CT, MRI, Ultrasound, Fluoroscopy). If it is a generic photo (like a cat, a logo, or a car), or a non-radiology medical image (like Dermoscopy or Fundoscopy), you MUST set "is_radiology_image" to false.

Return a JSON object with:
{{
  "is_radiology_image": true,
  "answer": "Direct, concise answer to the question. If it can be answered yes or no, start with yes or no.",
  "explanation": "Brief explanation (1-2 sentences)",
  "confidence": "high / medium / low"
}}

Return ONLY the JSON object, no additional text."""

    try:
        response = _get_client().chat.completions.create(
            model=MODELS["baseline"],
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
            **generation_kwargs("baseline"),

        )

        record_usage("baseline", MODELS["baseline"], response.usage)

        raw = strip_json_fences(response.choices[0].message.content or "")
        _log.debug("Raw baseline response: %s", raw[:500])
        result = json.loads(raw)
        
        val = result.get("is_radiology_image", result.get("is_medical_image", True))
        if isinstance(val, str):
            val = val.lower().strip() not in ("false", "no", "0")
            
        if not val:
            _log.warning("Non-radiology image detected in baseline. Aborting.")
            raise ValueError("The uploaded image does not appear to be a radiology image. Please upload a valid X-ray, CT, MRI, or Ultrasound.")

        result["status"] = "success"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result["duration_ms"] = elapsed_ms
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
            "confidence": "unverified"
        }
    except Exception as e:
        _log.error("Unexpected error in baseline: %s", e)
        return {
            "status": "error",
            "answer": "Baseline error.",
            "explanation": str(e),
            "confidence": "unverified"
        }
