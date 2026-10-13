import json
import time
from typing import Any, Dict

from haystack import component
from haystack.components.converters.image import ImageFileToImageContent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ByteStream, ChatMessage, TextContent
from haystack.utils import Secret

from config import OPENAI_API_KEY, MODELS, generation_kwargs
from logger import get_logger
from utils.json_utils import strip_json_fences
from utils.usage import record_usage

_log = get_logger("agents.look_agent")


class NonRadiologyImageError(ValueError):
    """Raised when the first look finds that the image is not a radiology image."""


@component
class LookAgent:
    """
    Perception agent for the agentic loop.

    Unlike ImageInterpreter (a question-agnostic description), it looks at the image
    *with the question in mind*, and can be called again with a `focus` (a region or
    structure the controller wants re-examined).
    """

    def __init__(self):
        self.generator = OpenAIChatGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model=MODELS["look"])
        self.image_converter = ImageFileToImageContent(detail="high", size=(1024, 1024))

    @component.output_types(result=Dict[str, Any])
    def run(self, image_bytes: bytes, question: str, mime_type: str = "image/png",
            focus: str = "", previous_observations: str = ""):
        focus = (focus or "").strip()
        focus_block = ""
        if focus:
            focus_block = f"""
FOCUSED RE-EXAMINATION: look specifically at "{focus}". Describe exactly what is (or is not) visible there,
including side (left/right), position and appearance. Earlier observations, which may be wrong or incomplete:
{previous_observations or "none"}
"""
        prompt = f"""You are a radiology perception agent. Examine the attached medical image in order to answer this question:

QUESTION: "{question}"
{focus_block}
First decide whether this is a radiology image (X-ray, CT, MRI, ultrasound, fluoroscopy). Photos, logos and
non-radiology medical images (dermoscopy, fundoscopy) are not.

Report only what is visible; do not answer with a diagnosis you cannot see. Note the patient's left appears on
the image's right on standard radiological views.

Return a JSON object:
{{
  "is_radiology_image": true,
  "modality": "e.g. X-ray, CT, MRI, Ultrasound",
  "body_region": "e.g. chest, head, abdomen",
  "view_or_sequence": "e.g. PA, axial, T2-weighted, or unknown",
  "visible_structures": ["anatomical structures visible"],
  "question_relevant_observations": ["concrete observations that bear on the question"],
  "focus_observation": "What the focused re-examination shows (empty if no focus was requested)",
  "notable_findings": ["other visible abnormalities or notable features"],
  "image_quality": "good / fair / poor",
  "uncertainties": ["what cannot be determined from this image"],
  "visual_summary": "2-3 sentence summary"
}}

Return ONLY the JSON object, no additional text."""

        started = time.perf_counter()
        _log.info("Look  focus=%r  question=%r", focus, question[:80])
        try:
            image_content = self.image_converter.run(
                sources=[ByteStream(data=image_bytes, mime_type=mime_type)]
            )["image_contents"][0]
            response = self.generator.run(
                messages=[ChatMessage.from_user(content_parts=[TextContent(text=prompt), image_content])],
                generation_kwargs=generation_kwargs("look"),
            )
            record_usage("look", MODELS["look"], response["replies"][0].meta.get("usage"))
            result = json.loads(strip_json_fences(response["replies"][0].text or ""))
        except json.JSONDecodeError as e:
            _log.error("Look returned unparseable JSON: %s", e)
            return {"result": {"status": "error", "error": "unparseable look response", "focus": focus}}
        except Exception as e:
            _log.error("Look failed: %s: %s", type(e).__name__, e)
            return {"result": {"status": "error", "error": str(e), "focus": focus}}

        is_radiology = result.get("is_radiology_image", True)
        if isinstance(is_radiology, str):
            is_radiology = is_radiology.strip().lower() not in {"false", "no", "0"}
        if not is_radiology and not focus:
            raise NonRadiologyImageError(
                "The uploaded image does not appear to be a radiology image. "
                "Please upload a valid X-ray, CT, MRI, or Ultrasound."
            )

        result["status"] = "success"
        result["focus"] = focus
        result["quality_warning"] = result.get("image_quality") == "poor"
        _log.info(
            "Look done  focus=%r  modality=%s  region=%s  relevant=%d  duration=%dms",
            focus, result.get("modality"), result.get("body_region"),
            len(result.get("question_relevant_observations", []) or []),
            int((time.perf_counter() - started) * 1000),
        )
        return {"result": result}
