import base64
import json
import time
from typing import Any, Dict

from haystack import component
from haystack.components.converters.image import ImageFileToImageContent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ByteStream, ChatMessage, ImageContent, TextContent
from haystack.utils import Secret

from config import OPENAI_API_KEY, MODELS, generation_kwargs
from logger import get_logger
from utils import image_zoom
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

    def _image_content(self, image_bytes: bytes, mime_type: str) -> ImageContent:
        """Resize via Haystack's converter; if it cannot read the file (unusual or mislabelled
        formats), send the original bytes instead of failing the look."""
        contents = self.image_converter.run(sources=[ByteStream(data=image_bytes, mime_type=mime_type)])["image_contents"]
        if contents:
            return contents[0]
        _log.warning("Image converter could not read the image; sending original bytes")
        return ImageContent(base64_image=base64.b64encode(image_bytes).decode("utf-8"),
                            mime_type=mime_type, detail="high")

    def _locate_and_zoom(self, image_bytes: bytes, mime_type: str, question: str, focus: str):
        """Zoom tool: find where `focus` is, then crop and enlarge it. None if it cannot be located or cropped."""
        prompt = f"""Locate "{focus}" on the attached radiology image (the question being answered is "{question}").

Return ONLY a JSON object: {{"found": true, "box": [x0, y0, x1, y1]}}, where the box encloses that region as
fractions (0-1) of the image width and height, measured from the top-left corner of the image as displayed.
The patient's left appears on the image's right on standard radiological views. Use "found": false if the
region is not in the image."""
        try:
            response = self.generator.run(
                messages=[ChatMessage.from_user(content_parts=[TextContent(text=prompt),
                                                               self._image_content(image_bytes, mime_type)])],
                generation_kwargs=generation_kwargs("look"),
            )
            record_usage("look", MODELS["look"], response["replies"][0].meta.get("usage"))
            located = json.loads(strip_json_fences(response["replies"][0].text or ""))
        except Exception as e:
            _log.warning("Zoom: could not locate %r: %s", focus, e)
            return None
        if not located.get("found", True):
            _log.info("Zoom: %r not found on the image; plain re-look", focus)
            return None
        zoomed = image_zoom.zoom(image_bytes, located.get("box"))
        if zoomed:
            _log.info("Zoom: %r located at %s", focus, zoomed[1])
        return zoomed

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
        zoomed = self._locate_and_zoom(image_bytes, mime_type, question, focus) if focus else None
        if zoomed:
            prompt += f"""

TWO IMAGES are attached: (1) the full image; (2) an ENLARGED CROP of the region where "{focus}" was located
(box {list(zoomed[1])}, as fractions of the full image's width and height from its top-left corner). Use the crop
for fine detail and the full image for context and laterality."""
        try:
            parts = [TextContent(text=prompt), self._image_content(image_bytes, mime_type)]
            if zoomed:
                parts.append(ImageContent(base64_image=base64.b64encode(zoomed[0]).decode("utf-8"),
                                          mime_type="image/png", detail="high"))
            response = self.generator.run(
                messages=[ChatMessage.from_user(content_parts=parts)],
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
        result["zoom_box"] = list(zoomed[1]) if zoomed else None
        result["quality_warning"] = result.get("image_quality") == "poor"
        _log.info(
            "Look done  focus=%r  modality=%s  region=%s  relevant=%d  duration=%dms",
            focus, result.get("modality"), result.get("body_region"),
            len(result.get("question_relevant_observations", []) or []),
            int((time.perf_counter() - started) * 1000),
        )
        return {"result": result}
