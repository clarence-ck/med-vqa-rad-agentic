import base64
import json
import time
from io import BytesIO
from typing import Dict, Any

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, TextContent, ImageContent
from haystack.utils import Secret
from PIL import Image

from config import OPENAI_API_KEY, VISION_MODEL
from utils.json_utils import strip_json_fences
from logger import get_logger

_log = get_logger("agents.image_interpreter")


@component
class ImageInterpreter:
    """
    Agent 1: Image Interpretation Agent
    Takes a medical image and extracts modality, anatomy, visible structures.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model=VISION_MODEL)

    @component.output_types(result=Dict[str, Any])
    def run(self, image_bytes: bytes, mime_type: str = "image/png"):
        prompt = """You are a medical image interpretation agent. Analyze this medical image and return a JSON object with the following fields:

{
  "modality": "The imaging modality (e.g., X-ray, CT, MRI, Ultrasound, Fundoscopy, Dermoscopy, etc.)",
  "body_region": "The anatomical region shown (e.g., chest, head, abdomen, knee, etc.)",
  "visible_structures": ["List of visible anatomical structures"],
  "visual_summary": "A 2-3 sentence summary of what is visible in the image",
  "image_quality": "good / fair / poor",
  "notable_findings": ["Any notable visual observations, without making diagnoses"]
}

Return ONLY the JSON object, no additional text."""

        _log.info("Running image interpretation (mime_type=%s, size=%d bytes)", mime_type, len(image_bytes))
        started = time.perf_counter()
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            messages = [
                ChatMessage.from_user(content_parts=[
                    TextContent(text=prompt),
                    ImageContent(base64_image=image_base64, mime_type=mime_type)
                ])
            ]
            response = self.generator.run(messages=messages, generation_kwargs={"max_completion_tokens": 3000})
            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            result["status"] = "success"
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _log.info(
                "Image interpretation done  modality=%s  body_region=%s  quality=%s  "
                "findings=%d  duration=%dms",
                result.get("modality", "unknown"),
                result.get("body_region", "unknown"),
                result.get("image_quality", "unknown"),
                len(result.get("notable_findings", [])),
                elapsed_ms,
            )
            return {"result": result}

        except json.JSONDecodeError as e:
            _log.error("JSON parse error: %s", e)
            _log.debug("Raw response: %s", raw if 'raw' in locals() else 'N/A')
            return {"result": {
                "status": "error",
                "error": "Failed to parse image interpretation response",
                "raw_response": raw if 'raw' in locals() else "",
                "modality": "unknown",
                "body_region": "unknown",
                "visible_structures": [],
                "visual_summary": "Image interpretation failed.",
                "image_quality": "unknown",
                "notable_findings": []
            }}
        except Exception as e:
            _log.error("Unexpected error: %s: %s", type(e).__name__, e)
            return {"result": {
                "status": "error",
                "error": str(e),
                "modality": "unknown",
                "body_region": "unknown",
                "visible_structures": [],
                "visual_summary": "Image interpretation failed.",
                "image_quality": "unknown",
                "notable_findings": []
            }}

# Wrapper function to maintain compatibility if called alone, though Pipeline is preferred.
def interpret_image(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    interpreter = ImageInterpreter()
    return interpreter.run(image_bytes=image_bytes, mime_type=mime_type)["result"]
