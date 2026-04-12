import json
import time
from typing import Dict, Any

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.converters.image import ImageFileToImageContent
from haystack.dataclasses import ChatMessage, TextContent, ImageContent, ByteStream
from haystack.utils import Secret

from config import OPENAI_API_KEY, VISION_MODEL
from utils.json_utils import strip_json_fences
from logger import get_logger

_log = get_logger("agents.image_interpreter")


@component
class ImageInterpreter:
    """
    Agent 1: Image Interpretation Agent
    Takes a medical image and extracts modality, anatomy, visible structures.
    Uses ImageFileToImageContent for standardized image processing.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model=VISION_MODEL)
        self.image_converter = ImageFileToImageContent(detail="high", size=(1024, 1024))

    @component.output_types(result=Dict[str, Any])
    def run(self, image_bytes: bytes, mime_type: str = "image/png"):
        prompt = """You are a medical image interpretation agent. Analyze this medical image and return a JSON object with the following fields.

IMPORTANT INSTRUCTION: You MUST evaluate if the image is a radiology image (e.g. X-ray, CT, MRI, Ultrasound, Fluoroscopy). If it is a generic photo (like a cat, a logo, or a car), or a non-radiology medical image (like Dermoscopy or Fundoscopy), you MUST set "is_radiology_image" to false.

{
  "is_radiology_image": true,
  "modality": "The imaging modality (e.g., X-ray, CT, MRI, Ultrasound, Fluoroscopy, etc.)",
  "body_region": "The anatomical region shown (e.g., chest, head, abdomen, knee, etc.)",
  "visible_structures": ["List of visible anatomical structures"],
  "visual_summary": "A 2-3 sentence summary of what is visible in the image",
  "image_quality": "good / fair / poor",
  "notable_findings": ["List of visual observations. Each should be a concrete, observable description without diagnostic terms. Focus on what you see, not what it means."]
}

Return ONLY the JSON object, no additional text."""

        _log.info("Running image interpretation (mime_type=%s, size=%d bytes)", mime_type, len(image_bytes))
        started = time.perf_counter()
        try:
            # Use ImageFileToImageContent for standardized image processing
            # Wrap bytes in ByteStream for the component
            stream = ByteStream(data=image_bytes, mime_type=mime_type)
            converter_result = self.image_converter.run(sources=[stream])
            image_content = converter_result["image_contents"][0]
            
            messages = [
                ChatMessage.from_user(content_parts=[
                    TextContent(text=prompt),
                    image_content  # Already ImageContent object from converter
                ])
            ]
            response = self.generator.run(messages=messages, generation_kwargs={"max_completion_tokens": 3000})
            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            
            val = result.get("is_radiology_image", result.get("is_medical_image", True))
            if isinstance(val, str):
                val = val.lower().strip() not in ("false", "no", "0")
                
            if not val:
                _log.warning("Non-radiology image detected. Aborting pipeline.")
                raise ValueError("The uploaded image does not appear to be a radiology image. Please upload a valid X-ray, CT, MRI, or Ultrasound.")

            result["status"] = "success"

            # IMAGE QUALITY GATE: Warn if image quality is poor
            image_quality = result.get("image_quality", "unknown")
            if image_quality == "poor":
                _log.warning("Image quality is poor — analysis may be unreliable. Consider providing a clearer image.")
                result["quality_warning"] = True
            else:
                result["quality_warning"] = False

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
            _log.debug("Visual summary: %s", result.get("visual_summary", ""))
            _log.debug("Notable findings: %s", result.get("notable_findings", []))
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
        except ValueError:
            # Let our intentional non-radiology blocker blow up the pipeline
            raise
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
