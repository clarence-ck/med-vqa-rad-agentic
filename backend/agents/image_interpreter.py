import base64
import json
from io import BytesIO
from typing import Dict, Any

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from PIL import Image

from config import OPENAI_API_KEY, OPENAI_MODEL
from utils.json_utils import strip_json_fences


@component
class ImageInterpreter:
    """
    Agent 1: Image Interpretation Agent
    Takes a medical image and extracts modality, anatomy, visible structures.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model=OPENAI_MODEL)

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

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ],
                },
            ]
            response = self.generator.run(messages=messages, generation_kwargs={"max_completion_tokens": 3000})
            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            result["status"] = "success"
            return {"result": result}

        except json.JSONDecodeError:
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
