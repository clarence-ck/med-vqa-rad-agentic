import base64
import json
from typing import Dict, Any

import httpx
from haystack import component

from config import LLAVA_API_BASE_URL, LLAVA_API_KEY, LLAVA_MODEL_ID, LLAVA_TIMEOUT_SECONDS
from utils.json_utils import strip_json_fences


@component
class ImageInterpreter:
    """
    Agent 1: Image Interpretation Agent
    Takes a medical image and extracts modality, anatomy, visible structures.
    """
    def __init__(self):
        self.api_base_url = LLAVA_API_BASE_URL.rstrip("/")
        self.api_key = LLAVA_API_KEY
        self.model_id = LLAVA_MODEL_ID
        self.timeout_seconds = LLAVA_TIMEOUT_SECONDS

    def _extract_text(self, payload: dict) -> str:
        """Handle common OpenAI-compatible and inference-server response shapes."""
        if not isinstance(payload, dict):
            return ""

        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message", {}) or {}
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )

        for key in ("generated_text", "output", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    return first.get("generated_text") or first.get("text") or ""

        return ""

    def _call_llava_api(self, prompt: str, image_bytes: bytes, mime_type: str) -> str:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 3000,
        }

        chat_completions_url = f"{self.api_base_url}/v1/chat/completions"
        legacy_generate_url = self.api_base_url

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(chat_completions_url, headers=headers, json=payload)
            if response.status_code == 404:
                legacy_payload = {
                    "inputs": {
                        "image": f"data:{mime_type};base64,{b64_image}",
                        "prompt": prompt,
                    },
                    "parameters": {
                        "max_new_tokens": 3000,
                        "temperature": 0,
                    },
                }
                response = client.post(legacy_generate_url, headers=headers, json=legacy_payload)

            response.raise_for_status()
            return self._extract_text(response.json())

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
            raw = self._call_llava_api(prompt, image_bytes, mime_type).strip()
            raw = strip_json_fences(raw)
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
