import json
import re
import time
from typing import Dict, Any

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.converters.image import ImageFileToImageContent
from haystack.dataclasses import ChatMessage, TextContent, ImageContent, ByteStream
from haystack.utils import Secret

from config import OPENAI_API_KEY, MODELS, generation_kwargs
from utils.json_utils import strip_json_fences
from utils.usage import record_usage
from logger import get_logger

_log = get_logger("agents.image_interpreter")

_APPROX_COUNT_PRIORITY = ("multiple", "several", "few")


_COUNT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}

_COUNT_LABELS = (
    "nodules?",
    "lesions?",
    "fractures?",
    "opacities?",
    "masses?",
    "calcifications?",
    "effusions?",
    "foreign bodies?",
    "foci",
    "focal opacities?",
)


def _normalize_count_value(value: str) -> int | str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in _COUNT_WORDS:
        return _COUNT_WORDS[text]
    if text in {"multiple", "few", "several"}:
        return text
    if text in {"a dozen", "dozen"}:
        return 12
    if text in {"half dozen", "half a dozen"}:
        return 6
    return None


def _words_to_int(value: str) -> int | str | None:
    text = str(value or "").strip().lower().replace("-", " ")
    if not text:
        return None
    if text in _COUNT_WORDS:
        return _COUNT_WORDS[text]
    parts = [part for part in text.split() if part]
    if len(parts) == 1:
        return _COUNT_WORDS.get(parts[0])
    if len(parts) == 2 and parts[0] in {"twenty", "thirty", "forty", "fifty"} and parts[1] in {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    }:
        return _COUNT_WORDS[parts[0]] + _COUNT_WORDS[parts[1]]
    return None


def _infer_count_from_text(text: str) -> dict:
    lowered = str(text or "").lower()
    if not lowered:
        return {}

    approx_terms = {"multiple", "few", "several"}
    paired_terms = {"bilateral", "both lungs", "both kidneys", "both sides", "paired", "pair of", "both"}
    single_terms = {"single", "solitary", "one", "only one"}

    patterns = [
        rf"\b(\d{{1,2}})\s+(?:visible\s+)?(?P<label>{'|'.join(_COUNT_LABELS)})\b",
        rf"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)(?:\s+(one|two|three|four|five|six|seven|eight|nine))?\s+(?:visible\s+)?(?P<label>{'|'.join(_COUNT_LABELS)})\b",
        rf"\b(multiple|few|several)\s+(?P<label>{'|'.join(_COUNT_LABELS)})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        raw_count = match.group(1)
        secondary = match.group(2) if match.lastindex and match.lastindex >= 2 else None
        count = _normalize_count_value(raw_count)
        if count is None:
            count = _words_to_int(raw_count)
        if isinstance(count, int) and secondary:
            secondary_value = _normalize_count_value(secondary)
            if isinstance(secondary_value, int):
                count = count + secondary_value
        return {
            "label": match.groupdict().get("label", match.group(0)),
            "count_estimate": count,
            "count_confidence": "high" if isinstance(count, int) and raw_count.isdigit() else "medium" if isinstance(count, int) else "low",
            "count_source": "text_inference",
        }

    if any(term in lowered for term in paired_terms):
        return {
            "label": "paired_structure",
            "count_estimate": 2,
            "count_confidence": "low",
            "count_source": "text_inference",
        }

    if any(term in lowered for term in single_terms):
        return {
            "label": "single_structure",
            "count_estimate": 1,
            "count_confidence": "low",
            "count_source": "text_inference",
        }

    range_match = re.search(r"\b(\d{1,2}|[a-z-]+)\s*(?:-|to|or)\s*(\d{1,2}|[a-z-]+)\b", lowered)
    if range_match:
        left_raw, right_raw = range_match.group(1), range_match.group(2)
        left_value = int(left_raw) if left_raw.isdigit() else _words_to_int(left_raw)
        right_value = int(right_raw) if right_raw.isdigit() else _words_to_int(right_raw)
        if left_value is not None and right_value is not None and 0 <= left_value <= 50 and 0 <= right_value <= 50:
            low, high = sorted((int(left_value), int(right_value)))
            return {
                "label": "range",
                "count_estimate": f"{low}-{high}",
                "count_confidence": "low",
                "count_source": "text_inference",
            }

    if any(term in lowered for term in approx_terms):
        approx_value = next((term for term in _APPROX_COUNT_PRIORITY if term in lowered), "multiple")
        return {
            "label": "approximate_count",
            "count_estimate": approx_value,
            "count_confidence": "low",
            "count_source": "text_inference",
        }

    return {}


def _build_count_metadata(result: dict) -> dict:
    visual_summary = str(result.get("visual_summary", "") or "")
    notable_findings = result.get("notable_findings", []) or []
    visible_structures = result.get("visible_structures", []) or []

    countable_objects = []
    count_signals = []
    scan_items = []
    for finding in notable_findings:
        if not isinstance(finding, str):
            continue
        scan_items.append(("notable_findings", finding))

    for structure in visible_structures:
        if not isinstance(structure, str):
            continue
        scan_items.append(("visible_structures", structure))

    if visual_summary:
        scan_items.append(("visual_summary", visual_summary))

    for source_name, text in scan_items:
        inferred = _infer_count_from_text(text)
        if inferred:
            countable_objects.append({"source": source_name, "text": text, **inferred})
            count_signals.append(inferred)

    count_estimate = result.get("count_estimate")
    count_confidence = result.get("count_confidence", "unknown")
    count_method = result.get("count_method", "none")

    if countable_objects:
        if count_estimate in {None, "", "unknown"}:
            exact_candidates = [obj.get("count_estimate") for obj in countable_objects if isinstance(obj.get("count_estimate"), int)]
            range_candidates = [obj.get("count_estimate") for obj in countable_objects if isinstance(obj.get("count_estimate"), str) and "-" in str(obj.get("count_estimate"))]
            approximate_candidates = [obj.get("count_estimate") for obj in countable_objects if obj.get("count_estimate") in {"multiple", "few", "several"}]

            unique_exact = sorted(set(exact_candidates))
            if len(unique_exact) == 1:
                count_estimate = unique_exact[0]
            elif len(unique_exact) > 1:
                count_estimate = "unknown"
            elif range_candidates:
                count_estimate = range_candidates[0]
            elif approximate_candidates:
                count_estimate = "multiple" if "multiple" in approximate_candidates else approximate_candidates[0]
            else:
                count_estimate = countable_objects[0].get("count_estimate")

        if count_confidence == "unknown":
            if isinstance(count_estimate, int) and any(signal.get("count_confidence") == "high" for signal in count_signals):
                count_confidence = "high"
            elif isinstance(count_estimate, int) and any(signal.get("count_confidence") == "medium" for signal in count_signals):
                count_confidence = "medium"
            elif isinstance(count_estimate, str) and "-" in count_estimate:
                count_confidence = "low"
            elif count_estimate in {"multiple", "few", "several"}:
                count_confidence = "low"
            elif count_estimate == "unknown":
                count_confidence = "low"
            else:
                count_confidence = countable_objects[0].get("count_confidence", "low")
        if count_method == "none":
            count_method = "text_inference"

    return {
        "countable_objects": countable_objects,
        "count_estimate": count_estimate,
        "count_confidence": count_confidence,
        "count_method": count_method,
    }


@component
class ImageInterpreter:
    """
    Agent 1: Image Interpretation Agent
    Takes a medical image and extracts modality, anatomy, visible structures.
    Uses ImageFileToImageContent for standardized image processing.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model=MODELS["look"])
        self.image_converter = ImageFileToImageContent(detail="high", size=(1024, 1024))

    @component.output_types(result=Dict[str, Any])
    def run(self, image_bytes: bytes, mime_type: str = "image/png"):
        prompt = """You are a medical image interpretation agent. Analyze this medical image and return a JSON object with the following fields.

IMPORTANT INSTRUCTION: You MUST evaluate if the image is a radiology image (e.g. X-ray, CT, MRI, Ultrasound, Fluoroscopy). If it is a generic photo (like a cat, a logo, or a car), or a non-radiology medical image (like Dermoscopy or Fundoscopy), you MUST set "is_radiology_image" to false.

If the image clearly contains countable repeated findings or objects, also provide count-related fields:
  - "countable_objects": list of objects/finding groups with a count estimate
  - "count_estimate": a number or approximate label if count is clear
  - "count_confidence": high / medium / low
  - "count_method": detector_based / visual_estimate / text_inference / none
Only include count fields when the count is reasonably supported by the image; otherwise omit them or set them conservatively.

{
  "is_radiology_image": true,
  "modality": "The imaging modality (e.g., X-ray, CT, MRI, Ultrasound, Fluoroscopy, etc.)",
  "body_region": "The anatomical region shown (e.g., chest, head, abdomen, knee, etc.)",
  "visible_structures": ["List of visible anatomical structures"],
  "visual_summary": "A 2-3 sentence summary of what is visible in the image",
  "image_quality": "good / fair / poor",
  "notable_findings": ["List of visual observations. Each should be a concrete, observable description without diagnostic terms. Focus on what you see, not what it means."],
  "countable_objects": [
    {
      "label": "what is being counted",
      "count_estimate": 1,
      "count_confidence": "high / medium / low",
      "count_method": "detector_based / visual_estimate / text_inference"
    }
  ],
  "count_estimate": "optional overall count estimate",
  "count_confidence": "optional overall confidence",
  "count_method": "optional overall method"
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
            response = self.generator.run(messages=messages, generation_kwargs=generation_kwargs("look"))
            record_usage("look", MODELS["look"], response["replies"][0].meta.get("usage"))
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

            count_metadata = _build_count_metadata(result)
            result.setdefault("countable_objects", count_metadata["countable_objects"])
            if result.get("count_estimate") in {None, "", "unknown"}:
                result["count_estimate"] = count_metadata["count_estimate"]
            if result.get("count_confidence", "unknown") == "unknown":
                result["count_confidence"] = count_metadata["count_confidence"]
            if result.get("count_method", "none") == "none":
                result["count_method"] = count_metadata["count_method"]

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _log.info(
                "Image interpretation done  modality=%s  body_region=%s  quality=%s  findings=%d  count=%s  duration=%dms",
                result.get("modality", "unknown"),
                result.get("body_region", "unknown"),
                result.get("image_quality", "unknown"),
                len(result.get("notable_findings", [])),
                result.get("count_estimate", "unknown"),
                elapsed_ms,
            )
            _log.debug("Visual summary: %s", result.get("visual_summary", ""))
            _log.debug("Notable findings: %s", result.get("notable_findings", []))
            _log.debug("Count metadata: %s", count_metadata)
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
                "notable_findings": [],
                "countable_objects": [],
                "count_estimate": "unknown",
                "count_confidence": "unknown",
                "count_method": "none",
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
                "notable_findings": [],
                "countable_objects": [],
                "count_estimate": "unknown",
                "count_confidence": "unknown",
                "count_method": "none",
            }}

# Wrapper function to maintain compatibility if called alone, though Pipeline is preferred.
def interpret_image(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    interpreter = ImageInterpreter()
    return interpreter.run(image_bytes=image_bytes, mime_type=mime_type)["result"]
