import json
import re
import time
from typing import Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import OPENAI_API_KEY, MODELS, generation_kwargs
from utils.json_utils import strip_json_fences
from utils.usage import record_usage
from logger import get_logger

_log = get_logger("agents.question_analyzer")


def _has_term(text: str, term: str) -> bool:
    """Whole-word/phrase match, so 'ct' does not fire inside 'fracture' or 'structure'."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _has_any(text: str, terms) -> bool:
    return any(_has_term(text, term) for term in terms)


@component
class QuestionAnalyzer:
    """
    Agent 2: Question Understanding Agent
    Classifies the user question into type, target entity, reasoning needed, and answer format.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=MODELS["analyze"],
        )
        self._allowed_question_subtypes = {
            "lesion_location",
            "generic_location",
            "finding_description",
            "counting_exact",
            "counting_approx",
            "yes_no_clinical",
            "presence_absence",
            "anatomy_identification",
            "modality_identification",
            "explanation_request",
            "other",
        }

    def _sanitize_question_subtype(self, subtype: str) -> str:
        value = (subtype or "").strip()
        return value if value in self._allowed_question_subtypes else ""

    def _infer_question_style(self, question: str) -> dict:
        q = (question or "").strip().lower()
        if not q:
            return {}

        anatomy_words = [
            "what organ", "which organ", "what body part", "which body part",
            "what structure", "which structure", "what anatomy", "anatomical",
            "what is shown", "what is visible", "what region", "body region",
            "identify the organ", "identify the structure", "identify the anatomy",
            "what is the name of", "what is the anatomical name of", "name this organ",
            "name this structure", "what part of the body", "which part of the body",
        ]
        modality_words = [
            "what modality", "which modality", "imaging modality", "scan type",
            "xray", "x-ray", "ct", "mri", "ultrasound", "radiograph", "plain film",
            "what kind of scan", "what type of scan", "what kind of image", "what type of image",
            "what is this scan", "what is this image", "what kind of xray", "what type of xray",
        ]
        explanation_words = [
            "why", "what does", "what is the significance", "what is going on",
            "describe the image", "explain", "interpret", "what could this be",
            "what does this mean", "what is the impression", "what is the diagnosis",
        ]

        starts_binary = bool(re.match(r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|should)\b", q))
        starts_counting = bool(re.match(r"^(how many|count|what number|what is the number|what is the count|number of)\b", q))
        starts_location = bool(re.match(r"^(where|which side|what side|on which side|where is|where are)\b", q))
        polarity_words = [
            "normal", "abnormal", "present", "absent", "visible", "seen", "identifiable",
            "well-defined", "well defined", "enlarged", "ring-enhancing", "ring enhancing",
            "hyperinflated", "shifted", "fracture", "lesion", "mass", "effusion", "pneumothorax",
        ]
        has_polarity = _has_any(q, polarity_words)
        counting_words = [
            "how many", "count", "number of", "number", "total", "total number", "many",
            "single", "double", "triple", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten",
        ]
        has_counting_signal = _has_any(q, counting_words)
        location_words = [
            "where", "located", "location", "side", "left", "right", "bilateral", "midline",
            "upper", "lower", "anterior", "posterior", "superior", "inferior", "medial", "lateral",
        ]
        has_location_signal = _has_any(q, location_words)
        explicit_choice = _has_any(q, [
            "normal or abnormal", "abnormal or normal", "present or absent", "absent or present",
            "benign or malignant", "malignant or benign", "left or right", "right or left",
        ])

        if starts_counting or (has_counting_signal and re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q)):
            return {
                "question_type": "counting",
                "expected_answer_format": "single_word",
                "reasoning_type": "visual_recognition",
                "question_clarity": "clear",
            }

        if starts_location or (has_location_signal and _has_any(q, ["where", "located", "side"])):
            return {
                "question_type": "location",
                "expected_answer_format": "short_phrase",
                "reasoning_type": "spatial_reasoning",
                "question_clarity": "clear",
            }

        if explicit_choice:
            return {
                "question_type": "abnormality_description",
                "expected_answer_format": "short_phrase",
                "reasoning_type": "comparison",
                "question_clarity": "clear",
            }

        if _has_any(q, modality_words) or (
            _has_any(q, {"xray", "x-ray", "ct", "mri", "ultrasound", "radiograph"})
            and _has_any(q, {"what", "which", "identify", "determine", "is this", "is it"})
        ):
            return {
                "question_type": "modality_identification",
                "expected_answer_format": "short_phrase",
                "reasoning_type": "visual_recognition",
                "question_clarity": "clear",
            }

        if _has_any(q, anatomy_words) or (
            _has_any(q, {"organ", "structure", "body part", "anatomy", "anatomical", "region"})
            and _has_any(q, {"what", "which", "identify", "name", "is this", "is it"})
        ):
            return {
                "question_type": "anatomy_identification",
                "expected_answer_format": "short_phrase",
                "reasoning_type": "visual_recognition",
                "question_clarity": "clear",
            }

        if _has_any(q, explanation_words) or (
            q.startswith("what") and _has_any(q, {"mean", "diagnosis", "impression", "caused", "cause"})
        ):
            return {
                "question_type": "explanation_request",
                "expected_answer_format": "sentence",
                "reasoning_type": "medical_knowledge",
                "question_clarity": "clear",
            }

        if starts_binary and (has_polarity or has_location_signal):
            # Keep binary yes/no questions strict, but do not let location-style or polarity-style wording
            # get downgraded into generic yes/no when the answer needs an anatomical phrase.
            if has_location_signal and not has_polarity:
                return {
                    "question_type": "location",
                    "expected_answer_format": "short_phrase",
                    "reasoning_type": "spatial_reasoning",
                    "question_clarity": "clear",
                }
            if has_polarity and not has_location_signal:
                return {
                    "question_type": "presence_absence",
                    "expected_answer_format": "yes_no",
                    "reasoning_type": "visual_recognition",
                    "question_clarity": "clear",
                }

        if starts_binary and has_polarity:
            return {
                "question_type": "yes_no",
                "expected_answer_format": "yes_no",
                "reasoning_type": "visual_recognition",
                "question_clarity": "clear",
            }

        if starts_binary:
            return {
                "question_type": "yes_no",
                "expected_answer_format": "yes_no",
                "reasoning_type": "visual_recognition",
                "question_clarity": "clear",
            }

        return {}

    def _infer_question_subtype(self, question: str, style_hint: dict, target_entity: str = "") -> str:
        q = (question or "").strip().lower()
        question_type = (style_hint or {}).get("question_type", "")
        target = (target_entity or "").strip().lower()

        lesion_terms = (
            "mass", "lesion", "nodule", "opacity", "tumor", "tumour", "masslike", "mass-like",
            "abnormality", "finding", "calcification", "effusion", "pneumothorax",
        )
        counting_cues = ("about", "approximately", "approx", "roughly", "estimate", "estimated", "around")

        if question_type == "counting":
            if _has_any(q, counting_cues):
                return "counting_approx"
            return "counting_exact"

        if question_type == "location":
            if _has_any(q, lesion_terms) or _has_any(target, lesion_terms):
                return "lesion_location"
            return "generic_location"

        if question_type == "abnormality_description":
            return "finding_description"

        if question_type in {"yes_no", "presence_absence"}:
            return "yes_no_clinical"

        if question_type == "anatomy_identification":
            return "anatomy_identification"

        if question_type == "modality_identification":
            return "modality_identification"

        if question_type == "explanation_request":
            return "explanation_request"

        return "other"

    def _resolve_question_subtype(self, question_type: str, model_subtype: str, heuristic_subtype: str) -> str:
        model_subtype = self._sanitize_question_subtype(model_subtype)
        heuristic_subtype = (heuristic_subtype or "").strip()

        if not heuristic_subtype or heuristic_subtype == "other":
            return model_subtype or heuristic_subtype or "other"

        if question_type == "counting":
            return heuristic_subtype
        if question_type == "location":
            return heuristic_subtype
        if question_type in {"yes_no", "presence_absence", "abnormality_description", "anatomy_identification", "modality_identification", "explanation_request"}:
            return heuristic_subtype

        return model_subtype or heuristic_subtype or "other"

    def _apply_style_override(self, result: dict, style_hint: dict) -> dict:
        if not style_hint:
            return result

        merged = dict(result or {})
        for key, value in style_hint.items():
            if value is not None:
                merged[key] = value
        return merged

    @component.output_types(result=Dict[str, Any])
    def run(self, question: str, image_interpretation: dict = None):
        visual_summary = ""
        if image_interpretation and isinstance(image_interpretation, dict):
            visual_summary = image_interpretation.get("visual_summary", "")

        style_hint = self._infer_question_style(question)

        _log.info("Analyzing question: %r", question[:120])
        _log.debug("Visual context provided: %s", bool(visual_summary))
        started = time.perf_counter()

        question_subtype = self._infer_question_subtype(question, style_hint)

        prompt = f"""You are a medical VQA question understanding agent. Analyze the following question asked about a medical image.

Question: "{question}"

Visual context (from image analysis): {visual_summary if visual_summary else "Not available"}

Return a JSON object with:
{{
  "question_type": "One of: anatomy_identification, modality_identification, presence_absence, location, counting, abnormality_description, explanation_request, yes_no, other",
  "question_subtype": "One of: lesion_location, generic_location, finding_description, counting_exact, counting_approx, yes_no_clinical, presence_absence, anatomy_identification, modality_identification, explanation_request, other",
  "target_entity": "What the question is asking about (e.g., 'body region', 'organ', 'modality', 'abnormality')",
  "reasoning_type": "One of: visual_recognition, spatial_reasoning, medical_knowledge, comparison, descriptive",
  "expected_answer_format": "One of: single_word, short_phrase, yes_no, sentence, paragraph",
  "requires_medical_knowledge": true/false,
  "question_clarity": "clear / ambiguous / unclear",
  "rephrased_question": "A clearer version of the question if ambiguous, otherwise repeat the original"
}}

IMPORTANT CLASSIFICATION RULES:
- If the question asks "is X A or B?" (e.g., "normal or abnormal", "benign or malignant", "left or right"), classify as: question_type="abnormality_description" or appropriate category, expected_answer_format="single_word" or "short_phrase". Do NOT use yes_no.
- Only use question_type="yes_no" and expected_answer_format="yes_no" for true binary questions that can be answered with just "yes" or "no" (e.g., "is there pneumonia?", "can you see the lesion?").

Return ONLY the JSON object, no additional text."""

        try:
            response = self.generator.run(
                messages=[ChatMessage.from_user(prompt)],
                generation_kwargs=generation_kwargs("analyze")
            )

            record_usage("analyze", MODELS["analyze"], response["replies"][0].meta.get("usage"))

            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            result = self._apply_style_override(result, style_hint)
            result["question_subtype"] = self._resolve_question_subtype(
                result.get("question_type", "other"),
                result.get("question_subtype"),
                question_subtype,
            )
            if result.get("question_subtype") not in self._allowed_question_subtypes:
                result["question_subtype"] = "other"
            result["status"] = "success"
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _log.info(
                "Question analysis done  type=%s  subtype=%s  reasoning=%s  med_knowledge=%s  clarity=%s  duration=%dms",
                result.get("question_type", "unknown"),
                result.get("question_subtype", "unknown"),
                result.get("reasoning_type", "unknown"),
                result.get("requires_medical_knowledge", "?"),
                result.get("question_clarity", "unknown"),
                elapsed_ms,
            )
            _log.debug(
                "Target entity: %s  Expected format: %s  Rephrased: %s",
                result.get("target_entity", "unknown"),
                result.get("expected_answer_format", "unknown"),
                result.get("rephrased_question", "")[:120],
            )
            return {"result": result}

        except json.JSONDecodeError:
            _log.error("Failed to parse question analysis JSON response")
            return {"result": {
                "status": "error",
                "error": "Failed to parse question analysis response",
                "question_type": "other",
                "question_subtype": question_subtype,
                "target_entity": "unknown",
                "reasoning_type": "visual_recognition",
                "expected_answer_format": "short_phrase",
                "requires_medical_knowledge": False,
                "question_clarity": "unclear",
                "rephrased_question": question
            }}
        except Exception as e:
            _log.error("Unexpected error in question analysis: %s", e)
            return {"result": {
                "status": "error",
                "error": str(e),
                "question_type": "other",
                "question_subtype": question_subtype,
                "target_entity": "unknown",
                "reasoning_type": "visual_recognition",
                "expected_answer_format": "short_phrase",
                "requires_medical_knowledge": False,
                "question_clarity": "unclear",
                "rephrased_question": question
            }}

# Wrapper function for compatibility if called directly
def analyze_question(question: str, visual_summary: str = "") -> dict:
    analyzer = QuestionAnalyzer()
    simulated_input = {"visual_summary": visual_summary} if visual_summary else None
    return analyzer.run(question=question, image_interpretation=simulated_input)["result"]
