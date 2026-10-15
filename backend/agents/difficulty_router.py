import json
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

_log = get_logger("agents.difficulty_router")


@component
class DifficultyRouter:
    """
    Difficulty routing (fixed pipeline only).

    Evaluates the complexity of the VQA question and determines the
    appropriate handling for the downstream agents:
    - How strict verification should be
    - What fallback policy to apply if confidence is low
    """

    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=MODELS["analyze"],
        )

    @component.output_types(result=Dict[str, Any])
    def run(self, question_analysis: dict = None, image_interpretation: dict = None):
        qa = question_analysis or {}
        ii = image_interpretation or {}

        question_type = qa.get("question_type", "unknown")
        question_subtype = qa.get("question_subtype", "other")
        requires_medical_knowledge = qa.get("requires_medical_knowledge", False)
        reasoning_type = qa.get("reasoning_type", "visual_recognition")
        question_clarity = qa.get("question_clarity", "clear")
        expected_answer_format = qa.get("expected_answer_format", "short_phrase")
        modality = ii.get("modality", "unknown")
        body_region = ii.get("body_region", "unknown")
        visual_summary = ii.get("visual_summary", "") or ""

        _log.info(
            "Routing difficulty  question_type=%s  question_subtype=%s  reasoning=%s  med_knowledge=%s  modality=%s  region=%s",
            question_type, question_subtype, reasoning_type, requires_medical_knowledge, modality, body_region,
        )

        # Direct visual questions need no LLM call to route: label them easy.
        DIRECT_VISUAL_TYPES = {"yes_no", "presence_absence", "counting"}
        DIRECT_VISUAL_SUBTYPES = {"counting_exact", "counting_approx", "yes_no_clinical"}
        if question_type in DIRECT_VISUAL_TYPES or question_subtype in DIRECT_VISUAL_SUBTYPES:
            _log.info(
                "Rule: question_type='%s' question_subtype='%s' is a direct visual question, routing easy",
                question_type,
                question_subtype,
            )
            return {"result": {
                "difficulty": "easy",
                "verification_policy": "standard",
                "fallback_policy": "none",
                "routing_rationale": f"Rule: {question_type} / {question_subtype} is a direct visual question",
                "status": "success"
            }}

        # HARD RULE: Poor image quality requires conservative handling
        image_quality = ii.get("image_quality", "unknown")
        quality_warning = ii.get("quality_warning", False)
        if quality_warning or image_quality == "poor":
            _log.info(
                "Hard rule: image_quality='%s' with quality_warning=%s, forcing hard/strict",
                image_quality, quality_warning
            )
            return {"result": {
                "difficulty": "hard",
                "verification_policy": "strict",
                "fallback_policy": "flag_for_review",
                "routing_rationale": f"Hard rule: poor image quality ({image_quality}) requires conservative handling with human review flag",
                "status": "success"
            }}

        started = time.perf_counter()

        prompt = f"""You are a medical VQA routing and planning agent.

Given the following question analysis and image interpretation, act like a professional radiologist/radiographer: judge the case by how conspicuous the pathology is, how complex the anatomical background is, and how much the image is affected by overlap, variants, or artifacts, then determine the appropriate handling.

QUESTION ANALYSIS:
- Question type: {question_type}
- Question subtype: {question_subtype}
- Requires medical knowledge: {requires_medical_knowledge}
- Reasoning type: {reasoning_type}
- Question clarity: {question_clarity}
- Expected answer format: {expected_answer_format}

IMAGE CONTEXT:
- Modality: {modality}
- Body region: {body_region}
- Visual summary: {visual_summary if visual_summary else "Not available"}

Return a JSON object with:
{{
  "difficulty": "One of: easy, medium, hard",
  "verification_policy": "One of: standard, strict",
  "fallback_policy": "One of: none, add_disclaimer, flag_for_review",
  "routing_rationale": "One sentence summarizing the routing decision"
}}

ROUTING RULES:
1. RADIOLOGY-STYLE DIFFICULTY LEVEL (Combine Question + Image Context)
- "easy": A classic spot diagnosis on a clean study. The abnormality is conspicuous, the anatomy is straightforward, and the answer should be evident on first pass: obvious fracture, large consolidation, large foreign body, textbook anatomy, or a direct yes/no answer on an uncomplicated image.
- "medium": A systematic search case. The finding is present but subtle, partially obscured, or easy to overlook, so the image requires deliberate step-by-step review: hairline or non-displaced fracture, small nodule, early disease, mild overlap, or an anatomic variant that must be separated from true pathology.
- "hard": A perception challenge. The abnormality is near the limits of visibility or buried in a noisy study, with blind-spot locations, tiny pneumothorax, subtle mediastinal change, ICU lines/tubes, summation effects, or multiple competing clues that must be integrated to avoid a false-negative interpretation.

2. STRICT POLICY MAPPING
Policies are strictly determined by the difficulty level:
- If difficulty="easy":   verification_policy="standard", fallback_policy="none"
- If difficulty="medium": verification_policy="standard", fallback_policy="add_disclaimer"
- If difficulty="hard":   verification_policy="strict",   fallback_policy="flag_for_review"

Return ONLY the JSON object, no additional text."""

        try:
            response = self.generator.run(
                messages=[ChatMessage.from_user(prompt)],
                generation_kwargs=generation_kwargs("analyze"),
            )
            record_usage("analyze", MODELS["analyze"], response["replies"][0].meta.get("usage"))
            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            result["status"] = "success"
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _log.info(
                "Routing decision  difficulty=%s  verification=%s  fallback=%s  duration=%dms",
                result.get("difficulty", "unknown"),
                result.get("verification_policy", "unknown"),
                result.get("fallback_policy", "unknown"),
                elapsed_ms,
            )
            _log.debug("Routing rationale: %s", result.get("routing_rationale", ""))
            return {"result": result}

        except json.JSONDecodeError:
            _log.error("Failed to parse routing decision JSON; applying fallback (difficulty=medium)")
            return {"result": {
                "status": "error",
                "difficulty": "medium",
                "verification_policy": "standard",
                "fallback_policy": "none",
                "routing_rationale": "Fallback routing applied due to JSON parse failure.",
            }}
        except Exception as e:
            _log.error("Unexpected error in difficulty routing: %s", e)
            return {"result": {
                "status": "error",
                "error": str(e),
                "difficulty": "medium",
                "verification_policy": "standard",
                "fallback_policy": "none",
                "routing_rationale": "Fallback routing applied due to exception.",
            }}
