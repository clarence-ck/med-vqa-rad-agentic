import json
import time
from typing import Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import OPENAI_API_KEY, OPENAI_MODEL
from utils.json_utils import strip_json_fences
from logger import get_logger

_log = get_logger("agents.difficulty_router")


@component
class DifficultyRouter:
    """
    Agent 3: Difficulty Routing / Planning Agent

    Evaluates the complexity of the VQA question and determines the
    appropriate pipeline depth for the downstream agents:
    - Whether knowledge retrieval is needed
    - How strict verification should be
    - What fallback policy to apply if confidence is low
    """

    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=OPENAI_MODEL,
        )

    @component.output_types(result=Dict[str, Any])
    def run(self, question_analysis: dict = None, image_interpretation: dict = None):
        qa = question_analysis or {}
        ii = image_interpretation or {}

        question_type = qa.get("question_type", "unknown")
        requires_medical_knowledge = qa.get("requires_medical_knowledge", False)
        reasoning_type = qa.get("reasoning_type", "visual_recognition")
        question_clarity = qa.get("question_clarity", "clear")
        expected_answer_format = qa.get("expected_answer_format", "short_phrase")
        modality = ii.get("modality", "unknown")
        body_region = ii.get("body_region", "unknown")
        visual_summary = ii.get("visual_summary", "") or ""

        _log.info(
            "Routing difficulty  question_type=%s  reasoning=%s  med_knowledge=%s  modality=%s  region=%s",
            question_type, reasoning_type, requires_medical_knowledge, modality, body_region,
        )

        # HARD RULE: Bypass retrieval for question types that are harmed by retrieval
        # These question types rely on direct visual interpretation and retrieval contamination
        # has been empirically shown to reduce accuracy (e.g., yes/no drops from 50% to 17%)
        RETRIEVAL_INCOMPATIBLE_TYPES = {"yes_no", "presence_absence", "location", "counting"}
        if question_type in RETRIEVAL_INCOMPATIBLE_TYPES:
            _log.info(
                "Hard rule: question_type='%s' is retrieval-incompatible, forcing easy/no-retrieval",
                question_type
            )
            return {"result": {
                "difficulty": "easy",
                "needs_retrieval": False,
                "retrieval_rationale": f"Question type '{question_type}' is incompatible with retrieval (binary/spatial/counting questions rely on direct visual interpretation)",
                "verification_policy": "standard",
                "fallback_policy": "none",
                "routing_rationale": f"Hard rule override: {question_type} questions bypass retrieval to avoid contamination",
                "status": "success"
            }}

        # HARD RULE: Poor image quality requires conservative handling
        # Don't compound bad image with questionable retrieval
        image_quality = ii.get("image_quality", "unknown")
        quality_warning = ii.get("quality_warning", False)
        if quality_warning or image_quality == "poor":
            _log.info(
                "Hard rule: image_quality='%s' with quality_warning=%s, forcing hard/strict/no-retrieval",
                image_quality, quality_warning
            )
            return {"result": {
                "difficulty": "hard",
                "needs_retrieval": False,
                "retrieval_rationale": "Poor image quality - retrieval would be unreliable, using strict visual-only analysis",
                "verification_policy": "strict",
                "fallback_policy": "flag_for_review",
                "routing_rationale": f"Hard rule: poor image quality ({image_quality}) requires conservative handling with human review flag",
                "status": "success"
            }}

        started = time.perf_counter()

        prompt = f"""You are a medical VQA routing and planning agent.

Given the following question analysis and image interpretation, act like a professional radiologist/radiographer: judge the case by how conspicuous the pathology is, how complex the anatomical background is, and how much the image is affected by overlap, variants, or artifacts, then determine the appropriate pipeline configuration.

QUESTION ANALYSIS:
- Question type: {question_type}
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
  "needs_retrieval": true or false,
  "retrieval_rationale": "One sentence explaining why retrieval is or is not needed",
  "verification_policy": "One of: standard, strict",
  "fallback_policy": "One of: none, add_disclaimer, flag_for_review",
  "routing_rationale": "One sentence summarizing the overall routing decision"

}}

ROUTING RULES:
1. ROUTING PRIORITY
- Prefer "easy" and no retrieval when the question is direct, image-grounded, and can be answered from the current visual analysis.
- Use retrieval only when the question genuinely needs broader medical context, is ambiguous, or cannot be answered confidently from the current image interpretation alone.

2. RADIOLOGY-STYLE DIFFICULTY LEVEL (Combine Question + Image Context)
- "easy": A classic spot diagnosis on a clean study. The abnormality is conspicuous, the anatomy is straightforward, and the answer should be evident on first pass: obvious fracture, large consolidation, large foreign body, textbook anatomy, or a direct yes/no answer on an uncomplicated image.
- "medium": A systematic search case. The finding is present but subtle, partially obscured, or easy to overlook, so the image requires deliberate step-by-step review: hairline or non-displaced fracture, small nodule, early disease, mild overlap, or an anatomic variant that must be separated from true pathology.
- "hard": A perception challenge. The abnormality is near the limits of visibility or buried in a noisy study, with blind-spot locations, tiny pneumothorax, subtle mediastinal change, ICU lines/tubes, summation effects, or multiple competing clues that must be integrated to avoid a false-negative interpretation.

3. STRICT POLICY MAPPING
Policies are strictly determined by the holistic difficulty level:
- If difficulty="easy":
   -> needs_retrieval=false
   -> verification_policy="standard"
   -> fallback_policy="none"
- If difficulty="medium":
   -> needs_retrieval=true
   -> verification_policy="standard"
   -> fallback_policy="add_disclaimer"
- If difficulty="hard":
   -> needs_retrieval=true
   -> verification_policy="strict"
   -> fallback_policy="flag_for_review"

Return ONLY the JSON object, no additional text."""

        try:
            response = self.generator.run(
                messages=[ChatMessage.from_user(prompt)],
                generation_kwargs={"max_completion_tokens": 500},
            )
            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            result["status"] = "success"
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _log.info(
                "Routing decision  difficulty=%s  needs_retrieval=%s  "
                "verification=%s  fallback=%s  duration=%dms",
                result.get("difficulty", "unknown"),
                result.get("needs_retrieval", "?"),
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
                "needs_retrieval": bool(requires_medical_knowledge),
                "retrieval_rationale": "Fallback: defaulting to requires_medical_knowledge flag due to parse error.",
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
                "needs_retrieval": bool(requires_medical_knowledge),
                "retrieval_rationale": "Fallback: defaulting to requires_medical_knowledge flag due to exception.",
                "verification_policy": "standard",
                "fallback_policy": "none",
                "routing_rationale": "Fallback routing applied due to exception.",
            }}
