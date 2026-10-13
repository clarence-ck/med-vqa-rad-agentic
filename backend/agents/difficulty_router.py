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

        # HARD RULE: Bypass retrieval only for question types/subtypes that are most harmed by it.
        # Visual-only categories can still use the normal router if the image is poor or the case is hard.
        RETRIEVAL_INCOMPATIBLE_TYPES = {"yes_no", "presence_absence", "counting"}
        RETRIEVAL_INCOMPATIBLE_SUBTYPES = {"counting_exact", "counting_approx", "yes_no_clinical"}
        if question_type in RETRIEVAL_INCOMPATIBLE_TYPES or question_subtype in RETRIEVAL_INCOMPATIBLE_SUBTYPES:
            _log.info(
                "Hard rule: question_type='%s' question_subtype='%s' is retrieval-incompatible, forcing easy/no-retrieval",
                question_type,
                question_subtype,
            )
            return {"result": {
                "difficulty": "easy",
                "needs_retrieval": False,
                "retrieval_rationale": (
                    f"Question type '{question_type}' / subtype '{question_subtype}' is incompatible with retrieval "
                    "(direct visual interpretation is more reliable than background knowledge for this question)"
                ),
                "verification_policy": "standard",
                "fallback_policy": "none",
                "routing_rationale": (
                    f"Hard rule override: {question_type} / {question_subtype} bypass retrieval to avoid contamination"
                ),
                "status": "success"
            }}

        if question_type == "location" and question_subtype in {"lesion_location", "generic_location"}:
            _log.info(
                "Subtype-aware location routing: question_subtype='%s' will be routed conservatively without a hard retrieval block",
                question_subtype,
            )

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
- Hard no-retrieval is reserved for yes/no, presence/absence, and counting questions.
- For location, finding-description, anatomy-identification, and explanation questions, decide case-by-case; retrieval is allowed when the image is subtle, the wording is ambiguous, or broader context would improve confidence.
- If a location question is explicitly about a lesion or mass, stay conservative, but do not hard-block retrieval unless the question is clearly direct and image-grounded.

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

4. SUBTYPE-SPECIFIC GUIDANCE
- lesion_location / generic_location: route conservatively, but allow retrieval when the lesion is subtle, the wording is ambiguous, or the image context is insufficient.
- finding_description: allow retrieval when the finding is subtle, broad, or harder to characterize from the image alone.
- counting_exact / counting_approx: hard no-retrieval.

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
