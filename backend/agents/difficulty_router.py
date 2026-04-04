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
        started = time.perf_counter()

        prompt = f"""You are a medical VQA routing and planning agent.

Given the following question analysis and image interpretation, evaluate the complexity of the question and determine the appropriate pipeline configuration.

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
ROUTING RULES:
1. HOLISTIC DIFFICULTY LEVEL (Combine Question + Image Context)
- "easy": Straightforward questions (e.g. modality, basic anatomy, simple counting) on clear, unambiguous images.
- "medium": Requires focused clinical inspection (e.g. finding a specific lesion or abnormality) OR a simpler question on a degraded or ambiguous image.
- "hard": Complex clinical reasoning (e.g. differential diagnosis, detailed pathophysiological explanation) OR a medium question on a highly ambiguous/complex case.

2. STRICT POLICY MAPPING
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
