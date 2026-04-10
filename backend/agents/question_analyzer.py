import json
import re
import time
from typing import Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import OPENAI_API_KEY, OPENAI_MODEL
from utils.json_utils import strip_json_fences
from logger import get_logger

_log = get_logger("agents.question_analyzer")


@component
class QuestionAnalyzer:
    """
    Agent 2: Question Understanding Agent
    Classifies the user question into type, target entity, reasoning needed, and answer format.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=OPENAI_MODEL,
        )

    def _infer_question_style(self, question: str) -> dict:
        q = (question or "").strip().lower()
        if not q:
            return {}

        starts_binary = bool(re.match(r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|should)\b", q))
        polarity_words = [
            "normal", "abnormal", "present", "absent", "visible", "seen", "identifiable",
            "well-defined", "well defined", "enlarged", "ring-enhancing", "ring enhancing",
            "hyperinflated", "shifted", "fracture", "lesion", "mass", "effusion", "pneumothorax",
        ]
        has_polarity = any(word in q for word in polarity_words)
        explicit_choice = any(pattern in q for pattern in [
            "normal or abnormal", "abnormal or normal", "present or absent", "absent or present",
            "benign or malignant", "malignant or benign", "left or right", "right or left",
        ])

        if explicit_choice:
            return {
                "question_type": "abnormality_description",
                "expected_answer_format": "short_phrase",
                "reasoning_type": "comparison",
                "question_clarity": "clear",
            }

        if starts_binary and has_polarity:
            return {
                "question_type": "yes_no",
                "expected_answer_format": "yes_no",
                "reasoning_type": "visual_recognition",
                "question_clarity": "clear",
            }

        return {}

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

        prompt = f"""You are a medical VQA question understanding agent. Analyze the following question asked about a medical image.

Question: "{question}"

Visual context (from image analysis): {visual_summary if visual_summary else "Not available"}

Return a JSON object with:
{{
  "question_type": "One of: anatomy_identification, modality_identification, presence_absence, location, counting, abnormality_description, explanation_request, yes_no, other",
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
                generation_kwargs={
                    "max_completion_tokens": 3000,
                }
            )

            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            result = self._apply_style_override(result, style_hint)
            result["status"] = "success"
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _log.info(
                "Question analysis done  type=%s  reasoning=%s  med_knowledge=%s  clarity=%s  duration=%dms",
                result.get("question_type", "unknown"),
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
