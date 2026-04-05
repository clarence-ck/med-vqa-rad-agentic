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

    @component.output_types(result=Dict[str, Any])
    def run(self, question: str, image_interpretation: dict = None):
        visual_summary = ""
        if image_interpretation and isinstance(image_interpretation, dict):
            visual_summary = image_interpretation.get("visual_summary", "")

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
