import json
from typing import Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import OPENAI_API_KEY, OPENAI_MODEL


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

            raw = (response["replies"][0].text or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                raw = raw.strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

            result = json.loads(raw)
            result["status"] = "success"
            return {"result": result}

        except json.JSONDecodeError:
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
