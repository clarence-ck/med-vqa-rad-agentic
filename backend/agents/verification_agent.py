import json
from typing import Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import OPENAI_API_KEY, OPENAI_MODEL
from utils.json_utils import strip_json_fences


@component
class VerificationAgent:
    """
    Agent 6: Verification Agent
    Checks consistency and believability of the generated answer after reasoning.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=OPENAI_MODEL,
        )

    @component.output_types(result=Dict[str, Any])
    def run(self, 
            answer_data: dict = None, 
            image_interpretation: dict = None,
            question_analysis: dict = None, 
            original_question: str = ""):

        ans = answer_data.get("result", answer_data) if answer_data else {}
        vs = image_interpretation.get("result", image_interpretation) if image_interpretation else {}
        qa = question_analysis.get("result", question_analysis) if question_analysis else {}

        if not isinstance(ans, dict): ans = {}
        if not isinstance(vs, dict): vs = {}
        if not isinstance(qa, dict): qa = {}

        prompt = f"""You are a medical VQA verification agent. Your job is to critically evaluate whether the following answer is consistent, grounded, and appropriate.

ORIGINAL QUESTION: "{original_question}"

GENERATED ANSWER: "{ans.get('answer', '')}"
EXPLANATION: "{ans.get('explanation', '')}"
STATED CONFIDENCE: {ans.get('answer_confidence', 'unknown')}

IMAGE ANALYSIS (from visual interpreter):
- Modality: {vs.get('modality', 'unknown')}
- Body region: {vs.get('body_region', 'unknown')}
- Visible structures: {', '.join(vs.get('visible_structures', []))}
- Visual summary: {vs.get('visual_summary', '')}

QUESTION ANALYSIS:
- Question type: {qa.get('question_type', 'unknown')}
- Expected format: {qa.get('expected_answer_format', 'unknown')}

Perform these checks:
1. Does the answer match the question type?
2. Does the explanation reference evidence actually seen in the image?
3. Is the answer too strong for the available evidence?
4. Is there any contradiction between the answer and the visual analysis?
5. Is the answer format appropriate for the question?

Return a JSON object with:
{{
  "verification_status": "verified / partially_verified / low_confidence / unsupported",
  "confidence_label": "High / Medium / Low",
  "checks": {{
    "answer_matches_question_type": true/false,
    "explanation_grounded_in_evidence": true/false,
    "answer_strength_appropriate": true/false,
    "no_contradictions": true/false,
    "format_appropriate": true/false
  }},
  "issues_found": ["List of any specific issues found, empty if none"],
  "fallback_recommendation": "none / rephrase_answer / add_disclaimer / flag_for_review",
  "verification_notes": "Brief explanation of the verification result",
  "suggested_disclaimer": "If needed, a safety disclaimer to append (empty string if not needed)"
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
            return {"result": result}

        except json.JSONDecodeError:
            return {"result": {
                "status": "error",
                "error": "Failed to parse verification response",
                "verification_status": "low_confidence",
                "confidence_label": "Low",
                "checks": {},
                "issues_found": ["Verification parsing failed"],
                "fallback_recommendation": "flag_for_review",
                "verification_notes": "Verification could not be completed.",
                "suggested_disclaimer": "This answer has not been verified."
            }}
        except Exception as e:
            return {"result": {
                "status": "error",
                "error": str(e),
                "verification_status": "low_confidence",
                "confidence_label": "Low",
                "checks": {},
                "issues_found": [str(e)],
                "fallback_recommendation": "flag_for_review",
                "verification_notes": "Verification failed due to an error.",
                "suggested_disclaimer": "This answer has not been verified."
            }}

# Wrapper function for compatibility
def verify_answer(answer_data: dict, visual_summary: dict,
                  question_analysis: dict, original_question: str) -> dict:
    agent = VerificationAgent()
    return agent.run(
        answer_data={"result": answer_data},
        image_interpretation={"result": visual_summary},
        question_analysis={"result": question_analysis},
        original_question=original_question
    )["result"]
