import json
from typing import Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import HF_TOKEN, TEXT_MODEL, TEXT_MODEL_BASE_URL
from utils.json_utils import strip_json_fences


@component
class ReasoningAgent:
    """
    Agent 4: Reasoning & Answer Agent
    Synthesizes all prior agent outputs into a final answer with explanation.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(HF_TOKEN),
            model=TEXT_MODEL,
            api_base_url=TEXT_MODEL_BASE_URL,
        )

    @component.output_types(result=Dict[str, Any])
    def run(self, 
            image_interpretation: dict = None, 
            question_analysis: dict = None,
            knowledge: dict = None, 
            original_question: str = "",
            retry_context: dict = None):

        # Default fallback values
        vs = image_interpretation.get("result", image_interpretation) if image_interpretation else {}
        qa = question_analysis.get("result", question_analysis) if question_analysis else {}
        kn = knowledge.get("result", knowledge) if knowledge else {}

        # Safely handle potential None returns from previous agents
        if not isinstance(vs, dict): vs = {}
        if not isinstance(qa, dict): qa = {}
        if not isinstance(kn, dict): kn = {}
        rc = retry_context if isinstance(retry_context, dict) else {}

        failure_categories = rc.get('failure_categories', []) if rc else []
        if isinstance(failure_categories, list):
            formatted_failure_categories = []
            for category in failure_categories:
                if isinstance(category, dict):
                    code = category.get('code') or category.get('label') or str(category)
                    severity = category.get('severity')
                    formatted_failure_categories.append(f"{code} ({severity})" if severity else str(code))
                else:
                    formatted_failure_categories.append(str(category))
            failure_categories_text = ', '.join(formatted_failure_categories) if formatted_failure_categories else 'none'
        else:
            failure_categories_text = str(failure_categories)

        retry_guidance = ""
        if rc:
            retry_guidance = f"""

RETRY CONTEXT:
- Previous verification status: {rc.get('verification_status', 'unknown')}
- Previous confidence: {rc.get('confidence', 'unknown')}
- Previous fallback recommendation: {rc.get('fallback_recommendation', 'none')}
- Previous failure categories: {failure_categories_text}
- Retry instruction: Re-evaluate the answer with extra caution. If the evidence is still insufficient, produce a more conservative answer or explicitly abstain.
"""

        question_type = qa.get('question_type', '')
        expected_format = qa.get('expected_answer_format', '')
        is_yes_no = question_type in ('yes_no', 'presence_absence') or expected_format == 'yes_no'

        yes_no_constraint = ""
        if is_yes_no:
            yes_no_constraint = """
ANSWER CONSTRAINT — YES/NO QUESTION:
The question requires a yes or no answer.
- The "answer" field MUST be exactly "yes" or "no" (lowercase, no punctuation, no other words).
- Do NOT answer with "yes, ..." or "no, ..." or any sentence.
- Do NOT use synonyms such as "present", "absent", "true", "false", "positive", "negative".
- If the visual evidence is genuinely insufficient to decide, use "answer_confidence": "low" and set "answer" to whichever is more defensible, stating the uncertainty in "limitations".
"""

        answer_field_desc = (
            '"answer": "MUST be exactly \\"yes\\" or \\"no\\" — no other words, no punctuation."'
            if is_yes_no else
            '"answer": "The direct answer to the question. Keep it concise and match the expected answer format."'
        )

        prompt = f"""You are a medical VQA reasoning agent. Synthesize the following information to answer a medical image question.

ORIGINAL QUESTION: "{original_question}"

IMAGE ANALYSIS:
- Modality: {vs.get('modality', 'unknown')}
- Body region: {vs.get('body_region', 'unknown')}
- Visible structures: {', '.join(vs.get('visible_structures', []))}
- Visual summary: {vs.get('visual_summary', 'Not available')}
- Notable findings: {', '.join(vs.get('notable_findings', []))}

QUESTION ANALYSIS:
- Question type: {qa.get('question_type', 'unknown')}
- Target entity: {qa.get('target_entity', 'unknown')}
- Expected answer format: {qa.get('expected_answer_format', 'short_phrase')}
- Rephrased question: {qa.get('rephrased_question', original_question)}

SUPPORTING KNOWLEDGE:
- Context: {kn.get('contextual_notes', 'Not available')}
- Key facts: {', '.join(kn.get('key_facts', []))}
- Normal findings: {kn.get('expected_normal_findings', 'Not available')}
{yes_no_constraint}{retry_guidance}

Return a JSON object with:
{{
  {answer_field_desc},
  "explanation": "A 2-4 sentence explanation justifying the answer. Ground it in visual evidence from the image analysis. Do not speculate beyond what was observed.",
  "reasoning_trace": "Brief step-by-step description of how you arrived at the answer",
  "evidence_used": ["List of specific visual/knowledge evidence that supports the answer"],
  "answer_confidence": "high / medium / low",
  "limitations": "Any limitations of this answer (e.g., image quality, ambiguous features)"
}}

IMPORTANT: Base your answer on the visual evidence provided. If the evidence is insufficient, say so.
Return ONLY the JSON object, no additional text."""

        try:
            response = self.generator.run(
                messages=[ChatMessage.from_user(prompt)],
                generation_kwargs={
                    "max_tokens": 3000,
                }
            )

            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            result["status"] = "success"
            return {"result": result}

        except json.JSONDecodeError:
            return {"result": {
                "status": "error",
                "error": "Failed to parse reasoning response",
                "answer": "Unable to generate an answer.",
                "explanation": "The reasoning agent was unable to produce a valid response.",
                "reasoning_trace": "",
                "evidence_used": [],
                "answer_confidence": "low",
                "limitations": "Reasoning failed"
            }}
        except Exception as e:
            return {"result": {
                "status": "error",
                "error": str(e),
                "answer": "Unable to generate an answer.",
                "explanation": "An error occurred during reasoning.",
                "reasoning_trace": "",
                "evidence_used": [],
                "answer_confidence": "low",
                "limitations": str(e)
            }}

# Wrapper function for compatibility
def generate_answer(visual_summary: dict, question_analysis: dict,
                    knowledge: dict, original_question: str, retry_context: dict = None) -> dict:
    agent = ReasoningAgent()
    return agent.run(
        image_interpretation={"result": visual_summary},
        question_analysis={"result": question_analysis},
        knowledge={"result": knowledge},
        original_question=original_question,
        retry_context=retry_context,
    )["result"]
