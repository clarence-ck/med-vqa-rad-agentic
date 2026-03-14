import json
from typing import Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import OPENAI_API_KEY, OPENAI_MODEL


@component
class ReasoningAgent:
    """
    Agent 4: Reasoning & Answer Agent
    Synthesizes all prior agent outputs into a final answer with explanation.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=OPENAI_MODEL,
        )

    @component.output_types(result=Dict[str, Any])
    def run(self, 
            image_interpretation: dict = None, 
            question_analysis: dict = None,
            knowledge: dict = None, 
            original_question: str = ""):

        # Default fallback values
        vs = image_interpretation.get("result", image_interpretation) if image_interpretation else {}
        qa = question_analysis.get("result", question_analysis) if question_analysis else {}
        kn = knowledge.get("result", knowledge) if knowledge else {}

        # Safely handle potential None returns from previous agents
        if not isinstance(vs, dict): vs = {}
        if not isinstance(qa, dict): qa = {}
        if not isinstance(kn, dict): kn = {}

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

Return a JSON object with:
{{
  "answer": "The direct answer to the question. Keep it concise and match the expected answer format.",
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
                    knowledge: dict, original_question: str) -> dict:
    agent = ReasoningAgent()
    return agent.run(
        image_interpretation={"result": visual_summary},
        question_analysis={"result": question_analysis},
        knowledge={"result": knowledge},
        original_question=original_question
    )["result"]
