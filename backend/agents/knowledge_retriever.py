import json
from typing import Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import HF_TOKEN, TEXT_MODEL, TEXT_MODEL_BASE_URL
from utils.json_utils import strip_json_fences


@component
class KnowledgeRetriever:
    """
    Agent 3: Knowledge Support Agent
    Retrieves supporting medical context relevant to the question and image.
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
            original_question: str = "",
            question_type: str = "",
            body_region: str = "",
            modality: str = "",
            visual_cues: list = None,
            target_entity: str = ""):
        
        # The Haystack pipeline always delivers result-wrapped dicts from upstream components.
        # Unwrap them directly; the bare-dict elif branches were dead code.
        if question_analysis:
            qa = question_analysis.get("result", question_analysis)
            question_type = question_type or qa.get("question_type", "")
            target_entity = target_entity or qa.get("target_entity", "")

        if image_interpretation:
            ii = image_interpretation.get("result", image_interpretation)
            body_region = body_region or ii.get("body_region", "")
            modality = modality or ii.get("modality", "")
            visual_cues = visual_cues or ii.get("visible_structures", [])

        cues_str = ", ".join(visual_cues) if visual_cues else "none"

        prompt = f"""You are a medical knowledge support agent. Provide concise, relevant medical context to help answer a question about a medical image.

Context:
- Original question: {original_question if original_question else "not provided"}
- Question type: {question_type}
- Body region: {body_region}
- Imaging modality: {modality}
- Visual cues observed: {cues_str}
- Target entity: {target_entity if target_entity else "not specified"}

Return a JSON object with:
{{
  "contextual_notes": "2-3 sentences of relevant medical background for this question type and anatomy",
  "key_facts": ["List of 3-5 concise medical facts relevant to the question"],
  "expected_normal_findings": "What would typically be seen in a normal image of this type",
  "common_pathologies": ["List of common abnormalities that might be seen in this type of image"],
  "terminology": {{"term": "definition"}} for 2-3 relevant medical terms
}}

Keep responses focused and educational. Do NOT attempt to diagnose.
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
                "error": "Failed to parse knowledge retrieval response",
                "contextual_notes": "Knowledge retrieval was not successful.",
                "key_facts": [],
                "expected_normal_findings": "",
                "common_pathologies": [],
                "terminology": {}
            }}
        except Exception as e:
            return {"result": {
                "status": "error",
                "error": str(e),
                "contextual_notes": "Knowledge retrieval failed.",
                "key_facts": [],
                "expected_normal_findings": "",
                "common_pathologies": [],
                "terminology": {}
            }}

# Wrapper function for compatibility
def retrieve_knowledge(question_type: str, body_region: str, modality: str,
                       visual_cues: list, target_entity: str = "") -> dict:
    retriever = KnowledgeRetriever()
    return retriever.run(
        question_type=question_type,
        body_region=body_region,
        modality=modality,
        visual_cues=visual_cues,
        target_entity=target_entity
    )["result"]
