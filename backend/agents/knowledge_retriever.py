import json
import time
from typing import Dict, Any, Optional
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import OPENAI_API_KEY, OPENAI_MODEL
from utils.json_utils import strip_json_fences
from logger import get_logger

_log = get_logger("agents.knowledge_retriever")

# Import advanced RAG
try:
    from agents.advanced_rag import get_rag_retriever, retrieve_knowledge_advanced
    _ADVANCED_RAG_AVAILABLE = True
    _log.info("Advanced RAG module loaded successfully")
except ImportError:
    _ADVANCED_RAG_AVAILABLE = False
    _log.warning("Advanced RAG not available, falling back to standard (LLM-only) knowledge retrieval")


@component
class KnowledgeRetriever:
    """
    Agent 4: Knowledge Support Agent (Advanced RAG Version)
    
    Retrieves supporting medical context using advanced RAG when the router requests retrieval:
    - Query rewriting and expansion
    - Hybrid retrieval (dense embeddings + BM25)
    - Metadata filtering by modality, body region, and finding category
    - Cross-encoder reranking
    - Self-reflection validation
    """
    
    def __init__(self, use_advanced_rag: bool = True):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=OPENAI_MODEL,
        )
        self.use_advanced_rag = use_advanced_rag and _ADVANCED_RAG_AVAILABLE
        self._rag_retriever = None

        # Lazy init RAG retriever on first use
        if self.use_advanced_rag:
            try:
                self._rag_retriever = get_rag_retriever()
                _log.info("Advanced RAG retriever initialized successfully")
            except Exception as e:
                _log.error("Advanced RAG init failed: %s — falling back to LLM-only mode", e)
                self.use_advanced_rag = False

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
        
        _log.info(
            "Knowledge retrieval starting  question=%r  modality=%s  body_region=%s  "
            "question_type=%s  target_entity=%s  advanced_rag=%s",
            original_question[:100] if original_question else "",
            modality or "unknown",
            body_region or "unknown",
            question_type or "unknown",
            target_entity or "unknown",
            self.use_advanced_rag,
        )
        started = time.perf_counter()

        # Advanced RAG: Retrieve similar cases from train set
        rag_context = ""
        similar_cases = []
        rag_metadata = {}

        if self.use_advanced_rag and original_question:
            try:
                finding_category = None
                if self._rag_retriever is not None:
                    finding_category = self._rag_retriever._infer_finding_category(
                        original_question, ''
                    )
                rag_result = retrieve_knowledge_advanced(
                    question=original_question,
                    modality=modality,
                    body_region=body_region,
                    finding_category=finding_category,
                    use_query_rewrite=True,
                    use_reranking=True,
                    use_self_reflection=True
                )

                similar_cases = rag_result.get("similar_cases", [])
                rag_metadata = {
                    "count": rag_result.get("count", 0),
                    "query_rewritten": rag_result.get("query_rewritten", False)
                }

                if similar_cases:
                    rag_context = self._format_similar_cases(similar_cases)
                    _log.info(
                        "Advanced RAG retrieved %d similar cases  "
                        "(query_rewritten=%s  finding_category=%s)",
                        len(similar_cases),
                        rag_metadata.get("query_rewritten", False),
                        finding_category or "none",
                    )
                else:
                    _log.info("Advanced RAG returned 0 similar cases for this question")

            except Exception as e:
                _log.error("Advanced RAG retrieval failed: %s — continuing without RAG context", e)
                rag_context = ""

        # Build enhanced prompt with RAG context
        prompt = self._build_prompt(
            original_question=original_question,
            question_type=question_type,
            body_region=body_region,
            modality=modality,
            cues_str=cues_str,
            target_entity=target_entity,
            rag_context=rag_context
        )

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

            # Add RAG metadata to result
            if similar_cases:
                result["rag_retrieval"] = {
                    "similar_cases": similar_cases,
                    "metadata": rag_metadata
                }

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _log.info(
                "Knowledge retrieval done  key_facts=%d  rag_cases=%d  duration=%dms",
                len(result.get("key_facts", [])),
                len(similar_cases),
                elapsed_ms,
            )
            return {"result": result}

        except json.JSONDecodeError:
            _log.error("Failed to parse knowledge retrieval JSON response")
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
            _log.error("Unexpected error in knowledge retrieval: %s", e)
            return {"result": {
                "status": "error",
                "error": str(e),
                "contextual_notes": "Knowledge retrieval failed.",
                "key_facts": [],
                "expected_normal_findings": "",
                "common_pathologies": [],
                "terminology": {}
            }}

    def _format_similar_cases(self, cases: list) -> str:
        """Format similar cases for prompt injection"""
        formatted = []
        for i, case in enumerate(cases, 1):
            q = case.get("question", "")
            a = case.get("answer", "")
            score = case.get("score", 0.0)
            meta = case.get("metadata", {})
            
            case_str = f"Case {i}:\n"
            case_str += f"  Question: {q}\n"
            case_str += f"  Answer: {a}\n"
            if meta.get("modality"):
                case_str += f"  Modality: {meta['modality']}\n"
            if meta.get("body_region"):
                case_str += f"  Region: {meta['body_region']}\n"
            case_str += f"  Relevance Score: {score:.3f}"
            
            formatted.append(case_str)
        
        return "\n\n".join(formatted)
    
    def _build_prompt(
        self,
        original_question: str,
        question_type: str,
        body_region: str,
        modality: str,
        cues_str: str,
        target_entity: str,
        rag_context: str
    ) -> str:
        """Build the knowledge retrieval prompt with RAG context"""
        
        rag_section = ""
        if rag_context:
            rag_section = f"""
RETRIEVED SIMILAR CASES (from training data):
{rag_context}

Use these similar cases as reference for the expected answer format and relevant medical concepts, but adapt to the current specific question.
"""
        
        return f"""You are a medical knowledge support agent. Provide concise, relevant medical context to help answer a question about a medical image.

CURRENT QUESTION CONTEXT:
- Original question: {original_question if original_question else "not provided"}
- Question type: {question_type}
- Body region: {body_region}
- Imaging modality: {modality}
- Visual cues observed: {cues_str}
- Target entity: {target_entity if target_entity else "not specified"}
{rag_section}
Return a JSON object with:
{{
  "contextual_notes": "2-3 sentences of relevant medical background for this question type and anatomy. Reference the retrieved similar cases if available.",
  "key_facts": ["List of 3-5 concise medical facts relevant to the question"],
  "expected_normal_findings": "What would typically be seen in a normal image of this type",
  "common_pathologies": ["List of common abnormalities that might be seen in this type of image"],
  "terminology": {{"term": "definition"}} for 2-3 relevant medical terms
}}

Keep responses focused and educational. Do NOT attempt to diagnose. Base your response on established medical knowledge and the patterns observed in similar cases.
Return ONLY the JSON object, no additional text."""

# Wrapper function for compatibility
def retrieve_knowledge(
    question_type: str = "",
    body_region: str = "",
    modality: str = "",
    visual_cues: list = None,
    target_entity: str = "",
    original_question: str = "",
    use_advanced_rag: bool = True
) -> dict:
    retriever = KnowledgeRetriever(use_advanced_rag=use_advanced_rag)
    return retriever.run(
        question_type=question_type,
        body_region=body_region,
        modality=modality,
        visual_cues=visual_cues,
        target_entity=target_entity,
        original_question=original_question
    )["result"]
