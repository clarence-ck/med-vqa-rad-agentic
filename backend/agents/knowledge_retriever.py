import json
import time
from typing import Dict, Any, Optional
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import OPENAI_API_KEY, MODELS, generation_kwargs
from utils.json_utils import strip_json_fences
from logger import get_logger

_log = get_logger("agents.knowledge_retriever")

# Import advanced RAG
try:
    from agents.advanced_rag import retrieve_knowledge_advanced
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
    - Hybrid retrieval (dense embeddings + BM25)
    - Image-grounded metadata filtering (modality, body region, finding category)
    - Cross-encoder reranking (mandatory)
    - Multi-signal confidence-based validation
    """
    
    def __init__(self, use_advanced_rag: bool = True):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=MODELS["lookup"],
        )
        self.use_advanced_rag = use_advanced_rag and _ADVANCED_RAG_AVAILABLE

    def _normalize_answer_label(self, answer: str) -> str:
        text = (answer or "").strip().lower()
        text = " ".join(text.split())
        if not text:
            return "unknown"

        if text.startswith("yes") or text in {"positive", "present", "true"}:
            return "yes"
        if text.startswith("no") or text in {"negative", "absent", "false"}:
            return "no"

        if text in {"left", "right", "bilateral", "midline"}:
            return text

        if text in {"normal", "abnormal"}:
            return text

        return text

    def _is_binary_or_location_query(self, question_type: str, answer_style: str) -> bool:
        q_type = (question_type or "").strip().lower()
        a_style = (answer_style or "").strip().lower()
        return q_type in {"yes_no", "presence_absence", "location"} or a_style in {"yes_no", "closed"}

    def _split_contradictory_cases(self, cases: list, question_type: str, answer_style: str) -> dict:
        ordered = list(cases or [])
        if not ordered:
            return {
                "supporting_cases": [],
                "contradictory_cases": [],
                "dominant_label": "unknown",
                "has_contradiction": False,
                "label_scores": {},
            }

        if not self._is_binary_or_location_query(question_type, answer_style):
            dominant_label = self._normalize_answer_label(ordered[0].get("answer", ""))
            return {
                "supporting_cases": ordered,
                "contradictory_cases": [],
                "dominant_label": dominant_label,
                "has_contradiction": False,
                "label_scores": {},
            }

        label_scores: dict[str, float] = {}
        for case in ordered:
            label = self._normalize_answer_label(case.get("answer", ""))
            score = float(case.get("score", 0.0) or 0.0)
            if label == "unknown":
                continue
            label_scores[label] = label_scores.get(label, 0.0) + max(score, 0.0)

        if label_scores:
            dominant_label = max(label_scores.items(), key=lambda item: (item[1], item[0]))[0]
        else:
            dominant_label = self._normalize_answer_label(ordered[0].get("answer", ""))

        supporting_cases = []
        contradictory_cases = []
        for case in ordered:
            label = self._normalize_answer_label(case.get("answer", ""))
            if label == dominant_label or label == "unknown":
                supporting_cases.append(case)
            else:
                contradictory_cases.append(case)

        return {
            "supporting_cases": supporting_cases,
            "contradictory_cases": contradictory_cases,
            "dominant_label": dominant_label,
            "has_contradiction": bool(contradictory_cases),
            "label_scores": label_scores,
        }

    def _format_retrieved_cases_section(self, title: str, cases: list) -> str:
        if not cases:
            return f"{title}: none"

        formatted = [f"{title}:"]
        for i, case in enumerate(cases, 1):
            score = case.get("score", 0.0)
            answer_label = self._normalize_answer_label(case.get("answer", ""))
            card_lines = [f"  KB Chunk {i}:"]

            meta = case.get("metadata", {})
            for label, key in [
                ("Concept Type", "concept_type"),
                ("Concept", "concept"),
                ("Target Entity", "target_entity"),
                ("Modality", "modality"),
                ("Body Region", "body_region"),
                ("Finding Category", "finding_category"),
                ("Laterality", "laterality"),
                ("Language", "language"),
                ("Question Family", "question_family"),
                ("Answer Style", "answer_style"),
            ]:
                value = meta.get(key)
                if value:
                    card_lines.append(f"    {label}: {value}")

            support_count = meta.get("support_count")
            if support_count is not None:
                card_lines.append(f"    Support Count: {support_count}")

            visual_cues = meta.get("visual_cues") or []
            if visual_cues:
                card_lines.append(f"    Visual Cues: {', '.join(visual_cues[:6])}")

            text_chunk = meta.get("text_chunk")
            if text_chunk:
                card_lines.append(f"    Text Chunk: {text_chunk}")

            card_lines.append(f"    Source Q: {case.get('question', '')}")
            card_lines.append(f"    Source A: {case.get('answer', '')}")
            card_lines.append(f"    Answer Label: {answer_label}")
            card_lines.append(f"    Relevance Score: {score:.3f}")
            formatted.append("\n".join(card_lines))

        return "\n\n".join(formatted)

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
            question_family = qa.get("question_family", qa.get("reasoning_type", ""))
            answer_style = qa.get("expected_answer_format", "")
        else:
            question_family = ""
            answer_style = ""

        if image_interpretation:
            ii = image_interpretation.get("result", image_interpretation)
            body_region = body_region or ii.get("body_region", "")
            modality = modality or ii.get("modality", "")
            visual_cues = visual_cues or ii.get("visible_structures", [])
            visual_cues = visual_cues or ii.get("notable_findings", [])

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
        grouped_cases = {
            "supporting_cases": [],
            "contradictory_cases": [],
            "dominant_label": "unknown",
            "has_contradiction": False,
        }
        rag_metadata = {
            "count": 0,
            "triggered": False,
            "dominant_label": "unknown",
            "has_contradiction": False,
            "supporting_count": 0,
            "contradictory_count": 0,
            "label_scores": {},
        }

        if self.use_advanced_rag and original_question:
            try:
                # Extract notable_findings from image interpretation for image-grounded retrieval
                notable_findings = None
                if image_interpretation:
                    ii = image_interpretation.get("result", image_interpretation)
                    notable_findings = ii.get("notable_findings")

                # RAG retrieval with image-grounded finding inference
                rag_result = retrieve_knowledge_advanced(
                    question=original_question,
                    modality=modality,
                    body_region=body_region,
                    notable_findings=notable_findings,
                    question_type=question_type,
                    question_family=question_family,
                    target_entity=target_entity,
                    answer_style=answer_style,
                    visual_cues=visual_cues,
                )

                similar_cases = rag_result.get("similar_cases", [])
                grouped_cases = self._split_contradictory_cases(similar_cases, question_type, answer_style)
                rag_metadata = {
                    "count": rag_result.get("count", 0),
                    "triggered": True,
                    "dominant_label": grouped_cases["dominant_label"],
                    "has_contradiction": grouped_cases["has_contradiction"],
                    "supporting_count": len(grouped_cases["supporting_cases"]),
                    "contradictory_count": len(grouped_cases["contradictory_cases"]),
                    "label_scores": grouped_cases.get("label_scores", {}),
                }

                if similar_cases:
                    support_section = self._format_retrieved_cases_section("SUPPORTING KB CHUNKS", grouped_cases["supporting_cases"])
                    conflict_section = self._format_retrieved_cases_section("CONTRADICTORY KB CHUNKS", grouped_cases["contradictory_cases"])
                    rag_context = f"{support_section}\n\n{conflict_section}"
                    _log.info(
                        "Advanced RAG retrieved %d similar cases (supporting=%d contradictory=%d)",
                        len(similar_cases),
                        len(grouped_cases["supporting_cases"]),
                        len(grouped_cases["contradictory_cases"]),
                    )
                    if grouped_cases["has_contradiction"]:
                        _log.warning(
                            "RAG contradiction detected: dominant_label=%s label_scores=%s",
                            grouped_cases["dominant_label"],
                            grouped_cases.get("label_scores", {}),
                        )
                else:
                    # The confidence gate rejected every candidate. Do not ask the LLM to
                    # invent "key facts" from general knowledge in place of evidence.
                    _log.info("Advanced RAG returned 0 similar cases; returning empty knowledge (no LLM synthesis)")
                    return {"result": self._no_evidence_result(
                        "success", "empty",
                        "No knowledge-base evidence passed the retrieval confidence gate; "
                        "the answer relies on image analysis only.",
                        rag_metadata,
                    )}

            except Exception as e:
                _log.error("Advanced RAG retrieval failed: %s", e)
                return {"result": self._no_evidence_result(
                    "error", "failed", f"Knowledge retrieval failed: {e}", rag_metadata,
                )}

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
                generation_kwargs=generation_kwargs("lookup")
            )

            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            result["status"] = "success"
            result["retrieval_status"] = "success"

            # Add RAG metadata to result (always include if RAG was triggered, even if no cases found)
            if self.use_advanced_rag:
                result["rag_retrieval"] = {
                    "similar_cases": similar_cases,
                    "supporting_cases": grouped_cases["supporting_cases"],
                    "contradictory_cases": grouped_cases["contradictory_cases"],
                    "dominant_label": grouped_cases["dominant_label"],
                    "has_contradiction": grouped_cases["has_contradiction"],
                    "label_scores": grouped_cases.get("label_scores", {}),
                    "metadata": rag_metadata
                }

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _log.info(
                "Knowledge retrieval done  retrieval_status=%s  key_facts=%d  rag_cases=%d  duration=%dms",
                result.get("retrieval_status", "success"),
                len(result.get("key_facts", [])),
                len(similar_cases),
                elapsed_ms,
            )
            _log.debug("Key facts: %s", result.get("key_facts", []))
            _log.debug("Contextual notes: %s", str(result.get("contextual_notes", ""))[:200])
            return {"result": result}

        except json.JSONDecodeError:
            _log.error("Failed to parse knowledge retrieval JSON response")
            return {"result": {
                "status": "error",
                "retrieval_status": "failed",
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
                "retrieval_status": "failed",
                "error": str(e),
                "contextual_notes": "Knowledge retrieval failed.",
                "key_facts": [],
                "expected_normal_findings": "",
                "common_pathologies": [],
                "terminology": {}
            }}

    def _no_evidence_result(self, status: str, retrieval_status: str, notes: str, rag_metadata: dict) -> dict:
        return {
            "status": status,
            "retrieval_status": retrieval_status,
            "contextual_notes": notes,
            "key_facts": [],
            "expected_normal_findings": "",
            "common_pathologies": [],
            "terminology": {},
            "rag_retrieval": {
                "similar_cases": [],
                "supporting_cases": [],
                "contradictory_cases": [],
                "dominant_label": "unknown",
                "has_contradiction": False,
                "label_scores": {},
                "metadata": rag_metadata,
            },
        }

    def _format_similar_cases(self, cases: list) -> str:
        """Format retrieved SLAKE KB concepts as compact evidence cards."""
        formatted = []
        for i, case in enumerate(cases, 1):
            meta = case.get("metadata", {})
            score = case.get("score", 0.0)
            card_lines = [f"KB Chunk {i}:"]

            for label, key in [
                ("Concept Type", "concept_type"),
                ("Concept", "concept"),
                ("Target Entity", "target_entity"),
                ("Modality", "modality"),
                ("Body Region", "body_region"),
                ("Finding Category", "finding_category"),
                ("Laterality", "laterality"),
                ("Language", "language"),
                ("Question Family", "question_family"),
                ("Answer Style", "answer_style"),
            ]:
                value = meta.get(key)
                if value:
                    card_lines.append(f"  {label}: {value}")

            support_count = meta.get("support_count")
            if support_count is not None:
                card_lines.append(f"  Support Count: {support_count}")

            visual_cues = meta.get("visual_cues") or []
            if visual_cues:
                card_lines.append(f"  Visual Cues: {', '.join(visual_cues[:6])}")

            text_chunk = meta.get("text_chunk")
            if text_chunk:
                card_lines.append(f"  Text Chunk: {text_chunk}")

            card_lines.append(f"  Source Q: {case.get('question', '')}")
            card_lines.append(f"  Source A: {case.get('answer', '')}")
            card_lines.append(f"  Relevance Score: {score:.3f}")
            formatted.append("\n".join(card_lines))

        return "\n\n".join(formatted)

    def _build_prompt(
        self,
        original_question: str,
        question_type: str,
        body_region: str,
        modality: str,
        cues_str: str,
        target_entity: str,
        rag_context: str,
    ) -> str:
        """Build the knowledge retrieval prompt with SLAKE concept context."""

        if rag_context:
            rag_section = f"""
RETRIEVED SLAKE CONCEPT CHUNKS (from KB):
{rag_context}

IMPORTANT: These are distilled medical concept chunks extracted from SLAKE. They provide anatomy, modality, abnormality, and visual-cue context. Do NOT treat them as the answer for the current image. Use them to support reasoning, not to replace image evidence.
"""
        else:
            rag_section = """
No usable retrieved KB concepts were found. Rely on the current question context and general medical knowledge only.
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
  "contextual_notes": "2-3 sentences of relevant medical background for this question type and anatomy. Prefer the retrieved SLAKE concept chunks if they match the same anatomy, modality, or abnormality.",
  "key_facts": ["List of 3-5 concise medical facts relevant to the question"],
  "expected_normal_findings": "What would typically be seen in a normal image of this type",
  "common_pathologies": ["List of common abnormalities that might be seen in this type of image"],
  "terminology": {{"term": "definition"}} for 2-3 relevant medical terms,
  "kb_interpretation": "One short sentence explaining how the SLAKE concept chunks relate to the current question"
}}

Keep responses focused and educational. Do NOT attempt to diagnose. Base your response on established medical knowledge and only the retrieved evidence that directly matches the current anatomy/pathology; ignore weak or loosely related cases.
Return ONLY the JSON object, no additional text."""

