import json
import re
import time
from typing import Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from config import OPENAI_API_KEY, MODELS, generation_kwargs
from utils.json_utils import strip_json_fences
from utils.answer_utils import leading_yes_no
from logger import get_logger

_log = get_logger("agents.reasoning_agent")


@component
class ReasoningAgent:
    """
    Agent 5: Reasoning & Answer Agent
    Synthesizes image analysis, question analysis, and optional retrieved knowledge into a final answer with explanation.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=MODELS["answer"],
        )

    def _normalize_yes_no_answer(self, answer: str) -> str:
        text = (answer or "").strip().lower()

        if not text:
            return answer

        leading = leading_yes_no(text)
        if leading:
            return leading

        explicit_rules = {
            "not present": "no",
            "not visible": "no",
            "not seen": "no",
            "not identifiable": "no",
            "not detected": "no",
            "not normal": "no",
            "not absent": "yes",
            "not negative": "yes",
            "absent": "no",
            "negative": "no",
            "cannot determine": "no",
            "indeterminate": "no",
            "no evidence": "no",
            "no sign": "no",
            "no definite": "no",
            "unremarkable": "no",
            "normal": "yes",
            "present": "yes",
            "visible": "yes",
            "seen": "yes",
            "identifiable": "yes",
            "well-defined": "yes",
            "well defined": "yes",
            "positive": "yes",
            "detected": "yes",
            "identified": "yes",
        }

        for phrase, normalized_answer in explicit_rules.items():
            if phrase in text:
                return normalized_answer

        return answer

    def _normalize_counting_answer(self, answer: str) -> str:
        text = (answer or "").strip().lower()
        if not text:
            return answer

        number_words = {
            "zero": 0,
            "one": 1,
            "single": 1,
            "two": 2,
            "double": 2,
            "three": 3,
            "triple": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
        }

        def _words_to_int(value: str) -> int | None:
            cleaned = value.replace("-", " ")
            parts = [part for part in cleaned.split() if part]
            if not parts:
                return None
            if len(parts) == 1:
                return number_words.get(parts[0])
            if len(parts) == 2 and parts[0] in {"twenty", "thirty", "forty", "fifty"} and parts[1] in {
                "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
            }:
                return number_words[parts[0]] + number_words[parts[1]]
            return None

        def _extract_numeric_token(value: str) -> str | None:
            match = re.search(r"\b(\d{1,2})\b", value)
            if match:
                parsed = int(match.group(1))
                if 0 <= parsed <= 50:
                    return str(parsed)
            return None

        if any(marker in text for marker in {" or ", " to ", "-"}):
            range_match = re.search(r"\b(\d{1,2}|[a-z-]+)\b\s*(?:-|to|or)\s*\b(\d{1,2}|[a-z-]+)\b", text)
            if range_match:
                left_raw, right_raw = range_match.group(1), range_match.group(2)
                left_value = int(left_raw) if left_raw.isdigit() else _words_to_int(left_raw)
                right_value = int(right_raw) if right_raw.isdigit() else _words_to_int(right_raw)
                if left_value is not None and right_value is not None and 0 <= left_value <= 50 and 0 <= right_value <= 50:
                    low, high = sorted((left_value, right_value))
                    return f"{low}-{high}"

        if text.isdigit():
            value = int(text)
            if 0 <= value <= 50:
                return str(value)

        word_value = _words_to_int(text)
        if word_value is not None and 0 <= word_value <= 50:
            return str(word_value)

        numeric_token = _extract_numeric_token(text)
        if numeric_token is not None:
            return numeric_token

        approx_prefixes = ("about ", "around ", "approximately ", "roughly ", "nearly ", "close to ")
        for prefix in approx_prefixes:
            if text.startswith(prefix):
                remainder = text[len(prefix):].strip()
                if remainder:
                    if remainder in {"a dozen", "dozen"}:
                        return "12"
                    remainder_value = int(remainder) if remainder.isdigit() else _words_to_int(remainder)
                    if remainder_value is not None and 0 <= remainder_value <= 50:
                        return str(remainder_value)

        if text in {"a dozen", "dozen"}:
            return "12"
        if text in {"half dozen", "half a dozen"}:
            return "6"

        if text in {"multiple", "few", "several", "cannot reliably count", "unable to count"}:
            return text
        return answer

    def _normalize_location_answer(self, answer: str, question_text: str, image_interpretation: dict, question_analysis: dict) -> str:
        text = (answer or "").strip().lower()
        question = (question_text or "").strip().lower()
        body_region = (image_interpretation or {}).get("body_region", "")
        target = (question_analysis or {}).get("target_entity", "")

        if not text:
            return answer

        direction_only = text in {"left", "right", "bilateral", "midline"}
        if direction_only:
            hints = [question, target.lower(), str(body_region).lower()]
            if any("lung" in hint for hint in hints):
                return f"{text} lung"
            if any("kidney" in hint for hint in hints):
                return f"{text} kidney"
            if any("hemisphere" in hint for hint in hints):
                return f"{text} hemisphere"
            if any("upper lobe" in hint for hint in hints):
                return f"{text} upper lobe"

        if body_region and text in {"brain", "chest", "abdomen", "pelvis", "head", "neck", "musculoskeletal"}:
            return body_region

        return answer

    def _normalize_final_answer(self, result: dict, qa: dict, vs: dict, original_question: str) -> dict:
        normalized = dict(result or {})
        answer = normalized.get("answer", "")
        question_type = (qa or {}).get("question_type", "")
        expected_format = (qa or {}).get("expected_answer_format", "")
        question_subtype = (qa or {}).get("question_subtype", "")

        if expected_format == "yes_no" or question_type in {"yes_no", "presence_absence"}:
            normalized["answer"] = self._normalize_yes_no_answer(answer)
        elif question_type == "counting" or question_subtype in {"counting_exact", "counting_approx"}:
            normalized["answer"] = self._normalize_counting_answer(answer)
        elif question_type == "location":
            normalized["answer"] = self._normalize_location_answer(answer, original_question, vs, qa)
        elif question_type == "anatomy_identification":
            body_region = (vs or {}).get("body_region", "")
            generic_labels = {
                "nervous system", "central nervous system", "respiratory system", "circulatory system",
                "musculoskeletal system", "digestive system", "urinary system", "reproductive system",
            }
            if body_region and (normalize := normalized.get("answer", "").strip().lower()) in generic_labels:
                normalized["answer"] = body_region

        return normalized

    def _build_style_guidance(self, question_type: str, expected_format: str, vs: dict, question_subtype: str = "") -> str:
        body_region = (vs or {}).get("body_region", "unknown")
        modality = (vs or {}).get("modality", "unknown")
        guidance = [
            "- Prefer the most direct, image-grounded answer that matches the question wording.",
            "- Avoid vague abstentions when the image analysis already supports a concrete response.",
        ]
        if expected_format == "yes_no" or question_type in {"yes_no", "presence_absence"}:
            guidance.append("- For yes/no questions, the answer must be exactly yes or no; do not use synonyms like normal/abnormal, present/absent, or positive/negative.")
        if question_type == "location":
            guidance.append("- For location questions, include the anatomical noun with the side when appropriate (for example, 'right lung' rather than just 'right').")
            if question_subtype == "lesion_location":
                guidance.append("- For lesion location questions, answer where the mass/lesion is located rather than whether a mass is visible; prefer an anatomical phrase like 'anterior mediastinum' when supported.")
        if question_type == "anatomy_identification":
            guidance.append(f"- For anatomy identification, prefer the most specific image-grounded anatomical label (current image region: {body_region}; modality: {modality}).")
        if question_subtype == "finding_description":
            guidance.append("- For finding description questions, describe the observable abnormality, its location, and its extent; avoid collapsing the answer into a generic 'abnormal' statement.")
        if question_subtype == "counting_exact":
            guidance.append("- For exact counting questions, provide a numeric answer when supported by the image; if the count is not clear, state the uncertainty rather than inventing a precise number.")
        if question_subtype == "counting_approx":
            guidance.append("- For approximate counting questions, a concise estimate such as 'multiple' or 'few' is acceptable if the image does not support an exact count.")
        if question_type == "counting":
            guidance.append("- If count metadata is low-confidence or text-inferred, prefer conservative wording such as 'multiple', 'few', or 'unable to determine' rather than a precise number.")
            guidance.append("- Only commit to an exact numeric count when the image analysis explicitly supports it.")
        return "\n".join(guidance)

    @component.output_types(result=Dict[str, Any])
    def run(self, 
            image_interpretation: dict = None, 
            question_analysis: dict = None,
            filtered_image_context: dict = None,
            knowledge: dict = None, 
            original_question: str = "",
            retry_context: dict = None):

        # Default fallback values
        vs = image_interpretation.get("result", image_interpretation) if image_interpretation else {}
        qa = question_analysis.get("result", question_analysis) if question_analysis else {}
        fc = filtered_image_context.get("result", filtered_image_context) if filtered_image_context else {}
        kn = knowledge.get("result", knowledge) if knowledge else {}

        # Safely handle potential None returns from previous agents
        if not isinstance(vs, dict): vs = {}
        if not isinstance(qa, dict): qa = {}
        if not isinstance(fc, dict): fc = {}
        if not isinstance(kn, dict): kn = {}
        rc = retry_context if isinstance(retry_context, dict) else {}

        is_retry = bool(rc)
        question_type = qa.get("question_type", "unknown")
        question_subtype = qa.get("question_subtype", "other")
        _log.info(
            "Reasoning agent starting  question=%r  is_retry=%s  "
            "question_type=%s  subtype=%s  yes_no=%s",
            original_question[:100] if original_question else "",
            is_retry,
            question_type,
            question_subtype,
            question_type in ("yes_no", "presence_absence"),
        )
        if is_retry:
            _log.info(
                "Retry context  prev_status=%s  prev_confidence=%s  fallback=%s",
                rc.get("verification_status", "unknown"),
                rc.get("confidence", "unknown"),
                rc.get("fallback_recommendation", "none"),
            )
        started = time.perf_counter()
        _log.debug("Knowledge input  key_facts=%d  rag_cases=%d",
                   len(kn.get("key_facts", [])),
                   len(kn.get("rag_retrieval", {}).get("similar_cases", [])))

        relevant_findings = fc.get("relevant_findings", []) if isinstance(fc, dict) else []
        relevant_structures = fc.get("relevant_structures", []) if isinstance(fc, dict) else []
        filtered_summary = fc.get("filtered_visual_summary", "") if isinstance(fc, dict) else ""
        if not isinstance(relevant_findings, list):
            relevant_findings = []
        if not isinstance(relevant_structures, list):
            relevant_structures = []

        countable_objects = vs.get("countable_objects", []) if isinstance(vs, dict) else []
        count_estimate = vs.get("count_estimate", "unknown") if isinstance(vs, dict) else "unknown"
        count_confidence = vs.get("count_confidence", "unknown") if isinstance(vs, dict) else "unknown"
        count_method = vs.get("count_method", "none") if isinstance(vs, dict) else "none"
        if not isinstance(countable_objects, list):
            countable_objects = []

        rag_retrieval = kn.get("rag_retrieval", {}) if isinstance(kn, dict) else {}
        supporting_cases = rag_retrieval.get("supporting_cases", []) if isinstance(rag_retrieval, dict) else []
        contradictory_cases = rag_retrieval.get("contradictory_cases", []) if isinstance(rag_retrieval, dict) else []
        has_contradiction = bool(rag_retrieval.get("has_contradiction", False)) if isinstance(rag_retrieval, dict) else False
        dominant_label = rag_retrieval.get("dominant_label", "unknown") if isinstance(rag_retrieval, dict) else "unknown"
        label_scores = rag_retrieval.get("label_scores", {}) if isinstance(rag_retrieval, dict) else {}

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

        contradiction_guidance = ""
        if has_contradiction:
            _log.warning("Processing RAG contradiction: dominant=%s supporting=%d contradictory=%d", dominant_label, len(supporting_cases), len(contradictory_cases))
            contradiction_guidance = f"""

RAG CONTRADICTION WARNING:
- Retrieved evidence contains conflicting answers for the same or a closely related query.
- Dominant retrieved label: {dominant_label}
- Score-weighted label support: {label_scores}
- Supporting cases: {len(supporting_cases)}
- Contradictory cases: {len(contradictory_cases)}
- Treat contradictory retrieved cases as lower-trust context.
- Prefer the current image analysis over retrieved cases whenever there is disagreement.
- Use retrieved knowledge only as background support, not as a source of truth.
"""

        expected_format = qa.get('expected_answer_format', '')
        is_yes_no = question_type in ('yes_no', 'presence_absence') or expected_format == 'yes_no'
        style_guidance = self._build_style_guidance(question_type, expected_format, vs, question_subtype)

        yes_no_constraint = ""
        if is_yes_no:
            _log.info("Yes/no constraint applied for question_type=%s", question_type)
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

QUESTION-FOCUSED IMAGE CONTEXT:
- Filtered visual summary: {filtered_summary or 'Not available'}
- Relevant findings: {', '.join(relevant_findings) if relevant_findings else 'Not available'}
- Relevant structures: {', '.join(relevant_structures) if relevant_structures else 'Not available'}

QUESTION ANALYSIS:
- Question type: {qa.get('question_type', 'unknown')}
- Question subtype: {qa.get('question_subtype', 'other')}
- Target entity: {qa.get('target_entity', 'unknown')}
- Expected answer format: {qa.get('expected_answer_format', 'short_phrase')}
- Rephrased question: {qa.get('rephrased_question', original_question)}

COUNTING SIGNALS:
- Count estimate: {count_estimate}
- Count confidence: {count_confidence}
- Count method: {count_method}
- Countable objects: {countable_objects}

COUNTING SAFETY RULES:
- If count confidence is low or count method is text_inference, avoid asserting a precise count unless the image analysis strongly supports it.
- Prefer conservative estimates or explicit uncertainty when the count signal is weak.

SUPPORTING KNOWLEDGE:
- Context: {kn.get('contextual_notes', 'Not available')}
- Key facts: {', '.join(kn.get('key_facts', []))}
- Normal findings: {kn.get('expected_normal_findings', 'Not available')}
{yes_no_constraint}{retry_guidance}{contradiction_guidance}

ANSWER STYLE RULES:
{style_guidance}

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
IMPORTANT: If the retrieved knowledge conflicts with the image analysis, trust the image analysis and down-weight the conflicting retrieval.
Return ONLY the JSON object, no additional text."""

        try:
            response = self.generator.run(
                messages=[ChatMessage.from_user(prompt)],
                generation_kwargs=generation_kwargs("answer")
            )

            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            result = self._normalize_final_answer(result, qa, vs, original_question)
            result["status"] = "success"
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _log.info(
                "Reasoning done  answer=%r  confidence=%s  duration=%dms",
                str(result.get("answer", ""))[:80],
                result.get("answer_confidence", "unknown"),
                elapsed_ms,
            )
            _log.debug("Reasoning trace: %s", result.get("reasoning_trace", ""))
            _log.debug("Evidence used: %s", result.get("evidence_used", []))
            _log.debug("Limitations: %s", result.get("limitations", ""))
            return {"result": result}

        except json.JSONDecodeError:
            _log.error("Failed to parse reasoning response JSON")
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
            _log.error("Unexpected error in reasoning: %s", e)
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
                    knowledge: dict, original_question: str, retry_context: dict = None,
                    filtered_image_context: dict = None) -> dict:
    agent = ReasoningAgent()
    return agent.run(
        image_interpretation={"result": visual_summary},
        question_analysis={"result": question_analysis},
        knowledge={"result": knowledge},
        original_question=original_question,
        retry_context=retry_context,
        filtered_image_context={"result": filtered_image_context} if filtered_image_context is not None else None,
    )["result"]
