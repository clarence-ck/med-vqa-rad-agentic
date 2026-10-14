import json
import re
import time
from typing import Dict, Any
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ImageContent, TextContent
from haystack.utils import Secret
from config import OPENAI_API_KEY, MODELS, generation_kwargs
from utils.json_utils import strip_json_fences
from utils.usage import record_usage
from utils.answer_utils import leading_yes_no
from logger import get_logger

_log = get_logger("agents.verification_agent")


@component
class VerificationAgent:
    """
    Agent 6: Verification Agent
    Checks consistency and believability of the generated answer after reasoning.
    """
    def __init__(self):
        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=MODELS["check"],
        )

    def _normalize_text(self, text: str) -> str:
        return (text or "").strip().lower()

    def _normalize_yes_no(self, text: str) -> str:
        normalized = self._normalize_text(text)
        return leading_yes_no(normalized) or normalized

    def _format_heuristics(self, answer: str, qa: dict, original_question: str,
                           count_estimate=None, count_confidence: str = "unknown",
                           count_method: str = "none") -> dict:
        question_type = (qa or {}).get("question_type", "")
        expected_format = (qa or {}).get("expected_answer_format", "")
        question_subtype = (qa or {}).get("question_subtype", "")
        answer_text = self._normalize_text(answer)
        question_text = self._normalize_text(original_question)

        checks = {
            "answer_matches_question_type": True,
            "explanation_grounded_in_evidence": True,
            "answer_strength_appropriate": True,
            "no_contradictions": True,
            "format_appropriate": True,
        }
        issues = []

        yes_no_expected = expected_format == "yes_no" or question_type in {"yes_no", "presence_absence"}
        if yes_no_expected:
            yes_no_answer = self._normalize_yes_no(answer_text)
            if yes_no_answer not in {"yes", "no"}:
                checks["format_appropriate"] = False
                issues.append("yes/no answer did not normalize to yes or no")

        if question_type == "location":
            if answer_text in {"left", "right", "bilateral", "midline"}:
                checks["format_appropriate"] = False
                issues.append("location answer is too terse")
            if question_subtype == "lesion_location":
                if any(phrase in answer_text for phrase in {
                    "no discrete mass is visible",
                    "no mass is visible",
                    "no focal mass is visible",
                    "no lesion is visible",
                    "no discrete lesion is visible",
                }):
                    checks["answer_strength_appropriate"] = False
                    issues.append("lesion location answer abstains instead of localizing the finding")
                if answer_text in {"mass", "lesion", "nodule", "opacity", "finding", "abnormality"}:
                    checks["format_appropriate"] = False
                    issues.append("lesion location answer is too generic")

        if question_subtype == "finding_description":
            generic_labels = {
                "abnormal", "abnormality", "finding", "mass", "lesion", "opacity", "nodule",
                "present", "seen", "visible", "detected", "identified",
            }
            if answer_text in generic_labels:
                checks["format_appropriate"] = False
                issues.append("finding description answer is too generic")

        if question_subtype == "counting_exact":
            if not re.search(r"\b\d+\b", answer_text) and answer_text not in {"multiple", "few", "several", "cannot reliably count", "unable to count", "uncertain"}:
                checks["format_appropriate"] = False
                issues.append("exact counting answer is not numeric")
            if count_estimate not in {None, "", "unknown"} and re.search(r"\b\d+\b", answer_text):
                answer_count = int(re.search(r"\b(\d+)\b", answer_text).group(1))
                if isinstance(count_estimate, int) and abs(answer_count - count_estimate) > 1 and count_confidence == "high":
                    checks["answer_strength_appropriate"] = False
                    issues.append("counting answer disagrees with image count estimate")
                if count_method == "text_inference" and count_confidence != "high":
                    checks["answer_strength_appropriate"] = False
                    issues.append("counting answer is too precise for a text-inferred estimate")

        if question_type == "anatomy_identification":
            generic_labels = {
                "nervous system", "central nervous system", "respiratory system", "circulatory system",
                "musculoskeletal system", "digestive system", "urinary system", "reproductive system",
            }
            if answer_text in generic_labels and any(token in question_text for token in {"brain", "lung", "kidney", "liver", "heart", "spleen"}):
                checks["format_appropriate"] = False
                issues.append("anatomy answer is too generic for the question")

        if "cannot determine" in answer_text or "unable to determine" in answer_text:
            checks["answer_strength_appropriate"] = False
            issues.append("answer is an abstention despite a concrete question")

        return {"checks": checks, "issues": issues}

    def _merge_checks(self, model_checks: dict, heuristic_checks: dict) -> dict:
        merged = {
            "answer_matches_question_type": bool(model_checks.get("answer_matches_question_type", True)) and bool(heuristic_checks.get("answer_matches_question_type", True)),
            "explanation_grounded_in_evidence": bool(model_checks.get("explanation_grounded_in_evidence", True)) and bool(heuristic_checks.get("explanation_grounded_in_evidence", True)),
            "answer_strength_appropriate": bool(model_checks.get("answer_strength_appropriate", True)) and bool(heuristic_checks.get("answer_strength_appropriate", True)),
            "no_contradictions": bool(model_checks.get("no_contradictions", True)) and bool(heuristic_checks.get("no_contradictions", True)),
            "format_appropriate": bool(model_checks.get("format_appropriate", True)) and bool(heuristic_checks.get("format_appropriate", True)),
        }
        return merged

    @component.output_types(result=Dict[str, Any])
    def run(self, 
            answer_data: dict = None, 
            image_interpretation: dict = None,
            question_analysis: dict = None,
            filtered_image_context: dict = None,
            original_question: str = "",
            image_bytes_b64: str = "",
            mime_type: str = "image/png"):
        """With image_bytes_b64 the verifier checks the answer against the image itself
        (a cross-modal check) and recommends the next action for the agent controller."""

        ans = answer_data.get("result", answer_data) if answer_data else {}
        vs = image_interpretation.get("result", image_interpretation) if image_interpretation else {}
        qa = question_analysis.get("result", question_analysis) if question_analysis else {}
        fc = filtered_image_context.get("result", filtered_image_context) if filtered_image_context else {}

        if not isinstance(ans, dict): ans = {}
        if not isinstance(vs, dict): vs = {}
        if not isinstance(qa, dict): qa = {}
        if not isinstance(fc, dict): fc = {}

        countable_objects = vs.get("countable_objects", []) if isinstance(vs, dict) else []
        count_estimate = vs.get("count_estimate", "unknown") if isinstance(vs, dict) else "unknown"
        count_confidence = vs.get("count_confidence", "unknown") if isinstance(vs, dict) else "unknown"
        count_method = vs.get("count_method", "none") if isinstance(vs, dict) else "none"
        if not isinstance(countable_objects, list):
            countable_objects = []

        _log.info(
            "Verifying answer=%r  question=%r  question_type=%s",
            str(ans.get("answer", ""))[:80],
            original_question[:80] if original_question else "",
            qa.get("question_type", "unknown"),
        )
        started = time.perf_counter()

        action_fields = ""
        if image_bytes_b64:
            action_fields = (
                ',\n  "recommended_action": "accept / relook / reanswer / abstain",'
                '\n  "relook_focus": "If relook: the specific region or structure to re-examine, else empty",'
                '\n  "reanswer_guidance": "If reanswer: what the next answer must fix, else empty"'
            )

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

QUESTION-FOCUSED IMAGE CONTEXT:
- Filtered visual summary: {fc.get('filtered_visual_summary', 'Not available')}
- Relevant findings: {', '.join(fc.get('relevant_findings', [])) if isinstance(fc.get('relevant_findings', []), list) else fc.get('relevant_findings', 'Not available')}
- Relevant structures: {', '.join(fc.get('relevant_structures', [])) if isinstance(fc.get('relevant_structures', []), list) else fc.get('relevant_structures', 'Not available')}

QUESTION ANALYSIS:
- Question type: {qa.get('question_type', 'unknown')}
- Question subtype: {qa.get('question_subtype', 'other')}
- Expected format: {qa.get('expected_answer_format', 'unknown')}

COUNTING SIGNALS:
- Count estimate: {count_estimate}
- Count confidence: {count_confidence}
- Count method: {count_method}
- Countable objects: {countable_objects}

IMPORTANT: Be stricter about answer format than content plausibility. If the answer is too generic, too terse, or not aligned with the expected format, mark it as not fully verified even if it sounds medically plausible.

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
  "suggested_disclaimer": "If needed, a safety disclaimer to append (empty string if not needed)"{action_fields}
}}

Return ONLY the JSON object, no additional text."""

        try:
            if image_bytes_b64:
                prompt += """

The medical image is attached. Judge the answer against the IMAGE ITSELF, not only against the text
analysis above: the text analysis can be incomplete or wrong. Then recommend the next action:
- "accept": the image supports the answer.
- "relook": the answer depends on a region or structure that needs a closer, focused look (name it in relook_focus).
- "reanswer": the evidence is adequate but the answer is wrong, unsupported, or badly formatted (say what to fix).
- "abstain": the image cannot answer the question reliably."""
                message = ChatMessage.from_user(content_parts=[
                    TextContent(text=prompt),
                    ImageContent(base64_image=image_bytes_b64, mime_type=mime_type, detail="high"),
                ])
            else:
                message = ChatMessage.from_user(prompt)
            response = self.generator.run(
                messages=[message],
                generation_kwargs=generation_kwargs("check")
            )

            record_usage("check", MODELS["check"], response["replies"][0].meta.get("usage"))

            raw = strip_json_fences(response["replies"][0].text or "")
            result = json.loads(raw)
            heuristic = self._format_heuristics(
                ans.get("answer", ""), qa, original_question,
                count_estimate=count_estimate,
                count_confidence=count_confidence,
                count_method=count_method,
            )
            result["checks"] = self._merge_checks(result.get("checks", {}) or {}, heuristic.get("checks", {}) or {})

            issues = list(result.get("issues_found", []) or [])
            for issue in heuristic.get("issues", []):
                if issue not in issues:
                    issues.append(issue)
            result["issues_found"] = issues

            if not result["checks"].get("format_appropriate", True) or not result["checks"].get("answer_matches_question_type", True):
                if result.get("verification_status") == "verified":
                    result["verification_status"] = "partially_verified"
                if result.get("confidence_label") == "High":
                    result["confidence_label"] = "Medium"

            result["status"] = "success"
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            checks = result.get("checks", {}) or {}
            passed = sum(1 for v in checks.values() if v)
            _log.info(
                "Verification done  status=%s  confidence=%s  checks=%d/%d  duration=%dms",
                result.get("verification_status", "unknown"),
                result.get("confidence_label", "unknown"),
                passed,
                len(checks),
                elapsed_ms,
            )
            issues = result.get("issues_found", [])
            if issues:
                _log.warning("Verification issues: %s", issues)
            _log.info("Fallback recommendation: %s", result.get("fallback_recommendation", "none"))
            _log.debug("Verification notes: %s", result.get("verification_notes", ""))
            return {"result": result}

        except json.JSONDecodeError:
            _log.error("Failed to parse verification response JSON")
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
            _log.error("Unexpected error in verification: %s", e)
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
