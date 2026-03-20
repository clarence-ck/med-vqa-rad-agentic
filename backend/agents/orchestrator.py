import time
from typing import Any, Dict
from haystack import Pipeline, component
from agents.image_interpreter import ImageInterpreter
from agents.question_analyzer import QuestionAnalyzer
from agents.knowledge_retriever import KnowledgeRetriever
from agents.reasoning_agent import ReasoningAgent
from agents.verification_agent import VerificationAgent


@component
class TimedImageInterpreter:
    def __init__(self):
        self.inner = ImageInterpreter()

    @component.output_types(result=Dict[str, Any], duration_ms=int)
    def run(self, image_bytes: bytes, mime_type: str = "image/png"):
        started_at = time.time()
        output = self.inner.run(image_bytes=image_bytes, mime_type=mime_type)
        duration_ms = int((time.time() - started_at) * 1000)
        return {
            "result": output["result"],
            "duration_ms": duration_ms,
        }


@component
class TimedQuestionAnalyzer:
    def __init__(self):
        self.inner = QuestionAnalyzer()

    @component.output_types(result=Dict[str, Any], duration_ms=int)
    def run(self, question: str, image_interpretation: dict = None):
        started_at = time.time()
        output = self.inner.run(question=question, image_interpretation=image_interpretation)
        duration_ms = int((time.time() - started_at) * 1000)
        return {
            "result": output["result"],
            "duration_ms": duration_ms,
        }


@component
class TimedKnowledgeRetriever:
    def __init__(self):
        self.inner = KnowledgeRetriever()

    @component.output_types(result=Dict[str, Any], duration_ms=int)
    def run(self,
            image_interpretation: dict = None,
            question_analysis: dict = None,
            original_question: str = "",
            question_type: str = "",
            body_region: str = "",
            modality: str = "",
            visual_cues: list = None,
            target_entity: str = ""):
        started_at = time.time()
        output = self.inner.run(
            image_interpretation=image_interpretation,
            question_analysis=question_analysis,
            original_question=original_question,
            question_type=question_type,
            body_region=body_region,
            modality=modality,
            visual_cues=visual_cues,
            target_entity=target_entity,
        )
        duration_ms = int((time.time() - started_at) * 1000)
        return {
            "result": output["result"],
            "duration_ms": duration_ms,
        }


@component
class TimedReasoningAgent:
    def __init__(self):
        self.inner = ReasoningAgent()

    @component.output_types(result=Dict[str, Any], duration_ms=int)
    def run(self,
            image_interpretation: dict = None,
            question_analysis: dict = None,
            knowledge: dict = None,
            original_question: str = "",
            retry_context: dict = None):
        started_at = time.time()
        output = self.inner.run(
            image_interpretation=image_interpretation,
            question_analysis=question_analysis,
            knowledge=knowledge,
            original_question=original_question,
            retry_context=retry_context,
        )
        duration_ms = int((time.time() - started_at) * 1000)
        return {
            "result": output["result"],
            "duration_ms": duration_ms,
        }


@component
class TimedVerificationAgent:
    def __init__(self):
        self.inner = VerificationAgent()

    @component.output_types(result=Dict[str, Any], duration_ms=int)
    def run(self,
            answer_data: dict = None,
            image_interpretation: dict = None,
            question_analysis: dict = None,
            original_question: str = ""):
        started_at = time.time()
        output = self.inner.run(
            answer_data=answer_data,
            image_interpretation=image_interpretation,
            question_analysis=question_analysis,
            original_question=original_question,
        )
        duration_ms = int((time.time() - started_at) * 1000)
        return {
            "result": output["result"],
            "duration_ms": duration_ms,
        }


def create_pipeline() -> Pipeline:
    """Create a fresh pipeline instance. Called per-request to avoid shared mutable state."""
    pipeline = Pipeline()
    pipeline.add_component("image_interpreter", TimedImageInterpreter())
    pipeline.add_component("question_analyzer", TimedQuestionAnalyzer())
    pipeline.add_component("knowledge_retriever", TimedKnowledgeRetriever())
    pipeline.add_component("reasoning_agent", TimedReasoningAgent())
    pipeline.add_component("verification_agent", TimedVerificationAgent())
    pipeline.connect("image_interpreter.result", "question_analyzer.image_interpretation")
    pipeline.connect("image_interpreter.result", "knowledge_retriever.image_interpretation")
    pipeline.connect("image_interpreter.result", "reasoning_agent.image_interpretation")
    pipeline.connect("image_interpreter.result", "verification_agent.image_interpretation")
    pipeline.connect("question_analyzer.result", "knowledge_retriever.question_analysis")
    pipeline.connect("question_analyzer.result", "reasoning_agent.question_analysis")
    pipeline.connect("question_analyzer.result", "verification_agent.question_analysis")
    pipeline.connect("knowledge_retriever.result", "reasoning_agent.knowledge")
    pipeline.connect("reasoning_agent.result", "verification_agent.answer_data")
    return pipeline


def _categorize_failure_modes(knowledge_result: dict, answer_result: dict, verification_result: dict) -> list:
    categories = []
    issues = [str(issue) for issue in verification_result.get("issues_found", [])]
    verification_status = verification_result.get("verification_status", "unknown")
    knowledge_status = knowledge_result.get("status", "unknown")
    reasoning_status = answer_result.get("status", "unknown")

    def add_category(code: str, label: str, description: str, severity: str):
        if any(category["code"] == code for category in categories):
            return
        categories.append({
            "code": code,
            "label": label,
            "description": description,
            "severity": severity,
        })

    if knowledge_status != "success":
        add_category(
            "knowledge_retrieval_failed",
            "Knowledge retrieval failed",
            "The support agent did not return reliable contextual notes or key facts.",
            "medium",
        )

    if reasoning_status != "success":
        add_category(
            "reasoning_generation_unstable",
            "Reasoning generation unstable",
            "The answer generation step did not complete cleanly.",
            "high",
        )

    if verification_status == "unsupported":
        add_category(
            "unsupported_claim",
            "Unsupported claim",
            "The final answer is not sufficiently grounded in the visual evidence.",
            "high",
        )
    elif verification_status == "low_confidence":
        add_category(
            "low_confidence",
            "Low confidence",
            "The system does not have enough evidence for a confident answer.",
            "high",
        )
    elif verification_status == "partially_verified":
        add_category(
            "partially_verified",
            "Partially verified",
            "Some parts of the answer appear plausible, but important uncertainty remains.",
            "medium",
        )

    issue_text = " ".join(issues).lower()
    if "contradiction" in issue_text:
        add_category(
            "visual_contradiction",
            "Visual contradiction",
            "The answer appears to conflict with the image interpretation output.",
            "high",
        )
    if "format" in issue_text or "question type" in issue_text:
        add_category(
            "question_answer_mismatch",
            "Question-answer mismatch",
            "The returned answer format or content may not align with the question type.",
            "medium",
        )
    if "parse" in issue_text:
        add_category(
            "parser_failure",
            "Parser failure",
            "A structured model response could not be parsed cleanly.",
            "high",
        )

    return categories


def _build_guardrail_payload(answer_result: dict, verification_result: dict, failure_categories: list) -> dict:
    answer_text = answer_result.get("answer", "Unable to generate an answer.")
    fallback_recommendation = verification_result.get("fallback_recommendation", "none")
    suggested_disclaimer = verification_result.get("suggested_disclaimer", "")
    verification_status = verification_result.get("verification_status", "low_confidence")
    severe_codes = {category["code"] for category in failure_categories if category["severity"] == "high"}

    guarded_answer = answer_text
    should_flag_for_review = fallback_recommendation == "flag_for_review" or bool(
        {"unsupported_claim", "visual_contradiction", "parser_failure"} & severe_codes
    )

    if verification_status in {"low_confidence", "unsupported"}:
        guarded_answer = "The available evidence is insufficient to provide a reliable answer."
        if answer_text:
            guarded_answer = f"{guarded_answer} Most likely response: {answer_text}"

    if not suggested_disclaimer and (verification_status in {"low_confidence", "unsupported"} or should_flag_for_review):
        suggested_disclaimer = "This response is uncertain and should be reviewed by a qualified expert."

    if fallback_recommendation == "none" and verification_status in {"low_confidence", "unsupported"}:
        fallback_recommendation = "add_disclaimer" if verification_status == "low_confidence" else "flag_for_review"

    return {
        "guarded_answer": guarded_answer,
        "fallback_recommendation": fallback_recommendation,
        "suggested_disclaimer": suggested_disclaimer,
        "should_flag_for_review": should_flag_for_review,
    }


def _build_self_consistency_signal(verification_result: dict) -> dict:
    checks = verification_result.get("checks", {}) or {}
    total_checks = len(checks)
    passed_checks = sum(1 for passed in checks.values() if passed)
    verification_status = verification_result.get("verification_status", "low_confidence")

    if verification_status == "verified" and total_checks and passed_checks == total_checks:
        level = "high"
    elif (
        total_checks > 0
        and verification_status in {"verified", "partially_verified"}
        and passed_checks >= max(1, total_checks - 1)
    ):
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
    }


def _build_cross_modal_agreement_signal(verification_result: dict) -> dict:
    checks = verification_result.get("checks", {}) or {}
    grounded = checks.get("explanation_grounded_in_evidence")
    contradiction_free = checks.get("no_contradictions")

    if grounded is True and contradiction_free is True:
        level = "high"
    elif grounded is False or contradiction_free is False:
        level = "low"
    else:
        level = "medium"

    return {
        "level": level,
        "grounded_in_evidence": grounded,
        "no_contradictions": contradiction_free,
    }


def _build_agent_trace(image_result: dict, question_result: dict, knowledge_result: dict,
                       answer_result: dict, verification_result: dict,
                       image_duration_ms: int, question_duration_ms: int,
                       knowledge_duration_ms: int, answer_duration_ms: int,
                       verification_duration_ms: int) -> list:
    return [
        {
            "agent": "Image Interpretation",
            "agent_id": 1,
            "status": "pass" if image_result.get("status") == "success" else "warning",
            "summary": image_result.get("visual_summary", "No summary produced."),
            "output": {
                "modality": image_result.get("modality", "unknown"),
                "body_region": image_result.get("body_region", "unknown"),
                "visible_structures": image_result.get("visible_structures", []),
            },
            "duration_ms": image_duration_ms,
        },
        {
            "agent": "Question Understanding",
            "agent_id": 2,
            "status": "pass" if question_result.get("status") == "success" else "warning",
            "summary": f"Type: {question_result.get('question_type', 'unknown')} | Format: {question_result.get('expected_answer_format', 'unknown')}",
            "output": {
                "question_type": question_result.get("question_type", "unknown"),
                "target_entity": question_result.get("target_entity", "unknown"),
                "expected_answer_format": question_result.get("expected_answer_format", "short_phrase"),
            },
            "duration_ms": question_duration_ms,
        },
        {
            "agent": "Knowledge Support",
            "agent_id": 3,
            "status": "pass" if knowledge_result.get("status") == "success" else "warning",
            "summary": knowledge_result.get("contextual_notes", "No context retrieved."),
            "output": {
                "key_facts": knowledge_result.get("key_facts", []),
            },
            "duration_ms": knowledge_duration_ms,
        },
        {
            "agent": "Reasoning & Answer",
            "agent_id": 4,
            "status": "pass" if answer_result.get("status") == "success" else "warning",
            "summary": f"Answer: {answer_result.get('answer', 'N/A')}",
            "output": {
                "answer": answer_result.get("answer", ""),
                "confidence": answer_result.get("answer_confidence", "low"),
            },
            "duration_ms": answer_duration_ms,
        },
        {
            "agent": "Verification",
            "agent_id": 5,
            "status": "pass" if verification_result.get("verification_status") == "verified" else "warning" if verification_result.get("verification_status") == "partially_verified" else "fail",
            "summary": f"Status: {verification_result.get('verification_status', 'unknown')} | Confidence: {verification_result.get('confidence_label', 'unknown')}",
            "output": {
                "verification_status": verification_result.get("verification_status", "unknown"),
                "confidence_label": verification_result.get("confidence_label", "Low"),
                "issues_found": verification_result.get("issues_found", []),
            },
            "duration_ms": verification_duration_ms,
        },
    ]


def _assemble_pipeline_payload(results: dict, question: str, started_at: float, retry_attempts: int = 0,
                               retry_reason: str = "") -> dict:
    image_result = results.get("image_interpreter", {}).get("result", {})
    image_duration_ms = results.get("image_interpreter", {}).get("duration_ms", 0)
    question_result = results.get("question_analyzer", {}).get("result", {})
    question_duration_ms = results.get("question_analyzer", {}).get("duration_ms", 0)
    knowledge_result = results.get("knowledge_retriever", {}).get("result", {})
    knowledge_duration_ms = results.get("knowledge_retriever", {}).get("duration_ms", 0)
    answer_result = results.get("reasoning_agent", {}).get("result", {})
    answer_duration_ms = results.get("reasoning_agent", {}).get("duration_ms", 0)
    verification_result = results.get("verification_agent", {}).get("result", {})
    verification_duration_ms = results.get("verification_agent", {}).get("duration_ms", 0)
    failure_categories = _categorize_failure_modes(
        knowledge_result=knowledge_result,
        answer_result=answer_result,
        verification_result=verification_result,
    )
    guardrail_payload = _build_guardrail_payload(
        answer_result=answer_result,
        verification_result=verification_result,
        failure_categories=failure_categories,
    )
    self_consistency_signal = _build_self_consistency_signal(verification_result)
    cross_modal_agreement = _build_cross_modal_agreement_signal(verification_result)
    agent_trace = _build_agent_trace(
        image_result=image_result,
        question_result=question_result,
        knowledge_result=knowledge_result,
        answer_result=answer_result,
        verification_result=verification_result,
        image_duration_ms=image_duration_ms,
        question_duration_ms=question_duration_ms,
        knowledge_duration_ms=knowledge_duration_ms,
        answer_duration_ms=answer_duration_ms,
        verification_duration_ms=verification_duration_ms,
    )
    total_ms = int((time.time() - started_at) * 1000)
    return {
        "answer": answer_result.get("answer", "Unable to generate an answer."),
        "guarded_answer": guardrail_payload.get("guarded_answer", answer_result.get("answer", "Unable to generate an answer.")),
        "explanation": answer_result.get("explanation", ""),
        "reasoning_trace": answer_result.get("reasoning_trace", ""),
        "evidence_used": answer_result.get("evidence_used", []),
        "confidence": verification_result.get("confidence_label", "Low"),
        "verification_status": verification_result.get("verification_status", "low_confidence"),
        "checks": verification_result.get("checks", {}),
        "verification_notes": verification_result.get("verification_notes", ""),
        "issues_found": verification_result.get("issues_found", []),
        "failure_categories": failure_categories,
        "self_consistency_signal": self_consistency_signal,
        "cross_modal_agreement": cross_modal_agreement,
        "fallback_recommendation": guardrail_payload.get("fallback_recommendation", verification_result.get("fallback_recommendation", "none")),
        "suggested_disclaimer": guardrail_payload.get("suggested_disclaimer", verification_result.get("suggested_disclaimer", "")),
        "should_flag_for_review": guardrail_payload.get("should_flag_for_review", False),
        "image_analysis": {
            "modality": image_result.get("modality", "unknown"),
            "body_region": image_result.get("body_region", "unknown"),
            "visible_structures": image_result.get("visible_structures", []),
            "visual_summary": image_result.get("visual_summary", ""),
            "image_quality": image_result.get("image_quality", "unknown"),
            "notable_findings": image_result.get("notable_findings", []),
        },
        "question_analysis": {
            "question_type": question_result.get("question_type", "unknown"),
            "target_entity": question_result.get("target_entity", "unknown"),
            "expected_answer_format": question_result.get("expected_answer_format", "short_phrase"),
            "question_clarity": question_result.get("question_clarity", "unclear"),
            "reasoning_type": question_result.get("reasoning_type", "visual_recognition"),
            "rephrased_question": question_result.get("rephrased_question", question),
        },
        "agent_trace": agent_trace,
        "total_duration_ms": total_ms,
        "retry_attempts": retry_attempts,
        "retry_reason": retry_reason,
    }


def _should_retry_pipeline(payload: dict) -> bool:
    verification_status = payload.get("verification_status", "low_confidence")
    if verification_status in {"low_confidence", "unsupported"}:
        return True
    if payload.get("should_flag_for_review"):
        return True
    failure_categories = payload.get("failure_categories", []) or []
    return any(category.get("severity") == "high" for category in failure_categories)


def _payload_score(payload: dict) -> tuple[int, int, int, int]:
    verification_status = payload.get("verification_status", "low_confidence")
    verification_rank = {
        "verified": 4,
        "partially_verified": 3,
        "low_confidence": 2,
        "unsupported": 1,
    }.get(verification_status, 0)
    confidence_rank = {
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }.get(str(payload.get("confidence", "Low")), 1)
    failure_categories = payload.get("failure_categories", []) or []
    high_severity_count = sum(1 for category in failure_categories if category.get("severity") == "high")
    review_flag = 1 if payload.get("should_flag_for_review") else 0
    return verification_rank, confidence_rank, -high_severity_count, -review_flag


def run_pipeline(image_bytes: bytes, question: str, mime_type: str = "image/png") -> dict:
    # Create a fresh pipeline per request — prevents shared mutable state under concurrency.
    pipeline_start = time.time()

    def _run_once(retry_context: dict | None = None) -> dict:
        pipeline = create_pipeline()
        return pipeline.run(
            {
                "image_interpreter": {
                    "image_bytes": image_bytes,
                    "mime_type": mime_type,
                },
                "question_analyzer": {
                    "question": question,
                },
                "knowledge_retriever": {
                    "original_question": question,
                },
                "reasoning_agent": {
                    "original_question": question,
                    "retry_context": retry_context,
                },
                "verification_agent": {
                    "original_question": question,
                },
            },
            include_outputs_from={
                "image_interpreter",
                "question_analyzer",
                "knowledge_retriever",
                "reasoning_agent",
                "verification_agent",
            },
        )

    first_results = _run_once()
    first_payload = _assemble_pipeline_payload(first_results, question, pipeline_start)

    retry_reason = ""
    retry_attempts = 0
    best_payload = first_payload

    if _should_retry_pipeline(first_payload):
        retry_attempts = 1
        retry_reason = (
            f"verification_status={first_payload.get('verification_status', 'unknown')}; "
            f"fallback={first_payload.get('fallback_recommendation', 'none')}; "
            f"flag_for_review={first_payload.get('should_flag_for_review', False)}"
        )
        retry_context = {
            "verification_status": first_payload.get("verification_status", "unknown"),
            "confidence": first_payload.get("confidence", "Low"),
            "fallback_recommendation": first_payload.get("fallback_recommendation", "none"),
            "failure_categories": first_payload.get("failure_categories", []),
        }
        second_results = _run_once(retry_context=retry_context)
        second_payload = _assemble_pipeline_payload(second_results, question, pipeline_start, retry_attempts=1, retry_reason=retry_reason)

        if _payload_score(second_payload) > _payload_score(first_payload):
            best_payload = second_payload
        else:
            best_payload = first_payload

    best_payload["retry_attempts"] = retry_attempts
    best_payload["retry_reason"] = retry_reason
    return best_payload
