import time
from typing import Any
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.utils import Secret
from agents.image_interpreter import ImageInterpreter
from agents.question_analyzer import QuestionAnalyzer
from agents.difficulty_router import DifficultyRouter
from agents.knowledge_retriever import KnowledgeRetriever
from agents.reasoning_agent import ReasoningAgent
from agents.verification_agent import VerificationAgent
from config import OPENAI_API_KEY, OPENAI_MODEL
from logger import get_logger

_log = get_logger("agents.orchestrator")


# ---------------------------------------------------------------------------
# Tool implementation functions — thin wrappers around each agent
# ---------------------------------------------------------------------------

class ImageBlockerAbort(BaseException):
    """A special exception that inherits from BaseException so Haystack's Agent 
    (which catches `Exception`) does not trap it. This allows us to instantly 
    abort the pipeline without Haystack stringifying binary images into tokens."""
    pass

def _tool_interpret_image(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    started = time.time()
    _log.info("[tool] interpret_image called")
    try:
        result = ImageInterpreter().run(image_bytes=image_bytes, mime_type=mime_type)["result"]
    except ValueError as e:
        # Trap the intentional blocker exception and escalate it as a BaseException
        raise ImageBlockerAbort(str(e))
        
    duration_ms = int((time.time() - started) * 1000)
    _log.info("[tool] interpret_image done  modality=%s  region=%s  duration=%dms",
              result.get("modality", "?"), result.get("body_region", "?"), duration_ms)
    return {"result": result, "duration_ms": duration_ms}


def _tool_analyze_question(question: str, image_interpretation: dict = None) -> dict:
    started = time.time()
    _log.info("[tool] analyze_question called  question=%r", question[:80])
    result = QuestionAnalyzer().run(question=question, image_interpretation=image_interpretation)["result"]
    duration_ms = int((time.time() - started) * 1000)
    _log.info("[tool] analyze_question done  type=%s  duration=%dms",
              result.get("question_type", "?"), duration_ms)
    return {"result": result, "duration_ms": duration_ms}


def _tool_route_difficulty(question_analysis: dict = None, image_interpretation: dict = None) -> dict:
    started = time.time()
    _log.info("[tool] route_difficulty called")
    result = DifficultyRouter().run(
        question_analysis=question_analysis,
        image_interpretation=image_interpretation,
    )["result"]
    duration_ms = int((time.time() - started) * 1000)
    _log.info("[tool] route_difficulty done  difficulty=%s  needs_retrieval=%s  duration=%dms",
              result.get("difficulty", "?"), result.get("needs_retrieval", "?"), duration_ms)
    return {"result": result, "duration_ms": duration_ms}


def _tool_retrieve_knowledge(image_interpretation: dict = None, question_analysis: dict = None,
                             question: str = "", routing_decision: dict = None) -> dict:
    if routing_decision and not routing_decision.get("needs_retrieval", True):
        _log.info("[tool] retrieve_knowledge SKIPPED (needs_retrieval=False)")
        return {"result": {
            "status": "skipped",
            "contextual_notes": "Retrieval skipped — routing decision: not needed.",
            "key_facts": [],
            "expected_normal_findings": "",
            "common_pathologies": [],
            "terminology": {},
        }, "duration_ms": 0}
    started = time.time()
    _log.info("[tool] retrieve_knowledge called")
    result = KnowledgeRetriever().run(
        image_interpretation=image_interpretation,
        question_analysis=question_analysis,
        original_question=question,
    )["result"]
    duration_ms = int((time.time() - started) * 1000)
    _log.info("[tool] retrieve_knowledge done  key_facts=%d  rag_cases=%d  duration=%dms",
              len(result.get("key_facts", [])),
              len(result.get("rag_retrieval", {}).get("similar_cases", [])),
              duration_ms)
    return {"result": result, "duration_ms": duration_ms}


def _tool_reason_answer(image_interpretation: dict = None, question_analysis: dict = None,
                        knowledge: dict = None, question: str = "",
                        retry_context: dict = None) -> dict:
    started = time.time()
    _log.info("[tool] reason_answer called  is_retry=%s", bool(retry_context))
    result = ReasoningAgent().run(
        image_interpretation=image_interpretation,
        question_analysis=question_analysis,
        knowledge=knowledge,
        original_question=question,
        retry_context=retry_context,
    )["result"]
    duration_ms = int((time.time() - started) * 1000)
    _log.info("[tool] reason_answer done  answer=%r  confidence=%s  duration=%dms",
              str(result.get("answer", ""))[:60], result.get("answer_confidence", "?"), duration_ms)
    return {"result": result, "duration_ms": duration_ms}


def _tool_verify_answer(answer_data: dict = None, image_interpretation: dict = None,
                        question_analysis: dict = None, question: str = "") -> dict:
    started = time.time()
    _log.info("[tool] verify_answer called")
    result = VerificationAgent().run(
        answer_data=answer_data,
        image_interpretation=image_interpretation,
        question_analysis=question_analysis,
        original_question=question,
    )["result"]
    duration_ms = int((time.time() - started) * 1000)
    _log.info("[tool] verify_answer done  status=%s  confidence=%s  duration=%dms",
              result.get("verification_status", "?"), result.get("confidence_label", "?"), duration_ms)
    return {"result": result, "duration_ms": duration_ms}


# ---------------------------------------------------------------------------
# Haystack Tool definitions — LLM-callable with state I/O bindings
# ---------------------------------------------------------------------------

_INTERPRET_IMAGE_TOOL = Tool(
    name="interpret_image",
    description=(
        "Analyze the medical image to extract modality, body region, visible structures, "
        "and a visual summary. Always call this first before any other tool."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    function=_tool_interpret_image,
    inputs_from_state={"image_bytes": "image_bytes", "mime_type": "mime_type"},
    outputs_to_state={
        "image_interpretation": {"source": "result"},
        "image_duration_ms": {"source": "duration_ms"},
    },
)

_ANALYZE_QUESTION_TOOL = Tool(
    name="analyze_question",
    description=(
        "Analyze the medical VQA question to classify its type, identify the target entity, "
        "and determine the expected answer format. Always call after interpret_image."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    function=_tool_analyze_question,
    inputs_from_state={"question": "question", "image_interpretation": "image_interpretation"},
    outputs_to_state={
        "question_analysis": {"source": "result"},
        "question_duration_ms": {"source": "duration_ms"},
    },
)

_ROUTE_DIFFICULTY_TOOL = Tool(
    name="route_difficulty",
    description=(
        "Evaluate the complexity of the question and determine the appropriate pipeline depth. "
        "Call this after analyze_question and before retrieve_knowledge. "
        "It returns needs_retrieval (whether to call retrieve_knowledge), "
        "verification_policy (standard or strict), and fallback_policy."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    function=_tool_route_difficulty,
    inputs_from_state={
        "question_analysis": "question_analysis",
        "image_interpretation": "image_interpretation",
    },
    outputs_to_state={
        "routing_decision": {"source": "result"},
        "routing_duration_ms": {"source": "duration_ms"},
    },
)

_RETRIEVE_KNOWLEDGE_TOOL = Tool(
    name="retrieve_knowledge",
    description=(
        "Retrieve supporting medical context and background knowledge relevant to the question. "
        "Call this ONLY when the routing_decision says needs_retrieval is true. "
        "Skip if routing_decision says needs_retrieval is false."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    function=_tool_retrieve_knowledge,
    inputs_from_state={
        "image_interpretation": "image_interpretation",
        "question_analysis": "question_analysis",
        "question": "question",
        "routing_decision": "routing_decision",
    },
    outputs_to_state={
        "knowledge": {"source": "result"},
        "knowledge_duration_ms": {"source": "duration_ms"},
    },
)

_REASON_ANSWER_TOOL = Tool(
    name="reason_answer",
    description=(
        "Synthesize image analysis, question understanding, and any retrieved knowledge "
        "to produce a final answer with explanation and confidence level. "
        "Call after analyze_question (and after retrieve_knowledge if it was invoked)."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    function=_tool_reason_answer,
    inputs_from_state={
        "image_interpretation": "image_interpretation",
        "question_analysis": "question_analysis",
        "knowledge": "knowledge",
        "question": "question",
        "retry_context": "retry_context",
    },
    outputs_to_state={
        "answer_data": {"source": "result"},
        "answer_duration_ms": {"source": "duration_ms"},
    },
)

_VERIFY_ANSWER_TOOL = Tool(
    name="verify_answer",
    description=(
        "Critically evaluate whether the generated answer is consistent, grounded in visual "
        "evidence, and appropriate for the question type. Always call this last to finalize."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    function=_tool_verify_answer,
    inputs_from_state={
        "answer_data": "answer_data",
        "image_interpretation": "image_interpretation",
        "question_analysis": "question_analysis",
        "question": "question",
    },
    outputs_to_state={
        "verification_result": {"source": "result"},
        "verification_duration_ms": {"source": "duration_ms"},
    },
)

_AGENT_SYSTEM_PROMPT = (
    "You are a medical visual question answering (VQA) orchestrator. "
    "You coordinate specialized tools to answer questions about medical images accurately and safely.\n\n"
    "PROCEDURE (follow in order):\n"
    "1. interpret_image — always required first.\n"
    "2. analyze_question — always required second.\n"
    "3. route_difficulty — always required third. Evaluates question complexity and sets "
    "needs_retrieval, verification_policy, and fallback_policy for downstream steps.\n"
    "4. retrieve_knowledge — call ONLY if routing_decision.needs_retrieval is true. "
    "Skip if routing_decision.needs_retrieval is false.\n"
    "5. reason_answer — always required.\n"
    "6. verify_answer — always required last; finalizes the session."
)

_STATE_SCHEMA = {
    "image_bytes": {"type": Any},
    "mime_type": {"type": str},
    "question": {"type": str},
    "image_interpretation": {"type": dict},
    "question_analysis": {"type": dict},
    "routing_decision": {"type": dict},
    "knowledge": {"type": dict},
    "answer_data": {"type": dict},
    "verification_result": {"type": dict},
    "image_duration_ms": {"type": int},
    "question_duration_ms": {"type": int},
    "routing_duration_ms": {"type": int},
    "knowledge_duration_ms": {"type": int},
    "answer_duration_ms": {"type": int},
    "verification_duration_ms": {"type": int},
    "retry_context": {"type": dict},
}


def _create_agent() -> Agent:
    return Agent(
        chat_generator=OpenAIChatGenerator(
            api_key=Secret.from_token(OPENAI_API_KEY),
            model=OPENAI_MODEL,
        ),
        system_prompt=_AGENT_SYSTEM_PROMPT,
        tools=[
            _INTERPRET_IMAGE_TOOL,
            _ANALYZE_QUESTION_TOOL,
            _ROUTE_DIFFICULTY_TOOL,
            _RETRIEVE_KNOWLEDGE_TOOL,
            _REASON_ANSWER_TOOL,
            _VERIFY_ANSWER_TOOL,
        ],
        exit_conditions=["verify_answer"],
        state_schema=_STATE_SCHEMA,
    )


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

    if categories:
        _log.debug("Failure categories detected: %s", [c['code'] for c in categories])
    return categories


def _build_guardrail_payload(answer_result: dict, verification_result: dict, failure_categories: list) -> dict:
    """Build guardrail output: guarded answer, disclaimer, flag-for-review signal."""
    answer = answer_result.get("answer", "Unable to generate an answer.")
    verification_status = verification_result.get("verification_status", "low_confidence")
    fallback_recommendation = verification_result.get("fallback_recommendation", "none")
    suggested_disclaimer = verification_result.get("suggested_disclaimer", "")
    high_severity = any(c.get("severity") == "high" for c in (failure_categories or []))

    # Surface a flag if the answer is untrustworthy
    should_flag_for_review = (
        verification_status in {"low_confidence", "unsupported"}
        or high_severity
    )

    # Add a default disclaimer when the answer is flagged and none was supplied
    if should_flag_for_review and not suggested_disclaimer:
        suggested_disclaimer = (
            "This answer has low verification confidence. "
            "Please review with a qualified medical professional."
        )

    # The guarded_answer is the same as answer unless the verification outright rejected it
    if verification_status == "unsupported":
        guarded_answer = (
            f"{answer} [Note: This answer could not be fully verified against the image evidence.]"
        )
    else:
        guarded_answer = answer

    _log.info(
        "Guardrail decision  flag_for_review=%s  fallback=%s  disclaimer=%s",
        should_flag_for_review,
        fallback_recommendation,
        bool(suggested_disclaimer),
    )
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

    _log.debug("Self-consistency signal: level=%s  passed=%d/%d", level, passed_checks, total_checks)
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

    _log.debug("Cross-modal agreement: level=%s  grounded=%s  no_contradictions=%s", level, grounded, contradiction_free)
    return {
        "level": level,
        "grounded_in_evidence": grounded,
        "no_contradictions": contradiction_free,
    }


def _build_agent_trace(image_result: dict, question_result: dict, routing_result: dict,
                       knowledge_result: dict, answer_result: dict, verification_result: dict,
                       image_duration_ms: int, question_duration_ms: int,
                       routing_duration_ms: int, knowledge_duration_ms: int,
                       answer_duration_ms: int, verification_duration_ms: int) -> list:
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
            "agent": "Difficulty Routing",
            "agent_id": 3,
            "status": "pass" if routing_result.get("status") == "success" else "warning",
            "summary": (
                f"Difficulty: {routing_result.get('difficulty', 'unknown')} | "
                f"Retrieval: {'yes' if routing_result.get('needs_retrieval') else 'no'} | "
                f"Verification: {routing_result.get('verification_policy', 'standard')}"
            ),
            "output": {
                "difficulty": routing_result.get("difficulty", "unknown"),
                "needs_retrieval": routing_result.get("needs_retrieval", False),
                "retrieval_rationale": routing_result.get("retrieval_rationale", ""),
                "verification_policy": routing_result.get("verification_policy", "standard"),
                "fallback_policy": routing_result.get("fallback_policy", "none"),
                "routing_rationale": routing_result.get("routing_rationale", ""),
            },
            "duration_ms": routing_duration_ms,
        },
        {
            "agent": "Knowledge Support",
            "agent_id": 4,
            "status": "skipped" if (knowledge_duration_ms == 0 and not knowledge_result.get("key_facts")) else ("pass" if knowledge_result.get("status") == "success" else "warning"),
            "summary": "Skipped — routing decision: retrieval not needed." if (knowledge_duration_ms == 0 and not knowledge_result.get("key_facts")) else knowledge_result.get("contextual_notes", "No context retrieved."),
            "output": {
                "key_facts": knowledge_result.get("key_facts", []),
            },
            "duration_ms": knowledge_duration_ms,
        },
        {
            "agent": "Reasoning & Answer",
            "agent_id": 5,
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
            "agent_id": 6,
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
    routing_result = results.get("difficulty_router", {}).get("result", {})
    routing_duration_ms = results.get("difficulty_router", {}).get("duration_ms", 0)
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
        routing_result=routing_result,
        knowledge_result=knowledge_result,
        answer_result=answer_result,
        verification_result=verification_result,
        image_duration_ms=image_duration_ms,
        question_duration_ms=question_duration_ms,
        routing_duration_ms=routing_duration_ms,
        knowledge_duration_ms=knowledge_duration_ms,
        answer_duration_ms=answer_duration_ms,
        verification_duration_ms=verification_duration_ms,
    )
    total_ms = int((time.time() - started_at) * 1000)
    _log.info(
        "Pipeline payload assembled  answer=%r  verification=%s  guardrail_flag=%s  consistency=%s  cross_modal=%s  duration=%dms",
        str(answer_result.get("answer", ""))[:60],
        verification_result.get("verification_status", "unknown"),
        guardrail_payload.get("should_flag_for_review", False),
        self_consistency_signal.get("level", "unknown"),
        cross_modal_agreement.get("level", "unknown"),
        total_ms,
    )
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
        "routing_decision": {
            "difficulty": routing_result.get("difficulty", "unknown"),
            "needs_retrieval": routing_result.get("needs_retrieval", False),
            "retrieval_rationale": routing_result.get("retrieval_rationale", ""),
            "verification_policy": routing_result.get("verification_policy", "standard"),
            "fallback_policy": routing_result.get("fallback_policy", "none"),
            "routing_rationale": routing_result.get("routing_rationale", ""),
        },
        "agent_trace": agent_trace,
        "total_duration_ms": total_ms,
        "retry_attempts": retry_attempts,
        "retry_reason": retry_reason,
    }


def _should_retry_pipeline(payload: dict) -> bool:
    verification_status = payload.get("verification_status", "low_confidence")
    flag_review = payload.get("should_flag_for_review", False)
    failure_categories = payload.get("failure_categories", []) or []
    high_severity = any(category.get("severity") == "high" for category in failure_categories)
    should_retry = (
        verification_status in {"low_confidence", "unsupported"}
        or flag_review
        or high_severity
    )
    _log.debug(
        "Retry evaluation  verification=%s  flag_review=%s  high_severity=%s  -> retry=%s",
        verification_status, flag_review, high_severity, should_retry,
    )
    return should_retry


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
    _log.debug(
        "Payload score  verification_rank=%d  confidence_rank=%d  high_failures=%d  review=%d",
        verification_rank, confidence_rank, high_severity_count, review_flag,
    )
    return verification_rank, confidence_rank, -high_severity_count, -review_flag


def run_pipeline(image_bytes: bytes, question: str, mime_type: str = "image/png") -> dict:
    pipeline_start = time.time()
    _log.info(
        "=== Agentic pipeline START  question=%r  mime_type=%s  size=%d bytes ===",
        question[:100], mime_type, len(image_bytes),
    )

    def _run_once(retry_context: dict | None = None) -> dict:
        _log.info("Running agentic pipeline (is_retry=%s)", bool(retry_context))
        agent = _create_agent()
        try:
            response = agent.run(
                messages=[ChatMessage.from_user(question)],
                image_bytes=image_bytes,
                mime_type=mime_type,
                question=question,
                retry_context=retry_context or {},
            )
        except ImageBlockerAbort as e:
            _log.warning("Agentic pipeline halted by ImageBlockerAbort: %s", e)
            return {
                "image_interpreter": {"result": {"status": "error", "error": str(e)}},
                "abort": str(e)
            }
            
        return {
            "image_interpreter": {
                "result": response.get("image_interpretation") or {},
                "duration_ms": response.get("image_duration_ms") or 0,
            },
            "question_analyzer": {
                "result": response.get("question_analysis") or {},
                "duration_ms": response.get("question_duration_ms") or 0,
            },
            "difficulty_router": {
                "result": response.get("routing_decision") or {},
                "duration_ms": response.get("routing_duration_ms") or 0,
            },
            "knowledge_retriever": {
                "result": response.get("knowledge") or {},
                "duration_ms": response.get("knowledge_duration_ms") or 0,
            },
            "reasoning_agent": {
                "result": response.get("answer_data") or {},
                "duration_ms": response.get("answer_duration_ms") or 0,
            },
            "verification_agent": {
                "result": response.get("verification_result") or {},
                "duration_ms": response.get("verification_duration_ms") or 0,
            },
        }

    first_results = _run_once()
    
    if first_results.get("abort"):
        return {
            "answer": first_results["abort"],
            "verification_status": "rejected",
            "confidence": "high",
            "should_flag_for_review": False,
            "fallback_recommendation": "none",
            "duration_ms": int((time.time() - pipeline_start) * 1000),
            "reasoning_trace": [],
            "evidence_used": ["Image interpretation failed validation"],
            "limitations": "Invalid image provided."
        }

    first_payload = _assemble_pipeline_payload(first_results, question, pipeline_start)
    _log.info(
        "Pipeline attempt 1 result  answer=%r  verification=%s  confidence=%s  flag_review=%s",
        str(first_payload.get("answer", ""))[:60],
        first_payload.get("verification_status", "unknown"),
        first_payload.get("confidence", "unknown"),
        first_payload.get("should_flag_for_review", False),
    )

    retry_reason = ""
    retry_attempts = 0
    best_payload = first_payload

    if _should_retry_pipeline(first_payload):
        retry_reason = (
            f"verification_status={first_payload.get('verification_status', 'unknown')}; "
            f"fallback={first_payload.get('fallback_recommendation', 'none')}; "
            f"flag_for_review={first_payload.get('should_flag_for_review', False)}"
        )
        _log.info("Pipeline RETRY triggered  reason: %s", retry_reason)
        retry_attempts = 1
        retry_context = {
            "verification_status": first_payload.get("verification_status", "unknown"),
            "confidence": first_payload.get("confidence", "Low"),
            "fallback_recommendation": first_payload.get("fallback_recommendation", "none"),
            "failure_categories": first_payload.get("failure_categories", []),
        }
        second_results = _run_once(retry_context=retry_context)
        second_payload = _assemble_pipeline_payload(
            second_results, question, pipeline_start, retry_attempts=1, retry_reason=retry_reason
        )
        _log.info(
            "Pipeline attempt 2 result  answer=%r  verification=%s  confidence=%s",
            str(second_payload.get("answer", ""))[:60],
            second_payload.get("verification_status", "unknown"),
            second_payload.get("confidence", "unknown"),
        )
        if _payload_score(second_payload) > _payload_score(first_payload):
            best_payload = second_payload
            _log.info("Pipeline RETRY winner: attempt 2 (higher score)")
        else:
            best_payload = first_payload
            _log.info("Pipeline RETRY winner: attempt 1 (kept original)")
    else:
        _log.info("Pipeline retry NOT needed")

    best_payload["retry_attempts"] = retry_attempts
    best_payload["retry_reason"] = retry_reason
    total_ms = int((time.time() - pipeline_start) * 1000)
    _log.info(
        "=== Agentic pipeline DONE  answer=%r  retries=%d  total_duration=%dms ===",
        str(best_payload.get("answer", ""))[:60],
        retry_attempts,
        total_ms,
    )
    return best_payload
