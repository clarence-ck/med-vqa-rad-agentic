"""
Agentic loop: a Haystack Agent controller that decides what to do next.

Tools (bound to one per-request LoopContext):
  analyze_question()        classify the question (type, expected format)
  look(focus="")            question-aware look at the image; with `focus`, a zoomed re-examination of that region
  answer(guidance="")       answer from the image + all observations so far
  check()                   blind reading of the image (once), then verify the latest answer; recommends the next action
  finish(abstain, reason)   end the episode

Unlike the fixed pipeline (orchestrator.run_pipeline), the tool sequence depends on what
`check` finds: accept -> finish; relook -> look(focus) -> answer -> check; reanswer ->
answer(guidance) -> check. Budgets cap re-looks and answers.
"""
import base64
import contextvars
import time
from concurrent.futures import ThreadPoolExecutor

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.utils import Secret

from agents.look_agent import LookAgent, NonRadiologyImageError
from agents.orchestrator import (
    ImageBlockerAbort,
    _build_cross_modal_agreement_signal,
    _build_guardrail_payload,
    _categorize_failure_modes,
    _normalize_final_answer,
    run_pipeline,
)
from agents.question_analyzer import QuestionAnalyzer
from agents.reasoning_agent import ReasoningAgent
from agents.verification_agent import VerificationAgent, blind_read
from config import AGENT_MODE, MODELS, OPENAI_API_KEY, SELF_CONSISTENCY_SAMPLES, generation_kwargs
from logger import get_logger
from utils import scoring
from utils.usage import record_usage

_log = get_logger("agents.agentic_loop")

MAX_LOOKS = 3        # 1 initial + 2 focused re-looks
MAX_ANSWERS = 3
MAX_AGENT_STEPS = 16
ABSTAIN_ANSWER = "Unable to determine from this image."
IMAGE_ANALYSIS_FAILED = "The image could not be analysed. Please try again or upload a different image."

_VERIFICATION_RANK = {"verified": 4, "partially_verified": 3, "low_confidence": 2, "unsupported": 1}
_CONFIDENCE_RANK = {"High": 3, "Medium": 2, "Low": 1}


def controller_path(actions: list[str], abstained: bool = False) -> str:
    """Summarise what the controller did: direct, relook, reanswer or abstained.

    `abstained` must be the episode outcome: a refused finish(abstain) call (e.g. the
    latest answer was still unchecked) appears in `actions` but is not an abstention.
    """
    if abstained:
        return "abstained"
    if any(action.startswith("look(") for action in actions):
        return "relook"
    if any(action.startswith("answer(guided") for action in actions):
        return "reanswer"
    return "direct"


BLIND_DISAGREEMENT = "an independent blind reading of the image disagrees with this answer"


def apply_blind_agreement(check: dict, answer: str, blind: dict, question_analysis: dict) -> dict:
    """Combine the answer-aware check with the blind reading (in place).

    The check only stays "verified" if an independent reading of the image, made without seeing the answer,
    agrees with it. On disagreement the status drops to at most partially_verified and an "accept" becomes a
    focused re-look. The blind answer itself is stored but never put in the text the controller sees.
    """
    if not blind or blind.get("status") != "success":
        check["blind_agreement"] = None
        return check
    qa = question_analysis or {}
    close_ended = qa.get("expected_answer_format") == "yes_no" or qa.get("question_type") in {"yes_no", "presence_absence"}
    agree = scoring.answers_agree(str(blind.get("answer", "")), str(answer or ""), close_ended, qa.get("question_type"))
    check["blind_agreement"] = agree
    if agree:
        return check
    if check.get("verification_status") == "verified":
        check["verification_status"] = "partially_verified"
    if check.get("confidence_label") == "High":
        check["confidence_label"] = "Medium"
    check["issues_found"] = list(check.get("issues_found") or []) + [BLIND_DISAGREEMENT]
    if check.get("recommended_action", "accept") == "accept":
        check["recommended_action"] = "relook"
        check["relook_focus"] = check.get("relook_focus") or qa.get("target_entity") or "the finding the question asks about"
    return check


def consistency_level(agreeing: int, total: int) -> str:
    """high = every resample agrees with the final answer; medium = at least half; low = fewer."""
    if total <= 0:
        return "not_run"
    if agreeing == total:
        return "high"
    if agreeing * 2 >= total:
        return "medium"
    return "low"


def select_best_attempt(checks: list[dict]) -> int:
    """Index of the answer to keep: best verification status, then confidence.

    Ties keep the earliest answer, so a later re-look or re-answer only replaces an
    answer when its check is strictly better (re-looks can also over-interpret).
    """
    def score(index: int) -> tuple:
        check = checks[index] or {}
        return (
            _VERIFICATION_RANK.get(check.get("verification_status"), 0),
            _CONFIDENCE_RANK.get(str(check.get("confidence_label")), 0),
            -index,
        )
    return max(range(len(checks)), key=score)

_SYSTEM_PROMPT = """You are the controller of a medical visual question answering agent.
Goal: give a correct, image-grounded answer to the user's question about a radiology image, verified against the image.

Tools:
- analyze_question: classify the question and its expected answer format.
- look(focus): examine the image with the question in mind. Leave focus empty for the first look; on later looks,
  name the region or structure to re-examine (e.g. "left costophrenic angle"): that region is located, cropped and
  enlarged, so the re-look sees it in more detail.
- answer(guidance): produce an answer from the image and all observations so far. Use guidance to say what the new
  answer must fix.
- check: verify the latest answer against the image. An independent reader first answers from the image alone,
  without seeing yours; if it disagrees, the answer is not verified. Returns a verdict and a recommended next action.
- finish(abstain, reason): end the episode. The best-verified checked answer is kept, or abstain.

How to work:
1. Call analyze_question and look first.
2. Call answer, then check.
3. Act on the check result: if it accepts, finish. If it recommends relook, call look with that focus (or a better
   one), then answer and check again. If it recommends reanswer, call answer with guidance, then check again.
4. At most 2 focused looks and 3 answers. Always check the latest answer before finishing. Abstain only if the image
   genuinely cannot answer the question; otherwise finish with the best answer."""


class LoopContext:
    """Per-request working memory shared by the tools."""

    def __init__(self, image_bytes: bytes, question: str, mime_type: str):
        self.image_bytes = image_bytes
        self.image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        self.mime_type = mime_type
        self.question = question
        self.question_analysis: dict = {}
        self.looks: list[dict] = []
        self.answers: list[dict] = []
        self.checks: list[dict] = []          # checks[i] verifies answers[i]
        self.trace: list[dict] = []
        self.actions: list[str] = []
        self.abstained = False
        self.blind: dict | None = None        # blind reading, made once per episode

    # -- helpers -------------------------------------------------------------
    def _record(self, agent: str, status: str, summary: str, output: dict, started: float) -> None:
        self.trace.append({
            "agent": agent,
            "agent_id": len(self.trace) + 1,
            "status": status,
            "summary": summary,
            "output": output,
            "duration_ms": int((time.time() - started) * 1000),
        })

    def observations(self) -> dict:
        """Merge all looks into the image_interpretation shape the reasoner/verifier expect."""
        if not self.looks:
            return {}
        first = self.looks[0]
        findings, structures = [], []
        for look in self.looks:
            prefix = f"[focus: {look['focus']}] " if look.get("focus") else ""
            findings += [prefix + str(x) for x in (look.get("question_relevant_observations") or [])]
            if look.get("focus_observation"):
                findings.append(prefix + str(look["focus_observation"]))
            findings += [str(x) for x in (look.get("notable_findings") or [])]
            structures += [str(x) for x in (look.get("visible_structures") or []) if str(x) not in structures]
        return {
            "status": "success",
            "modality": first.get("modality", "unknown"),
            "body_region": first.get("body_region", "unknown"),
            "visible_structures": structures,
            "notable_findings": list(dict.fromkeys(findings)),
            "visual_summary": " ".join(str(look.get("visual_summary", "")) for look in self.looks).strip(),
            "image_quality": first.get("image_quality", "unknown"),
            "quality_warning": bool(first.get("quality_warning", False)),
        }

    # -- tools -----------------------------------------------------------------
    def analyze_question(self) -> str:
        self.actions.append("analyze_question")
        started = time.time()
        if not self.question_analysis:
            self.question_analysis = QuestionAnalyzer().run(question=self.question,
                                                            image_interpretation=self.observations())["result"]
            qa = self.question_analysis
            self._record("Question Understanding", "pass" if qa.get("status") == "success" else "warning",
                         f"Type: {qa.get('question_type')} | Format: {qa.get('expected_answer_format')}",
                         {k: qa.get(k) for k in ("question_type", "question_subtype", "target_entity", "expected_answer_format")},
                         started)
        qa = self.question_analysis
        return f"question_type={qa.get('question_type')}; expected_answer_format={qa.get('expected_answer_format')}; target={qa.get('target_entity')}"

    def look(self, focus: str = "") -> str:
        self.actions.append(f"look({focus})" if focus else "look")
        if len(self.looks) >= MAX_LOOKS:
            return "Look budget used up. Answer with the observations you have, check, then finish."
        started = time.time()
        previous = "; ".join(self.observations().get("notable_findings", [])[:12])
        try:
            result = LookAgent().run(image_bytes=self.image_bytes, question=self.question, mime_type=self.mime_type,
                                     focus=focus, previous_observations=previous)["result"]
        except NonRadiologyImageError as e:
            raise ImageBlockerAbort(str(e))
        if result.get("status") != "success" and not self.looks:
            # Without a successful first look the radiology check never ran: do not answer blind.
            raise ImageBlockerAbort(IMAGE_ANALYSIS_FAILED)
        self.looks.append(result)
        label = f"Focused Look: {focus}" if focus else "Image Interpretation"
        observed = result.get("focus_observation") or "; ".join(result.get("question_relevant_observations", []) or [])
        self._record(label, "pass" if result.get("status") == "success" else "warning",
                     observed or result.get("visual_summary", "No observation produced."),
                     {k: result.get(k) for k in ("modality", "body_region", "view_or_sequence", "focus_observation",
                                                 "question_relevant_observations", "uncertainties", "image_quality",
                                                 "zoom_box")},
                     started)
        if result.get("status") != "success":
            return f"Look failed: {result.get('error', 'unknown error')}"
        return f"Observed: {observed or result.get('visual_summary', '')}"[:700]

    def answer(self, guidance: str = "") -> str:
        self.actions.append("answer(guided)" if guidance else "answer")
        if len(self.answers) >= MAX_ANSWERS:
            return "Answer budget used up. Check the latest answer if unchecked, then finish."
        if not self.question_analysis:
            self.analyze_question()
        if not self.looks:
            self.look()
        started = time.time()
        result = ReasoningAgent().run(
            image_interpretation=self.observations(),
            question_analysis=self.question_analysis,
            original_question=self.question,
            image_bytes_b64=self.image_b64,
            mime_type=self.mime_type,
            guidance=guidance,
        )["result"]
        self.answers.append(result)
        self._record("Reasoning & Answer", "pass" if result.get("status") == "success" else "warning",
                     f"Answer: {result.get('answer', 'N/A')}",
                     {"answer": result.get("answer", ""), "confidence": result.get("answer_confidence", "low"),
                      "guidance": guidance}, started)
        return f"answer={result.get('answer', '')!r}; confidence={result.get('answer_confidence', 'unknown')}"

    def check(self) -> str:
        self.actions.append("check")
        if not self.answers:
            return "Nothing to check yet: call answer first."
        if len(self.checks) == len(self.answers):
            return "The latest answer is already checked. Act on that result or finish."
        started = time.time()
        result = VerificationAgent().run(
            answer_data=self.answers[-1],
            image_interpretation=self.observations(),
            question_analysis=self.question_analysis,
            original_question=self.question,
            image_bytes_b64=self.image_b64,
            mime_type=self.mime_type,
        )["result"]
        apply_blind_agreement(result, self.answers[-1].get("answer", ""), self.blind_reading(), self.question_analysis)
        self.checks.append(result)
        status = result.get("verification_status", "unknown")
        action = result.get("recommended_action", "accept" if status == "verified" else "reanswer")
        self._record("Verification",
                     "pass" if status == "verified" else "warning" if status == "partially_verified" else "fail",
                     f"Status: {status} | Next: {action}",
                     {"verification_status": status, "confidence_label": result.get("confidence_label"),
                      "issues_found": result.get("issues_found", []), "recommended_action": action,
                      "relook_focus": result.get("relook_focus", ""),
                      "reanswer_guidance": result.get("reanswer_guidance", "")}, started)
        issues = "; ".join(str(i) for i in (result.get("issues_found") or [])[:3])
        return (f"verification_status={status}; recommended_action={action}; "
                f"relook_focus={result.get('relook_focus', '')!r}; reanswer_guidance={result.get('reanswer_guidance', '')!r}; "
                f"issues={issues!r}")[:900]

    def blind_reading(self) -> dict:
        """The verifier model's own answer from the image and question alone, read once per episode.

        Kept out of everything the controller and the answerer see, so the loop has to resolve a
        disagreement by looking again with its own model, not by copying the verifier's answer.
        """
        if self.blind is None:
            if not self.question_analysis:
                self.analyze_question()
            started = time.time()
            self.blind = blind_read(self.image_b64, self.mime_type, self.question, self.question_analysis)
            ok = self.blind.get("status") == "success"
            self._record("Blind Reading", "pass" if ok else "warning",
                         f"Independent answer from the image alone: {self.blind.get('answer')!r}" if ok
                         else "Blind reading failed; the check falls back to the answer-aware review only",
                         {"answer": self.blind.get("answer", ""), "explanation": self.blind.get("explanation", "")},
                         started)
        return self.blind

    def self_consistency(self, final_answer: str, samples: int) -> dict:
        """Re-ask the reasoner `samples` times independently and measure agreement with the final answer."""
        if samples <= 0 or self.abstained or not final_answer:
            return {"level": "not_run", "method": "resampling", "agreeing_samples": 0, "total_samples": 0, "samples": []}
        started = time.time()
        observations = self.observations()

        def one_sample(_):
            return ReasoningAgent().run(
                image_interpretation=observations,
                question_analysis=self.question_analysis,
                original_question=self.question,
                image_bytes_b64=self.image_b64,
                mime_type=self.mime_type,
            )["result"].get("answer", "")

        # copy_context() so usage tracking (a context variable) follows the worker threads
        with ThreadPoolExecutor(max_workers=samples) as pool:
            futures = [pool.submit(contextvars.copy_context().run, one_sample, i) for i in range(samples)]
            answers = [f.result() for f in futures]

        qa = self.question_analysis
        close_ended = qa.get("expected_answer_format") == "yes_no" or qa.get("question_type") in {"yes_no", "presence_absence"}
        agreeing = sum(
            1 for answer in answers
            if scoring.answers_agree(str(answer), str(final_answer), close_ended, qa.get("question_type"))
        )
        level = consistency_level(agreeing, len(answers))
        self._record("Self-Consistency", "pass" if level == "high" else "warning" if level == "medium" else "fail",
                     f"{agreeing}/{len(answers)} independent re-answers agree with the final answer",
                     {"level": level, "final_answer": final_answer, "samples": answers}, started)
        return {"level": level, "method": "resampling", "agreeing_samples": agreeing,
                "total_samples": len(answers), "samples": answers}

    def finish(self, abstain: bool = False, reason: str = "") -> str:
        self.actions.append("finish(abstain)" if abstain else "finish")
        if self.answers and len(self.checks) < len(self.answers) and not abstain:
            return "The latest answer has not been checked. Call check before finishing."
        self.abstained = bool(abstain)
        self._record("Finish", "pass", f"{'Abstained' if abstain else 'Accepted latest answer'}"
                     f"{': ' + reason if reason else ''}", {"abstain": bool(abstain), "reason": reason}, time.time())
        return "Finished."


def _tools(ctx: LoopContext) -> list[Tool]:
    no_params = {"type": "object", "properties": {}, "required": []}
    return [
        Tool(name="analyze_question", description="Classify the question type and expected answer format.",
             parameters=no_params, function=ctx.analyze_question),
        Tool(name="look", description="Examine the image with the question in mind. Pass focus to zoom into and re-examine a specific region or structure.",
             parameters={"type": "object", "properties": {"focus": {"type": "string", "description": "Region or structure to re-examine; empty for the first look."}}, "required": []},
             function=ctx.look),
        Tool(name="answer", description="Answer the question from the image and all observations so far.",
             parameters={"type": "object", "properties": {"guidance": {"type": "string", "description": "What the new answer must fix (from the check result); empty for the first answer."}}, "required": []},
             function=ctx.answer),
        Tool(name="check", description="Verify the latest answer against the image (blind reading first, then an answer-aware review); returns a verdict and a recommended next action.",
             parameters=no_params, function=ctx.check),
        Tool(name="finish", description="End the episode with the latest checked answer, or abstain.",
             parameters={"type": "object", "properties": {"abstain": {"type": "boolean"}, "reason": {"type": "string"}}, "required": []},
             function=ctx.finish),
    ]


def _image_analysis(ctx: LoopContext) -> dict:
    first = ctx.looks[0] if ctx.looks else {}
    obs = ctx.observations()
    return {
        "modality": first.get("modality", "unknown"),
        "body_region": first.get("body_region", "unknown"),
        "visible_structures": obs.get("visible_structures", []),
        "visual_summary": first.get("visual_summary", ""),
        "image_quality": first.get("image_quality", "unknown"),
        "quality_warning": bool(first.get("quality_warning", False)),
        "notable_findings": obs.get("notable_findings", []),
    }


def _rejected_payload(message: str, started: float) -> dict:
    # A failed analysis (API/image error) is retryable; a non-radiology rejection is a real answer.
    failed = message == IMAGE_ANALYSIS_FAILED
    return {
        "api_errors": 1 if failed else 0,
        "answer": message, "guarded_answer": message, "explanation": "", "reasoning_trace": "",
        "evidence_used": [], "confidence": "unknown", "verification_status": "error" if failed else "rejected", "checks": {},
        "verification_notes": "", "issues_found": [], "failure_categories": [],
        "self_consistency_signal": {"level": "low", "passed_checks": 0, "total_checks": 0},
        "cross_modal_agreement": {"level": "low", "grounded_in_evidence": None, "no_contradictions": None},
        "fallback_recommendation": "none", "suggested_disclaimer": "", "should_flag_for_review": False,
        "image_analysis": {}, "question_analysis": {}, "agent_trace": [], "agent_mode": "loop",
        "controller_actions": [], "total_duration_ms": int((time.time() - started) * 1000),
        "retry_attempts": 0, "retry_reason": "",
    }


def run_agentic_loop(image_bytes: bytes, question: str, mime_type: str = "image/png") -> dict:
    started = time.time()
    _log.info("=== Agentic loop START  question=%r ===", question[:100])
    ctx = LoopContext(image_bytes, question, mime_type)
    agent = Agent(
        chat_generator=OpenAIChatGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model=MODELS["controller"],
                                           generation_kwargs=generation_kwargs("controller")),
        system_prompt=_SYSTEM_PROMPT,
        tools=_tools(ctx),
        exit_conditions=["finish"],
        max_agent_steps=MAX_AGENT_STEPS,
    )
    try:
        response = agent.run(messages=[ChatMessage.from_user(question)])
        for message in response.get("messages", []) or []:
            if message.role.value == "assistant" and message.meta.get("usage"):
                record_usage("controller", MODELS["controller"], message.meta["usage"])
    except ImageBlockerAbort as e:
        _log.warning("Agentic loop halted: %s", e)
        return _rejected_payload(str(e), started)

    # Safety net: the controller must leave a checked answer behind.
    if not ctx.answers:
        _log.warning("Controller finished without an answer; answering now")
        ctx.answer()
    if len(ctx.checks) < len(ctx.answers):
        _log.warning("Latest answer was not checked; checking now")
        ctx.check()

    best = select_best_attempt(ctx.checks)
    if best != len(ctx.answers) - 1:
        _log.info("Keeping answer %d of %d (better verified than the latest)", best + 1, len(ctx.answers))
    answer_result = dict(ctx.answers[best])
    verification_result = ctx.checks[best]
    if ctx.abstained:
        # Never normalise an abstention: the yes/no normaliser would turn "unable to determine" into "no".
        answer_text = ABSTAIN_ANSWER
    else:
        answer_text = _normalize_final_answer(answer_result.get("answer", ""), ctx.question_analysis, _image_analysis(ctx))
    answer_result["answer"] = answer_text

    self_consistency = ctx.self_consistency(answer_text, SELF_CONSISTENCY_SAMPLES)

    failure_categories = _categorize_failure_modes(answer_result=answer_result, verification_result=verification_result)
    if self_consistency["level"] == "low":
        failure_categories.append({
            "code": "self_inconsistent",
            "label": "Self-inconsistent",
            "description": "Independent re-answers mostly disagree with the final answer.",
            "severity": "medium",
        })
    guardrail = _build_guardrail_payload(answer_result=answer_result, verification_result=verification_result,
                                         failure_categories=failure_categories)
    if self_consistency["level"] == "low":
        guardrail["should_flag_for_review"] = True
        guardrail["suggested_disclaimer"] = guardrail["suggested_disclaimer"] or (
            "Repeated answers to this question were inconsistent. Please review with a qualified medical professional."
        )
    relooks = sum(1 for look in ctx.looks if look.get("focus"))
    reanswers = max(0, len(ctx.answers) - 1)
    qa = ctx.question_analysis
    payload = {
        "answer": answer_text,
        "guarded_answer": ABSTAIN_ANSWER if ctx.abstained else _normalize_final_answer(guardrail["guarded_answer"], qa, _image_analysis(ctx)),
        "explanation": answer_result.get("explanation", ""),
        "reasoning_trace": answer_result.get("reasoning_trace", ""),
        "evidence_used": answer_result.get("evidence_used", []),
        "confidence": verification_result.get("confidence_label", "Low"),
        "verification_status": verification_result.get("verification_status", "low_confidence"),
        "checks": verification_result.get("checks", {}),
        "verification_notes": verification_result.get("verification_notes", ""),
        "issues_found": verification_result.get("issues_found", []),
        "failure_categories": failure_categories,
        "self_consistency_signal": self_consistency,
        "cross_modal_agreement": _build_cross_modal_agreement_signal(verification_result),
        "fallback_recommendation": guardrail["fallback_recommendation"],
        "suggested_disclaimer": guardrail["suggested_disclaimer"],
        "should_flag_for_review": guardrail["should_flag_for_review"],
        "image_analysis": _image_analysis(ctx),
        "question_analysis": {
            "question_type": qa.get("question_type", "unknown"),
            "question_subtype": qa.get("question_subtype", "unknown"),
            "target_entity": qa.get("target_entity", "unknown"),
            "expected_answer_format": qa.get("expected_answer_format", "short_phrase"),
            "question_clarity": qa.get("question_clarity", "unclear"),
            "reasoning_type": qa.get("reasoning_type", "visual_recognition"),
            "rephrased_question": qa.get("rephrased_question", question),
        },
        "agent_trace": ctx.trace,
        "agent_mode": "loop",
        "controller_actions": ctx.actions,
        "blind_reading": {"answer": (ctx.blind or {}).get("answer", ""),
                          "agrees_with_final": verification_result.get("blind_agreement")},
        "zoomed_looks": sum(1 for look in ctx.looks if look.get("zoom_box")),
        "abstained": ctx.abstained,
        "controller_path": controller_path(ctx.actions, ctx.abstained),
        "api_errors": sum(
            1 for result in [ctx.question_analysis, *ctx.looks, *ctx.answers, *ctx.checks]
            if isinstance(result, dict) and result.get("status") == "error"
        ),
        "selected_attempt": best + 1,
        "attempts": len(ctx.answers),
        "total_duration_ms": int((time.time() - started) * 1000),
        "retry_attempts": relooks + reanswers,
        "retry_reason": "; ".join(
            f"{c.get('recommended_action')}" + (f"({c.get('relook_focus')})" if c.get("relook_focus") else "")
            for c in ctx.checks[:-1]
        ),
    }
    _log.info("=== Agentic loop DONE  answer=%r  verification=%s  actions=%s  duration=%dms ===",
              answer_text[:60], payload["verification_status"], ctx.actions, payload["total_duration_ms"])
    return payload


def run_agentic(image_bytes: bytes, question: str, mime_type: str = "image/png", mode: str | None = None) -> dict:
    """Entry point for the agentic system: the controller loop (default) or the fixed pipeline (ablation)."""
    selected = (mode or AGENT_MODE).strip().lower()
    if selected == "pipeline":
        payload = run_pipeline(image_bytes, question, mime_type)
        payload.setdefault("agent_mode", "pipeline")
        return payload
    return run_agentic_loop(image_bytes, question, mime_type)
