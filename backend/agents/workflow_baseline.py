import re
import time
from agents.image_interpreter import interpret_image
from agents.question_analyzer import analyze_question
from logger import get_logger

_log = get_logger("agents.workflow_baseline")

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "were", "be", "there", "any", "this", "that",
    "image", "scan", "shown", "visible", "seen", "does", "do", "has", "have",
}


def _clean_value(value: str, fallback: str = "unknown") -> str:
    if not value:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", str(value or "").lower()).strip()


def _tokenize(value: str) -> set[str]:
    return {
        token for token in _normalize_text(value).split()
        if token and token not in STOPWORDS
    }


def _find_matching_items(query: str, items: list[str]) -> list[str]:
    query_normalized = _normalize_text(query)
    query_tokens = _tokenize(query)
    matches = []
    for item in items:
        item_normalized = _normalize_text(item)
        item_tokens = _tokenize(item)
        if query_normalized and query_normalized in item_normalized:
            matches.append(item)
            continue
        if query_tokens and item_tokens and (query_tokens <= item_tokens or item_tokens <= query_tokens):
            matches.append(item)
    return matches


def _has_target_match(query: str, evidence_text: str, evidence_items: list[str]) -> bool:
    query_normalized = _normalize_text(query)
    if not query_normalized or query_normalized == "unknown":
        return False
    if query_normalized in evidence_text:
        return True
    return len(_find_matching_items(query, evidence_items)) > 0


def _derive_presence_answer(question: str, target_entity: str, evidence_text: str,
                            evidence_items: list[str], notable_findings: list[str],
                            image_status: str) -> tuple[str, str]:
    query_text = target_entity if target_entity not in {"unknown", "abnormality"} else question
    generic_abnormality_query = any(
        term in _normalize_text(question)
        for term in ["abnormal", "finding", "patholog", "lesion", "fracture", "mass", "effusion"]
    )
    if _has_target_match(query_text, evidence_text, evidence_items):
        return "Yes", "high" if notable_findings else "medium"
    if generic_abnormality_query:
        return ("Yes", "medium") if notable_findings else ("No", "medium" if image_status == "success" else "low")
    return ("No", "low") if image_status == "success" else ("Unable to determine", "low")


def workflow_answer(image_bytes: bytes, question: str, mime_type: str = "image/png",
                    precomputed_image_result: dict | None = None) -> dict:
    """Run the hand-designed non-agentic workflow baseline.

    Args:
        image_bytes: Raw image bytes.
        question: The user question.
        mime_type: MIME type of the image.
        precomputed_image_result: Optional pre-computed image interpretation dict.
            Pass this to avoid a second LLM call when the caller already has one
            (e.g. from the agentic pipeline run).
    """
    started = time.perf_counter()
    try:
        image_result = precomputed_image_result if precomputed_image_result is not None else interpret_image(image_bytes, mime_type)
    except Exception as e:
        _log.error("Image interpretation failed in workflow: %s", e)
        return {
            "status": "error",
            "answer": "Workflow failed: image interpretation error.",
            "explanation": str(e),
            "confidence": "unverified",
            "workflow_trace": {},
        }
    try:
        question_result = analyze_question(question, image_result.get("visual_summary", ""))
    except Exception as e:
        _log.error("Question analysis failed in workflow: %s", e)
        return {
            "status": "error",
            "answer": "Workflow failed: question analysis error.",
            "explanation": str(e),
            "confidence": "unverified",
            "workflow_trace": {},
        }

    _log.info(
        "Workflow baseline  question=%r  modality=%s  body_region=%s  question_type=%s",
        question[:100],
        image_result.get("modality", "unknown"),
        image_result.get("body_region", "unknown"),
        question_result.get("question_type", "unknown"),
    )

    modality = _clean_value(image_result.get("modality"))
    body_region = _clean_value(image_result.get("body_region"))
    visible_structures = image_result.get("visible_structures", []) or []
    notable_findings = image_result.get("notable_findings", []) or []
    visual_summary = _clean_value(image_result.get("visual_summary"), "No visual summary available.")
    question_type = _clean_value(question_result.get("question_type"), "other")
    target_entity = _clean_value(question_result.get("target_entity"), "unknown")
    question_clarity = _clean_value(question_result.get("question_clarity"), "unclear")
    rephrased_question = _clean_value(question_result.get("rephrased_question"), question)
    image_status = image_result.get("status", "warning")
    evidence_items = notable_findings + visible_structures + [body_region, modality]
    evidence_text = " ".join(
        part for part in [
            _normalize_text(visual_summary),
            _normalize_text(" ".join(notable_findings)),
            _normalize_text(" ".join(visible_structures)),
            _normalize_text(body_region),
            _normalize_text(modality),
        ]
        if part
    )
    matched_findings = _find_matching_items(target_entity, notable_findings)
    matched_structures = _find_matching_items(target_entity, visible_structures)
    _log.debug("Matched findings: %s  Matched structures: %s", matched_findings, matched_structures)

    answer = "Unable to determine from the current workflow."
    confidence = "low"

    if question_type == "modality_identification":
        _log.debug("Branch taken: modality_identification")
        answer = modality
        confidence = "high" if modality != "unknown" else "low"
    elif question_type == "anatomy_identification":
        _log.debug("Branch taken: anatomy_identification")
        answer = body_region
        confidence = "high" if body_region != "unknown" else "low"
    elif question_type in {"yes_no", "presence_absence"}:
        _log.debug("Branch taken: yes_no/presence_absence")
        answer, confidence = _derive_presence_answer(
            question=rephrased_question,
            target_entity=target_entity,
            evidence_text=evidence_text,
            evidence_items=evidence_items,
            notable_findings=notable_findings,
            image_status=image_status,
        )
    elif question_type == "location":
        _log.debug("Branch taken: location")
        if matched_findings:
            answer = body_region if body_region != "unknown" else matched_findings[0]
            confidence = "medium"
        elif notable_findings:
            answer = body_region if body_region != "unknown" else notable_findings[0]
            confidence = "medium"
        else:
            answer = body_region
            confidence = "low" if body_region == "unknown" else "medium"
    elif question_type == "counting":
        _log.debug("Branch taken: counting")
        if matched_findings:
            answer = str(len(matched_findings))
            confidence = "high" if len(matched_findings) == 1 else "medium"
        elif notable_findings:
            answer = str(len(notable_findings))
            confidence = "medium"
        elif visible_structures:
            answer = str(len(visible_structures))
            confidence = "low"
    elif question_type == "abnormality_description":
        _log.debug("Branch taken: abnormality_description")
        if matched_findings:
            answer = "; ".join(matched_findings[:2])
            confidence = "high"
        elif notable_findings:
            answer = "; ".join(notable_findings[:2])
            confidence = "medium"
        else:
            answer = visual_summary
            confidence = "low"
    elif question_type == "explanation_request":
        _log.debug("Branch taken: explanation_request")
        parts = []
        if modality != "unknown":
            parts.append(f"This is a {modality} image")
        if body_region != "unknown":
            parts.append(f"of the {body_region}")
        if notable_findings:
            parts.append(f"showing {'; '.join(notable_findings[:3])}")
        elif visible_structures:
            parts.append(f"with visible structures: {', '.join(visible_structures[:3])}")
        if parts:
            answer = ". ".join(parts) + "."
            confidence = "medium"
        else:
            answer = visual_summary
            confidence = "low"
    else:
        _log.debug("Branch taken: fallback/other")
        if target_entity in {"modality", "imaging modality"}:
            answer = modality
            confidence = "medium" if modality != "unknown" else "low"
        elif target_entity in {"body region", "organ", "anatomy"}:
            answer = body_region
            confidence = "medium" if body_region != "unknown" else "low"
        elif matched_findings:
            answer = "; ".join(matched_findings[:2])
            confidence = "medium"
        elif matched_structures:
            answer = "; ".join(matched_structures[:2])
            confidence = "medium"
        else:
            answer = visual_summary
            confidence = "low"

    explanation_parts = [
        f"Workflow used image interpretation ({modality}, {body_region})",
        f"question type: {question_type}",
    ]
    if question_clarity != "clear":
        explanation_parts.append(f"question clarity: {question_clarity}")
    if matched_findings:
        explanation_parts.append(f"matched findings: {'; '.join(matched_findings[:2])}")
    elif notable_findings:
        explanation_parts.append(f"findings: {'; '.join(notable_findings[:2])}")
    elif matched_structures:
        explanation_parts.append(f"matched structures: {', '.join(matched_structures[:4])}")
    elif visible_structures:
        explanation_parts.append(f"visible structures: {', '.join(visible_structures[:4])}")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    _log.info(
        "Workflow baseline done  answer=%r  confidence=%s  question_type=%s  duration=%dms",
        answer[:80],
        confidence,
        question_type,
        elapsed_ms,
    )
    return {
        "status": "success" if image_status == "success" else "warning",
        "answer": answer,
        "explanation": ". ".join(explanation_parts) + ".",
        "confidence": confidence,
        "duration_ms": elapsed_ms,
        "workflow_trace": {
            "image_result": image_result,
            "question_result": question_result,
        },
    }
