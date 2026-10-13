"""
Shared answer-normalisation helpers used by the agents and the evaluator.
"""
import re

_LEADING_YES_NO = re.compile(r"^\W*(yes|no)(?![a-z0-9])", re.IGNORECASE)


def leading_yes_no(text: str) -> str | None:
    """Return 'yes'/'no' if the text starts with that whole word, else None.

    Matches "no", "No.", "yes, there is…" but not "normal", "nodule" or "none".
    """
    match = _LEADING_YES_NO.match(str(text or "").strip())
    return match.group(1).lower() if match else None
