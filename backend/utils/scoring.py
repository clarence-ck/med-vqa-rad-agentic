"""
Answer-matching rules for the evaluation scorer.

Kept free of model loading so the rules can be unit-tested; the caller supplies a
``similarity`` callable (prediction, ground_truth) -> cosine similarity.

Why not plain embedding similarity: S-PubMedBert-MS-MARCO scores "left lung" vs
"right lung" at 0.95 and "brain" vs "liver" at 0.91, so a single 0.85 threshold
accepts many wrong short answers. Short answers are therefore matched on content
words (with aliases and plural folding), laterality conflicts are always wrong,
and embeddings are only a strict fallback for synonyms or longer answers.
"""
import re
from typing import Callable

from utils.answer_utils import leading_yes_no

SHORT_ANSWER_MAX_TOKENS = 3
SHORT_ANSWER_SEMANTIC_THRESHOLD = 0.95
LONG_ANSWER_SEMANTIC_THRESHOLD = 0.90
LONG_ANSWER_MIN_TOKEN_RECALL = 0.6

BINARY_OPPOSITES = {
    "normal": "abnormal", "abnormal": "normal",
    "present": "absent", "absent": "present",
    "positive": "negative", "negative": "positive",
    "benign": "malignant", "malignant": "benign",
    "acute": "chronic", "chronic": "acute",
    "diffuse": "focal", "focal": "diffuse",
}

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "is", "are", "was", "were", "this",
    "that", "these", "those", "image", "there", "and", "with", "its", "it", "be", "by",
}

# Applied after lower-casing and replacing '-' and '/' with spaces.
_ALIASES = [
    (r"\bcomputed tomography\b|\bcat scan\b", "ct"),
    (r"\bmagnetic resonance(?: imaging)?\b|\bmr\b", "mri"),
    (r"\bx ray\b|\bxray\b|\bradiograph(?:y|s)?\b|\bplain film\b", "xray"),
    (r"\bultrasonograph(?:y)?\b|\bsonograph(?:y|ic)?\b", "ultrasound"),
]

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    """Lower-case, alias-normalised, plural-folded content words."""
    lowered = str(text or "").lower().replace("-", " ").replace("/", " ")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    for pattern, replacement in _ALIASES:
        lowered = re.sub(pattern, replacement, lowered)
    return [_singular(token) for token in lowered.split() if token not in _STOPWORDS]


def _laterality(tokens: set[str]) -> set[str]:
    sides = tokens & {"left", "right"}
    if tokens & {"bilateral", "both"}:
        sides = {"left", "right"}
    return sides


def extract_number(text: str) -> int | None:
    digit_match = re.search(r"\d+", str(text or ""))
    if digit_match:
        return int(digit_match.group())
    for word in re.findall(r"[a-z]+", str(text or "").lower()):
        if word in _NUMBER_WORDS:
            return _NUMBER_WORDS[word]
    return None


def is_close_ended(ground_truth: str) -> bool:
    """Yes/no or a single binary medical assessment (normal/abnormal, …)."""
    text = " ".join(str(ground_truth or "").lower().split()).strip(" .,;:!?")
    return text in {"yes", "no"} or text in BINARY_OPPOSITES


def match_close_ended(prediction: str, ground_truth: str) -> bool:
    gt = " ".join(str(ground_truth or "").lower().split()).strip(" .,;:!?")
    if gt in {"yes", "no"}:
        return leading_yes_no(prediction) == gt
    pred_tokens = set(content_tokens(prediction))
    opposite = BINARY_OPPOSITES.get(gt)
    return gt in pred_tokens and (opposite is None or opposite not in pred_tokens)


def match_open_ended(prediction: str, ground_truth: str,
                     similarity: Callable[[str, str], float] | None = None) -> bool:
    if not prediction or not ground_truth:
        return prediction == ground_truth

    gt_tokens = content_tokens(ground_truth)
    pred_tokens = content_tokens(prediction)
    gt_set, pred_set = set(gt_tokens), set(pred_tokens)
    if not gt_set:
        return prediction.strip().lower() == ground_truth.strip().lower()

    gt_sides, pred_sides = _laterality(gt_set), _laterality(pred_set)
    if gt_sides and pred_sides and gt_sides != pred_sides:
        return False

    # "No ventricular abnormality seen" denies the finding; it cannot match a positive answer.
    if leading_yes_no(prediction) == "no" and leading_yes_no(ground_truth) != "no":
        return False

    if gt_set <= pred_set:
        return True

    if len(gt_tokens) <= SHORT_ANSWER_MAX_TOKENS:
        return similarity is not None and similarity(prediction, ground_truth) >= SHORT_ANSWER_SEMANTIC_THRESHOLD

    recall = len(gt_set & pred_set) / len(gt_set)
    if recall >= LONG_ANSWER_MIN_TOKEN_RECALL:
        return True
    return similarity is not None and similarity(prediction, ground_truth) >= LONG_ANSWER_SEMANTIC_THRESHOLD


def answers_match(prediction: str, ground_truth: str, close_ended: bool, question_type: str | None = None,
                  similarity: Callable[[str, str], float] | None = None) -> bool:
    if close_ended:
        return match_close_ended(prediction, ground_truth)
    if question_type == "counting":
        pred_num, gt_num = extract_number(prediction), extract_number(ground_truth)
        if pred_num is not None and gt_num is not None:
            return pred_num == gt_num
    return match_open_ended(prediction, ground_truth, similarity)


def answers_agree(first: str, second: str, close_ended: bool, question_type: str | None = None,
                  similarity: Callable[[str, str], float] | None = None) -> bool:
    """Symmetric agreement between two system answers (used for disagreement analysis)."""
    if close_ended:
        first_yes_no, second_yes_no = leading_yes_no(first), leading_yes_no(second)
        if first_yes_no or second_yes_no:
            return first_yes_no == second_yes_no
    return (answers_match(first, second, False, question_type, similarity)
            or answers_match(second, first, False, question_type, similarity))
