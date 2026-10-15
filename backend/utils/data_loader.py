import json
import os
import random
from pathlib import Path
from config import DATA_DIR


# Excluded sample IDs (problematic cases)
# 2070: Problematic question and answer pair (ground truth issue)
EXCLUDED_SAMPLE_IDS = {2070}

# Seed for the evaluation sample drawn from the evaluation split (used only when there is no eval_set.json).
SAMPLE_SEED = int(os.getenv("EVAL_SAMPLE_SEED", "42"))

# The evaluation questions, as an explicit list (written by scripts/make_eval_set.py).
EVAL_SET_PATH = Path(__file__).resolve().parents[2] / "experiments" / "eval_set.json"


def eval_set_ids() -> list[int] | None:
    try:
        return [int(i) for i in json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))["ids"]]
    except (OSError, ValueError, KeyError):
        return None


def eval_split() -> str:
    """The dataset split the evaluation questions come from ("all" for the full dataset; test by default)."""
    try:
        return str(json.loads(EVAL_SET_PATH.read_text(encoding="utf-8")).get("split") or "test").lower()
    except (OSError, ValueError):
        return "test"


QUESTION_TYPE_ORDER = [
    "anatomy_identification",
    "modality_identification",
    "yes_no",
    "counting",
    "location",
    "abnormality_description",
    "other",
]


def load_dataset():
    """Load VQA-RAD dataset from the data directory."""
    json_path = os.path.join(DATA_DIR, "vqa_rad.json")
    if not os.path.exists(json_path):
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def infer_split(image_name: str) -> str:
    name = (image_name or "").lower()
    if name.startswith("test_"):
        return "test"
    if name.startswith("train_"):
        return "train"
    return "unknown"


def _resolve_split(item: dict, image_name: str) -> str:
    return (item.get("split") or infer_split(image_name) or "unknown").lower()


def get_sample_cases(limit: int = 1000) -> list:
    """Return the evaluation questions (experiments/eval_set.json) for the Case Explorer and the evaluation scripts."""
    data = load_dataset()
    if not data:
        return _get_fallback_samples()

    split = eval_split()
    buckets = {}
    for i, item in enumerate(data):
        if i in EXCLUDED_SAMPLE_IDS:
            continue
        q_type = _classify_question(item.get("question", ""))
        image_name = item.get("image_name", item.get("image", ""))
        sample_split = _resolve_split(item, image_name)
        if split != "all" and sample_split != split:
            continue
        sample = {
            "id": i,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "image_name": image_name,
            "question_type": q_type,
            "split": sample_split,
        }
        buckets.setdefault(q_type, []).append(sample)

    if not buckets:
        return []

    pool = [sample for bucket in buckets.values() for sample in bucket]
    pool.sort(key=lambda sample: sample["id"])
    listed = eval_set_ids()
    if listed is not None:
        # The explicit evaluation set (experiments/eval_set.json); grouped by question type below for the dropdown.
        by_id = {sample["id"]: sample for sample in pool}
        chosen = [by_id[i] for i in listed if i in by_id][:limit]
    else:
        # Seeded random sample of the split, so results are not biased by which questions sort first.
        chosen = random.Random(SAMPLE_SEED).sample(pool, min(limit, len(pool)))
    type_rank = {question_type: index for index, question_type in enumerate(QUESTION_TYPE_ORDER)}
    chosen.sort(key=lambda sample: (type_rank.get(sample["question_type"], len(type_rank)),
                                    (sample.get("question") or "").lower()))
    return chosen


def get_sample_by_id(sample_id: int) -> dict | None:
    """Return a specific sample case by index."""
    if sample_id in EXCLUDED_SAMPLE_IDS:
        return None
    data = load_dataset()
    if not data or sample_id < 0 or sample_id >= len(data):
        return None

    item = data[sample_id]
    image_name = item.get("image_name", item.get("image", ""))
    sample_split = _resolve_split(item, image_name)
    split = eval_split()
    if split != "all" and sample_split != split:
        return None
    image_path = os.path.join(DATA_DIR, "images", image_name)

    return {
        "id": sample_id,
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "image_name": image_name,
        "image_path": image_path if os.path.exists(image_path) else None,
        "question_type": _classify_question(item.get("question", "")),
        "split": sample_split,
    }


def get_image_bytes(image_name: str) -> bytes | None:
    """Read image bytes from the data directory."""
    image_path = os.path.join(DATA_DIR, "images", image_name)
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return f.read()


def _classify_question(question: str) -> str:
    """Simple rule-based question type classification for display."""
    q = question.lower().strip()
    if any(w in q for w in ["what organ", "what part", "what body", "what region",
                             "which organ", "which body"]):
        return "anatomy_identification"
    if any(w in q for w in ["what modality", "what type of image", "what imaging",
                             "what kind of image", "what is the imaging"]):
        return "modality_identification"
    if q.startswith(("is ", "are ", "does ", "do ", "was ", "has ", "have ")):
        return "yes_no"
    if any(w in q for w in ["how many", "number of", "count"]):
        return "counting"
    if any(w in q for w in ["where is", "where are", "location", "located"]):
        return "location"
    if any(w in q for w in ["abnormal", "finding", "diagnosis", "patholog"]):
        return "abnormality_description"
    return "other"


def _get_fallback_samples() -> list:
    """Provide fallback sample cases when dataset is not yet downloaded."""
    return [
        {
            "id": -1,
            "question": "What part of the body is shown in this image?",
            "answer": "(upload your own image)",
            "image_name": "",
            "question_type": "anatomy_identification",
            "split": "unknown",
        },
        {
            "id": -2,
            "question": "What imaging modality is used?",
            "answer": "(upload your own image)",
            "image_name": "",
            "question_type": "modality_identification",
            "split": "unknown",
        },
        {
            "id": -3,
            "question": "Is there any abnormality visible?",
            "answer": "(upload your own image)",
            "image_name": "",
            "question_type": "yes_no",
            "split": "unknown",
        },
    ]
