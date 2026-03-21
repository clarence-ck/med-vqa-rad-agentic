import json
import os
from config import DATA_DIR


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


def get_sample_cases(limit: int = 20) -> list:
    """Return a curated set of sample cases for the UI dropdown."""
    data = load_dataset()
    if not data:
        return _get_fallback_samples()

    buckets = {}
    for i, item in enumerate(data):
        q_type = _classify_question(item.get("question", ""))
        image_name = item.get("image_name", item.get("image", ""))
        sample_split = item.get("split") or infer_split(image_name)
        sample = {
            "id": i,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "image_name": image_name,
            "question_type": q_type,
            "split": sample_split,
        }
        buckets.setdefault(q_type, []).append(sample)

    samples = []
    bucket_order = list(buckets.keys())
    while len(samples) < limit and any(buckets.get(question_type) for question_type in bucket_order):
        for question_type in bucket_order:
            bucket = buckets.get(question_type) or []
            if not bucket:
                continue
            samples.append(bucket.pop(0))
            if len(samples) >= limit:
                break

    return samples


def get_sample_by_id(sample_id: int) -> dict | None:
    """Return a specific sample case by index."""
    data = load_dataset()
    if not data or sample_id < 0 or sample_id >= len(data):
        return None

    item = data[sample_id]
    image_name = item.get("image_name", item.get("image", ""))
    image_path = os.path.join(DATA_DIR, "images", image_name)

    return {
        "id": sample_id,
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "image_name": image_name,
        "image_path": image_path if os.path.exists(image_path) else None,
        "question_type": _classify_question(item.get("question", "")),
        "split": item.get("split") or infer_split(image_name),
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
