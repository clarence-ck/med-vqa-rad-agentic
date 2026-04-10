"""
Build a concept-level SLAKE knowledge base for text-only RAG.

This script converts the downloaded SLAKE JSON into a normalized KB that is
better suited for retrieval than raw QA pairs. It focuses on concept-level
medical knowledge such as anatomy, modality, abnormalities, laterality,
and visual cues.

Usage:
    python scripts/build_slake_kb.py
    python scripts/build_slake_kb.py --input data/slake/slake.json --output-dir data/slake/kb
    python scripts/build_slake_kb.py --language en
    python scripts/build_slake_kb.py --language all

Default behavior:
    - uses the train split only
    - uses English-only rows unless --language all is passed
    - writes JSONL + summary JSON
    - groups repeated concept signatures into aggregated KB chunks
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


MODALITY_ALIASES = {
    "mri": "MRI",
    "magnetic resonance": "MRI",
    "ct": "CT",
    "computed tomography": "CT",
    "x-ray": "X-Ray",
    "xray": "X-Ray",
    "radiograph": "X-Ray",
    "plain film": "X-Ray",
    "ultrasound": "Ultrasound",
    "sonogram": "Ultrasound",
    "sonography": "Ultrasound",
    "fundus": "Fundus",
    "mammogram": "Mammography",
    "mammography": "Mammography",
    "dermoscopy": "Dermoscopy",
    "fluoroscopy": "Fluoroscopy",
    "angiography": "Angiography",
    "angiogram": "Angiography",
    "pet": "Nuclear/PET",
    "scintigraphy": "Nuclear",
}

BODY_REGION_ALIASES = {
    "chest": "chest",
    "thorax": "chest",
    "lung": "chest",
    "pulmonary": "chest",
    "pleural": "chest",
    "mediastin": "chest",
    "heart": "chest",
    "cardiac": "chest",
    "abdomen": "abdomen",
    "abdominal": "abdomen",
    "liver": "abdomen",
    "hepatic": "abdomen",
    "kidney": "abdomen",
    "renal": "abdomen",
    "spleen": "abdomen",
    "pancrea": "abdomen",
    "bowel": "abdomen",
    "intestin": "abdomen",
    "pelvis": "pelvis",
    "pelvic": "pelvis",
    "bladder": "pelvis",
    "uterus": "pelvis",
    "prostate": "pelvis",
    "head": "head",
    "brain": "head",
    "cranial": "head",
    "cerebral": "head",
    "cerebell": "head",
    "skull": "head",
    "neck": "neck",
    "cervical": "neck",
    "laryngeal": "neck",
    "spine": "musculoskeletal",
    "vertebra": "musculoskeletal",
    "bone": "musculoskeletal",
    "joint": "musculoskeletal",
    "fracture": "musculoskeletal",
    "shoulder": "musculoskeletal",
    "hip": "musculoskeletal",
    "knee": "musculoskeletal",
    "eye": "eye",
    "ocular": "eye",
    "retina": "eye",
    "fundus": "eye",
    "skin": "skin",
    "dermal": "skin",
}

FINDING_ALIASES = {
    "normal": "normal",
    "unremarkable": "normal",
    "no abnormality": "normal",
    "no finding": "normal",
    "fracture": "fracture",
    "broken": "fracture",
    "hemorrhage": "hemorrhage",
    "bleed": "hemorrhage",
    "hematoma": "hemorrhage",
    "infarct": "infarct",
    "ischemi": "infarct",
    "stroke": "infarct",
    "mass": "mass",
    "tumor": "mass",
    "tumour": "mass",
    "nodule": "mass",
    "lesion": "lesion",
    "effusion": "effusion",
    "fluid": "effusion",
    "edema": "edema",
    "oedema": "edema",
    "swelling": "edema",
    "pneumonia": "pneumonia",
    "consolidation": "pneumonia",
    "opacity": "opacity",
    "infiltrate": "opacity",
    "atelectasis": "atelectasis",
    "collapse": "atelectasis",
    "pneumothorax": "pneumothorax",
    "device": "device",
    "catheter": "device",
    "tube": "device",
}

CONCEPT_DEFINITIONS = {
    "modality": {
        "MRI": "Magnetic resonance imaging; often used for soft tissue contrast and detailed anatomy.",
        "CT": "Computed tomography; commonly used for cross-sectional anatomy and density-based findings.",
        "X-Ray": "Projection radiography; commonly used for chest, bone, and general structural assessment.",
        "Ultrasound": "Ultrasound imaging; used for soft tissue and fluid assessment in real time.",
        "Fundus": "Fundus imaging; used for the retina and optic disc.",
        "Mammography": "Breast imaging modality used to screen for masses, calcifications, and architectural distortion.",
        "Dermoscopy": "Skin imaging modality used to inspect lesions and pigmented structures.",
        "Fluoroscopy": "Real-time X-ray imaging used for motion and contrast studies.",
        "Angiography": "Vascular imaging used to inspect blood vessels and flow-related abnormalities.",
        "Nuclear": "Functional imaging with radiotracers used to study organ activity.",
        "Nuclear/PET": "Functional imaging that highlights metabolic or physiologic activity.",
    },
    "presence_absence": "Binary question pattern asking whether a structure or abnormality is present.",
    "counting": "Counting question pattern asking how many visible objects, structures, or findings are present.",
    "laterality": "Laterality concept indicating left, right, or bilateral location.",
}

VISUAL_CUE_TERMS = [
    "enlarged", "small", "large", "mild", "severe", "diffuse", "focal", "ring-enhancing", "ring enhancing",
    "hyperdense", "hypodense", "hyperintense", "hypointense", "opacity", "calcification", "effusion",
    "consolidation", "fracture", "lesion", "mass", "nodule", "edema", "swelling", "collapse",
    "pneumothorax", "fluid", "normal", "abnormal", "visible", "seen", "identifiable",
]

TARGET_PATTERNS = [
    r"contain\s+([^?.,;:]+)",
    r"present\s+([^?.,;:]+)",
    r"visible\s+([^?.,;:]+)",
    r"seen\s+([^?.,;:]+)",
    r"is\s+([^?.,;:]+?)\s+in\s+the\s+image",
    r"where\s+is\s+([^?.,;:]+)",
    r"what\s+is\s+the\s+main\s+([^?.,;:]+)",
    r"what\s+organ\s+is\s+([^?.,;:]+)",
]

QUESTION_FAMILY_PATTERNS = [
    ("modality_identification", ["what modality", "what is the imaging modality", "what type of imaging", "how was the image taken"]),
    ("anatomy_identification", ["which part", "what organ", "what body part", "what anatomy", "what organ system"]),
    ("presence_absence", ["does the picture contain", "is there", "are there", "can you see", "contain"]),
    ("counting", ["how many", "number of"]),
    ("location", ["where", "which side", "left or right", "right or left"]),
    ("abnormality_description", ["what is abnormal", "what abnormality", "what is wrong", "what findings", "describe the"]),
    ("explanation_request", ["why", "how", "explain", "describe"]),
]

YES_NO_STARTERS = ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "will ", "would ", "has ", "have ", "should ")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def detect_language(text: str) -> str:
    q = text or ""
    if re.search(r"[\u4e00-\u9fff]", q):
        return "zh"
    if q and all(ord(ch) < 128 for ch in q if not ch.isspace()):
        return "en"
    return "unknown"


def canonical_from_aliases(text: str, aliases: Dict[str, str], default: str = "unknown") -> str:
    low = normalize(text)
    for alias, canonical in aliases.items():
        if alias in low:
            return canonical
    return default


def infer_question_family(question: str, answer_type: str) -> str:
    q = normalize(question)
    if not q:
        return "other"
    if answer_type.upper() == "CLOSED" or q.startswith(YES_NO_STARTERS):
        if any(tok in q for tok in ["how many", "number of"]):
            return "counting"
        if any(tok in q for tok in ["contain", "present", "visible", "seen", "identifiable"]):
            return "presence_absence"
    for family, patterns in QUESTION_FAMILY_PATTERNS:
        if any(pattern in q for pattern in patterns):
            return family
    return "other"


def infer_modality(text: str) -> str:
    return canonical_from_aliases(text, MODALITY_ALIASES)


def infer_body_region(text: str) -> str:
    return canonical_from_aliases(text, BODY_REGION_ALIASES)


def infer_finding_category(text: str) -> str:
    return canonical_from_aliases(text, FINDING_ALIASES, default="other")


def infer_laterality(question: str, answer: str) -> str:
    q = normalize(question)
    a = normalize(answer)
    combined = f"{q} {a}"
    if "bilateral" in combined or "both sides" in combined:
        return "bilateral"
    has_left = bool(re.search(r"\bleft\b", combined))
    has_right = bool(re.search(r"\bright\b", combined))
    if has_left and has_right:
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return "unknown"


def extract_target_entity(question: str) -> str:
    q = normalize(question)
    if not q:
        return "unknown"

    for pattern in TARGET_PATTERNS:
        match = re.search(pattern, q)
        if match:
            target = match.group(1).strip()
            target = re.sub(r"^(the|a|an)\s+", "", target)
            target = re.sub(r"\s+(image|picture|scan|photo)$", "", target)
            target = target.rstrip("?.,;:")
            if target:
                return target

    return "unknown"


def infer_answer_style(answer: str) -> str:
    ans = normalize(answer)
    if ans in {"yes", "no", "y", "n", "true", "false", "left", "right", "bilateral", "normal", "abnormal"}:
        return "closed"
    if len(ans.split()) <= 3:
        return "short"
    return "open"


def infer_concept_type(question_family: str, answer_type: str, answer: str, modality: str, body_region: str, finding_category: str, target_entity: str) -> str:
    ans = normalize(answer)
    if question_family == "modality_identification":
        return "modality"
    if question_family == "counting" or re.fullmatch(r"\d+", ans):
        return "counting"
    if question_family == "presence_absence" or (answer_type.upper() == "CLOSED" and ans in {"yes", "no", "true", "false", "是的", "不是", "有", "没有", "包含", "不包含"}):
        if ans in {"yes", "no"}:
            return "presence_absence"
    if question_family == "location" or infer_laterality(question_family, answer) != "unknown":
        return "laterality"
    if question_family == "anatomy_identification":
        return "anatomy"
    if question_family == "abnormality_description":
        return "abnormality"
    if question_family == "explanation_request":
        return "explanation"
    if target_entity != "unknown" and question_family in {"presence_absence", "location", "anatomy_identification", "abnormality_description", "other"}:
        if target_entity in {"yes", "no"}:
            return "presence_absence"
        if any(token in target_entity for token in ["left", "right", "bilateral"]):
            return "laterality"
        if any(token in target_entity for token in ["modality", "ct", "mri", "x-ray", "xray"]):
            return "modality"
        if infer_body_region(target_entity) != "unknown":
            return "anatomy"
    if body_region != "unknown":
        return "anatomy"
    if finding_category != "other":
        return "abnormality"
    if modality != "unknown":
        return "modality"
    return "other"


def infer_visual_cues(question: str, answer: str, finding_category: str) -> List[str]:
    combined = normalize(f"{question} {answer} {finding_category}")
    cues = [term for term in VISUAL_CUE_TERMS if term in combined]
    if finding_category != "other" and finding_category not in cues:
        cues.append(finding_category)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for cue in cues:
        if cue not in seen:
            out.append(cue)
            seen.add(cue)
    return out


def infer_concept_label(concept_type: str, question: str, answer: str, modality: str, body_region: str, finding_category: str, target_entity: str) -> str:
    ans = normalize(answer)
    if concept_type == "modality":
        return modality if modality != "unknown" else (ans.upper() if ans else "unknown")
    if concept_type == "counting":
        return ans or "counting"
    if concept_type == "presence_absence":
        if target_entity != "unknown":
            return target_entity
        return finding_category if finding_category != "other" else (body_region if body_region != "unknown" else "presence")
    if concept_type == "laterality":
        lat = infer_laterality(question, answer)
        return lat if lat != "unknown" else (ans or "laterality")
    if concept_type == "anatomy":
        if target_entity != "unknown":
            return target_entity
        return body_region if body_region != "unknown" else (ans or "anatomy")
    if concept_type == "abnormality":
        if target_entity != "unknown":
            return target_entity
        return finding_category if finding_category != "other" else (ans or "abnormality")
    if concept_type == "explanation":
        return ans[:80] if ans else "explanation"
    return ans[:80] if ans else "unknown"


def build_text_chunk(concept_type: str, concept: str, modality: str, body_region: str, finding_category: str, laterality: str, answer_type: str, visual_cues: List[str], target_entity: str) -> str:
    parts = [f"Concept type: {concept_type}", f"Concept: {concept}"]
    if target_entity != "unknown" and target_entity != concept:
        parts.append(f"Target entity: {target_entity}")

    if modality != "unknown":
        parts.append(f"Modality: {modality}")
    if body_region != "unknown":
        parts.append(f"Body region: {body_region}")
    if finding_category != "other":
        parts.append(f"Finding category: {finding_category}")
    if laterality != "unknown":
        parts.append(f"Laterality: {laterality}")
    if answer_type:
        parts.append(f"Answer type: {answer_type}")
    if visual_cues:
        parts.append(f"Visual cues: {', '.join(visual_cues[:6])}")

    definition = CONCEPT_DEFINITIONS.get(concept_type, {})
    if isinstance(definition, dict) and concept in definition:
        parts.append(f"Medical note: {definition[concept]}")
    elif isinstance(definition, str):
        parts.append(f"Medical note: {definition}")

    if concept_type == "modality" and concept in definition:
        pass
    elif concept_type == "abnormality":
        parts.append("Medical note: This chunk should be used as a concept cue, not as a final answer.")
    elif concept_type == "anatomy":
        parts.append("Medical note: Use the organ/body-region concept to support visual grounding.")
    elif concept_type == "presence_absence":
        parts.append("Medical note: Use this as a binary evidence cue for yes/no questions.")

    return " ".join(parts)


def iter_records(rows: Iterable[dict], splits: set[str]) -> Iterable[dict]:
    for row in rows:
        if row.get("split") in splits:
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a concept-level SLAKE KB for RAG.")
    parser.add_argument("--input", default=os.path.join("data", "slake", "slake.json"), help="Path to the downloaded SLAKE JSON")
    parser.add_argument("--output-dir", default=os.path.join("data", "slake", "kb"), help="Directory to write KB files")
    parser.add_argument("--splits", default="train", help="Comma-separated splits to include (default: train)")
    parser.add_argument("--language", default="en", help="Language to include in the KB (default: en). Use 'all' to keep bilingual rows.")
    parser.add_argument("--min-support", type=int, default=1, help="Minimum aggregate support count to keep a KB chunk")
    parser.add_argument("--no-deduplicate", action="store_true", help="Disable concept-level deduplication/aggregation")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    splits = {part.strip() for part in args.splits.split(",") if part.strip()}
    filtered_rows = list(iter_records(rows, splits))
    language_filter = (args.language or "en").strip().lower()
    if language_filter != "all":
        filtered_rows = [row for row in filtered_rows if (row.get("lang") or row.get("language") or detect_language(f"{row.get('question', '')} {row.get('answer', '')}")) == language_filter]

    grouped: dict[Tuple[str, str, str, str, str, str, str], dict] = {}
    ordered_records: List[dict] = []

    for row in filtered_rows:
        question = row.get("question", "")
        answer = row.get("answer", "")
        answer_type = row.get("answer_type", "")
        language = row.get("language") or row.get("lang") or detect_language(f"{question} {answer}")
        target_entity = extract_target_entity(question)
        modality = infer_modality(f"{question} {answer} {row.get('modality', '')}")
        body_region = infer_body_region(f"{question} {answer}")
        finding_category = infer_finding_category(f"{question} {answer}")
        laterality = infer_laterality(question, answer)
        question_family = infer_question_family(question, answer_type)
        concept_type = infer_concept_type(question_family, answer_type, answer, modality, body_region, finding_category, target_entity)
        concept = infer_concept_label(concept_type, question, answer, modality, body_region, finding_category, target_entity)
        visual_cues = infer_visual_cues(question, answer, finding_category)
        answer_style = infer_answer_style(answer)

        record = {
            "source": "slake",
            "split": row.get("split", "train"),
            "kb_id": None,
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "language": language,
            "concept_type": concept_type,
            "concept": concept,
            "modality": modality,
            "body_region": body_region,
            "finding_category": finding_category,
            "laterality": laterality,
            "question_family": question_family,
            "target_entity": target_entity,
            "answer_style": answer_style,
            "visual_cues": visual_cues,
            "keywords": sorted({
                *(normalize(question).split()),
                *(normalize(answer).split()),
                *(visual_cues or []),
                modality.lower() if modality != "unknown" else "",
                body_region,
                finding_category,
                laterality,
                target_entity,
                language,
                concept_type,
            } - {""}),
            "metadata_confidence": 0.78,
            "source_question": question,
            "source_answer": answer,
        }
        record["text_chunk"] = build_text_chunk(
            concept_type=concept_type,
            concept=concept,
            modality=modality,
            body_region=body_region,
            finding_category=finding_category,
            laterality=laterality,
            answer_type=answer_type,
            visual_cues=visual_cues,
            target_entity=target_entity,
        )

        if args.no_deduplicate:
            ordered_records.append(record)
            continue

        key = (concept_type, concept, modality, body_region, finding_category, laterality, answer_type, language)
        agg = grouped.get(key)
        if agg is None:
            agg = {
                **record,
                "support_count": 0,
                "question_samples": [],
                "answer_samples": [],
                "split_counts": Counter(),
                "visual_cues_counts": Counter(),
            }
            grouped[key] = agg
            ordered_records.append(agg)

        agg["support_count"] += 1
        agg["split_counts"][row.get("split", "train")] += 1
        if question and len(agg["question_samples"]) < 5:
            agg["question_samples"].append(question)
        if answer and len(agg["answer_samples"]) < 5:
            agg["answer_samples"].append(answer)
        for cue in visual_cues:
            agg["visual_cues_counts"][cue] += 1

    # Finalize aggregated fields
    final_records: List[dict] = []
    for idx, rec in enumerate(ordered_records):
        if rec.get("support_count", 0) < args.min_support:
            continue
        item = dict(rec)
        item["kb_id"] = f"slake_kb_{idx:06d}"
        if "split_counts" in item:
            item["split_counts"] = dict(item["split_counts"])
        if "visual_cues_counts" in item:
            item["visual_cues_counts"] = dict(item["visual_cues_counts"])
        final_records.append(item)

    jsonl_path = os.path.join(output_dir, "slake_kb.jsonl")
    json_path = os.path.join(output_dir, "slake_kb.json")
    summary_path = os.path.join(output_dir, "slake_kb_summary.json")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in final_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)

    summary = {
        "source": "slake",
        "input_path": input_path,
        "splits": sorted(splits),
        "language_filter": language_filter,
        "input_rows": len(rows),
        "filtered_rows": len(filtered_rows),
        "kb_records": len(final_records),
        "concept_types": Counter(item["concept_type"] for item in final_records),
        "language_counts": Counter(item.get("language", "unknown") for item in final_records),
        "modality_counts": Counter(item["modality"] for item in final_records),
        "body_region_counts": Counter(item["body_region"] for item in final_records),
        "finding_category_counts": Counter(item["finding_category"] for item in final_records),
        "answer_type_counts": Counter(item["answer_type"] for item in final_records),
        "output_files": {
            "jsonl": jsonl_path,
            "json": json_path,
        },
    }
    summary["concept_types"] = dict(summary["concept_types"])
    summary["language_counts"] = dict(summary["language_counts"])
    summary["modality_counts"] = dict(summary["modality_counts"])
    summary["body_region_counts"] = dict(summary["body_region_counts"])
    summary["finding_category_counts"] = dict(summary["finding_category_counts"])
    summary["answer_type_counts"] = dict(summary["answer_type_counts"])

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Built SLAKE KB from {len(filtered_rows)} rows")
    print(f"KB records: {len(final_records)}")
    print(f"Language filter: {language_filter}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {summary_path}")
    print("Top concept types:")
    for k, v in Counter(item["concept_type"] for item in final_records).most_common(10):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
