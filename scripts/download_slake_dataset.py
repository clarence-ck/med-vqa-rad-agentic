"""
Download the SLAKE dataset from Hugging Face and organize it
into the data/slake/ directory.

Usage:
    python scripts/download_slake_dataset.py
"""
import json
import os
import re
import sys


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'slake')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    print("Downloading SLAKE dataset from Hugging Face...")
    ds = load_dataset("BoKelvin/SLAKE")

    all_items = []

    def detect_language(item):
        """Infer language from explicit metadata or text content."""
        lang = item.get("lang") or item.get("language") or item.get("lang_code")
        if isinstance(lang, str) and lang.strip():
            return lang.strip().lower()

        question = str(item.get("question", ""))
        answer = str(item.get("answer", ""))
        combined = f"{question} {answer}"

        # Detect Chinese characters directly
        if re.search(r"[\u4e00-\u9fff]", combined):
            return "zh"

        # Fallback: treat as English if the text is mostly ASCII/Latin medical text
        if combined and all(ord(ch) < 128 for ch in combined if not ch.isspace()):
            return "en"

        return "unknown"

    for split_name in ds:
        split = ds[split_name]
        print(f"Processing split: {split_name} ({len(split)} items)")

        for i, item in enumerate(split):
            question = item.get("question", "")
            answer = item.get("answer", "")
            image = item.get("image", None)

            # Include additional SLAKE-specific metadata if available
            question_type = item.get("question_type", "")
            answer_type = item.get("answer_type", "")
            lang = detect_language(item)

            image_name = f"{split_name}_{i:04d}.png"

            if image is not None:
                image_path = os.path.join(images_dir, image_name)
                if not os.path.exists(image_path):
                    image.save(image_path)

            entry = {
                "id": len(all_items),
                "split": split_name,
                "question": question,
                "answer": answer,
                "image_name": image_name,
                "question_type": question_type,
                "answer_type": answer_type,
                "lang": lang,
            }

            # Add modality and body part if available (SLAKE-specific fields)
            if "modality" in item:
                entry["modality"] = item.get("modality", "")
            if "body_part" in item:
                entry["body_part"] = item.get("body_part", "")
            if "disease" in item:
                entry["disease"] = item.get("disease", "")

            all_items.append(entry)

    json_path = os.path.join(output_dir, "slake.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(all_items)} items to {json_path}")
    print(f"Images saved to {images_dir}")

    # Print summary by language and split
    print(f"\nLanguage breakdown by split:")
    split_lang_counts: Dict[str, Dict[str, int]] = {}
    for item in all_items:
        split = item.get("split", "unknown")
        lang = item.get("lang", "unknown")
        if split not in split_lang_counts:
            split_lang_counts[split] = {}
        split_lang_counts[split][lang] = split_lang_counts[split].get(lang, 0) + 1

    for split in sorted(split_lang_counts.keys()):
        counts = split_lang_counts[split]
        en = counts.get("en", 0)
        zh = counts.get("zh", 0)
        unknown = counts.get("unknown", 0)
        print(f"  {split}:")
        print(f"    English (en): {en} items")
        print(f"    Chinese (zh): {zh} items")
        if unknown:
            print(f"    Unknown: {unknown} items")

    # Total breakdown
    en_count = sum(1 for item in all_items if item.get("lang") == "en")
    zh_count = sum(1 for item in all_items if item.get("lang") == "zh")
    unknown_count = sum(1 for item in all_items if item.get("lang") == "unknown")
    print(f"\nTotal:")
    print(f"  English (en): {en_count} items")
    print(f"  Chinese (zh): {zh_count} items")
    if unknown_count:
        print(f"  Unknown: {unknown_count} items")

    if zh_count == 0:
        print("\nWarning: No Chinese rows were detected in this downloaded snapshot.")
        print("This HF export may be English-only, or the Chinese metadata may not be preserved in the loaded split.")


if __name__ == "__main__":
    main()
