"""
Download the SLAKE dataset from Hugging Face and organize it
into the data/slake/ directory.

Usage:
    python scripts/download_slake_dataset.py
"""
import json
import os
import re
import shutil
import sys
import urllib.request
import urllib.error
import zipfile


def download_file(url: str, output_path: str) -> bool:
    """Download a file from URL to the specified path."""
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"Failed to download {url}: {e}")
        return False


def extract_required_images(zip_path: str, images_dir: str, required_images: set[str]) -> dict[str, int]:
    """Extract only the required images from the SLAKE archive."""
    stats = {"downloaded": 0, "existing": 0, "missing": 0}

    def normalize(name: str) -> str:
        return name.replace("\\", "/").lstrip("./")

    required_images = {normalize(name) for name in required_images if name}
    if not required_images:
        return stats

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive_lookup = {}
        archive_names = []
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            normalized = normalize(member)
            archive_names.append(normalized)
            archive_lookup[normalized] = member

        for image_name in sorted(required_images):
            target_path = os.path.join(images_dir, image_name)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if os.path.exists(target_path):
                stats["existing"] += 1
                continue

            member_name = archive_lookup.get(image_name)
            if member_name is None:
                suffix = "/" + image_name
                for candidate in archive_names:
                    if candidate == image_name or candidate.endswith(suffix):
                        member_name = archive_lookup.get(candidate)
                        break

            if member_name is None:
                stats["missing"] += 1
                print(f"      Warning: image not found in archive: {image_name}")
                continue

            with archive.open(member_name) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            stats["downloaded"] += 1

    return stats


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
    required_images = set()

    def detect_language(item):
        """Infer language from explicit metadata or text content."""
        lang = item.get("q_lang") or item.get("lang") or item.get("language") or item.get("lang_code")
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

        for item in split:
            question = item.get("question", "")
            answer = item.get("answer", "")
            image_name = item.get("img_name") or item.get("image_name") or ""
            if image_name:
                required_images.add(image_name)

            # Include additional SLAKE-specific metadata if available
            question_type = item.get("content_type", item.get("question_type", ""))
            answer_type = item.get("answer_type", "")
            lang = detect_language(item)

            entry = {
                "id": len(all_items),
                "split": split_name,
                "question": question,
                "answer": answer,
                "image_name": image_name,
                "question_type": question_type,
                "answer_type": answer_type,
                "lang": lang,
                "img_id": item.get("img_id", ""),
                "qid": item.get("qid", ""),
                "base_type": item.get("base_type", ""),
                "content_type": item.get("content_type", ""),
                "location": item.get("location", ""),
                "triple": item.get("triple", []),
            }

            # Add modality and body part if available (SLAKE-specific fields)
            if "modality" in item:
                entry["modality"] = item.get("modality", "")
            if "body_part" in item:
                entry["body_part"] = item.get("body_part", "")
            if "disease" in item:
                entry["disease"] = item.get("disease", "")

            all_items.append(entry)

    imgs_zip_path = os.path.join(output_dir, "imgs.zip")
    if not os.path.exists(imgs_zip_path):
        print("Downloading SLAKE images archive (imgs.zip) from Hugging Face...")
        imgs_zip_url = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip?download=true"
        if not download_file(imgs_zip_url, imgs_zip_path):
            print("Error: failed to download imgs.zip")
            sys.exit(1)

    print(f"Extracting {len(required_images)} referenced images from imgs.zip...")
    image_stats = extract_required_images(imgs_zip_path, images_dir, required_images)

    json_path = os.path.join(output_dir, "slake.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(all_items)} items to {json_path}")
    print(f"Images saved to {images_dir}")
    print(f"\nImage extract stats: {image_stats['downloaded']} extracted, {image_stats['existing']} existing, {image_stats['missing']} missing")

    # Print summary by language and split
    print(f"\nLanguage breakdown by split:")
    split_lang_counts = {}
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
