"""
Download the VQA-RAD dataset from Hugging Face and organize it
into the data/vqa-rad/ directory.

Usage:
    python scripts/download_dataset.py
"""
import json
import os
import sys

def main():
    try:
        # Pick up HF_TOKEN (optional) from the project .env for authenticated downloads.
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    except ImportError:
        pass
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'vqa-rad')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    print("Downloading VQA-RAD dataset from Hugging Face...")
    ds = load_dataset("flaviagiammarino/vqa-rad")

    all_items = []

    for split_name in ds:
        split = ds[split_name]
        print(f"Processing split: {split_name} ({len(split)} items)")

        for i, item in enumerate(split):
            question = item.get("question", "")
            answer = item.get("answer", "")
            image = item.get("image", None)

            image_name = f"{split_name}_{i:04d}.png"

            if image is not None:
                image_path = os.path.join(images_dir, image_name)
                if not os.path.exists(image_path):
                    image.save(image_path)

            all_items.append({
                "id": len(all_items),
                "split": split_name,
                "question": question,
                "answer": answer,
                "image_name": image_name,
            })

    json_path = os.path.join(output_dir, "vqa_rad.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(all_items)} items to {json_path}")
    print(f"Images saved to {images_dir}")


if __name__ == "__main__":
    main()
