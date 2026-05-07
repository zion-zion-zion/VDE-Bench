"""Full-page text editing driven by pre-generated text instructions.

Given a JSON file of ``{image_path, instruction}`` records (produced by
``generate_text_instruction.py``), ask an image-editing LLM to apply each
instruction to the *full* page (no cropping) and save the result.

Usage
-----
    python edit_text_fullpage.py \
        --instruction_json outputs/text_instructions.json \
        --output_dir       outputs/text_edited \
        --output_json      outputs/text_edited.json \
        --model gemini-3-pro-image
"""

from __future__ import annotations

import argparse
import json
import os

from PIL import Image
from tqdm import tqdm

from llm_client import LLMClient, image_to_base64
from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instruction_json", required=True,
                   help="JSON list of {'image_path': ..., 'instruction': ...}.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--tmp_dir", default="tmp")
    p.add_argument("--model", required=True)
    p.add_argument("--edit_type_label", default="text modify",
                   help="Label stored in each output record.")
    p.add_argument("--api_base", default=None)
    p.add_argument("--api_key", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = LLMClient(api_base=args.api_base, api_key=args.api_key)
    ensure_dir(args.output_dir)
    ensure_dir(args.tmp_dir)

    with open(args.instruction_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} instructions.")

    records = []
    tmp_img_path = os.path.join(args.tmp_dir, "tmp_full.jpg")

    for sample in tqdm(samples, desc="Processing", ncols=100):
        img_path = sample["image_path"]
        img_name = os.path.basename(img_path)

        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        # Re-encode to JPEG for transport.
        Image.open(img_path).convert("RGB").save(tmp_img_path)

        prompt = sample["instruction"]
        edited_img, _ = client.edit_image(
            image_b64=image_to_base64(tmp_img_path),
            prompt=prompt,
            model=args.model,
        )
        if edited_img is None:
            print(f"[WARN] Edit failed: {img_path}")
            continue

        save_path = os.path.join(args.output_dir, f"{img_name}_modified.png")
        edited_img.save(save_path)

        records.append({
            "text": prompt,
            "image_path": save_path,
            "original_image_path": img_path,
            "edit_type": args.edit_type_label,
            "lang": sample.get("language"),
            "data_source": sample.get("data_source"),
            "width": None,
            "height": None,
            "poly": None,
        })

    if os.path.exists(tmp_img_path):
        os.remove(tmp_img_path)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {args.output_json}")


if __name__ == "__main__":
    main()
