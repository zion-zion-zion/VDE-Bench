"""Crop each table region, ask an image-editing LLM to modify it, and paste it back.

This is the first iteration of the VDE-Bench table-editing pipeline. It uses
the short-form prompts (``PROMPT_TABLE_DELETE`` / ``PROMPT_TABLE_ADD``) so the
model returns *both* the modified crop and a natural-language instruction in
a single round-trip.

Usage
-----
    python edit_table_crop.py \
        --source_json /path/to/OmniDocBench.json \
        --image_root  /path/to/source/images \
        --output_dir  outputs/table_edited \
        --output_json outputs/table_edited.json \
        --model gemini-3-pro-image \
        --num_samples 150
"""

from __future__ import annotations

import argparse
import json
import os
import random

from PIL import Image
from tqdm import tqdm

from llm_client import LLMClient, image_to_base64
from prompts import PROMPT_TABLE_DELETE, PROMPT_TABLE_ADD, PROMPT_TABLE_COLOR
from utils import ensure_dir, poly2bbox


PROMPTS = [PROMPT_TABLE_DELETE, PROMPT_TABLE_ADD, PROMPT_TABLE_COLOR]
EDIT_TYPE_BY_INDEX = {0: "text delete", 1: "text add", 2: "table color modify"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_json", required=True, help="Path to OmniDocBench-style JSON.")
    p.add_argument("--image_root", required=True, help="Directory containing the source images referenced in the JSON.")
    p.add_argument("--output_dir", required=True, help="Where to save the edited full-page images.")
    p.add_argument("--output_json", required=True, help="Path to write the per-edit metadata JSON.")
    p.add_argument("--tmp_dir", default="tmp", help="Scratch directory for cropped patches.")
    p.add_argument("--model", required=True, help="Image-editing model name (passed verbatim to the LLM endpoint).")
    p.add_argument("--num_samples", type=int, default=None, help="Randomly sample this many pages (default: use all).")
    p.add_argument("--max_prompt_index", type=int, default=1, help="Use prompt indices 0..max_prompt_index inclusive.")
    p.add_argument("--api_base", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    client = LLMClient(api_base=args.api_base, api_key=args.api_key)
    ensure_dir(args.output_dir)
    ensure_dir(args.tmp_dir)

    with open(args.source_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} source pages.")
    if args.num_samples is not None and len(samples) > args.num_samples:
        samples = random.sample(samples, args.num_samples)
        print(f"Randomly sampled {len(samples)} pages.")

    records = []
    for sample in tqdm(samples, desc="Processing", ncols=100):
        img_name = os.path.basename(sample["page_info"]["image_path"])
        img_path = os.path.join(args.image_root, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        page_img = Image.open(img_path).convert("RGB")

        for i, anno in enumerate(sample["layout_dets"]):
            if anno["category_type"] != "table":
                continue

            x1, y1, x2, y2 = poly2bbox(anno["poly"])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop_img = page_img.crop((x1, y1, x2, y2))

            tmp_crop_path = os.path.join(args.tmp_dir, f"tmp_{img_name}_table_{i}.png")
            crop_img.save(tmp_crop_path)

            index = random.randint(0, args.max_prompt_index)
            prompt = PROMPTS[index]
            edit_type = EDIT_TYPE_BY_INDEX[index]

            edited_crop, llm_text = client.edit_image(
                image_b64=image_to_base64(tmp_crop_path),
                prompt=prompt,
                model=args.model,
            )

            if edited_crop is None:
                print(f"[WARN] Edit failed: {tmp_crop_path}")
                os.remove(tmp_crop_path)
                continue

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            edited_crop = edited_crop.resize((w, h))

            merged = page_img.copy()
            merged.paste(edited_crop, (x1, y1))

            save_path = os.path.join(args.output_dir, f"{img_name}_merged_{index}.png")
            merged.save(save_path)

            records.append({
                "text": (llm_text or "").strip(),
                "image_path": save_path,
                "original_image_path": img_path,
                "edit_type": edit_type,
                "lang": sample["page_info"]["page_attribute"]["language"],
                "data_source": sample["page_info"]["page_attribute"]["data_source"],
                "width": sample["page_info"]["width"],
                "height": sample["page_info"]["height"],
                "poly": anno["poly"],
            })

            os.remove(tmp_crop_path)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {args.output_json}")


if __name__ == "__main__":
    main()
