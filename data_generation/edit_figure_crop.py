"""Crop each figure region, ask an image-editing LLM to modify it, and paste it back.

Generates three variants per figure: text modification, colour modification,
and figure-type modification.

Usage
-----
    python edit_figure_crop.py \
        --source_json /path/to/OmniDocBench.json \
        --image_root  /path/to/source/images \
        --output_dir  outputs/figure_edited \
        --output_json outputs/figure_edited.json \
        --model gemini-3-pro-image
"""

from __future__ import annotations

import argparse
import json
import os

from PIL import Image
from tqdm import tqdm

from llm_client import LLMClient, image_to_base64
from prompts import PROMPT_FIGURE_TEXT, PROMPT_FIGURE_COLOR, PROMPT_FIGURE_TYPE
from utils import ensure_dir, poly2bbox


PROMPTS = [PROMPT_FIGURE_TEXT, PROMPT_FIGURE_COLOR, PROMPT_FIGURE_TYPE]
EDIT_TYPES = ["text modify", "figure color modify", "figure type modify"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_json", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--tmp_dir", default="tmp")
    p.add_argument("--model", required=True)
    p.add_argument("--api_base", default=None)
    p.add_argument("--api_key", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = LLMClient(api_base=args.api_base, api_key=args.api_key)

    ensure_dir(args.output_dir)
    ensure_dir(args.tmp_dir)

    with open(args.source_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} pages.")

    records = []
    for sample in tqdm(samples, desc="Processing", ncols=100):
        img_name = os.path.basename(sample["page_info"]["image_path"])
        img_path = os.path.join(args.image_root, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        page_img = Image.open(img_path).convert("RGB")

        for i, anno in enumerate(sample["layout_dets"]):
            if anno["category_type"] != "figure":
                continue

            x1, y1, x2, y2 = poly2bbox(anno["poly"])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop_img = page_img.crop((x1, y1, x2, y2))

            tmp_crop_path = os.path.join(args.tmp_dir, f"tmp_{img_name}_figure_{i}.png")
            crop_img.save(tmp_crop_path)

            for index, prompt in enumerate(PROMPTS):
                edited_crop, llm_text = client.edit_image(
                    image_b64=image_to_base64(tmp_crop_path),
                    prompt=prompt,
                    model=args.model,
                )
                if edited_crop is None:
                    print(f"[WARN] Edit failed for prompt {index}: {tmp_crop_path}")
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
                    "edit_type": EDIT_TYPES[index],
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
