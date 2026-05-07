"""Merge multiple edit-info JSONs and copy their images into a single directory.

For each input record with ``image_input`` and ``image_output`` fields, this
script:

  1. Looks up the image file next to ``image_output`` *but with the same
     basename as* ``image_input``.  (This handles cases where the post-edit
     image lives in a different directory.)
  2. Copies it to ``output_image_dir`` with a new name of ``<id>_<original>``.
  3. Rewrites the item's ``image_output`` to point at the new location.
  4. Concatenates all such rewritten items into a single output JSON.

Usage
-----
    python merge_and_copy_images.py \
        --input_json text_info_a.json text_info_b.json \
        --output_image_dir outputs/text_output_all \
        --output_json      outputs/text_info_all.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_json", nargs="+", required=True)
    p.add_argument("--output_image_dir", required=True)
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def process_json(json_path: str, output_image_dir: str, merged: List[Dict]) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        image_output = item.get("image_output")
        image_input = item.get("image_input")
        item_id = item.get("id")

        if not image_output or not image_input or not item_id:
            continue

        output_dir = os.path.dirname(image_output)
        input_filename = os.path.basename(image_input)
        img_path = os.path.join(output_dir, input_filename)

        new_img_name = f"{item_id}_{input_filename}"
        new_img_path = os.path.join(output_image_dir, new_img_name)

        if os.path.exists(img_path):
            shutil.copy2(img_path, new_img_path)
        else:
            print(f"[WARN] Source image not found: {img_path}")
            continue

        item["image_output"] = new_img_path
        merged.append(item)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_image_dir, exist_ok=True)
    merged: List[Dict] = []
    for json_path in args.input_json:
        process_json(json_path, args.output_image_dir, merged)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Done! Merged {len(merged)} items into {args.output_json}")


if __name__ == "__main__":
    main()
