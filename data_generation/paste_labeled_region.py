"""Paste human-labeled regions from an LLM-edited image back into the original.

For each record with ``label_output`` bboxes (in percentage coordinates),
crop the edited image at those bboxes and overwrite the corresponding
regions of the (resized) original image.  Useful when you want the final
benchmark sample to keep the original's resolution and only replace the
regions the human annotator actually marked as correct.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from PIL import Image


def read_image(image_path: str) -> Optional[Image.Image]:
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"[ERROR] Failed to read image ({image_path}): {e}")
        return None


def process_item(item: Dict, resize_dir: str, final_dir: str, crop_dir: str) -> None:
    input_path = item.get("image_input")
    output_path = item.get("image_output")
    labels: List[Dict] = item.get("label_output", [])

    if not input_path or not output_path or not labels:
        print(f"[WARN] Missing fields, skipping: {item.get('data_id')}")
        return

    img_input = read_image(input_path)
    img_output = read_image(output_path)
    if img_input is None or img_output is None:
        print(f"[WARN] Skipping {item.get('data_id')} (could not read images)")
        return

    img_input = img_input.convert("RGB")
    img_output = img_output.convert("RGB")
    img_input = img_input.resize(img_output.size)
    out_w, out_h = img_output.size

    basename = os.path.basename(input_path)
    resized_save_path = os.path.join(resize_dir, basename)
    img_input.save(resized_save_path)
    print(f"[INFO] Saved resized input -> {resized_save_path}")

    for idx, label in enumerate(labels):
        x = int(label["x"] * out_w / 100)
        y = int(label["y"] * out_h / 100)
        w = int(label["width"] * out_w / 100)
        h = int(label["height"] * out_h / 100)

        crop_region = img_output.crop((x, y, x + w, y + h))
        crop_name = os.path.splitext(basename)[0] + f"_crop_{idx}.png"
        crop_save_path = os.path.join(crop_dir, crop_name)
        crop_region.save(crop_save_path)
        print(f"[INFO] Saved crop -> {crop_save_path}")

        img_input.paste(crop_region, (x, y))

    final_save_path = os.path.join(final_dir, basename)
    img_input.save(final_save_path)
    print(f"[INFO] Saved final -> {final_save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_json", required=True, help="JSON with records containing image_input, image_output and label_output.")
    p.add_argument("--resize_dir", required=True, help="Where resized originals are saved.")
    p.add_argument("--final_dir", required=True, help="Where pasted-back images are saved.")
    p.add_argument("--crop_dir", required=True, help="Where per-region crops are saved.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.resize_dir, exist_ok=True)
    os.makedirs(args.final_dir, exist_ok=True)
    os.makedirs(args.crop_dir, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        print(f"\n===== Processing {i + 1}/{len(data)} =====")
        try:
            process_item(item, args.resize_dir, args.final_dir, args.crop_dir)
        except Exception as e:
            print(f"[ERROR] {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
