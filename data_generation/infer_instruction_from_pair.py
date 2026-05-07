"""Reverse-engineer a single natural-language instruction from a (before, after) pair.

Given a JSON file where each record already has ``original_image_path`` and
``image_path`` fields (produced by any of the ``edit_*`` scripts) and an
optional short local reference in ``text``, ask a vision LLM to describe the
edit.  The result is written back into the same ``text`` field of each record.

Usage
-----
    python infer_instruction_from_pair.py \
        --input_json  outputs/table_edited.json \
        --output_json outputs/table_edited_with_text.json \
        --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

from llm_client import LLMClient, image_to_base64
from prompts import INFER_INSTRUCTION_PROMPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api_base", default=None)
    p.add_argument("--api_key", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = LLMClient(api_base=args.api_base, api_key=args.api_key)

    with open(args.input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} pairs.")

    for sample in tqdm(samples, desc="Processing", ncols=100):
        original_path = sample.get("original_image_path")
        modified_path = sample.get("image_path")
        reference = sample.get("text", "")
        if not original_path or not modified_path:
            continue
        if not (os.path.exists(original_path) and os.path.exists(modified_path)):
            continue

        prompt = INFER_INSTRUCTION_PROMPT.format(reference=reference)
        llm_text = client.chat_with_images(
            prompt=prompt,
            image_b64_list=[image_to_base64(original_path), image_to_base64(modified_path)],
            model=args.model,
        )
        if llm_text:
            sample["text"] = llm_text.strip()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
