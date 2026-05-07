"""Generate table-editing instructions from source document images.

Randomly picks one of three prompt types for each page that contains a table:
text-addition, text-deletion, or table-structure-modification.  The output
JSON feeds straight into ``edit_table_fullpage.py``.

Usage
-----
    python generate_table_instruction.py \
        --source_json /path/to/OmniDocBench.json \
        --image_root  /path/to/source/images \
        --output_json outputs/table_instructions.json \
        --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import random

from tqdm import tqdm

from llm_client import LLMClient, image_to_base64
from prompts import INSTR_TABLE_ADD, INSTR_TABLE_DELETE, INSTR_TABLE_STRUCTURE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_json", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--api_base", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_prompt(rand_val: float) -> str:
    if rand_val > 0.66:
        return INSTR_TABLE_DELETE
    elif rand_val < 0.33:
        return INSTR_TABLE_ADD
    else:
        return INSTR_TABLE_STRUCTURE


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    client = LLMClient(api_base=args.api_base, api_key=args.api_key)

    with open(args.source_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if args.num_samples:
        samples = random.sample(samples, min(args.num_samples, len(samples)))
    print(f"Loaded {len(samples)} pages.")

    records = []
    for sample in tqdm(samples, desc="Processing", ncols=100):
        has_table = any(a.get("category_type") == "table" for a in sample.get("layout_dets", []))
        if not has_table:
            continue

        img_name = os.path.basename(sample["page_info"]["image_path"])
        img_path = os.path.join(args.image_root, img_name)
        if not os.path.exists(img_path):
            continue

        prompt = pick_prompt(random.random())
        llm_text = client.chat_with_images(
            prompt=prompt,
            image_b64_list=[image_to_base64(img_path)],
            model=args.model,
        )
        if not llm_text:
            continue

        records.append({
            "image_path": img_path,
            "instruction": llm_text.strip(),
            "language": sample["page_info"]["page_attribute"].get("language"),
            "data_source": sample["page_info"]["page_attribute"].get("data_source"),
        })

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} instructions to {args.output_json}")


if __name__ == "__main__":
    main()
