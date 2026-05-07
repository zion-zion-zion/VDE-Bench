"""Classify each natural-language editing instruction into a coarse type.

Reads a JSON of edit records with an ``instruction`` field and writes back
an ``instruction type`` field whose value is one of:

    - text deletion
    - text addition
    - table structure edit

Usage
-----
    python classify_instruction.py \
        --input_json  outputs/table_edited_labeled.json \
        --output_json outputs/table_edited_labeled_typed.json \
        --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

from llm_client import LLMClient
from prompts import CLASSIFY_INSTRUCTION_PROMPT


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
    print(f"Classifying {len(samples)} instructions.")

    out = []
    for sample in tqdm(samples, desc="Classifying", ncols=100):
        instruction = sample.get("instruction", "")
        if not instruction:
            continue

        response = client.chat(
            prompt=CLASSIFY_INSTRUCTION_PROMPT.format(instruction=instruction),
            model=args.model,
        )
        itype = (response or "").strip()

        out.append({**sample, "instruction type": itype})

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
