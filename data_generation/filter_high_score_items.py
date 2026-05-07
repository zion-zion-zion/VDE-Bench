"""Keep only items with top human-review scores.

Scans a directory of per-item JSON files, keeps the records that received
score_instruction=3 AND score_satisfaction=3 (the highest on a 1-3 scale),
and emits a single aggregated JSON.
"""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, help="Directory of per-item JSON files.")
    p.add_argument("--output_json", required=True)
    p.add_argument("--score_value", default="3",
                   help="Required value for both score fields (string).")
    return p.parse_args()


def collect_high_score_data(input_dir: str, output_path: str, score_value: str) -> None:
    kept = []

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue
        full_path = os.path.join(input_dir, filename)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {filename}: {e}")
            continue

        def _match(item) -> bool:
            return (
                item.get("score_instruction") == score_value
                and item.get("score_satisfaction") == score_value
            )

        if isinstance(data, list):
            kept.extend(item for item in data if _match(item))
        elif isinstance(data, dict) and _match(data):
            kept.append(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"Kept {len(kept)} items -> {output_path}")


def main() -> None:
    args = parse_args()
    collect_high_score_data(args.input_dir, args.output_json, args.score_value)


if __name__ == "__main__":
    main()
