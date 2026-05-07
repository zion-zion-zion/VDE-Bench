"""Intersect two edit JSONs on the ``image_input`` field.

Useful when you have a superset JSON (``--source``) and a smaller
filtered/labeled JSON (``--filter``) and want to keep only the superset
rows whose ``image_input`` appears in the filter set.
"""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="JSON file to filter.")
    p.add_argument("--filter", required=True, help="JSON file whose image_input values define the keep-list.")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.source, "r", encoding="utf-8") as f:
        source = json.load(f)
    with open(args.filter, "r", encoding="utf-8") as f:
        filt = json.load(f)

    keep_ids = {item["image_input"] for item in filt if "image_input" in item}
    result = [item for item in source if item.get("image_input") in keep_ids]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Kept {len(result)}/{len(source)} items -> {args.output}")


if __name__ == "__main__":
    main()
