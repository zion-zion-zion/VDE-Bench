"""Stand-alone text-region evaluation for VDE-Bench.

Evaluates each ``label_output`` region in a ``text_info`` JSON by comparing
the GT and predicted OCR blocks that overlap with the annotated region.

Unlike ``eval_unified.py`` this script defines *per-label* success criteria
(e.g. for ``text deletion`` tasks, success is "no predicted text inside the
region") and reports success rates broken down by instruction type.
"""

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from math import exp, log
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_iou(box_a: Box, box_b: Box) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))

    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def convert_label_to_bbox(label: Dict) -> Box:
    """Convert label_output (percentage) to absolute bbox (x1, y1, x2, y2)."""
    x_pct = label["x"]
    y_pct = label["y"]
    width_pct = label["width"]
    height_pct = label["height"]
    orig_w = label["original_width"]
    orig_h = label["original_height"]

    x1 = (x_pct / 100.0) * orig_w
    y1 = (y_pct / 100.0) * orig_h
    x2 = x1 + (width_pct / 100.0) * orig_w
    y2 = y1 + (height_pct / 100.0) * orig_h

    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def load_ocr_blocks(ocr_path: Path) -> List[Tuple[Box, str]]:
    """Load blocks from OCR result JSON file."""
    try:
        with ocr_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    blocks = []
    for block in data.get("parsing_res_list", []):
        bbox = block.get("block_bbox")
        if not bbox or len(bbox) != 4:
            continue
        box = tuple(float(v) for v in bbox)
        text = block.get("block_content") or ""
        blocks.append((box, text))
    return blocks


def find_overlapping_blocks(
    label_bbox: Box,
    ocr_blocks: List[Tuple[Box, str]],
    iou_threshold: float = 0.1,
) -> List[Tuple[Box, str, float]]:
    """Find OCR blocks that overlap with the label region."""
    overlapping = []
    for ocr_bbox, text in ocr_blocks:
        iou = compute_iou(label_bbox, ocr_bbox)
        if iou >= iou_threshold:
            overlapping.append((ocr_bbox, text, iou))
    overlapping.sort(key=lambda x: x[2], reverse=True)
    return overlapping


def extract_text_from_region(
    label_bbox: Box,
    ocr_blocks: List[Tuple[Box, str]],
    iou_threshold: float = 0.1,
) -> str:
    """Concatenate text from all OCR blocks overlapping with *label_bbox*."""
    overlapping = find_overlapping_blocks(label_bbox, ocr_blocks, iou_threshold)
    texts = [text for _, text, _ in overlapping if text.strip()]
    return " ".join(texts).strip()


# ---------------------------------------------------------------------------
# Text similarity metrics
# ---------------------------------------------------------------------------

def compute_text_similarity(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, text_a, text_b).ratio()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i]
        for j, cb in enumerate(b, start=1):
            curr_row.append(min(
                curr_row[j - 1] + 1,
                prev_row[j] + 1,
                prev_row[j - 1] + (ca != cb),
            ))
        prev_row = curr_row
    return prev_row[-1]


def compute_cdm(candidate: str, reference: str) -> float:
    if not candidate and not reference:
        return 1.0
    max_len = max(len(candidate), len(reference))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - levenshtein_distance(candidate, reference) / max_len)


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    eps = 1e-9
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    if not cand_tokens and not ref_tokens:
        return 1.0
    if not cand_tokens or not ref_tokens:
        return 0.0
    precisions: List[float] = []
    for n in range(1, max_n + 1):
        cc = _ngram_counts(cand_tokens, n)
        rc = _ngram_counts(ref_tokens, n)
        if not cc:
            precisions.append(eps)
            continue
        overlap = sum((cc & rc).values())
        total = sum(cc.values())
        precisions.append(max((overlap + 1.0) / (total + 1.0), eps))
    geo_mean = exp(sum(log(p) for p in precisions) / max_n)
    cand_len, ref_len = len(cand_tokens), len(ref_tokens)
    bp = 1.0 if cand_len > ref_len else (exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0)
    return bp * geo_mean


# ---------------------------------------------------------------------------
# Per-label evaluation
# ---------------------------------------------------------------------------

def evaluate_label_region(
    label: Dict,
    gt_blocks: List[Tuple[Box, str]],
    pred_blocks: List[Tuple[Box, str]],
    instruction_type: str,
    iou_threshold: float = 0.1,
) -> Dict:
    """Evaluate a single ``label_output`` region."""
    label_bbox = convert_label_to_bbox(label)

    gt_text = extract_text_from_region(label_bbox, gt_blocks, iou_threshold)
    pred_text = extract_text_from_region(label_bbox, pred_blocks, iou_threshold)

    gt_overlapping = find_overlapping_blocks(label_bbox, gt_blocks, iou_threshold)
    pred_overlapping = find_overlapping_blocks(label_bbox, pred_blocks, iou_threshold)

    text_sim = compute_text_similarity(gt_text, pred_text)
    cdm_score = compute_cdm(pred_text, gt_text)
    bleu_score = compute_bleu(pred_text, gt_text)

    itype = instruction_type.lower()
    is_deletion = "deletion" in itype

    if is_deletion:
        pred_has_text = len(pred_text.strip()) > 0
        success = not pred_has_text
    else:
        success = text_sim >= 0.8

    gt_max_iou = gt_overlapping[0][2] if gt_overlapping else 0.0
    pred_max_iou = pred_overlapping[0][2] if pred_overlapping else 0.0

    return {
        "success": success,
        "gt_text": gt_text,
        "pred_text": pred_text,
        "text_similarity": text_sim,
        "cdm_score": cdm_score,
        "bleu_score": bleu_score,
        "gt_num_blocks": len(gt_overlapping),
        "pred_num_blocks": len(pred_overlapping),
        "gt_max_iou": gt_max_iou,
        "pred_max_iou": pred_max_iou,
        "label_bbox": list(label_bbox),
    }


# ---------------------------------------------------------------------------
# Recursive directory index (cached) + fuzzy file matching
# ---------------------------------------------------------------------------

_dir_index_cache: Dict[str, Dict[str, Path]] = {}


def _build_dir_index(json_dir: Path) -> Dict[str, Path]:
    """Recursively scan *json_dir* and cache a filename -> full_path mapping."""
    key = str(json_dir)
    if key in _dir_index_cache:
        return _dir_index_cache[key]

    index: Dict[str, Path] = {}
    for p in json_dir.rglob("*.json"):
        if p.is_file():
            index[p.name] = p
    _dir_index_cache[key] = index
    print(f"Built directory index for {json_dir}: {len(index)} JSON files found")
    return index


def _strip_id_prefix(filename: str, entry_id: Optional[int] = None) -> str:
    """Strip the numeric id prefix from an image_output filename."""
    import re
    if entry_id is not None:
        prefix = str(entry_id)
        if filename.startswith(prefix):
            rest = filename[len(prefix):]
            if rest.startswith("_"):
                return rest[1:]
            return rest
    m = re.match(r"^\d+_?", filename)
    if m:
        return filename[m.end():]
    return filename


def find_json_file(
    image_path: str,
    json_dir: Path,
    entry_id: Optional[int] = None,
    suffix_hint: Optional[str] = None,
) -> Optional[Path]:
    """Find the OCR JSON file corresponding to *image_path* in *json_dir* (recursive)."""
    file_index = _build_dir_index(json_dir)

    raw_name = Path(image_path).name
    original_name = _strip_id_prefix(raw_name, entry_id)
    base_names = list(dict.fromkeys([original_name, raw_name]))

    for base in base_names:
        patterns = [
            f"{base}_0.json",
            f"{base}_modified_0.json",
            f"{base}_merged_0_0.json",
            f"{base}_res.json",
            f"{base}.json",
            base.replace(".jpg", ".json").replace(".png", ".json"),
        ]

        if suffix_hint:
            preferred = [p for p in patterns if suffix_hint in p]
            others = [p for p in patterns if suffix_hint not in p]
            patterns = preferred + others

        for pattern in patterns:
            if pattern in file_index:
                return file_index[pattern]

    for base in base_names:
        stem = Path(base).stem
        if suffix_hint:
            hint_matches = [
                path for name, path in file_index.items()
                if stem in name and suffix_hint in name
            ]
            if hint_matches:
                return hint_matches[0]
        any_matches = [
            path for name, path in file_index.items() if stem in name
        ]
        if any_matches:
            return any_matches[0]

    return None


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

def evaluate_text_info(
    text_info_path: Path,
    gt_dir: Path,
    pred_dir: Path,
    iou_threshold: float = 0.1,
) -> Dict:
    """Evaluate all ``label_output`` regions in a text-info JSON."""
    with text_info_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    total_labels = 0
    successful_labels = 0

    deletion_results = []
    addition_results = []
    modify_results = []
    table_edit_results = []

    for entry in data:
        image_output = entry.get("image_output", "")
        image_input = entry.get("image_input", "")
        if not image_output and not image_input:
            continue

        entry_id = entry.get("id")

        gt_search_path = image_input or image_output
        pred_search_path = image_input or image_output

        gt_file = find_json_file(
            gt_search_path, gt_dir, entry_id, suffix_hint="_modified_0"
        )
        pred_file = find_json_file(
            pred_search_path, pred_dir, entry_id, suffix_hint="_modified_0"
        )

        if gt_file is None:
            gt_file = find_json_file(
                image_output, gt_dir, entry_id, suffix_hint="_modified_0"
            )
        if pred_file is None:
            pred_file = find_json_file(
                image_output, pred_dir, entry_id, suffix_hint="_modified_0"
            )

        if gt_file is None:
            print(f"Warning: GT file not found for {image_output} (id: {entry_id})")
            continue

        gt_blocks = load_ocr_blocks(gt_file)

        if pred_file is None:
            print(f"Info: Prediction file not found for {image_output} (id: {entry_id}), "
                  f"treating as empty prediction.")
            pred_blocks = []
        else:
            pred_blocks = load_ocr_blocks(pred_file)

        labels = entry.get("label_output", [])
        instruction_type = entry.get("instruction type") or entry.get("edit_type", "")

        for label in labels:
            total_labels += 1
            eval_result = evaluate_label_region(
                label, gt_blocks, pred_blocks, instruction_type, iou_threshold
            )
            eval_result.update({
                "entry_id": entry_id,
                "image_output": image_output,
                "instruction": entry.get("instruction", ""),
                "instruction_type": instruction_type,
                "gt_file": str(gt_file),
                "pred_file": str(pred_file) if pred_file else None,
                "pred_missing": pred_file is None,
            })
            results.append(eval_result)

            if eval_result["success"]:
                successful_labels += 1

            itype_lower = instruction_type.lower()
            if "deletion" in itype_lower:
                deletion_results.append(eval_result)
            elif "addition" in itype_lower:
                addition_results.append(eval_result)
            elif "table" in itype_lower:
                table_edit_results.append(eval_result)
            else:
                modify_results.append(eval_result)

    # ---- aggregate metrics ----
    def _rate(success_list):
        if not success_list:
            return 0.0
        return sum(1 for r in success_list if r["success"]) / len(success_list)

    def _avg(key):
        return sum(r[key] for r in results) / len(results) if results else 0.0

    pred_missing_count = sum(1 for r in results if r.get("pred_missing", False))

    return {
        "total_labels": total_labels,
        "overall_success_rate": successful_labels / total_labels if total_labels else 0.0,
        "deletion_count": len(deletion_results),
        "deletion_success_rate": _rate(deletion_results),
        "addition_count": len(addition_results),
        "addition_success_rate": _rate(addition_results),
        "modify_count": len(modify_results),
        "modify_success_rate": _rate(modify_results),
        "table_edit_count": len(table_edit_results),
        "table_edit_success_rate": _rate(table_edit_results),
        "avg_text_similarity": _avg("text_similarity"),
        "avg_cdm": _avg("cdm_score"),
        "avg_bleu": _avg("bleu_score"),
        "avg_gt_max_iou": _avg("gt_max_iou"),
        "avg_pred_max_iou": _avg("pred_max_iou"),
        "pred_missing_count": pred_missing_count,
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate label_output regions from a text-info JSON "
            "by comparing GT and prediction OCR results."
        )
    )
    parser.add_argument("--text_info", type=Path, required=True,
                        help="Path to text_info_all.json (or a similarly-shaped file).")
    parser.add_argument("--gt_dir", type=Path, required=True,
                        help="Directory containing GT OCR result JSON files (searched recursively).")
    parser.add_argument("--pred_dir", type=Path, required=True,
                        help="Directory containing prediction OCR result JSON files (searched recursively).")
    parser.add_argument("--iou_threshold", type=float, default=0.1,
                        help="Minimum IoU to consider overlap (default: 0.1).")
    parser.add_argument("--output", type=Path,
                        help="Optional: path to save detailed results JSON.")

    args = parser.parse_args()

    print(f"Loading text info from: {args.text_info}")
    print(f"GT directory: {args.gt_dir}")
    print(f"Prediction directory: {args.pred_dir}")
    print(f"IoU threshold: {args.iou_threshold}\n")

    results = evaluate_text_info(
        args.text_info, args.gt_dir, args.pred_dir, args.iou_threshold
    )

    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total label regions evaluated : {results['total_labels']}")
    print(f"Overall success rate          : {results['overall_success_rate']:.4f}\n")
    print(f"Text deletion tasks           : {results['deletion_count']}")
    print(f"  Deletion success rate       : {results['deletion_success_rate']:.4f}\n")
    print(f"Text addition tasks           : {results['addition_count']}")
    print(f"  Addition success rate       : {results['addition_success_rate']:.4f}\n")
    print(f"Text modify tasks             : {results['modify_count']}")
    print(f"  Modify success rate         : {results['modify_success_rate']:.4f}\n")
    print(f"Table structure edit tasks    : {results['table_edit_count']}")
    print(f"  Table edit success rate     : {results['table_edit_success_rate']:.4f}\n")
    print(f"Average text similarity       : {results['avg_text_similarity']:.4f}")
    print(f"Average CDM                   : {results['avg_cdm']:.4f}")
    print(f"Average BLEU                  : {results['avg_bleu']:.4f}")
    print(f"Average GT max IoU            : {results['avg_gt_max_iou']:.4f}")
    print(f"Average Pred max IoU          : {results['avg_pred_max_iou']:.4f}")
    print(f"Prediction files missing      : {results['pred_missing_count']}")
    print("=" * 60)

    if args.output:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
