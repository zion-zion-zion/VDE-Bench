#!/usr/bin/env python3
"""
Unified evaluation script for VDE-Bench.

Supports two modes:
  - Global mode (default): Directly compares ALL Pred OCR blocks with GT OCR blocks
    using greedy IoU matching.
  - Local mode: Uses annotation info JSON to identify edited regions, then only
    evaluates OCR blocks that overlap with those annotated regions.

Usage example (global mode)
---------------------------
python eval_unified.py \
    --mode global \
    --groups \
        text  /path/to/text_gt_ocr  /path/to/text_pred_ocr \
        table /path/to/table_gt_ocr  /path/to/table_pred_ocr \
    --iou_threshold 0.1 \
    --output eval_results.json

Usage example (local mode)
--------------------------
python eval_unified.py \
    --mode local \
    --groups \
        text  /path/to/text_info.json  /path/to/text_gt_ocr  /path/to/text_pred_ocr \
        table /path/to/table_info.json /path/to/table_gt_ocr  /path/to/table_pred_ocr \
    --iou_threshold 0.1 \
    --output eval_results.json
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from html.parser import HTMLParser
from math import exp, log
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
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


def _boxes_have_intersection(box_a: Box, box_b: Box) -> bool:
    """Check if two boxes have any intersection (non-zero overlap)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    return inter_x2 > inter_x1 and inter_y2 > inter_y1


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

# OCR block: (bbox, text, block_label, raw_content)
OCRBlock = Tuple[Box, str, str, str]  # (bbox, plain_text, label, raw_content)


def _strip_html_tags(html: str) -> str:
    """Return plain text from an HTML string (remove all tags)."""
    return re.sub(r"<[^>]+>", " ", html).strip()


def load_ocr_blocks(ocr_path: Path) -> List[OCRBlock]:
    """Load blocks from an OCR result JSON file.

    Returns list of (bbox, plain_text, block_label, raw_content).
    """
    try:
        with ocr_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    blocks: List[OCRBlock] = []
    for block in data.get("parsing_res_list", []):
        bbox = block.get("block_bbox")
        if not bbox or len(bbox) != 4:
            continue
        box = tuple(float(v) for v in bbox)
        raw_content = block.get("block_content") or ""
        label = block.get("block_label") or ""
        # For table blocks, derive plain text from HTML
        if label == "table" and raw_content.strip().startswith("<"):
            plain_text = _strip_html_tags(raw_content)
        else:
            plain_text = raw_content
        blocks.append((box, plain_text, label, raw_content))
    return blocks


# ---------------------------------------------------------------------------
# Block matching: greedy IoU-based matching between GT and Pred blocks
# ---------------------------------------------------------------------------


def match_blocks_greedy(
    gt_blocks: List[OCRBlock],
    pred_blocks: List[OCRBlock],
    iou_threshold: float = 0.1,
) -> List[Tuple[int, int, float]]:
    """Match GT blocks to Pred blocks using greedy max-IoU strategy.

    Returns list of (gt_idx, pred_idx, iou) for matched pairs.
    Each GT block is matched to at most one Pred block and vice versa.
    Only pairs with IoU >= iou_threshold are considered.
    """
    if not gt_blocks or not pred_blocks:
        return []

    # Compute all pairwise IoUs
    pairs = []
    for gi, (gt_box, _, _, _) in enumerate(gt_blocks):
        for pi, (pred_box, _, _, _) in enumerate(pred_blocks):
            iou = compute_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                pairs.append((gi, pi, iou))

    # Sort by IoU descending (greedy)
    pairs.sort(key=lambda x: x[2], reverse=True)

    matched_gt = set()
    matched_pred = set()
    result = []

    for gi, pi, iou in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        result.append((gi, pi, iou))
        matched_gt.add(gi)
        matched_pred.add(pi)

    return result


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
    """Character Distance Metric: 1 - normalised edit distance."""
    if not candidate and not reference:
        return 1.0
    max_len = max(len(candidate), len(reference))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - levenshtein_distance(candidate, reference) / max_len)


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Lightweight BLEU-4."""
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
# TEDS (Tree Edit Distance-based Similarity)
# ---------------------------------------------------------------------------


class _TableHTMLParser(HTMLParser):
    """Parse an HTML table string into a simple tree structure."""

    def __init__(self) -> None:
        super().__init__()
        self._root: Optional[Dict[str, Any]] = None
        self._stack: List[Dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        node: Dict[str, Any] = {
            "tag": tag,
            "attrs": dict(attrs),
            "children": [],
            "text": "",
        }
        if self._stack:
            self._stack[-1]["children"].append(node)
        else:
            self._root = node
        self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if self._stack:
            self._stack[-1]["text"] += data

    def get_tree(self) -> Optional[Dict[str, Any]]:
        return self._root


def _parse_html_table(html: str) -> Optional[Dict[str, Any]]:
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    return parser.get_tree()


def _tree_size(node: Optional[Dict[str, Any]]) -> int:
    if node is None:
        return 0
    return 1 + sum(_tree_size(c) for c in node.get("children", []))


def _node_match_cost(n1: Dict[str, Any], n2: Dict[str, Any]) -> float:
    if n1["tag"] != n2["tag"]:
        return 1.0
    t1 = n1.get("text", "").strip()
    t2 = n2.get("text", "").strip()
    if t1 == t2:
        return 0.0
    if not t1 and not t2:
        return 0.0
    max_len = max(len(t1), len(t2))
    if max_len == 0:
        return 0.0
    dist = levenshtein_distance(t1, t2)
    return dist / max_len


def _simple_tree_edit_distance(
    tree1: Optional[Dict[str, Any]],
    tree2: Optional[Dict[str, Any]],
) -> float:
    if tree1 is None and tree2 is None:
        return 0.0
    if tree1 is None:
        return float(_tree_size(tree2))
    if tree2 is None:
        return float(_tree_size(tree1))

    root_cost = _node_match_cost(tree1, tree2)

    c1 = tree1.get("children", [])
    c2 = tree2.get("children", [])
    n, m = len(c1), len(c2)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _tree_size(c1[i - 1])
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _tree_size(c2[j - 1])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_del = dp[i - 1][j] + _tree_size(c1[i - 1])
            cost_ins = dp[i][j - 1] + _tree_size(c2[j - 1])
            cost_match = dp[i - 1][j - 1] + _simple_tree_edit_distance(c1[i - 1], c2[j - 1])
            dp[i][j] = min(cost_del, cost_ins, cost_match)

    return root_cost + dp[n][m]


def compute_teds(pred_html: str, gt_html: str) -> float:
    """Compute TEDS between two HTML table strings. Returns [0, 1]."""
    if not pred_html and not gt_html:
        return 1.0
    if not pred_html or not gt_html:
        return 0.0

    tree_pred = _parse_html_table(pred_html)
    tree_gt = _parse_html_table(gt_html)

    if tree_pred is None and tree_gt is None:
        return 1.0
    if tree_pred is None or tree_gt is None:
        return 0.0

    ted = _simple_tree_edit_distance(tree_pred, tree_gt)
    size_pred = _tree_size(tree_pred)
    size_gt = _tree_size(tree_gt)
    max_size = max(size_pred, size_gt)
    if max_size == 0:
        return 1.0
    return max(0.0, 1.0 - ted / max_size)


# ---------------------------------------------------------------------------
# Evaluate a single matched block pair
# ---------------------------------------------------------------------------


def evaluate_block_pair(
    gt_block: OCRBlock,
    pred_block: OCRBlock,
    iou: float,
) -> Dict:
    """Evaluate metrics for a matched (GT block, Pred block) pair."""
    gt_box, gt_text, gt_label, gt_raw = gt_block
    pred_box, pred_text, pred_label, pred_raw = pred_block

    text_sim = compute_text_similarity(gt_text, pred_text)
    cdm_score = compute_cdm(pred_text, gt_text)
    bleu_score = compute_bleu(pred_text, gt_text)

    # TEDS: only for table blocks
    gt_html = gt_raw if gt_label == "table" and gt_raw.strip().startswith("<") else ""
    pred_html = pred_raw if pred_label == "table" and pred_raw.strip().startswith("<") else ""
    has_table = bool(gt_html) or bool(pred_html)
    teds_score = compute_teds(pred_html, gt_html) if has_table else None

    return {
        "iou": iou,
        "gt_text": gt_text,
        "pred_text": pred_text,
        "text_similarity": text_sim,
        "cdm_score": cdm_score,
        "bleu_score": bleu_score,
        "teds_score": teds_score,
        "has_table": has_table,
        "gt_label": gt_label,
        "pred_label": pred_label,
        "gt_bbox": list(gt_box),
        "pred_bbox": list(pred_box),
    }


# ---------------------------------------------------------------------------
# Annotation helpers (for local mode)
# ---------------------------------------------------------------------------


def load_annotation_info(info_json: Path) -> List[Dict]:
    """Load annotation info JSON file."""
    with info_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _convert_label_to_pixel_bbox(label: Dict, img_w: int, img_h: int) -> Box:
    """Convert a percentage-based label bbox to pixel coordinates (x1, y1, x2, y2)."""
    x_pct = label["x"]
    y_pct = label["y"]
    w_pct = label["width"]
    h_pct = label["height"]

    x1 = x_pct / 100.0 * img_w
    y1 = y_pct / 100.0 * img_h
    x2 = (x_pct + w_pct) / 100.0 * img_w
    y2 = (y_pct + h_pct) / 100.0 * img_h

    return (x1, y1, x2, y2)


def get_annotation_bboxes(
    entry: Dict, img_w: Optional[int] = None, img_h: Optional[int] = None
) -> List[Box]:
    """Extract pixel-coordinate bboxes from an annotation entry's label_output.

    If img_w/img_h not provided, tries to get from label_output items
    (original_width, original_height).
    """
    labels = entry.get("label_output", [])
    if not labels:
        return []

    # Try to get image dimensions from label items if not provided
    if img_w is None or img_h is None:
        for lbl in labels:
            if "original_width" in lbl and "original_height" in lbl:
                img_w = lbl["original_width"]
                img_h = lbl["original_height"]
                break

    if img_w is None or img_h is None:
        return []

    bboxes = []
    for lbl in labels:
        if "x" in lbl and "y" in lbl and "width" in lbl and "height" in lbl:
            bbox = _convert_label_to_pixel_bbox(lbl, img_w, img_h)
            bboxes.append(bbox)
    return bboxes


def filter_blocks_by_annotation(
    blocks: List[OCRBlock], annotation_bboxes: List[Box]
) -> List[OCRBlock]:
    """Filter OCR blocks to only those that intersect with any annotation bbox."""
    if not annotation_bboxes:
        return blocks

    filtered = []
    for block in blocks:
        block_box = block[0]  # (x1, y1, x2, y2)
        for ann_box in annotation_bboxes:
            if _boxes_have_intersection(block_box, ann_box):
                filtered.append(block)
                break
    return filtered


def build_annotation_lookup(entries: List[Dict]) -> Dict[str, Dict]:
    """Build a lookup from image_output basename (stem) to annotation entry."""
    lookup: Dict[str, Dict] = {}
    for entry in entries:
        if "image_output" not in entry:
            continue
        out_path = entry["image_output"]
        out_basename = os.path.basename(out_path)
        out_stem = os.path.splitext(out_basename)[0]
        lookup[out_basename] = entry
        lookup[out_stem] = entry
    return lookup


def _find_annotation_for_ocr_file(
    ocr_filename: str, annotation_lookup: Dict[str, Dict]
) -> Optional[Dict]:
    """Find annotation entry for an OCR filename."""
    name = ocr_filename
    if name.endswith(".json"):
        name = name[:-5]

    if name in annotation_lookup:
        return annotation_lookup[name]

    base_one_suffix = re.sub(r"_\d+$", "", name)
    if base_one_suffix in annotation_lookup:
        return annotation_lookup[base_one_suffix]
    for ext in [".png", ".jpg", ".jpeg"]:
        if base_one_suffix + ext in annotation_lookup:
            return annotation_lookup[base_one_suffix + ext]

    base = re.sub(r"_(merged|modified|res)(_\d+)*$", "", name)
    if base in annotation_lookup:
        return annotation_lookup[base]
    for ext in [".png", ".jpg", ".jpeg"]:
        if base + ext in annotation_lookup:
            return annotation_lookup[base + ext]

    base_no_num = re.sub(r"_\d+$", "", base)
    if base_no_num != base:
        if base_no_num in annotation_lookup:
            return annotation_lookup[base_no_num]
        for ext in [".png", ".jpg", ".jpeg"]:
            if base_no_num + ext in annotation_lookup:
                return annotation_lookup[base_no_num + ext]

    match = re.match(r"^(\d+)_(.+)$", name)
    if match:
        id_prefix = match.group(1)
        for key in annotation_lookup:
            if key.startswith(id_prefix + "_"):
                return annotation_lookup[key]

    return None


# ---------------------------------------------------------------------------
# File matching between GT and Pred directories
# ---------------------------------------------------------------------------


def build_file_index(directory: Path) -> Dict[str, Path]:
    """Build a filename -> path index for all JSON files in directory (recursive)."""
    index: Dict[str, Path] = {}
    for p in directory.rglob("*.json"):
        if p.is_file():
            index[p.name] = p
    return index


def _extract_base_stem(filename: str) -> str:
    """Extract base stem from OCR filename for fuzzy matching."""
    name = filename
    if name.endswith(".json"):
        name = name[:-5]
    stripped = re.sub(r"_(merged|modified|res)(_\d+)*$", "", name)
    if stripped != name:
        return stripped
    stripped = re.sub(r"_(\d+)$", "", name)
    return stripped


def match_files(gt_dir: Path, pred_dir: Path) -> List[Tuple[Path, Path, str]]:
    """Match GT and Pred OCR files by filename."""
    gt_index = build_file_index(gt_dir)
    pred_index = build_file_index(pred_dir)

    print(f"  GT directory: {len(gt_index)} JSON files")
    print(f"  Pred directory: {len(pred_index)} JSON files")

    matched = []
    for filename, gt_path in gt_index.items():
        if filename in pred_index:
            matched.append((gt_path, pred_index[filename], filename))

    # Stem-based matching for files with different suffixes
    gt_stems = {}
    for filename, gt_path in gt_index.items():
        stem = _extract_base_stem(filename)
        if stem not in gt_stems:
            gt_stems[stem] = (filename, gt_path)

    pred_stems = {}
    for filename, pred_path in pred_index.items():
        stem = _extract_base_stem(filename)
        if stem not in pred_stems:
            pred_stems[stem] = (filename, pred_path)

    matched_gt_files = {m[0] for m in matched}
    matched_pred_files = {m[1] for m in matched}

    for stem, (gt_fname, gt_path) in gt_stems.items():
        if gt_path in matched_gt_files:
            continue
        if stem in pred_stems:
            pred_fname, pred_path = pred_stems[stem]
            if pred_path in matched_pred_files:
                continue
            matched.append((gt_path, pred_path, f"{gt_fname} <-> {pred_fname}"))
            matched_gt_files.add(gt_path)
            matched_pred_files.add(pred_path)

    print(f"  Matched file pairs: {len(matched)}")
    return matched


# ---------------------------------------------------------------------------
# Core evaluation for one group - GLOBAL mode
# ---------------------------------------------------------------------------


def evaluate_group(
    gt_dir: Path,
    pred_dir: Path,
    category: str,
    iou_threshold: float = 0.1,
) -> List[Dict]:
    """Evaluate one data group by matching files and blocks directly (global mode)."""
    file_pairs = match_files(gt_dir, pred_dir)

    all_results: List[Dict] = []

    for gt_path, pred_path, match_name in file_pairs:
        gt_blocks = load_ocr_blocks(gt_path)
        pred_blocks = load_ocr_blocks(pred_path)

        if not gt_blocks and not pred_blocks:
            continue

        matches = match_blocks_greedy(gt_blocks, pred_blocks, iou_threshold)

        for gt_idx, pred_idx, iou in matches:
            res = evaluate_block_pair(gt_blocks[gt_idx], pred_blocks[pred_idx], iou)
            res.update({
                "category": category,
                "gt_file": str(gt_path),
                "pred_file": str(pred_path),
                "match_name": match_name,
                "gt_block_idx": gt_idx,
                "pred_block_idx": pred_idx,
            })
            all_results.append(res)

        # Track unmatched GT blocks (missed by prediction)
        matched_gt_indices = {m[0] for m in matches}
        for gi, gt_block in enumerate(gt_blocks):
            if gi not in matched_gt_indices:
                gt_box, gt_text, gt_label, _ = gt_block
                all_results.append({
                    "iou": 0.0,
                    "gt_text": gt_text,
                    "pred_text": "",
                    "text_similarity": 0.0,
                    "cdm_score": 0.0,
                    "bleu_score": 0.0,
                    "teds_score": None,
                    "has_table": gt_label == "table",
                    "gt_label": gt_label,
                    "pred_label": "",
                    "gt_bbox": list(gt_box),
                    "pred_bbox": [],
                    "category": category,
                    "gt_file": str(gt_path),
                    "pred_file": str(pred_path),
                    "match_name": match_name,
                    "gt_block_idx": gi,
                    "pred_block_idx": -1,
                    "unmatched": "gt",
                })

        # Track unmatched Pred blocks (extra predictions)
        matched_pred_indices = {m[1] for m in matches}
        for pi, pred_block in enumerate(pred_blocks):
            if pi not in matched_pred_indices:
                pred_box, pred_text, pred_label, _ = pred_block
                all_results.append({
                    "iou": 0.0,
                    "gt_text": "",
                    "pred_text": pred_text,
                    "text_similarity": 0.0,
                    "cdm_score": 0.0,
                    "bleu_score": 0.0,
                    "teds_score": None,
                    "has_table": pred_label == "table",
                    "gt_label": "",
                    "pred_label": pred_label,
                    "gt_bbox": [],
                    "pred_bbox": list(pred_box),
                    "category": category,
                    "gt_file": str(gt_path),
                    "pred_file": str(pred_path),
                    "match_name": match_name,
                    "gt_block_idx": -1,
                    "pred_block_idx": pi,
                    "unmatched": "pred",
                })

    return all_results


# ---------------------------------------------------------------------------
# Core evaluation for one group - LOCAL mode
# ---------------------------------------------------------------------------


def evaluate_group_local(
    info_json: Path,
    gt_dir: Path,
    pred_dir: Path,
    category: str,
    iou_threshold: float = 0.1,
) -> List[Dict]:
    """Evaluate one data group in local mode (only blocks overlapping annotation)."""
    print(f"  Loading annotation info: {info_json}")
    entries = load_annotation_info(info_json)
    annotation_lookup = build_annotation_lookup(entries)
    print(f"  Loaded {len(entries)} annotation entries")

    file_pairs = match_files(gt_dir, pred_dir)

    all_results: List[Dict] = []
    files_with_annotation = 0
    files_without_annotation = 0

    for gt_path, pred_path, match_name in file_pairs:
        ann_entry = _find_annotation_for_ocr_file(gt_path.name, annotation_lookup)
        if ann_entry is None:
            files_without_annotation += 1
            continue

        files_with_annotation += 1

        ann_bboxes = get_annotation_bboxes(ann_entry)
        if not ann_bboxes:
            continue

        gt_blocks = load_ocr_blocks(gt_path)
        pred_blocks = load_ocr_blocks(pred_path)

        if not gt_blocks and not pred_blocks:
            continue

        gt_blocks_filtered = filter_blocks_by_annotation(gt_blocks, ann_bboxes)
        pred_blocks_filtered = filter_blocks_by_annotation(pred_blocks, ann_bboxes)

        if not gt_blocks_filtered and not pred_blocks_filtered:
            continue

        matches = match_blocks_greedy(gt_blocks_filtered, pred_blocks_filtered, iou_threshold)

        for gt_idx, pred_idx, iou in matches:
            res = evaluate_block_pair(
                gt_blocks_filtered[gt_idx], pred_blocks_filtered[pred_idx], iou
            )
            res.update({
                "category": category,
                "gt_file": str(gt_path),
                "pred_file": str(pred_path),
                "match_name": match_name,
                "gt_block_idx": gt_idx,
                "pred_block_idx": pred_idx,
                "mode": "local",
            })
            all_results.append(res)

        matched_gt_indices = {m[0] for m in matches}
        for gi, gt_block in enumerate(gt_blocks_filtered):
            if gi not in matched_gt_indices:
                gt_box, gt_text, gt_label, _ = gt_block
                all_results.append({
                    "iou": 0.0,
                    "gt_text": gt_text,
                    "pred_text": "",
                    "text_similarity": 0.0,
                    "cdm_score": 0.0,
                    "bleu_score": 0.0,
                    "teds_score": None,
                    "has_table": gt_label == "table",
                    "gt_label": gt_label,
                    "pred_label": "",
                    "gt_bbox": list(gt_box),
                    "pred_bbox": [],
                    "category": category,
                    "gt_file": str(gt_path),
                    "pred_file": str(pred_path),
                    "match_name": match_name,
                    "gt_block_idx": gi,
                    "pred_block_idx": -1,
                    "unmatched": "gt",
                    "mode": "local",
                })

        matched_pred_indices = {m[1] for m in matches}
        for pi, pred_block in enumerate(pred_blocks_filtered):
            if pi not in matched_pred_indices:
                pred_box, pred_text, pred_label, _ = pred_block
                all_results.append({
                    "iou": 0.0,
                    "gt_text": "",
                    "pred_text": pred_text,
                    "text_similarity": 0.0,
                    "cdm_score": 0.0,
                    "bleu_score": 0.0,
                    "teds_score": None,
                    "has_table": pred_label == "table",
                    "gt_label": "",
                    "pred_label": pred_label,
                    "gt_bbox": [],
                    "pred_bbox": list(pred_box),
                    "category": category,
                    "gt_file": str(gt_path),
                    "pred_file": str(pred_path),
                    "match_name": match_name,
                    "gt_block_idx": -1,
                    "pred_block_idx": pi,
                    "unmatched": "pred",
                    "mode": "local",
                })

    print(f"  Files with annotation: {files_with_annotation}, "
          f"without annotation (skipped): {files_without_annotation}")

    return all_results


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------


def _safe_avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(results: List[Dict]) -> Dict:
    """Compute aggregate metrics for a list of per-block results."""
    if not results:
        return {
            "total_blocks": 0,
            "gt_blocks": 0,
            "matched_blocks": 0,
            "unmatched_gt": 0,
            "unmatched_pred": 0,
            "avg_iou": 0.0,
            "avg_text_sim": 0.0,
            "avg_cdm": 0.0,
            "avg_bleu": 0.0,
            "avg_teds": None,
            "teds_count": 0,
        }

    matched = [r for r in results if r.get("iou", 0) > 0]
    unmatched_gt = [r for r in results if r.get("unmatched") == "gt"]
    unmatched_pred = [r for r in results if r.get("unmatched") == "pred"]

    gt_based = matched + unmatched_gt
    iou_values = [r["iou"] for r in gt_based]
    sim_values = [r["text_similarity"] for r in matched]
    cdm_values = [r["cdm_score"] for r in matched]
    bleu_values = [r["bleu_score"] for r in matched]
    teds_values = [r["teds_score"] for r in matched if r.get("teds_score") is not None]

    return {
        "total_blocks": len(results),
        "gt_blocks": len(gt_based),
        "matched_blocks": len(matched),
        "unmatched_gt": len(unmatched_gt),
        "unmatched_pred": len(unmatched_pred),
        "avg_iou": _safe_avg(iou_values),
        "avg_text_sim": _safe_avg(sim_values),
        "avg_cdm": _safe_avg(cdm_values),
        "avg_bleu": _safe_avg(bleu_values),
        "avg_teds": _safe_avg(teds_values) if teds_values else None,
        "teds_count": len(teds_values),
    }


def _print_section(title: str, agg: Dict) -> None:
    teds_str = (
        f", TEDS={agg['avg_teds']:.4f}({agg['teds_count']})"
        if agg.get("avg_teds") is not None
        else ""
    )
    gt_blocks = agg["gt_blocks"]
    matched = agg["matched_blocks"]
    print(f"  {title}: gt_blocks={gt_blocks}, matched={matched}, "
          f"unmatched_gt={agg['unmatched_gt']}, unmatched_pred={agg['unmatched_pred']}")
    print(f"    IoU={agg['avg_iou']:.4f}(/{gt_blocks}), "
          f"Sim={agg['avg_text_sim']:.4f}, "
          f"CDM={agg['avg_cdm']:.4f}, "
          f"BLEU={agg['avg_bleu']:.4f}"
          f"{teds_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified VDE-Bench evaluation (Pred vs GT block matching).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Global mode (triples: CATEGORY GT_DIR PRED_DIR)
  python eval_unified.py --mode global \\
      --groups \\
          text  /path/to/text_gt_ocr  /path/to/text_pred_ocr \\
          table /path/to/table_gt_ocr  /path/to/table_pred_ocr \\
      --iou_threshold 0.1 --output results.json

  # Local mode (quadruples: CATEGORY INFO_JSON GT_DIR PRED_DIR)
  python eval_unified.py --mode local \\
      --groups \\
          text  /path/to/text_info.json  /path/to/text_gt_ocr  /path/to/text_pred_ocr \\
          table /path/to/table_info.json /path/to/table_gt_ocr  /path/to/table_pred_ocr \\
      --iou_threshold 0.1 --output results.json
""",
    )
    parser.add_argument(
        "--mode", choices=["global", "local"], default="global",
        help="Evaluation mode: 'global' compares all blocks; "
             "'local' only evaluates blocks overlapping with annotated regions.",
    )
    parser.add_argument(
        "--groups", nargs="+", required=True,
        help="Global mode: repeating triples CATEGORY GT_DIR PRED_DIR. "
             "Local mode: repeating quadruples CATEGORY INFO_JSON GT_DIR PRED_DIR.",
    )
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, help="Save detailed JSON results")

    args = parser.parse_args()

    is_local = args.mode == "local"

    tokens = args.groups
    if is_local:
        if len(tokens) % 4 != 0:
            parser.error("In local mode, --groups must be multiples of 4: "
                         "CATEGORY INFO_JSON GT_DIR PRED_DIR")
        groups = []
        for i in range(0, len(tokens), 4):
            groups.append({
                "category": tokens[i],
                "info_json": Path(tokens[i + 1]),
                "gt_dir": Path(tokens[i + 2]),
                "pred_dir": Path(tokens[i + 3]),
            })
    else:
        if len(tokens) % 3 != 0:
            parser.error("In global mode, --groups must be multiples of 3: "
                         "CATEGORY GT_DIR PRED_DIR")
        groups = []
        for i in range(0, len(tokens), 3):
            groups.append({
                "category": tokens[i],
                "gt_dir": Path(tokens[i + 1]),
                "pred_dir": Path(tokens[i + 2]),
            })

    all_results: List[Dict] = []
    for g in groups:
        print(f"\n{'='*60}")
        print(f"Evaluating category: {g['category']} (mode={args.mode})")
        print(f"  gt_dir  : {g['gt_dir']}")
        print(f"  pred_dir: {g['pred_dir']}")
        if is_local:
            print(f"  info_json: {g['info_json']}")
        print(f"{'='*60}")

        if is_local:
            results = evaluate_group_local(
                g["info_json"], g["gt_dir"], g["pred_dir"],
                g["category"], args.iou_threshold,
            )
        else:
            results = evaluate_group(
                g["gt_dir"], g["pred_dir"],
                g["category"], args.iou_threshold,
            )
        all_results.extend(results)
        matched_count = sum(1 for r in results if r.get("iou", 0) > 0)
        print(f"  -> {len(results)} total block entries, {matched_count} matched pairs")

    if not all_results:
        print("\nNo blocks were evaluated. Check your paths.")
        return

    by_category: Dict[str, List[Dict]] = defaultdict(list)
    by_block_label: Dict[str, List[Dict]] = defaultdict(list)

    for r in all_results:
        by_category[r["category"]].append(r)
        label = r.get("gt_label") or r.get("pred_label") or "unknown"
        by_block_label[label].append(r)

    overall = _aggregate(all_results)

    print(f"\n{'='*70}")
    print(f"OVERALL (mode={args.mode})")
    print(f"{'='*70}")
    _print_section("All", overall)

    print(f"\n{'-'*70}")
    print("BY CATEGORY")
    print(f"{'-'*70}")
    for cat in sorted(by_category):
        _print_section(cat, _aggregate(by_category[cat]))

    print(f"\n{'-'*70}")
    print("BY BLOCK LABEL")
    print(f"{'-'*70}")
    for label in sorted(by_block_label):
        _print_section(label, _aggregate(by_block_label[label]))

    print(f"\n{'='*70}")

    if args.output:
        summary = {
            "mode": args.mode,
            "overall": overall,
            "by_category": {k: _aggregate(v) for k, v in sorted(by_category.items())},
            "by_block_label": {k: _aggregate(v) for k, v in sorted(by_block_label.items())},
            "details": all_results,
        }
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
