#!/usr/bin/env python3
"""
Image Quality Evaluation Script (Unified).

Evaluate images in `eval_img_dir` against ground-truth images in `gt_img_dir`
using five metrics: PSNR, SSIM, LPIPS, FID, and CLIP Score.

Supports two modes:
  - Global mode (default): compare full images.
  - Local mode (--annotation_json per group): crop annotated regions and compare patches.

Supports multiple evaluation groups (e.g. text + table) via --groups parameter.

Usage (multi-group, local mode):
    python evaluate.py \
        --groups \
            text  /path/to/text_info_all.json  /path/to/text_gt_images  /path/to/text_eval_images \
            table /path/to/merged_with_label_output.json  /path/to/table_gt_images  /path/to/table_eval_images \
        [--padding 0] \
        [--device cuda] \
        [--batch_size 32] \
        [--clip_model openai/clip-vit-base-patch32] \
        [--output_json results.json]

Usage (single-group, global mode - backward compatible):
    python evaluate.py \
        --gt_img_dir /path/to/gt_images \
        --eval_img_dir /path/to/eval_images \
        [--annotation_json /path/to/annotation.json] \
        [--output_json results.json]

Image matching strategy:
    Images are matched by filename. Only files present in BOTH directories
    are evaluated. Supported formats: .png, .jpg, .jpeg, .bmp, .tiff, .webp
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from metrics import (
    compute_psnr,
    compute_ssim,
    LPIPSMetric,
    FIDMetric,
    CLIPScoreMetric,
)

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# Minimum patch size required by LPIPS (AlexNet backbone has multiple pooling layers)
MIN_PATCH_SIZE = 64


def _collect_images_recursive(directory: str) -> Dict[str, str]:
    """Recursively collect all image files in a directory tree."""
    files = {}
    for root, _dirs, filenames in os.walk(directory):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                if f not in files:
                    files[f] = os.path.join(root, f)
    return files


def find_matched_pairs(gt_dir: str, eval_dir: str) -> List[Tuple[str, str]]:
    """Find image pairs that exist in both directories (matched by stem name)."""
    gt_files = _collect_images_recursive(gt_dir)
    eval_files = _collect_images_recursive(eval_dir)

    gt_stems = {os.path.splitext(f)[0]: f for f in gt_files}
    eval_stems = {os.path.splitext(f)[0]: f for f in eval_files}
    common_stems = sorted(set(gt_stems.keys()) & set(eval_stems.keys()))

    if not common_stems:
        return []

    return [(gt_files[gt_stems[s]], eval_files[eval_stems[s]]) for s in common_stems]


def load_image(path: str) -> np.ndarray:
    """Load an image as a numpy uint8 RGB array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def resize_to_match(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resize pred to match gt dimensions if they differ."""
    if gt.shape[:2] != pred.shape[:2]:
        h, w = gt.shape[:2]
        pred_pil = Image.fromarray(pred).resize((w, h), Image.LANCZOS)
        pred = np.array(pred_pil)
    return gt, pred


# ---------------------------------------------------------------------------
# Local mode helpers
# ---------------------------------------------------------------------------

def _strip_id_prefix(name: str) -> str:
    """Strip leading numeric ID prefix (e.g. '49073_foo' -> 'foo')."""
    m = re.match(r'^\d+_(.+)$', name)
    return m.group(1) if m else name


def load_annotations(annotation_json: str) -> Tuple[Dict, Dict]:
    """Load annotation JSON and build lookups keyed by image_output basename."""
    with open(annotation_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    bbox_lookup = {}
    entry_lookup = {}
    for entry in data:
        if "image_output" not in entry or "label_output" not in entry:
            continue
        out_basename = os.path.basename(entry["image_output"])
        bbox_lookup[out_basename] = entry["label_output"]
        entry_lookup[out_basename] = entry
    return bbox_lookup, entry_lookup


def percent_bbox_to_pixel(bbox: Dict, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
    """Convert a percentage-based bbox to pixel coordinates, clamped to image."""
    x_pct = bbox["x"]
    y_pct = bbox["y"]
    w_pct = bbox["width"]
    h_pct = bbox["height"]

    x1 = int(x_pct / 100.0 * img_w)
    y1 = int(y_pct / 100.0 * img_h)
    x2 = int((x_pct + w_pct) / 100.0 * img_w)
    y2 = int((y_pct + h_pct) / 100.0 * img_h)

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))

    return x1, y1, x2, y2


def merge_bboxes(bboxes: List[Dict], img_h: int, img_w: int,
                 padding: int = 0) -> Tuple[int, int, int, int]:
    """Merge multiple bboxes into a single enclosing box (with optional padding)."""
    all_x1, all_y1, all_x2, all_y2 = [], [], [], []
    for bbox in bboxes:
        x1, y1, x2, y2 = percent_bbox_to_pixel(bbox, img_h, img_w)
        all_x1.append(x1)
        all_y1.append(y1)
        all_x2.append(x2)
        all_y2.append(y2)

    merged_x1 = max(0, min(all_x1) - padding)
    merged_y1 = max(0, min(all_y1) - padding)
    merged_x2 = min(img_w, max(all_x2) + padding)
    merged_y2 = min(img_h, max(all_y2) + padding)

    return merged_x1, merged_y1, merged_x2, merged_y2


def crop_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return img[y1:y2, x1:x2].copy()


def ensure_min_patch_size(gt_img: np.ndarray, pred_img: np.ndarray,
                          x1: int, y1: int, x2: int, y2: int,
                          img_h: int, img_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure cropped patches meet the min size required by LPIPS."""
    gt_patch = crop_region(gt_img, x1, y1, x2, y2)
    pred_patch = crop_region(pred_img, x1, y1, x2, y2)

    patch_h, patch_w = gt_patch.shape[:2]
    if patch_h < MIN_PATCH_SIZE or patch_w < MIN_PATCH_SIZE:
        need_h = max(MIN_PATCH_SIZE - (y2 - y1), 0)
        need_w = max(MIN_PATCH_SIZE - (x2 - x1), 0)

        new_y1 = max(0, y1 - need_h // 2)
        new_y2 = min(img_h, y2 + (need_h - need_h // 2))
        new_x1 = max(0, x1 - need_w // 2)
        new_x2 = min(img_w, x2 + (need_w - need_w // 2))

        if new_y2 - new_y1 < MIN_PATCH_SIZE:
            if new_y1 == 0:
                new_y2 = min(img_h, MIN_PATCH_SIZE)
            else:
                new_y1 = max(0, new_y2 - MIN_PATCH_SIZE)
        if new_x2 - new_x1 < MIN_PATCH_SIZE:
            if new_x1 == 0:
                new_x2 = min(img_w, MIN_PATCH_SIZE)
            else:
                new_x1 = max(0, new_x2 - MIN_PATCH_SIZE)

        gt_patch = crop_region(gt_img, new_x1, new_y1, new_x2, new_y2)
        pred_patch = crop_region(pred_img, new_x1, new_y1, new_x2, new_y2)

        ph, pw = gt_patch.shape[:2]
        if ph < MIN_PATCH_SIZE or pw < MIN_PATCH_SIZE:
            target_h = max(ph, MIN_PATCH_SIZE)
            target_w = max(pw, MIN_PATCH_SIZE)
            gt_patch = np.array(Image.fromarray(gt_patch).resize(
                (target_w, target_h), Image.LANCZOS))
            pred_patch = np.array(Image.fromarray(pred_patch).resize(
                (target_w, target_h), Image.LANCZOS))

    return gt_patch, pred_patch


def build_annotation_index(bbox_lookup: Dict, entry_lookup: Dict) -> Dict:
    """Build multi-way lookup indices for tolerant filename matching."""
    ann_by_gt_name = {}
    ann_by_gt_stem = {}
    ann_by_input_stem = {}

    for out_basename, bboxes in bbox_lookup.items():
        entry = entry_lookup[out_basename]
        ann_by_gt_name[out_basename] = (out_basename, bboxes)
        out_stem = os.path.splitext(out_basename)[0]
        ann_by_gt_stem[out_stem] = (out_basename, bboxes)

        in_basename = os.path.basename(entry.get("image_input", ""))
        in_stem = os.path.splitext(in_basename)[0]
        ann_by_input_stem[in_stem] = (out_basename, bboxes)

    return {
        'by_gt_name': ann_by_gt_name,
        'by_gt_stem': ann_by_gt_stem,
        'by_input_stem': ann_by_input_stem,
    }


def find_annotation_for_pair(gt_filename: str, ann_index: Dict) -> Optional[Tuple[str, List]]:
    """Find annotation entry for a matched GT filename (multi-strategy fallback)."""
    gt_stem = os.path.splitext(gt_filename)[0]
    stripped_gt_stem = _strip_id_prefix(gt_stem)

    return (ann_index['by_gt_name'].get(gt_filename)
            or ann_index['by_gt_stem'].get(gt_stem)
            or ann_index['by_input_stem'].get(stripped_gt_stem)
            or ann_index['by_input_stem'].get(gt_stem))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_metrics(records: List[Dict]) -> Dict:
    """Compute aggregate metrics for a list of per-image records."""
    if not records:
        return {
            "num_images": 0,
            "PSNR_mean": 0.0, "PSNR_std": 0.0,
            "SSIM_mean": 0.0, "SSIM_std": 0.0,
            "LPIPS_mean": 0.0, "LPIPS_std": 0.0,
            "CLIP_Score_mean": 0.0, "CLIP_Score_std": 0.0,
        }

    psnr_list = [r["PSNR"] for r in records]
    ssim_list = [r["SSIM"] for r in records]
    lpips_list = [r["LPIPS"] for r in records]
    clip_list = [r["CLIP_Score"] for r in records]

    # Filter out capped PSNR values (100.0 = identical images) for clean stats
    finite_psnr = [v for v in psnr_list if v < 100.0]
    if not finite_psnr:
        finite_psnr = psnr_list

    return {
        "num_images": len(records),
        "PSNR_mean": round(float(np.mean(finite_psnr)), 4),
        "PSNR_std": round(float(np.std(finite_psnr)), 4),
        "SSIM_mean": round(float(np.mean(ssim_list)), 4),
        "SSIM_std": round(float(np.std(ssim_list)), 4),
        "LPIPS_mean": round(float(np.mean(lpips_list)), 4),
        "LPIPS_std": round(float(np.std(lpips_list)), 4),
        "CLIP_Score_mean": round(float(np.mean(clip_list)), 4),
        "CLIP_Score_std": round(float(np.std(clip_list)), 4),
    }


# ---------------------------------------------------------------------------
# Core evaluation for one group
# ---------------------------------------------------------------------------

def evaluate_group(
    gt_img_dir: str,
    eval_img_dir: str,
    category: str,
    device: str = "cuda",
    clip_model: str = "openai/clip-vit-base-patch32",
    annotation_json: Optional[str] = None,
    padding: int = 0,
    lpips_metric: Optional[LPIPSMetric] = None,
    clip_metric: Optional[CLIPScoreMetric] = None,
) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray]]:
    """Evaluate one group of images."""
    local_mode = annotation_json is not None

    pairs = find_matched_pairs(gt_img_dir, eval_img_dir)
    if not pairs:
        print(f"  [WARN] No matching image pairs found for category '{category}'!")
        return [], [], []
    print(f"  [INFO] Found {len(pairs)} matched image pairs.")

    pair_annotations: Dict[str, Tuple[str, List]] = {}
    if local_mode:
        bbox_lookup, entry_lookup = load_annotations(annotation_json)
        print(f"  [INFO] Loaded annotations for {len(bbox_lookup)} images.")
        ann_index = build_annotation_index(bbox_lookup, entry_lookup)

        annotated_pairs = []
        no_annotation = 0
        for gt_path, eval_path in pairs:
            gt_filename = os.path.basename(gt_path)
            ann = find_annotation_for_pair(gt_filename, ann_index)
            if ann:
                pair_annotations[gt_path] = ann
                annotated_pairs.append((gt_path, eval_path))
            else:
                no_annotation += 1

        if no_annotation > 0:
            print(f"  [WARN] {no_annotation} image pairs have no matching annotation (skipped).")
        pairs = annotated_pairs
        if not pairs:
            print(f"  [WARN] No annotated image pairs found for category '{category}'!")
            return [], [], []
        print(f"  [INFO] {len(pairs)} pairs with annotations will be evaluated.")

    per_image_results = []
    all_gt_images = []
    all_pred_images = []

    mode_desc = "local" if local_mode else "global"
    for gt_path, eval_path in tqdm(pairs, desc=f"  [{category}] {mode_desc}"):
        gt_img = load_image(gt_path)
        pred_img = load_image(eval_path)
        gt_img, pred_img = resize_to_match(gt_img, pred_img)
        filename = os.path.basename(gt_path)

        if local_mode:
            out_basename, bboxes = pair_annotations[gt_path]
            img_h, img_w = gt_img.shape[:2]
            x1, y1, x2, y2 = merge_bboxes(bboxes, img_h, img_w, padding=padding)

            gt_patch = crop_region(gt_img, x1, y1, x2, y2)
            if gt_patch.shape[0] < 2 or gt_patch.shape[1] < 2:
                continue

            gt_patch, pred_patch = ensure_min_patch_size(
                gt_img, pred_img, x1, y1, x2, y2, img_h, img_w)

            gt_eval = gt_patch
            pred_eval = pred_patch
        else:
            gt_eval = gt_img
            pred_eval = pred_img

        all_gt_images.append(gt_eval)
        all_pred_images.append(pred_eval)

        psnr_val = compute_psnr(gt_eval, pred_eval)
        if np.isinf(psnr_val):
            psnr_val = 100.0
        ssim_val = compute_ssim(gt_eval, pred_eval)
        lpips_val = lpips_metric.compute(gt_eval, pred_eval)
        clip_val = clip_metric.compute(gt_eval, pred_eval)

        record = {
            "filename": filename,
            "category": category,
            "PSNR": round(psnr_val, 4),
            "SSIM": round(ssim_val, 4),
            "LPIPS": round(lpips_val, 4),
            "CLIP_Score": round(clip_val, 4),
        }
        if local_mode:
            record["crop_box"] = [x1, y1, x2, y2]
            record["crop_size"] = [x2 - x1, y2 - y1]
            record["num_regions"] = len(bboxes)
        per_image_results.append(record)

    return per_image_results, all_gt_images, all_pred_images


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_multi_group(
    groups: List[Dict],
    device: str = "cuda",
    batch_size: int = 32,
    clip_model: str = "openai/clip-vit-base-patch32",
    output_json: Optional[str] = None,
    padding: int = 0,
) -> Dict:
    """Run full evaluation pipeline for multiple groups."""
    print("=" * 70)
    print("  Image Quality Evaluation (Multi-Group)")
    print("=" * 70)
    for g in groups:
        print(f"  Group [{g['category']}]:")
        print(f"    GT dir       : {g['gt_img_dir']}")
        print(f"    Eval dir     : {g['eval_img_dir']}")
        if g.get('annotation_json'):
            print(f"    Annotation   : {g['annotation_json']}")
    print(f"  Device         : {device}")
    print(f"  Padding        : {padding} px")
    print("=" * 70)

    print("\n[INFO] Initializing LPIPS model ...")
    lpips_metric = LPIPSMetric(net="alex", device=device)

    print("[INFO] Initializing CLIP model ...")
    clip_metric = CLIPScoreMetric(model_name=clip_model, device=device)

    print("[INFO] Initializing FID model (InceptionV3) ...")
    fid_metric = FIDMetric(device=device)

    all_per_image_results: List[Dict] = []
    all_gt_images: List[np.ndarray] = []
    all_pred_images: List[np.ndarray] = []
    group_gt_images: Dict[str, List[np.ndarray]] = {}
    group_pred_images: Dict[str, List[np.ndarray]] = {}

    for g in groups:
        print(f"\n{'='*60}")
        print(f"Evaluating category: {g['category']}")
        print(f"{'='*60}")

        results, gt_imgs, pred_imgs = evaluate_group(
            gt_img_dir=g['gt_img_dir'],
            eval_img_dir=g['eval_img_dir'],
            category=g['category'],
            device=device,
            clip_model=clip_model,
            annotation_json=g.get('annotation_json'),
            padding=padding,
            lpips_metric=lpips_metric,
            clip_metric=clip_metric,
        )

        all_per_image_results.extend(results)
        all_gt_images.extend(gt_imgs)
        all_pred_images.extend(pred_imgs)
        group_gt_images[g['category']] = gt_imgs
        group_pred_images[g['category']] = pred_imgs

        print(f"  -> {len(results)} images evaluated")

    if not all_per_image_results:
        print("\n[ERROR] No valid image pairs to evaluate!")
        sys.exit(1)

    print("\n[INFO] Computing FID scores ...")
    overall_fid = 0.0
    if len(all_gt_images) >= 2:
        overall_fid = fid_metric.compute(all_gt_images, all_pred_images, batch_size=batch_size)
    else:
        print("  [WARN] Not enough images for FID computation.")

    category_fid: Dict[str, float] = {}
    for cat, gt_imgs in group_gt_images.items():
        pred_imgs = group_pred_images[cat]
        category_fid[cat] = (
            fid_metric.compute(gt_imgs, pred_imgs, batch_size=batch_size)
            if len(gt_imgs) >= 2 else 0.0
        )

    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_per_image_results:
        by_category[r["category"]].append(r)

    overall_agg = _aggregate_metrics(all_per_image_results)
    overall_agg["FID"] = round(overall_fid, 4)

    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"  Number of image pairs : {overall_agg['num_images']}")
    print(f"  PSNR  (mean +/- std)  : {overall_agg['PSNR_mean']:.4f} +/- {overall_agg['PSNR_std']:.4f}")
    print(f"  SSIM  (mean +/- std)  : {overall_agg['SSIM_mean']:.4f} +/- {overall_agg['SSIM_std']:.4f}")
    print(f"  LPIPS (mean +/- std)  : {overall_agg['LPIPS_mean']:.4f} +/- {overall_agg['LPIPS_std']:.4f}")
    print(f"  CLIP  (mean +/- std)  : {overall_agg['CLIP_Score_mean']:.4f} +/- {overall_agg['CLIP_Score_std']:.4f}")
    print(f"  FID                   : {overall_agg['FID']:.4f}")

    print(f"\n{'-'*70}")
    print("BY CATEGORY")
    print(f"{'-'*70}")
    for cat in sorted(by_category):
        cat_agg = _aggregate_metrics(by_category[cat])
        cat_agg["FID"] = round(category_fid.get(cat, 0.0), 4)
        print(f"  [{cat}] (n={cat_agg['num_images']}): "
              f"PSNR={cat_agg['PSNR_mean']:.4f}, "
              f"SSIM={cat_agg['SSIM_mean']:.4f}, "
              f"LPIPS={cat_agg['LPIPS_mean']:.4f}, "
              f"CLIP={cat_agg['CLIP_Score_mean']:.4f}, "
              f"FID={cat_agg['FID']:.4f}")

    print(f"\n{'='*70}")
    print("  Note: PSNR/SSIM/CLIP_Score -> higher is better")
    print("        LPIPS/FID            -> lower is better")
    print(f"{'='*70}")

    results = {
        "summary": {
            "overall": overall_agg,
            "by_category": {},
        },
        "per_image": all_per_image_results,
    }
    for cat in sorted(by_category):
        cat_agg = _aggregate_metrics(by_category[cat])
        cat_agg["FID"] = round(category_fid.get(cat, 0.0), 4)
        results["summary"]["by_category"][cat] = cat_agg

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Results saved to: {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate image quality: PSNR, SSIM, LPIPS, FID, CLIP Score. "
                    "Supports global (full-image) and local (region-based) modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Multi-group usage:
  python evaluate.py \\
      --groups \\
          text  /path/to/text_info.json  /path/to/text_gt_imgs  /path/to/text_eval_imgs \\
          table /path/to/table_info.json /path/to/table_gt_imgs /path/to/table_eval_imgs \\
      --padding 0 --device cuda

Single-group (backward compatible):
  python evaluate.py \\
      --gt_img_dir /path/to/gt \\
      --eval_img_dir /path/to/eval \\
      [--annotation_json /path/to/ann.json]
""",
    )

    parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Repeating quadruples: CATEGORY ANNOTATION_JSON GT_IMG_DIR EVAL_IMG_DIR. "
             "ANNOTATION_JSON can be 'none' for global mode.",
    )

    parser.add_argument("--gt_img_dir", type=str, default=None)
    parser.add_argument("--eval_img_dir", type=str, default=None)
    parser.add_argument("--annotation_json", type=str, default=None)

    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--padding", type=int, default=0)

    args = parser.parse_args()

    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.groups:
        tokens = args.groups
        if len(tokens) % 4 != 0:
            parser.error("--groups must be multiples of 4: CATEGORY ANNOTATION_JSON GT_IMG_DIR EVAL_IMG_DIR")

        groups = []
        for i in range(0, len(tokens), 4):
            ann_json = tokens[i + 1]
            if ann_json.lower() in ("none", "null", "-"):
                ann_json = None
            groups.append({
                "category": tokens[i],
                "annotation_json": ann_json,
                "gt_img_dir": tokens[i + 2],
                "eval_img_dir": tokens[i + 3],
            })

        for g in groups:
            if not os.path.isdir(g['gt_img_dir']):
                print(f"[ERROR] GT directory does not exist: {g['gt_img_dir']}")
                sys.exit(1)
            if not os.path.isdir(g['eval_img_dir']):
                print(f"[ERROR] Eval directory does not exist: {g['eval_img_dir']}")
                sys.exit(1)
            if g['annotation_json'] and not os.path.isfile(g['annotation_json']):
                print(f"[ERROR] Annotation JSON does not exist: {g['annotation_json']}")
                sys.exit(1)

        evaluate_multi_group(
            groups=groups, device=args.device, batch_size=args.batch_size,
            clip_model=args.clip_model, output_json=args.output_json,
            padding=args.padding,
        )

    elif args.gt_img_dir and args.eval_img_dir:
        if not os.path.isdir(args.gt_img_dir):
            print(f"[ERROR] GT directory does not exist: {args.gt_img_dir}")
            sys.exit(1)
        if not os.path.isdir(args.eval_img_dir):
            print(f"[ERROR] Eval directory does not exist: {args.eval_img_dir}")
            sys.exit(1)
        if args.annotation_json and not os.path.isfile(args.annotation_json):
            print(f"[ERROR] Annotation JSON does not exist: {args.annotation_json}")
            sys.exit(1)

        groups = [{
            "category": "default",
            "annotation_json": args.annotation_json,
            "gt_img_dir": args.gt_img_dir,
            "eval_img_dir": args.eval_img_dir,
        }]

        evaluate_multi_group(
            groups=groups, device=args.device, batch_size=args.batch_size,
            clip_model=args.clip_model, output_json=args.output_json,
            padding=args.padding,
        )

    else:
        parser.error("Must provide either --groups or both --gt_img_dir and --eval_img_dir.")


if __name__ == "__main__":
    main()
