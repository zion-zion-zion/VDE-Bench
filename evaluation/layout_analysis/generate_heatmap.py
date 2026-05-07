#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Layout Evaluation System -- Heatmap Generator
=============================================
Generates a comprehensive set of layout evaluation heatmaps from
OmniDocBench.json (or any OmniDocBench-shaped annotation file), covering
multiple evaluation dimensions:

1. Element Density:     area-ratio based density -- avg fraction of page area
                        occupied by each category_type, per data_source
2. Z-Score Deviation:   column-wise z-score of density (highlights anomalies)
3. Layout Complexity:   avg distinct element types per page, per data_source
4. Element Richness:    avg total element count per page, per data_source
5. Shannon Entropy:     distribution uniformity of element types, per data_source
6. Text/Visual Ratio:   area-based proportion of text vs visual elements
7. Raw Count:           absolute count of each category_type, per data_source
8. Row-Normalized %:    percentage composition within each data_source

Usage
-----
    python generate_heatmap.py \
        --omnidoc_json /path/to/OmniDocBench.json \
        --output_dir   ./heatmaps
"""

import argparse
import json
import os
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Element classification ─────────────────────────────────────────────────
TEXT_ELEMENTS = {
    "text_block", "title", "header", "footer", "page_number",
    "figure_caption", "table_caption", "table_footnote",
    "page_header", "page_footer", "code_txt",
}
VISUAL_ELEMENTS = {
    "figure", "table", "figure_body", "table_body",
    "formula_isolated", "formula_caption",
    "code_block", "seal", "chart",
}
# Elements to skip (sub-elements, not standalone layout elements)
SKIP_ELEMENTS = {"text_span"}


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_omnidoc(omnidoc_path: str) -> list:
    """Load OmniDocBench.json."""
    print(f"Loading OmniDocBench from {omnidoc_path} ...")
    with open(omnidoc_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} pages.")
    return data


def _poly_area(poly: list) -> float:
    """Compute area of a quadrilateral given 8 coordinates [x1,y1,...,x4,y4]."""
    if len(poly) != 8:
        return 0.0
    xs = [poly[0], poly[2], poly[4], poly[6]]
    ys = [poly[1], poly[3], poly[5], poly[7]]
    n = 4
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def build_page_records(omnidoc: list) -> list:
    """Parse each page into a structured record."""
    records = []
    for page in omnidoc:
        page_info = page.get("page_info", {})
        page_attr = page_info.get("page_attribute", {})
        data_source = page_attr.get("data_source", "unknown")
        layout = page_attr.get("layout", "unknown")

        page_w = page_info.get("width", 0)
        page_h = page_info.get("height", 0)
        page_area = float(page_w * page_h) if (page_w and page_h) else 0.0

        cat_counter: Counter = Counter()
        cat_areas: dict = defaultdict(float)
        for det in page.get("layout_dets", []):
            cat_type = det.get("category_type", "unknown")
            if cat_type in SKIP_ELEMENTS:
                continue
            cat_counter[cat_type] += 1
            poly = det.get("poly", [])
            if poly:
                cat_areas[cat_type] += _poly_area(poly)

        records.append({
            "data_source": data_source,
            "layout": layout,
            "category_counts": cat_counter,
            "category_areas": dict(cat_areas),
            "page_area": page_area,
        })
    return records


# ═══════════════════════════════════════════════════════════════════════════
# Metric Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_density_matrix(records: list):
    """Compute area-ratio based element density matrix."""
    ds_pages = defaultdict(list)
    all_cats = set()
    for rec in records:
        ds_pages[rec["data_source"]].append(rec)
        all_cats.update(rec["category_counts"].keys())

    cats = sorted(all_cats)
    dss = sorted(ds_pages.keys())

    matrix = np.zeros((len(dss), len(cats)), dtype=float)
    for i, ds in enumerate(dss):
        pages = ds_pages[ds]
        if not pages:
            continue
        for j, cat in enumerate(cats):
            area_ratios = []
            for rec in pages:
                pa = rec["page_area"]
                if pa <= 0:
                    continue
                elem_area = rec["category_areas"].get(cat, 0.0)
                area_ratios.append(elem_area / pa)
            if area_ratios:
                matrix[i, j] = np.mean(area_ratios)

    return matrix, dss, cats


def compute_zscore_matrix(density_matrix: np.ndarray) -> np.ndarray:
    """Column-wise z-score: z_{ds,cat} = (d - mu) / sigma."""
    mu = density_matrix.mean(axis=0)
    sigma = density_matrix.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (density_matrix - mu) / sigma


def compute_complexity_richness(records: list):
    """Complexity = avg distinct types/page; Richness = avg total count/page."""
    ds_pages = defaultdict(list)
    for rec in records:
        ds_pages[rec["data_source"]].append(rec["category_counts"])

    complexity, richness = {}, {}
    for ds, pages in ds_pages.items():
        if not pages:
            complexity[ds] = richness[ds] = 0.0
            continue
        complexity[ds] = np.mean([
            len([v for v in p.values() if v > 0]) for p in pages
        ])
        richness[ds] = np.mean([sum(p.values()) for p in pages])

    return complexity, richness


def compute_entropy(records: list):
    """Shannon entropy of element type distribution per data_source."""
    ds_pages = defaultdict(list)
    for rec in records:
        ds_pages[rec["data_source"]].append(rec["category_counts"])

    entropy = {}
    for ds, pages in ds_pages.items():
        total_counts: Counter = Counter()
        for p in pages:
            total_counts += p
        total = sum(total_counts.values())
        if total == 0:
            entropy[ds] = 0.0
            continue
        h = 0.0
        for cnt in total_counts.values():
            if cnt > 0:
                prob = cnt / total
                h -= prob * np.log2(prob)
        entropy[ds] = h
    return entropy


def compute_text_visual_ratio(records: list):
    """Compute area-based text vs visual element ratio per data_source."""
    ds_pages = defaultdict(list)
    for rec in records:
        ds_pages[rec["data_source"]].append(rec)

    ratios = {}
    for ds, pages in ds_pages.items():
        text_area_total = visual_area_total = 0.0
        valid_pages = 0
        for rec in pages:
            pa = rec["page_area"]
            if pa <= 0:
                continue
            valid_pages += 1
            for elem, area in rec["category_areas"].items():
                if elem in TEXT_ELEMENTS:
                    text_area_total += area / pa
                elif elem in VISUAL_ELEMENTS:
                    visual_area_total += area / pa
        n = valid_pages if valid_pages > 0 else 1
        text_avg = text_area_total / n
        visual_avg = visual_area_total / n
        total = text_avg + visual_avg
        ratios[ds] = {
            "text_area_per_page": text_avg,
            "visual_area_per_page": visual_avg,
            "text_ratio": text_avg / total if total > 0 else 0,
            "visual_ratio": visual_avg / total if total > 0 else 0,
        }
    return ratios


# ═══════════════════════════════════════════════════════════════════════════
# Plotting Functions
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig, output_dir, filename):
    out = os.path.join(output_dir, filename)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_density_zscore_heatmap(density, zscore, dss, cats, output_dir):
    fig, ax = plt.subplots(figsize=(24, 8))
    annot = np.array([
        [f"{zscore[i, j]:.2f}" for j in range(len(cats))]
        for i in range(len(dss))
    ])
    sns.heatmap(
        zscore, xticklabels=cats, yticklabels=dss,
        annot=annot, fmt="", cmap="PiYG_r", center=0,
        linewidths=0.5, linecolor="gray",
        cbar_kws={"label": "Z-Score (deviation from element mean)"},
        ax=ax,
    )
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    _save(fig, output_dir, "eval_1_density_zscore.png")


def plot_raw_count_heatmap(records, dss, cats, output_dir):
    agg = defaultdict(Counter)
    for rec in records:
        agg[rec["data_source"]] += rec["category_counts"]

    matrix = np.zeros((len(dss), len(cats)), dtype=float)
    for i, ds in enumerate(dss):
        for j, cat in enumerate(cats):
            matrix[i, j] = agg[ds].get(cat, 0)

    fig, ax = plt.subplots(
        figsize=(max(16, len(cats) * 1.3), max(8, len(dss) * 0.7))
    )
    sns.heatmap(
        matrix, annot=True, fmt=".0f",
        xticklabels=cats, yticklabels=dss,
        cmap="YlOrRd", linewidths=0.5, ax=ax,
    )
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    _save(fig, output_dir, "eval_2_element_count.png")


def plot_normalized_heatmap(records, dss, cats, output_dir):
    agg = defaultdict(Counter)
    for rec in records:
        agg[rec["data_source"]] += rec["category_counts"]

    matrix = np.zeros((len(dss), len(cats)), dtype=float)
    for i, ds in enumerate(dss):
        for j, cat in enumerate(cats):
            matrix[i, j] = agg[ds].get(cat, 0)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_pct = matrix / row_sums * 100

    fig, ax = plt.subplots(
        figsize=(max(16, len(cats) * 1.3), max(8, len(dss) * 0.7))
    )
    sns.heatmap(
        matrix_pct, annot=True, fmt=".1f",
        xticklabels=cats, yticklabels=dss,
        cmap="YlGnBu", linewidths=0.5, ax=ax,
    )
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    _save(fig, output_dir, "eval_3_element_composition.png")


def plot_complexity_richness_entropy(complexity, richness, entropy, output_dir):
    dss = sorted(complexity.keys())
    metrics = ["Complexity\n(distinct types/page)", "Richness\n(total elems/page)", "Entropy\n(bits)"]

    matrix = np.zeros((len(dss), 3), dtype=float)
    for i, ds in enumerate(dss):
        matrix[i, 0] = complexity[ds]
        matrix[i, 1] = richness[ds]
        matrix[i, 2] = entropy[ds]

    matrix_norm = matrix.copy()
    for j in range(3):
        col = matrix[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            matrix_norm[:, j] = (col - cmin) / (cmax - cmin)
        else:
            matrix_norm[:, j] = 0.5

    annot = np.array([
        [f"{matrix[i, j]:.2f}" for j in range(3)]
        for i in range(len(dss))
    ])

    fig, ax = plt.subplots(figsize=(8, max(6, len(dss) * 0.7)))
    sns.heatmap(
        matrix_norm, annot=annot, fmt="",
        xticklabels=metrics, yticklabels=dss,
        cmap="RdYlGn", linewidths=0.5, ax=ax,
        vmin=0, vmax=1,
        cbar_kws={"label": "Normalized Score (0=min, 1=max)"},
    )
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    _save(fig, output_dir, "eval_4_complexity_richness_entropy.png")


def plot_text_visual_heatmap(ratios, output_dir):
    dss = sorted(ratios.keys())
    cols = [
        "Text Area\nper Page", "Visual Area\nper Page",
        "Text\nRatio (%)", "Visual\nRatio (%)",
    ]

    matrix = np.zeros((len(dss), 4), dtype=float)
    for i, ds in enumerate(dss):
        r = ratios[ds]
        matrix[i, 0] = r["text_area_per_page"]
        matrix[i, 1] = r["visual_area_per_page"]
        matrix[i, 2] = r["text_ratio"] * 100
        matrix[i, 3] = r["visual_ratio"] * 100

    annot = np.array([
        [f"{matrix[i, j]:.1f}" for j in range(4)]
        for i in range(len(dss))
    ])

    matrix_norm = matrix.copy()
    for j in range(4):
        col = matrix[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            matrix_norm[:, j] = (col - cmin) / (cmax - cmin)
        else:
            matrix_norm[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(10, max(6, len(dss) * 0.7)))
    sns.heatmap(
        matrix_norm, annot=annot, fmt="",
        xticklabels=cols, yticklabels=dss,
        cmap="coolwarm", linewidths=0.5, ax=ax,
        vmin=0, vmax=1,
        cbar_kws={"label": "Normalized (0=min, 1=max)"},
    )
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    _save(fig, output_dir, "eval_5_text_visual_balance.png")


def plot_comprehensive_score(density, dss, cats, complexity, richness, entropy, ratios, output_dir):
    metric_names = [
        "Avg Area Density", "Complexity", "Richness",
        "Entropy", "Text Ratio", "Visual Ratio",
    ]

    matrix = np.zeros((len(dss), len(metric_names)), dtype=float)
    for i, ds in enumerate(dss):
        matrix[i, 0] = density[i].mean()
        matrix[i, 1] = complexity.get(ds, 0)
        matrix[i, 2] = richness.get(ds, 0)
        matrix[i, 3] = entropy.get(ds, 0)
        r = ratios.get(ds, {})
        matrix[i, 4] = r.get("text_ratio", 0) * 100
        matrix[i, 5] = r.get("visual_ratio", 0) * 100

    annot = np.array([
        [f"{matrix[i, j]:.2f}" for j in range(len(metric_names))]
        for i in range(len(dss))
    ])

    matrix_norm = matrix.copy()
    for j in range(len(metric_names)):
        col = matrix[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            matrix_norm[:, j] = (col - cmin) / (cmax - cmin)
        else:
            matrix_norm[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(12, max(6, len(dss) * 0.7)))
    sns.heatmap(
        matrix_norm, annot=annot, fmt="",
        xticklabels=metric_names, yticklabels=dss,
        cmap="viridis", linewidths=0.5, ax=ax,
        vmin=0, vmax=1,
        cbar_kws={"label": "Normalized Score (0=min, 1=max)"},
    )
    plt.xticks(rotation=15, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    _save(fig, output_dir, "eval_6_comprehensive_score.png")


def plot_page_count_and_layout(records, output_dir):
    agg = defaultdict(Counter)
    for rec in records:
        agg[rec["data_source"]][rec["layout"]] += 1

    all_layouts = set()
    for counter in agg.values():
        all_layouts.update(counter.keys())
    layouts = sorted(all_layouts)
    dss = sorted(agg.keys())

    matrix = np.zeros((len(dss), len(layouts)), dtype=float)
    for i, ds in enumerate(dss):
        for j, lay in enumerate(layouts):
            matrix[i, j] = agg[ds].get(lay, 0)

    fig, ax = plt.subplots(
        figsize=(max(10, len(layouts) * 2), max(6, len(dss) * 0.7))
    )
    sns.heatmap(
        matrix, annot=True, fmt=".0f",
        xticklabels=layouts, yticklabels=dss,
        cmap="PuBuGn", linewidths=0.5, ax=ax,
    )
    plt.xticks(rotation=30, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    _save(fig, output_dir, "eval_7_page_layout_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(records, dss, cats, density, complexity, richness, entropy, ratios):
    print("\n" + "=" * 80)
    print("  LAYOUT EVALUATION SUMMARY")
    print("=" * 80)

    ds_count = Counter(r["data_source"] for r in records)
    print(f"\n{'Data Source':<25} {'Pages':>8} {'Complexity':>12} {'Richness':>10} "
          f"{'Entropy':>9} {'TextArea%':>10} {'VisArea%':>10}")
    print("-" * 90)
    for ds in dss:
        r = ratios.get(ds, {})
        print(f"{ds:<25} {ds_count[ds]:>8} {complexity[ds]:>12.2f} {richness[ds]:>10.2f} "
              f"{entropy[ds]:>9.2f} {r.get('text_ratio', 0)*100:>9.1f}% "
              f"{r.get('visual_ratio', 0)*100:>9.1f}%")

    print(f"\n  Total pages: {len(records)}")
    print(f"  Data sources: {len(dss)}")
    print(f"  Category types: {len(cats)}")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omnidoc_json", required=True,
                   help="Path to OmniDocBench.json (or an OmniDocBench-shaped JSON).")
    p.add_argument("--output_dir", default="./heatmaps",
                   help="Where heatmap PNGs and metrics JSON will be written.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    omnidoc = load_omnidoc(args.omnidoc_json)
    records = build_page_records(omnidoc)

    print("\nComputing metrics ...")
    density, dss, cats = compute_density_matrix(records)
    zscore = compute_zscore_matrix(density)
    complexity, richness = compute_complexity_richness(records)
    entropy = compute_entropy(records)
    ratios = compute_text_visual_ratio(records)

    print_summary(records, dss, cats, density, complexity, richness, entropy, ratios)

    print("\nGenerating layout evaluation heatmaps ...")
    plot_density_zscore_heatmap(density, zscore, dss, cats, args.output_dir)
    plot_raw_count_heatmap(records, dss, cats, args.output_dir)
    plot_normalized_heatmap(records, dss, cats, args.output_dir)
    plot_complexity_richness_entropy(complexity, richness, entropy, args.output_dir)
    plot_text_visual_heatmap(ratios, args.output_dir)
    plot_comprehensive_score(density, dss, cats, complexity, richness, entropy, ratios, args.output_dir)
    plot_page_count_and_layout(records, args.output_dir)

    metrics_json = {
        "density_area_ratio": {
            ds: {cat: float(density[i, j]) for j, cat in enumerate(cats)}
            for i, ds in enumerate(dss)
        },
        "zscore": {
            ds: {cat: float(zscore[i, j]) for j, cat in enumerate(cats)}
            for i, ds in enumerate(dss)
        },
        "complexity": {k: float(v) for k, v in complexity.items()},
        "richness": {k: float(v) for k, v in richness.items()},
        "entropy": {k: float(v) for k, v in entropy.items()},
        "text_visual_ratio": ratios,
    }
    metrics_path = os.path.join(args.output_dir, "layout_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {metrics_path}")

    print(f"\nAll heatmaps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
