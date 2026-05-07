"""Report generation for layout evaluation results."""

from collections import defaultdict
from typing import Dict, List

import numpy as np

from constants import get_category


def generate_report(
    element_data: Dict[str, Dict[str, int]],
    density_matrix: np.ndarray,
    zscore_matrix: np.ndarray,
    categories: List[str],
    elements: List[str],
    complexity: Dict[str, float],
    richness: Dict[str, float],
    entropy: Dict[str, float],
    ratios: Dict[str, Dict[str, float]],
    output_path: str,
):
    """Generate a comprehensive text report of the layout evaluation."""
    lines = []
    lines.append("=" * 80)
    lines.append("  DOCUMENT LAYOUT EVALUATION REPORT")
    lines.append("=" * 80)

    # 1. Dataset overview
    category_counts = defaultdict(int)
    for filename in element_data:
        category_counts[get_category(filename)] += 1

    lines.append(f"\n1. DATASET OVERVIEW")
    lines.append(f"   Total pages: {len(element_data)}")
    lines.append(f"   Categories: {len(category_counts)}")
    lines.append(f"   Element types detected: {len(elements)}")
    lines.append(f"\n   Pages per category:")
    for cat in sorted(category_counts.keys()):
        lines.append(f"     {cat:<25s} {category_counts[cat]:>5d} pages")

    # 2. Element density table
    lines.append(f"\n2. ELEMENT DENSITY (elements per page)")
    hdr = f"   {'Category':<25s} " + " ".join(f"{e[:8]:>10s}" for e in elements)
    lines.append(hdr)
    lines.append("   " + "-" * (25 + 11 * len(elements)))
    for i, cat in enumerate(categories):
        row = f"   {cat:<25s} " + " ".join(
            f"{density_matrix[i, j]:>10.2f}" for j in range(len(elements))
        )
        lines.append(row)

    # 3. Z-score table
    lines.append(f"\n3. Z-SCORE NORMALIZATION")
    lines.append(hdr)
    lines.append("   " + "-" * (25 + 11 * len(elements)))
    for i, cat in enumerate(categories):
        row = f"   {cat:<25s} " + " ".join(
            f"{zscore_matrix[i, j]:>10.2f}" for j in range(len(elements))
        )
        lines.append(row)

    # 4. Per-element statistics
    mu = density_matrix.mean(axis=0)
    sigma = density_matrix.std(axis=0)
    lines.append(f"\n4. PER-ELEMENT STATISTICS")
    lines.append(
        f"   {'Element':<20s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}"
    )
    lines.append("   " + "-" * 60)
    for j, elem in enumerate(elements):
        lines.append(
            f"   {elem:<20s} {mu[j]:>10.2f} {sigma[j]:>10.2f} "
            f"{density_matrix[:, j].min():>10.2f} {density_matrix[:, j].max():>10.2f}"
        )

    # 5. Complexity & Richness & Entropy
    lines.append(f"\n5. LAYOUT COMPLEXITY & RICHNESS & ENTROPY")
    lines.append(
        f"   {'Category':<25s} {'Complexity':>12s} {'Richness':>12s} {'Entropy':>12s}"
    )
    lines.append("   " + "-" * 61)
    for cat in sorted(complexity.keys()):
        lines.append(
            f"   {cat:<25s} {complexity[cat]:>12.2f} "
            f"{richness[cat]:>12.2f} {entropy[cat]:>12.2f}"
        )

    # 6. Text vs Visual ratio
    lines.append(f"\n6. TEXT vs. VISUAL ELEMENT RATIO")
    lines.append(
        f"   {'Category':<25s} {'Text/pg':>10s} {'Visual/pg':>10s} "
        f"{'Text%':>10s} {'Visual%':>10s}"
    )
    lines.append("   " + "-" * 65)
    for cat in sorted(ratios.keys()):
        r = ratios[cat]
        lines.append(
            f"   {cat:<25s} {r['text_count']:>10.2f} {r['visual_count']:>10.2f} "
            f"{r['text_ratio']:>9.1%} {r['visual_ratio']:>9.1%}"
        )

    # 7. Key findings
    lines.append(f"\n7. KEY FINDINGS")

    max_comp_cat = max(complexity, key=complexity.get)
    min_comp_cat = min(complexity, key=complexity.get)
    lines.append(
        f"   - Most complex layout:  {max_comp_cat} "
        f"(complexity={complexity[max_comp_cat]:.2f})"
    )
    lines.append(
        f"   - Simplest layout:      {min_comp_cat} "
        f"(complexity={complexity[min_comp_cat]:.2f})"
    )

    max_vis_cat = max(ratios, key=lambda c: ratios[c]["visual_ratio"])
    max_txt_cat = max(ratios, key=lambda c: ratios[c]["text_ratio"])
    lines.append(
        f"   - Most visual-heavy:    {max_vis_cat} "
        f"(visual={ratios[max_vis_cat]['visual_ratio']:.1%})"
    )
    lines.append(
        f"   - Most text-heavy:      {max_txt_cat} "
        f"(text={ratios[max_txt_cat]['text_ratio']:.1%})"
    )

    max_ent_cat = max(entropy, key=entropy.get)
    min_ent_cat = min(entropy, key=entropy.get)
    lines.append(
        f"   - Most diverse layout:  {max_ent_cat} "
        f"(entropy={entropy[max_ent_cat]:.2f} bits)"
    )
    lines.append(
        f"   - Least diverse layout: {min_ent_cat} "
        f"(entropy={entropy[min_ent_cat]:.2f} bits)"
    )

    # Notable z-score outliers
    lines.append(f"\n   Notable outliers (|z| > 1.5):")
    for i, cat in enumerate(categories):
        for j, elem in enumerate(elements):
            z = zscore_matrix[i, j]
            if abs(z) > 1.5:
                direction = "HIGH" if z > 0 else "LOW"
                lines.append(
                    f"     {cat} x {elem}: z={z:.2f} "
                    f"({direction}, density={density_matrix[i, j]:.2f})"
                )

    lines.append("\n" + "=" * 80)

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[Saved] Evaluation report -> {output_path}")
    print(report_text)
