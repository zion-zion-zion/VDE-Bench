"""Visualization functions for layout evaluation results."""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from constants import get_category

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_density_heatmap(
    density_matrix: np.ndarray,
    zscore_matrix: np.ndarray,
    categories: List[str],
    elements: List[str],
    output_path: str,
):
    """Heatmap with z-score coloring and density value annotations."""
    mu = density_matrix.mean(axis=0)
    elem_labels = [f"{e}\n(mu={mu[j]:.2f})" for j, e in enumerate(elements)]

    fig, ax = plt.subplots(
        figsize=(max(14, len(elements) * 1.2), max(8, len(categories) * 0.6))
    )

    annot = np.array([
        [f"{density_matrix[i, j]:.2f}" for j in range(len(elements))]
        for i in range(len(categories))
    ])

    sns.heatmap(
        zscore_matrix,
        xticklabels=elem_labels,
        yticklabels=categories,
        annot=annot,
        fmt="",
        cmap="PiYG_r",
        center=0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Z-Score (deviation from element mean)"},
        ax=ax,
    )

    ax.set_title(
        "Document Layout Element Distribution\n"
        "(Color: Z-Score | Annotation: Elements per Page)",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.set_xlabel("Layout Element Type", fontsize=11)
    ax.set_ylabel("Document Category", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Density heatmap -> {output_path}")


def plot_complexity_bar(
    complexity: Dict[str, float],
    richness: Dict[str, float],
    output_path: str,
):
    """Plot layout complexity and richness as grouped bar chart."""
    categories = sorted(complexity.keys())
    comp_vals = [complexity[c] for c in categories]
    rich_vals = [richness[c] for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(12, len(categories) * 0.8), 6))
    ax1.bar(x - width / 2, comp_vals, width,
            label="Complexity (distinct types/page)", color="#4C72B0", alpha=0.85)
    ax1.set_ylabel("Complexity (distinct element types per page)", color="#4C72B0", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, rich_vals, width,
            label="Richness (total elements/page)", color="#DD8452", alpha=0.85)
    ax2.set_ylabel("Richness (total elements per page)", color="#DD8452", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax1.set_title("Layout Complexity & Richness by Document Category",
                  fontsize=13, fontweight="bold", pad=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Complexity bar chart -> {output_path}")


def plot_entropy_bar(entropy: Dict[str, float], output_path: str):
    """Plot Shannon entropy of element distribution per category."""
    categories = sorted(entropy.keys())
    values = [entropy[c] for c in categories]

    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.8), 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
    bars = ax.barh(categories, values, color=colors, edgecolor="gray", linewidth=0.5)

    ax.set_xlabel("Shannon Entropy (bits)", fontsize=11)
    ax.set_title("Layout Element Distribution Entropy by Category",
                 fontsize=13, fontweight="bold", pad=15)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Entropy bar chart -> {output_path}")


def plot_text_visual_ratio(ratios: Dict[str, Dict[str, float]], output_path: str):
    """Plot text vs. visual element ratio as stacked horizontal bar chart."""
    categories = sorted(ratios.keys())
    text_ratios = [ratios[c]["text_ratio"] for c in categories]
    visual_ratios = [ratios[c]["visual_ratio"] for c in categories]

    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.8), 5))
    x = np.arange(len(categories))
    ax.barh(x, text_ratios, label="Text Elements", color="#5B9BD5", edgecolor="white")
    ax.barh(x, visual_ratios, left=text_ratios,
            label="Visual Elements", color="#ED7D31", edgecolor="white")

    ax.set_yticks(x)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel("Proportion", fontsize=11)
    ax.set_title("Text vs. Visual Element Ratio by Category",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1)

    for i, (tr, vr) in enumerate(zip(text_ratios, visual_ratios)):
        if tr > 0.05:
            ax.text(tr / 2, i, f"{tr:.0%}", ha="center", va="center", fontsize=8, color="white")
        if vr > 0.05:
            ax.text(tr + vr / 2, i, f"{vr:.0%}", ha="center", va="center", fontsize=8, color="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Text/Visual ratio chart -> {output_path}")


def plot_category_distribution(element_data: Dict[str, Dict[str, int]], output_path: str):
    """Plot the number of pages per category as a pie chart."""
    category_counts = defaultdict(int)
    for filename in element_data:
        category_counts[get_category(filename)] += 1

    categories = sorted(category_counts.keys())
    counts = [category_counts[c] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    wedges, texts, autotexts = ax.pie(
        counts, labels=categories, autopct="%1.1f%%",
        colors=colors, pctdistance=0.85, startangle=90,
    )
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(8)

    ax.set_title("Document Category Distribution", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Category distribution -> {output_path}")
