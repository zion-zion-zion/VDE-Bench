"""
Krippendorff's Alpha with Bootstrap Confidence Intervals and Visualization.

This module computes Krippendorff's Alpha to measure inter-annotator agreement,
estimates confidence intervals via bootstrap resampling, and generates
publication-quality figures (dot plot with CI error bars).

Krippendorff's Alpha is more general than Fleiss' Kappa:
  - Supports ordinal, interval, and ratio scales (not just nominal).
  - Handles missing data naturally.
  - Works with any number of raters (not necessarily the same across items).

Usage:
    python krippendorff_alpha.py --input ratings.csv --dimensions spatial textual style
    python krippendorff_alpha.py --input ratings.json --plot output_figure.pdf
    python krippendorff_alpha.py --demo
"""

import argparse
import csv
import json
import os
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================
#   Krippendorff's Alpha
# =============================
def _coincidence_matrix(reliability_data, value_domain):
    """Build the coincidence (agreement) matrix from reliability data."""
    V = len(value_domain)
    val_to_idx = {v: i for i, v in enumerate(value_domain)}
    o_matrix = np.zeros((V, V), dtype=np.float64)

    for item_ratings in reliability_data:
        valid = [r for r in item_ratings if not (isinstance(r, float) and np.isnan(r))]
        m_u = len(valid)
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(m_u):
                if i == j:
                    continue
                c = val_to_idx[valid[i]]
                k = val_to_idx[valid[j]]
                o_matrix[c][k] += 1.0 / (m_u - 1)

    return o_matrix


def compute_krippendorff_alpha(reliability_data, value_domain=None, level="ordinal"):
    """Compute Krippendorff's Alpha."""
    if value_domain is None:
        all_vals = set()
        for item in reliability_data:
            for v in item:
                if not (isinstance(v, float) and np.isnan(v)):
                    all_vals.add(v)
        value_domain = sorted(all_vals)

    V = len(value_domain)
    if V < 2:
        return 1.0

    o_matrix = _coincidence_matrix(reliability_data, value_domain)
    n_c = np.sum(o_matrix, axis=1)
    n_total = np.sum(n_c)

    if n_total == 0:
        return 0.0

    def delta_sq(c, k):
        if level == "nominal":
            return 0.0 if c == k else 1.0
        elif level == "ordinal":
            if c > k:
                c, k = k, c
            cum = sum(n_c[g] for g in range(c, k + 1)) - (n_c[c] + n_c[k]) / 2.0
            return cum ** 2
        elif level == "interval":
            return (value_domain[c] - value_domain[k]) ** 2
        elif level == "ratio":
            vc = value_domain[c]
            vk = value_domain[k]
            if vc + vk == 0:
                return 0.0
            return ((vc - vk) / (vc + vk)) ** 2
        else:
            raise ValueError(f"Unknown level: {level}")

    D_o = 0.0
    for c in range(V):
        for k in range(V):
            D_o += o_matrix[c][k] * delta_sq(c, k)
    D_o /= n_total

    D_e = 0.0
    for c in range(V):
        for k in range(V):
            D_e += n_c[c] * n_c[k] * delta_sq(c, k)
    D_e /= (n_total * (n_total - 1))

    if D_e == 0:
        return 1.0

    return 1.0 - D_o / D_e


# =============================
#   Bootstrap CI
# =============================
def bootstrap_alpha(reliability_data, value_domain=None, level="ordinal",
                    n_bootstrap=1000, confidence=0.95, seed=42):
    """Estimate Krippendorff's Alpha with bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    N = len(reliability_data)

    alpha = compute_krippendorff_alpha(reliability_data, value_domain, level)

    boot_alphas = []
    for _ in range(n_bootstrap):
        indices = rng.choice(N, size=N, replace=True)
        boot_data = [reliability_data[i] for i in indices]
        try:
            a = compute_krippendorff_alpha(boot_data, value_domain, level)
            boot_alphas.append(a)
        except Exception:
            continue

    boot_alphas = np.array(boot_alphas)

    lower_pct = (1.0 - confidence) / 2.0 * 100
    upper_pct = (1.0 + confidence) / 2.0 * 100
    ci_lower = np.percentile(boot_alphas, lower_pct)
    ci_upper = np.percentile(boot_alphas, upper_pct)

    return alpha, ci_lower, ci_upper, boot_alphas


def interpret_alpha(alpha):
    """Krippendorff (2004) recommendation."""
    if alpha >= 0.800:
        return "Reliable"
    elif alpha >= 0.667:
        return "Tentatively Acceptable"
    else:
        return "Unreliable"


# =============================
#   Data Loading
# =============================
def load_ratings_from_csv(csv_path, dimensions=None):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file {csv_path} is empty.")

    all_cols = list(rows[0].keys())
    id_cols = {"item_id", "rater_id"}
    if dimensions is None:
        dimensions = [c for c in all_cols if c not in id_cols]

    items = OrderedDict()
    for row in rows:
        item_id = row.get("item_id", "")
        items.setdefault(item_id, []).append(row)

    ratings_dict = {}
    for dim in dimensions:
        dim_ratings = []
        for _item_id, rater_rows in items.items():
            scores = []
            for r in rater_rows:
                val = r.get(dim, "")
                if val == "" or val is None:
                    scores.append(np.nan)
                else:
                    scores.append(int(val))
            dim_ratings.append(scores)
        ratings_dict[dim] = dim_ratings
    return ratings_dict


def load_ratings_from_json(json_path, dimensions=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"JSON file {json_path} is empty.")

    if dimensions is None:
        first_rating = data[0]["ratings"][0]
        dimensions = [k for k in first_rating.keys() if k != "rater_id"]

    ratings_dict = {}
    for dim in dimensions:
        dim_ratings = []
        for item in data:
            scores = []
            for r in item["ratings"]:
                val = r.get(dim, None)
                scores.append(np.nan if val is None else val)
            dim_ratings.append(scores)
        ratings_dict[dim] = dim_ratings
    return ratings_dict


# =============================
#   Visualization
# =============================
def plot_alpha_with_ci(results, output_path="krippendorff_alpha_ci.pdf",
                       title=None, figsize=(7, 4.5), dpi=300,
                       display_names=None):
    """Dot plot with CI error bars + reliability bands."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "figure.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    dims = list(results.keys())
    n_dims = len(dims)

    alphas = [results[d]["alpha"] for d in dims]
    ci_lowers = [results[d]["ci_lower"] for d in dims]
    ci_uppers = [results[d]["ci_upper"] for d in dims]

    err_lower = [alphas[i] - ci_lowers[i] for i in range(n_dims)]
    err_upper = [ci_uppers[i] - alphas[i] for i in range(n_dims)]

    fig, ax = plt.subplots(figsize=figsize)

    ax.axvspan(-0.1, 0.667, alpha=0.08, color="#E74C3C", zorder=0)
    ax.axvspan(0.667, 0.800, alpha=0.08, color="#F39C12", zorder=0)
    ax.axvspan(0.800, 1.05, alpha=0.08, color="#27AE60", zorder=0)

    ax.axvline(x=0.667, color="#E67E22", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.axvline(x=0.800, color="#27AE60", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)

    display_names = display_names or {}
    y_positions = np.arange(n_dims)
    colors = []
    for a in alphas:
        if a >= 0.800:
            colors.append("#27AE60")
        elif a >= 0.667:
            colors.append("#F39C12")
        else:
            colors.append("#E74C3C")

    for i in range(n_dims):
        ax.errorbar(
            alphas[i], y_positions[i],
            xerr=[[err_lower[i]], [err_upper[i]]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=2.0,
            capsize=5,
            capthick=1.5,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            f"  {alphas[i]:.3f}",
            xy=(ci_uppers[i], y_positions[i]),
            va="center", ha="left",
            fontsize=12, fontweight="bold",
            color=colors[i],
            zorder=6,
        )

    y_labels = [display_names.get(d, d) for d in dims]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel(r"Krippendorff's $\alpha$")

    if title:
        ax.set_title(title)

    legend_patches = [
        mpatches.Patch(facecolor="#E74C3C", alpha=0.15, edgecolor="#E74C3C",
                       label=r"$\alpha < 0.667$ (Unreliable)"),
        mpatches.Patch(facecolor="#F39C12", alpha=0.15, edgecolor="#F39C12",
                       label=r"$0.667 \leq \alpha < 0.800$ (Tentative)"),
        mpatches.Patch(facecolor="#27AE60", alpha=0.15, edgecolor="#27AE60",
                       label=r"$\alpha \geq 0.800$ (Reliable)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        fancybox=True,
    )

    ax.xaxis.grid(True, linestyle=":", alpha=0.4)
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Figure saved to {output_path}")


def plot_bootstrap_distribution(results, output_path="bootstrap_dist.pdf",
                                figsize=(8, 3.5), dpi=300, display_names=None):
    """Bootstrap distribution of alpha for each dimension."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "figure.dpi": dpi,
        "savefig.bbox": "tight",
    })

    dims = list(results.keys())
    n_dims = len(dims)
    display_names = display_names or {}

    fig, axes = plt.subplots(1, n_dims, figsize=figsize, sharey=True)
    if n_dims == 1:
        axes = [axes]

    palette = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12"]

    for i, dim in enumerate(dims):
        ax = axes[i]
        boot_alphas = results[dim].get("boot_alphas", np.array([]))
        alpha = results[dim]["alpha"]
        ci_lower = results[dim]["ci_lower"]
        ci_upper = results[dim]["ci_upper"]

        color = palette[i % len(palette)]

        if len(boot_alphas) > 0:
            ax.hist(boot_alphas, bins=40, color=color, alpha=0.4,
                    edgecolor=color, linewidth=0.5, density=True)

        ax.axvline(alpha, color=color, linewidth=2, linestyle="-",
                   label=rf"$\alpha = {alpha:.3f}$")
        ax.axvline(ci_lower, color=color, linewidth=1.2, linestyle="--", alpha=0.7)
        ax.axvline(ci_upper, color=color, linewidth=1.2, linestyle="--", alpha=0.7)
        ax.axvspan(ci_lower, ci_upper, alpha=0.12, color=color)

        ax.set_title(display_names.get(dim, dim), fontsize=13, fontweight="bold")
        ax.set_xlabel(r"$\alpha$")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=10, loc="upper left", frameon=True, framealpha=0.8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Bootstrap distribution figure saved to {output_path}")


# =============================
#   Main
# =============================
def evaluate_agreement(input_path=None, ratings_dict=None, dimensions=None,
                       value_domain=None, level="ordinal",
                       n_bootstrap=1000, confidence=0.95, seed=42):
    if ratings_dict is None:
        if input_path is None:
            raise ValueError("Either input_path or ratings_dict must be provided.")
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".csv":
            ratings_dict = load_ratings_from_csv(input_path, dimensions)
        elif ext == ".json":
            ratings_dict = load_ratings_from_json(input_path, dimensions)
        else:
            raise ValueError(f"Unsupported file format: {ext}.")

    results = OrderedDict()
    for dim, reliability_data in ratings_dict.items():
        N = len(reliability_data)
        n_raters = len(reliability_data[0]) if N > 0 else 0

        print(f"\n{'='*55}")
        print(f"  Dimension: {dim}")
        print(f"  Items: {N}, Raters per item: {n_raters}")
        print(f"  Level: {level}, Bootstrap: {n_bootstrap}, CI: {confidence*100:.0f}%")

        alpha, ci_lower, ci_upper, boot_alphas = bootstrap_alpha(
            reliability_data, value_domain, level, n_bootstrap, confidence, seed
        )

        interpretation = interpret_alpha(alpha)
        results[dim] = {
            "alpha": round(alpha, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "interpretation": interpretation,
            "boot_alphas": boot_alphas,
            "num_items": N,
            "num_raters": n_raters,
            "confidence": confidence,
        }

        print(f"  Krippendorff's Alpha: {alpha:.4f}")
        print(f"  {confidence*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Interpretation: {interpretation}")

    return results


def save_results(results, output_path):
    serializable = OrderedDict()
    for dim, res in results.items():
        serializable[dim] = {k: v for k, v in res.items() if k != "boot_alphas"}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


def run_demo(output_dir=None):
    print("=" * 60)
    print("  Krippendorff's Alpha Demo (Synthetic Data)")
    print("=" * 60)

    np.random.seed(42)

    N, n_raters = 100, 5
    categories = [1, 2, 3]

    def _gen(noise_rate):
        base = np.random.choice(categories, size=N)
        data = []
        for i in range(N):
            ratings = []
            for _ in range(n_raters):
                if np.random.random() < noise_rate:
                    delta = np.random.choice([-1, 1])
                    noisy = int(np.clip(base[i] + delta, min(categories), max(categories)))
                    ratings.append(noisy)
                else:
                    ratings.append(int(base[i]))
            data.append(ratings)
        return data

    ratings_dict = OrderedDict([
        ("spatial", _gen(0.05)),
        ("textual", _gen(0.08)),
        ("style",   _gen(0.10)),
    ])

    results = evaluate_agreement(
        ratings_dict=ratings_dict,
        value_domain=categories,
        level="ordinal",
        n_bootstrap=2000,
        confidence=0.95,
        seed=42,
    )

    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    for dim, res in results.items():
        print(f"  {dim:>10s}: alpha={res['alpha']:.4f}  "
              f"95% CI=[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]  "
              f"({res['interpretation']})")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    fig1_path = os.path.join(output_dir, "krippendorff_alpha_ci.pdf")
    fig2_path = os.path.join(output_dir, "bootstrap_distribution.pdf")
    plot_alpha_with_ci(results, output_path=fig1_path,
                       title="Inter-Annotator Agreement (Krippendorff's $\\alpha$)")
    plot_bootstrap_distribution(results, output_path=fig2_path)
    print(f"\nDemo complete. Figures saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--dimensions", nargs="+", default=None)
    parser.add_argument("--level", type=str, default="ordinal",
                        choices=["nominal", "ordinal", "interval", "ratio"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--plot-dist", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--demo", action="store_true")

    args = parser.parse_args()

    if args.demo or args.input is None:
        if args.input is None and not args.demo:
            print("No input file specified. Running demo instead...\n")
        run_demo()
        return

    results = evaluate_agreement(
        input_path=args.input,
        dimensions=args.dimensions,
        level=args.level,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )

    if args.output:
        save_results(results, args.output)

    if args.plot:
        plot_alpha_with_ci(results, output_path=args.plot,
                           title="Inter-Annotator Agreement (Krippendorff's $\\alpha$)")
    if args.plot_dist:
        plot_bootstrap_distribution(results, output_path=args.plot_dist)

    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    for dim, res in results.items():
        ci = res["confidence"] * 100
        print(f"  {dim}: alpha={res['alpha']:.4f}  "
              f"{ci:.0f}% CI=[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]  "
              f"({res['interpretation']})")


if __name__ == "__main__":
    main()
