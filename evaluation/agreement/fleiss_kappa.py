"""Fleiss' Kappa computation for inter-annotator agreement evaluation.

This module computes Fleiss' Kappa to measure the reliability of agreement
among a fixed number of raters assigning categorical ratings to a set of
items.

Usage
-----
    python fleiss_kappa.py --input ratings.csv --dimensions spatial textual style
    python fleiss_kappa.py --input ratings.json --dimensions spatial textual style
    python fleiss_kappa.py --demo
"""

import argparse
import csv
import json
import os
from collections import OrderedDict

import numpy as np

try:
    from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss_kappa
    from statsmodels.stats.inter_rater import aggregate_raters
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# =============================
#   Fleiss' Kappa (Manual)
# =============================
def compute_fleiss_kappa(rating_matrix):
    """Compute Fleiss' Kappa from a category-count matrix.

    Args:
        rating_matrix: np.ndarray of shape (N, k), where N is the number of
                       items and k is the number of categories. Each entry
                       rating_matrix[i][j] is the number of raters who
                       assigned item i to category j.

    Returns:
        kappa: float, the Fleiss' Kappa value.
    """
    rating_matrix = np.asarray(rating_matrix, dtype=np.float64)
    N, k = rating_matrix.shape
    n = rating_matrix[0].sum()  # raters per item

    if n <= 1:
        raise ValueError("Need at least 2 raters per item to compute Fleiss' Kappa.")

    P_i = (1.0 / (n * (n - 1))) * (np.sum(rating_matrix ** 2, axis=1) - n)
    P_bar = np.mean(P_i)
    p_j = np.sum(rating_matrix, axis=0) / (N * n)
    P_e_bar = np.sum(p_j ** 2)

    if abs(1.0 - P_e_bar) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else 0.0

    kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)
    return kappa


def raw_ratings_to_matrix(raw_ratings, categories=None):
    """Convert raw ratings to a category-count matrix."""
    raw_ratings = np.asarray(raw_ratings)
    if categories is None:
        categories = sorted(set(raw_ratings.flatten()))

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    N, n = raw_ratings.shape
    k = len(categories)
    matrix = np.zeros((N, k), dtype=np.int64)

    for i in range(N):
        for j in range(n):
            val = raw_ratings[i, j]
            if val in cat_to_idx:
                matrix[i, cat_to_idx[val]] += 1

    return matrix, categories


# =============================
#   Fleiss' Kappa (statsmodels)
# =============================
def compute_fleiss_kappa_statsmodels(raw_ratings):
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required: pip install statsmodels")
    table, _ = aggregate_raters(raw_ratings)
    return sm_fleiss_kappa(table, method="fleiss")


# =============================
#   Interpretation
# =============================
def interpret_kappa(kappa):
    """Landis & Koch (1977) scale."""
    if kappa < 0:
        return "Poor (less than chance agreement)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


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
            scores = [int(r[dim]) for r in rater_rows]
            dim_ratings.append(scores)
        ratings_dict[dim] = np.array(dim_ratings)
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
            scores = [r[dim] for r in item["ratings"]]
            dim_ratings.append(scores)
        ratings_dict[dim] = np.array(dim_ratings)
    return ratings_dict


# =============================
#   Main Evaluation Pipeline
# =============================
def evaluate_agreement(input_path, dimensions=None, categories=None, use_statsmodels=False):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".csv":
        ratings_dict = load_ratings_from_csv(input_path, dimensions)
    elif ext == ".json":
        ratings_dict = load_ratings_from_json(input_path, dimensions)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json.")

    results = OrderedDict()
    for dim, raw_ratings in ratings_dict.items():
        N, n = raw_ratings.shape
        print(f"\n{'='*50}")
        print(f"Dimension: {dim}")
        print(f"  Items: {N}, Raters per item: {n}")

        if use_statsmodels and HAS_STATSMODELS:
            kappa = compute_fleiss_kappa_statsmodels(raw_ratings)
        else:
            matrix, _cats = raw_ratings_to_matrix(raw_ratings, categories)
            kappa = compute_fleiss_kappa(matrix)

        interpretation = interpret_kappa(kappa)
        results[dim] = {
            "kappa": round(kappa, 4),
            "interpretation": interpretation,
            "num_items": N,
            "num_raters": n,
        }
        print(f"  Fleiss' Kappa: {kappa:.4f}")
        print(f"  Agreement Level: {interpretation}")

    return results


def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


# =============================
#   Demo with Synthetic Data
# =============================
def run_demo():
    print("=" * 60)
    print("  Fleiss' Kappa Demo (Synthetic Data)")
    print("=" * 60)

    np.random.seed(42)
    N, n = 100, 3
    categories = [1, 2, 3]

    base_scores = np.random.choice(categories, size=N)
    high_agree = np.column_stack([base_scores] * n)
    noise_mask = np.random.random((N, n)) < 0.10
    noise_vals = np.random.choice(categories, size=(N, n))
    high_agree = np.where(noise_mask, noise_vals, high_agree)

    matrix_high, _ = raw_ratings_to_matrix(high_agree, categories)
    kappa_high = compute_fleiss_kappa(matrix_high)
    print(f"\n[High Agreement] Fleiss' Kappa = {kappa_high:.4f}")
    print(f"  Interpretation: {interpret_kappa(kappa_high)}")

    low_agree = np.random.choice(categories, size=(N, n))
    matrix_low, _ = raw_ratings_to_matrix(low_agree, categories)
    kappa_low = compute_fleiss_kappa(matrix_low)
    print(f"\n[Low Agreement (Random)] Fleiss' Kappa = {kappa_low:.4f}")
    print(f"  Interpretation: {interpret_kappa(kappa_low)}")

    if HAS_STATSMODELS:
        kappa_sm = compute_fleiss_kappa_statsmodels(high_agree)
        print(f"\n[Statsmodels Verification] Kappa = {kappa_sm:.4f}")
        print(f"  Match: {'Yes' if abs(kappa_high - kappa_sm) < 0.001 else 'No'}")
    else:
        print("\n[Note] statsmodels not installed, skipping verification.")

    print("\n" + "=" * 60)


# =============================
#   CLI
# =============================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--dimensions", nargs="+", default=None)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--use-statsmodels", action="store_true")
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
        categories=args.categories,
        use_statsmodels=args.use_statsmodels,
    )

    if args.output:
        save_results(results, args.output)
    else:
        print("\n\n" + "=" * 50)
        print("  Summary")
        print("=" * 50)
        for dim, res in results.items():
            print(f"  {dim}: kappa={res['kappa']:.4f} ({res['interpretation']})")


if __name__ == "__main__":
    main()
