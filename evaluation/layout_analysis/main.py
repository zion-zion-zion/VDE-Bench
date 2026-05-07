"""
Document Layout Evaluation System - Main Entry Point.

Usage:
    # Step 1: Extract layout elements from images
    python main.py extract --image_dir /path/to/images --output ./layout_elements.json \
        --model <model_name> --port 8000

    # Step 2: Evaluate and visualize (from cached extraction results)
    python main.py evaluate --input ./layout_elements.json --output_dir ./layout_results

    # Or run both steps together:
    python main.py run --image_dir /path/to/images --output_dir ./layout_results \
        --model <model_name> --port 8000
"""

import os
import json
import argparse
from typing import Dict

from extraction import extract_all_elements
from metrics import (
    compute_density_matrix,
    compute_zscore_matrix,
    compute_layout_complexity,
    compute_element_richness,
    compute_layout_entropy,
    compute_visual_density,
)
from visualization import (
    plot_density_heatmap,
    plot_complexity_bar,
    plot_entropy_bar,
    plot_text_visual_ratio,
    plot_category_distribution,
)
from report import generate_report


def run_extraction(args):
    """Run the element extraction step."""
    results = extract_all_elements(
        image_dir=args.image_dir,
        model=args.model,
        port=args.port,
        api_key=args.api_key,
        workers=args.workers,
        max_samples=args.max_samples,
        host=getattr(args, "host", "localhost"),
    )

    output_path = (
        args.output
        if hasattr(args, "output") and args.output
        else os.path.join(args.output_dir, "layout_elements.json")
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Extraction results -> {output_path}")
    return results


def run_evaluation(element_data: Dict[str, Dict[str, int]], output_dir: str):
    """Run the evaluation and visualization step."""
    os.makedirs(output_dir, exist_ok=True)

    density_matrix, categories, elements = compute_density_matrix(element_data)
    zscore_matrix = compute_zscore_matrix(density_matrix)
    complexity = compute_layout_complexity(element_data)
    richness = compute_element_richness(element_data)
    entropy_vals = compute_layout_entropy(element_data)
    ratios = compute_visual_density(element_data)

    plot_density_heatmap(
        density_matrix, zscore_matrix, categories, elements,
        os.path.join(output_dir, "heatmap_density_zscore.png"),
    )
    plot_complexity_bar(
        complexity, richness,
        os.path.join(output_dir, "bar_complexity_richness.png"),
    )
    plot_entropy_bar(
        entropy_vals,
        os.path.join(output_dir, "bar_entropy.png"),
    )
    plot_text_visual_ratio(
        ratios,
        os.path.join(output_dir, "bar_text_visual_ratio.png"),
    )
    plot_category_distribution(
        element_data,
        os.path.join(output_dir, "pie_category_distribution.png"),
    )

    generate_report(
        element_data, density_matrix, zscore_matrix,
        categories, elements, complexity, richness, entropy_vals, ratios,
        os.path.join(output_dir, "layout_evaluation_report.txt"),
    )

    metrics = {
        "density": {
            cat: {elem: float(density_matrix[i, j]) for j, elem in enumerate(elements)}
            for i, cat in enumerate(categories)
        },
        "zscore": {
            cat: {elem: float(zscore_matrix[i, j]) for j, elem in enumerate(elements)}
            for i, cat in enumerate(categories)
        },
        "complexity": {k: float(v) for k, v in complexity.items()},
        "richness": {k: float(v) for k, v in richness.items()},
        "entropy": {k: float(v) for k, v in entropy_vals.items()},
        "text_visual_ratio": ratios,
    }
    metrics_path = os.path.join(output_dir, "layout_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Metrics JSON -> {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Document Layout Evaluation System")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # Extract sub-command
    p_extract = subparsers.add_parser("extract", help="Extract layout elements from images")
    p_extract.add_argument("--image_dir", type=str, required=True)
    p_extract.add_argument("--output", type=str, default="layout_elements.json")
    p_extract.add_argument("--model", type=str, required=True)
    p_extract.add_argument("--host", type=str, default="localhost")
    p_extract.add_argument("--port", type=int, default=8000)
    p_extract.add_argument("--api_key", type=str, default="EMPTY")
    p_extract.add_argument("--workers", type=int, default=32)
    p_extract.add_argument("--max_samples", type=int, default=None)

    # Evaluate sub-command
    p_eval = subparsers.add_parser("evaluate", help="Evaluate from cached extraction")
    p_eval.add_argument("--input", type=str, required=True)
    p_eval.add_argument("--output_dir", type=str, default="./layout_results")

    # Run sub-command (extract + evaluate)
    p_run = subparsers.add_parser("run", help="Full pipeline (extract + evaluate)")
    p_run.add_argument("--image_dir", type=str, required=True)
    p_run.add_argument("--output_dir", type=str, default="./layout_results")
    p_run.add_argument("--model", type=str, required=True)
    p_run.add_argument("--host", type=str, default="localhost")
    p_run.add_argument("--port", type=int, default=8000)
    p_run.add_argument("--api_key", type=str, default="EMPTY")
    p_run.add_argument("--workers", type=int, default=32)
    p_run.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    if args.command == "extract":
        run_extraction(args)

    elif args.command == "evaluate":
        print(f"Loading extraction results from: {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            element_data = json.load(f)
        print(f"Loaded {len(element_data)} pages.")
        run_evaluation(element_data, args.output_dir)

    elif args.command == "run":
        element_data = run_extraction(args)
        run_evaluation(element_data, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
