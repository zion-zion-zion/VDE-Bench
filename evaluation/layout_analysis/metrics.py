"""Evaluation metrics for document layout analysis."""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from constants import get_category


def compute_density_matrix(
    element_data: Dict[str, Dict[str, int]],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute the element density matrix d_{c,e}.

    d_{c,e} = average count of element e per page in category c.
    """
    category_pages: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    for filename, counts in element_data.items():
        cat = get_category(filename)
        category_pages[cat].append(counts)

    all_elements = set()
    for counts in element_data.values():
        all_elements.update(counts.keys())

    elements = sorted(all_elements)
    categories = sorted(category_pages.keys())

    density_matrix = np.zeros((len(categories), len(elements)))
    for i, cat in enumerate(categories):
        pages = category_pages[cat]
        if not pages:
            continue
        for j, elem in enumerate(elements):
            total = sum(p.get(elem, 0) for p in pages)
            density_matrix[i, j] = total / len(pages)

    return density_matrix, categories, elements


def compute_zscore_matrix(density_matrix: np.ndarray) -> np.ndarray:
    """Column-wise (per-element) z-score normalization."""
    mu = density_matrix.mean(axis=0)
    sigma = density_matrix.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (density_matrix - mu) / sigma


def compute_layout_complexity(element_data: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Layout Complexity = average number of distinct element types per page."""
    category_pages: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    for filename, counts in element_data.items():
        category_pages[get_category(filename)].append(counts)

    complexity = {}
    for cat, pages in category_pages.items():
        if not pages:
            complexity[cat] = 0.0
            continue
        distinct_counts = [len([v for v in p.values() if v > 0]) for p in pages]
        complexity[cat] = np.mean(distinct_counts)
    return complexity


def compute_element_richness(element_data: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Element Richness = average total number of elements per page."""
    category_pages: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    for filename, counts in element_data.items():
        category_pages[get_category(filename)].append(counts)

    richness = {}
    for cat, pages in category_pages.items():
        if not pages:
            richness[cat] = 0.0
            continue
        richness[cat] = np.mean([sum(p.values()) for p in pages])
    return richness


def compute_layout_entropy(element_data: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Shannon entropy of element type distribution per category."""
    category_pages: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    for filename, counts in element_data.items():
        category_pages[get_category(filename)].append(counts)

    entropy = {}
    for cat, pages in category_pages.items():
        if not pages:
            entropy[cat] = 0.0
            continue

        total_counts = defaultdict(int)
        for p in pages:
            for elem, cnt in p.items():
                total_counts[elem] += cnt

        total = sum(total_counts.values())
        if total == 0:
            entropy[cat] = 0.0
            continue

        h = 0.0
        for cnt in total_counts.values():
            if cnt > 0:
                prob = cnt / total
                h -= prob * np.log2(prob)
        entropy[cat] = h
    return entropy


def compute_visual_density(
    element_data: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, float]]:
    """Compute text vs. visual element ratio per category."""
    TEXT_ELEMENTS = {"text_block", "title", "header", "footer", "page_number",
                     "caption", "footnote", "code_block"}
    VISUAL_ELEMENTS = {"table", "figure", "formula", "list", "sidebar",
                       "logo", "separator", "watermark"}

    category_pages: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    for filename, counts in element_data.items():
        category_pages[get_category(filename)].append(counts)

    ratios = {}
    for cat, pages in category_pages.items():
        text_total, visual_total = 0, 0
        for p in pages:
            for elem, cnt in p.items():
                if elem in TEXT_ELEMENTS:
                    text_total += cnt
                elif elem in VISUAL_ELEMENTS:
                    visual_total += cnt

        total = text_total + visual_total
        ratios[cat] = {
            "text_count": text_total / len(pages) if pages else 0,
            "visual_count": visual_total / len(pages) if pages else 0,
            "text_ratio": text_total / total if total > 0 else 0,
            "visual_ratio": visual_total / total if total > 0 else 0,
        }
    return ratios
