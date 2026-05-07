"""Document Layout Evaluation System."""

from .constants import ELEMENT_TYPES, CATEGORY_RULES, EXTRACTION_PROMPT
from .extraction import extract_all_elements, parse_element_counts
from .metrics import (
    compute_density_matrix,
    compute_zscore_matrix,
    compute_layout_complexity,
    compute_element_richness,
    compute_layout_entropy,
    compute_visual_density,
)
from .visualization import (
    plot_density_heatmap,
    plot_complexity_bar,
    plot_entropy_bar,
    plot_text_visual_ratio,
    plot_category_distribution,
)
from .report import generate_report
