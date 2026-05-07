"""Constants and configuration for the layout evaluation system."""

# Layout element types to detect
ELEMENT_TYPES = [
    "text_block",       # Regular paragraph / body text
    "title",            # Title / heading
    "header",           # Page header
    "footer",           # Page footer
    "page_number",      # Page number
    "table",            # Table
    "figure",           # Figure / image / chart / diagram
    "formula",          # Mathematical formula / equation
    "list",             # Bulleted or numbered list
    "caption",          # Figure / table caption
    "footnote",         # Footnote
    "code_block",       # Code snippet
    "sidebar",          # Sidebar / margin note
    "watermark",        # Watermark
    "logo",             # Logo
    "separator",        # Horizontal / vertical separator line
]

# Category extraction rules: prefix -> category name
#
# NOTE: these prefixes are specific to how VDE-Bench's raw document images
# are named (inherited from the OmniDocBench source).  Adjust the list if
# you are bringing your own dataset.
CATEGORY_RULES = [
    ("book_en_",           "Book (EN)"),
    ("book_zh_",           "Book (ZH)"),
    ("PPT_",               "PPT / Slides"),
    ("scihub_",            "Academic Paper"),
    ("yanbaopptmerge_",    "Report PPT"),
    ("yanbaor2_",          "Report R2"),
    ("color_textbook_",    "Color Textbook"),
    ("docstructbench_",    "DocStructBench"),
    ("eastmoney_",         "Financial Report"),
    ("exam_paper_",        "Exam Paper"),
    ("jiaocaineedrop_",    "Textbook (Ext)"),
    ("jiaocai_",           "Textbook"),
    ("magazine_",          "Magazine"),
    ("newspaper_",         "Newspaper"),
    ("notes_",             "Notes"),
]

# Prompt for the vision LLM to extract layout elements
EXTRACTION_PROMPT = """Count the layout elements on this document page image.

Element types:
text_block, title, header, footer, page_number, table, figure, formula, list, caption, footnote, code_block, sidebar, watermark, logo, separator

Rules:
- Do NOT transcribe, describe, or output any actual text content from the image.
- Do NOT explain your reasoning.
- ONLY output a single JSON object mapping element type names to their integer counts.
- Only include element types that are present (count > 0).

Output ONLY valid JSON, nothing else:
{"text_block": 3, "title": 1, "figure": 2}
"""


def get_category(filename: str) -> str:
    """Determine the document category from filename prefix."""
    for prefix, category in CATEGORY_RULES:
        if filename.startswith(prefix):
            return category
    return "Unknown"
