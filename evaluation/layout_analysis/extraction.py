"""Layout element extraction using vision LLM."""

import os
import re
import json
import glob
import time
import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from tqdm import tqdm

from constants import ELEMENT_TYPES, EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alias map: maps common VLM output variations to canonical element types.
# ---------------------------------------------------------------------------
ALIAS_MAP = {}
for _et in ELEMENT_TYPES:
    ALIAS_MAP[_et] = _et
    ALIAS_MAP[_et.replace("_", "")] = _et
    ALIAS_MAP[_et.replace("_", " ")] = _et

_EXTRA_ALIASES = {
    "text": "text_block", "paragraph": "text_block", "body": "text_block",
    "body_text": "text_block", "body text": "text_block", "paragraphs": "text_block",
    "text_blocks": "text_block", "text blocks": "text_block",
    "heading": "title", "headings": "title", "titles": "title",
    "section_title": "title", "section title": "title",
    "subtitle": "title", "sub_title": "title",
    "page_header": "header", "page header": "header", "running_header": "header",
    "page_footer": "footer", "page footer": "footer", "running_footer": "footer",
    "page number": "page_number", "pagenumber": "page_number",
    "page_no": "page_number", "page no": "page_number",
    "tables": "table",
    "image": "figure", "images": "figure", "figures": "figure",
    "chart": "figure", "charts": "figure", "diagram": "figure",
    "illustration": "figure", "picture": "figure", "photo": "figure",
    "equation": "formula", "equations": "formula", "formulas": "formula",
    "math": "formula", "math_formula": "formula", "mathematical_formula": "formula",
    "bullet_list": "list", "bullet list": "list", "numbered_list": "list",
    "numbered list": "list", "bulleted_list": "list", "lists": "list",
    "captions": "caption", "figure_caption": "caption", "table_caption": "caption",
    "footnotes": "footnote", "endnote": "footnote", "endnotes": "footnote",
    "code": "code_block", "code block": "code_block", "codeblock": "code_block",
    "code_snippet": "code_block", "code snippet": "code_block",
    "margin_note": "sidebar", "margin note": "sidebar", "sidebars": "sidebar",
    "watermarks": "watermark",
    "logos": "logo", "emblem": "logo",
    "separators": "separator", "divider": "separator", "line": "separator",
    "horizontal_line": "separator", "horizontal line": "separator",
    "separator_line": "separator",
}
ALIAS_MAP.update(_EXTRA_ALIASES)


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg",
        ".gif": "gif", ".webp": "webp", ".bmp": "bmp",
    }
    return mime_map.get(ext, "png")


def _normalize_key(key: str) -> Optional[str]:
    k = key.lower().strip()
    if k in ALIAS_MAP:
        return ALIAS_MAP[k]
    if k.endswith("s") and k[:-1] in ALIAS_MAP:
        return ALIAS_MAP[k[:-1]]
    return None


def parse_element_counts(response_text: str) -> Dict[str, int]:
    """Parse the LLM response to extract element counts."""
    if not response_text or not response_text.strip():
        return {}

    for match in re.findall(r'\{[^{}]*\}', response_text, re.DOTALL):
        try:
            obj = json.loads(match)
            counts: Dict[str, int] = {}
            for key, val in obj.items():
                canonical = _normalize_key(key)
                if canonical is not None:
                    try:
                        v = int(val)
                        if v > 0:
                            counts[canonical] = counts.get(canonical, 0) + v
                    except (ValueError, TypeError):
                        continue
            if counts:
                return counts
        except (json.JSONDecodeError, ValueError):
            continue

    counts: Dict[str, int] = {}
    for alias, canonical in ALIAS_MAP.items():
        escaped = re.escape(alias)
        m = re.search(
            rf'["\']?{escaped}["\']?\s*:\s*(\d+)',
            response_text,
            re.IGNORECASE,
        )
        if m:
            v = int(m.group(1))
            if v > 0:
                counts[canonical] = max(counts.get(canonical, 0), v)

    return counts


def extract_elements_from_image(
    image_path: str,
    client,
    model: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Dict[str, int]:
    """Use a vision LLM to extract layout elements from a single image."""
    img_base64 = encode_image_base64(image_path)
    img_ext = get_image_extension(image_path)
    fname = os.path.basename(image_path)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_ext};base64,{img_base64}"
                            },
                        },
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                }],
                temperature=0.0,
                max_tokens=30000,
            )
            content = response.choices[0].message.content
            result = parse_element_counts(content)

            if not result:
                logger.warning(
                    "[PARSE_EMPTY] %s (attempt %d/%d) - raw response:\n%s",
                    fname, attempt, max_retries, content,
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    continue

            return result

        except Exception as e:
            logger.error(
                "[REQUEST_ERROR] %s (attempt %d/%d): %s",
                fname, attempt, max_retries, e,
            )
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)

    logger.error("[FAILED] %s - all %d attempts failed, returning empty.", fname, max_retries)
    return {}


def extract_all_elements(
    image_dir: str,
    model: str,
    port: int,
    api_key: str = "EMPTY",
    workers: int = 32,
    max_samples: Optional[int] = None,
    host: str = "localhost",
) -> Dict[str, Dict[str, int]]:
    """Extract layout elements from all images in the directory."""
    from openai import OpenAI

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    client = OpenAI(api_key=api_key, base_url=f"http://{host}:{port}/v1")

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    if max_samples:
        image_files = image_files[:max_samples]

    print(f"Found {len(image_files)} images to process (workers={workers}).")

    results = {}
    success_count = 0
    empty_count = 0
    error_files = []

    def process_one(img_path):
        fname = os.path.basename(img_path)
        counts = extract_elements_from_image(img_path, client, model)
        return fname, counts

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, p): p for p in image_files}
        with tqdm(total=len(image_files), desc="Extracting layout elements") as pbar:
            for future in as_completed(futures):
                fname, counts = future.result()
                results[fname] = counts
                if counts:
                    success_count += 1
                else:
                    empty_count += 1
                    error_files.append(fname)
                pbar.update(1)

    total = len(image_files)
    print(f"\n{'='*60}")
    print(f"  Extraction Summary")
    print(f"{'='*60}")
    print(f"  Total images:     {total}")
    if total:
        print(f"  Successful:       {success_count} ({100*success_count/total:.1f}%)")
        print(f"  Empty/Failed:     {empty_count} ({100*empty_count/total:.1f}%)")
    print(f"{'='*60}")

    if error_files and len(error_files) <= 20:
        print(f"\n  Empty result files:")
        for ef in error_files:
            print(f"    - {ef}")
    elif error_files:
        print(f"\n  First 20 empty result files (of {len(error_files)}):")
        for ef in error_files[:20]:
            print(f"    - {ef}")

    return results
