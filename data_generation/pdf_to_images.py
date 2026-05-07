"""Convert every PDF in a directory to one JPEG per page."""

from __future__ import annotations

import argparse
import os

import fitz  # PyMuPDF


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf_dir", required=True, help="Directory containing PDF files.")
    p.add_argument("--output_dir", default=None,
                   help="Output directory.  Defaults to <pdf_dir>/images.")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def pdf_to_jpgs(pdf_dir: str, output_dir: str, dpi: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(pdf_dir):
        if not file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, file)
        pdf_name = os.path.splitext(file)[0]
        print(f"Processing: {file}")

        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            out_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num + 1}.jpg")
            pix.save(out_path)
        doc.close()
        print(f"Finished: {file}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.pdf_dir, "images")
    pdf_to_jpgs(args.pdf_dir, output_dir, args.dpi)


if __name__ == "__main__":
    main()
