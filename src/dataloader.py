"""
data_loader.py
----------------
Extracts Telugu text from the Mahabharatam PDF via OCR (EasyOCR).

Why OCR and not direct text extraction:
The source PDF is typeset in legacy non-Unicode Telugu fonts (Praveena,
Priyaanka — Modular Infotech family). Direct text extraction (PyMuPDF,
pdfplumber, PyPDF2) pulls out the raw character codes, which are garbage —
the fonts remap Telugu glyphs onto arbitrary Latin codepoints purely for
*visual rendering*. OCR sidesteps this entirely by reading the rendered
glyphs as pixels, independent of the underlying (broken) character codes.

Output: a JSONL file, one JSON object per line:
    {"page": 5, "text": "..."}

JSONL (not a single JSON array) so that:
  - progress is saved incrementally — a crash/interrupt mid-run doesn't
    lose everything
  - reruns can skip already-processed pages (checkpointing)
"""

import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image
import io

# ---- config ----------------------------------------------------------
PDF_PATH = "../data/MAHABHARATAM.pdf"
OUTPUT_PATH = "data/mahabharatam_ocr.jsonl"
DPI = 300
LANG = ["te"]            # EasyOCR uses "te" for Telugu (not "tel")
START_PAGE = 0
END_PAGE = None
GPU = False               # set True if you have a working CUDA GPU
# -----------------------------------------------------------------------

# Load the reader once (loads/downloads model weights on first run)
print("Loading EasyOCR model (first run downloads weights, may take a bit)...")
reader = easyocr.Reader(LANG, gpu=GPU)


def already_processed_pages(output_path: str) -> set:
    done = set()
    p = Path(output_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done.add(obj["page"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def ocr_page(page: "fitz.Page", dpi: int) -> str:
    """Rasterize a single PDF page and OCR it with EasyOCR."""
    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    result = reader.readtext(np.array(img), detail=0, paragraph=True)
    return "\n".join(result).strip()


def main():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    end_page = END_PAGE if END_PAGE is not None else total_pages - 1

    done = already_processed_pages(OUTPUT_PATH)
    if done:
        print(f"Resuming: {len(done)} pages already processed, skipping those.")

    pages_to_process = [p for p in range(START_PAGE, end_page + 1) if p not in done]
    print(f"Processing {len(pages_to_process)} pages (of {total_pages} total)...")

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for i, page_num in enumerate(pages_to_process):
            t0 = time.time()
            page = doc[page_num]
            text = ocr_page(page, DPI)

            record = {"page": page_num, "text": text}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            elapsed = time.time() - t0
            print(f"[{i+1}/{len(pages_to_process)}] page {page_num} "
                  f"-> {len(text)} chars in {elapsed:.1f}s")

    print(f"Done. Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()