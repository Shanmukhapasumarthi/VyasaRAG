import json
import re
from pathlib import Path

# ---- config -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "mahabharatam_ocr.jsonl"
OUTPUT_PATH = DATA_DIR / "mahabharatam_clean.jsonl"
CONTENT_START_PAGE = 18   # <- VERIFY this against your own full OCR output
# ---------------------------------------------------------------------

FOOTER_PATTERN = re.compile(r"మహాభారతం.{0,15}$")
CHAPTER_PATTERN = re.compile(r"అధ్యాయం\s*-?\s*(\d+)")


def strip_footer(text: str) -> str:
    """Remove the trailing running-footer line if present."""
    lines = text.split("\n")
    while lines and (lines[-1].strip() == "" or FOOTER_PATTERN.search(lines[-1])):
        lines.pop()
    return "\n".join(lines).strip()


def detect_chapter(text: str):
    """
    Look for a chapter marker ('అధ్యాయం - N') in this page's text.
    Returns (chapter_num, chapter_title) if found, else (None, None).
    Chapter title is assumed to be the next non-empty line after the marker.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = CHAPTER_PATTERN.search(line)
        if m:
            chapter_num = int(m.group(1))
            title = ""
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    title = lines[j].strip()
                    break
            return chapter_num, title
    return None, None


def main():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    current_chapter_num = None
    current_chapter_title = None

    with open(INPUT_PATH, "r", encoding="utf-8") as in_f, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:

        for line in in_f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            page_num = record["page"]
            raw_text = record["text"]

            if page_num < CONTENT_START_PAGE:
                continue  # skip front matter / TOC

            cleaned = strip_footer(raw_text)
            if not cleaned:
                continue  # skip pages that are now empty after cleaning

            ch_num, ch_title = detect_chapter(cleaned)
            if ch_num is not None:
                current_chapter_num = ch_num
                current_chapter_title = ch_title

            out_record = {
                "page": page_num,
                "chapter_num": current_chapter_num,
                "chapter_title": current_chapter_title,
                "text": cleaned,
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"Cleaned output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()