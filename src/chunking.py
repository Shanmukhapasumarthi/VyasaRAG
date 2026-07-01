"""
chunking.py
------------
Recursive semantic chunking on the cleaned, chapter-tagged OCR output
from clean_text.py.

Approach:
1. Group pages by chapter (consecutive pages sharing the same chapter_num
   are concatenated into one continuous text blob first). This matters
   because OCR'd text per page can end mid-sentence purely due to where
   the PDF page happened to break — chunking page-by-page would bake that
   artificial break into your chunks. Chunking at the chapter level avoids
   that.
2. Recursively split each chapter's text on a hierarchy of separators
   (paragraph -> sentence -> word), so chunks break at natural semantic
   boundaries wherever possible instead of at an arbitrary character count.
3. Greedily merge the resulting pieces into ~MAX_TOKENS-sized chunks, with
   OVERLAP_TOKENS of trailing context carried into the next chunk (so a
   sentence/idea split across a chunk boundary isn't completely lost to
   retrieval).
4. Each chunk keeps chapter_num, chapter_title, and the source page range
   it was drawn from, as metadata for later citation in answers.

Token counts use the actual BGE-M3 tokenizer (same one used in embedding.py)
so chunk sizing matches what the embedding model will actually see — not an
approximation. First run will download the tokenizer from Hugging Face
(needs internet once; cached locally after that).

Input:  data/mahabharatam_clean.jsonl  (from clean_text.py)
Output: data/mahabharatam_chunks.jsonl
        {"chunk_id": 0, "chapter_num": 1, "chapter_title": "...",
         "pages": [18, 19], "token_count": 487, "text": "..."}
"""

import json
import re
from pathlib import Path
from transformers import AutoTokenizer

# ---- config -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "mahabharatam_clean.jsonl"
OUTPUT_PATH = DATA_DIR / "mahabharatam_chunks.jsonl"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"   # must match embedding.py
MAX_TOKENS = 500
OVERLAP_TOKENS = 50
# ---------------------------------------------------------------------

# Hierarchy of separators, tried in order: paragraph break first, then
# Telugu/standard sentence enders, then a hard fallback to plain newline
# and whitespace if a "paragraph" turns out to be one giant run-on block.
SEPARATORS = ["\n\n", "। ", ". ", "! ", "? ", "\n", " "]
PAGE_MARKER_RE = re.compile(r"\[\[PAGE:(\d+)\]\]")

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_text_recursive(text: str, separators: list, max_tokens: int) -> list:
    """Recursively split on a hierarchy of separators until every piece
    fits within max_tokens. Falls back to a hard token-based slice if no
    separator helps (e.g. one giant run-on sentence with no punctuation)."""
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    if not separators:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return [
            tokenizer.decode(tokens[i:i + max_tokens])
            for i in range(0, len(tokens), max_tokens)
        ]

    sep = separators[0]
    parts = text.split(sep)
    if len(parts) == 1:
        return split_text_recursive(text, separators[1:], max_tokens)

    pieces = [p + sep for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])

    result = []
    for piece in pieces:
        if count_tokens(piece) <= max_tokens:
            result.append(piece)
        else:
            result.extend(split_text_recursive(piece, separators[1:], max_tokens))
    return result


def merge_with_overlap(units: list, max_tokens: int, overlap_tokens: int) -> list:
    """Greedily pack small units into ~max_tokens chunks, carrying a tail
    of the previous chunk forward as overlap for retrieval continuity."""
    chunks = []
    current, current_tok = [], 0

    for unit in units:
        u_tok = count_tokens(unit)
        if current and current_tok + u_tok > max_tokens:
            chunks.append("".join(current))
            overlap, overlap_tok = [], 0
            for u in reversed(current):
                t = count_tokens(u)
                if overlap_tok + t > overlap_tokens:
                    break
                overlap.insert(0, u)
                overlap_tok += t
            current, current_tok = overlap, overlap_tok
        current.append(unit)
        current_tok += u_tok

    if current:
        chunks.append("".join(current))
    return chunks


def extract_pages_and_strip(chunk_text: str):
    pages = sorted(set(int(p) for p in PAGE_MARKER_RE.findall(chunk_text)))
    clean = PAGE_MARKER_RE.sub("", chunk_text).strip()
    return pages, clean


def group_by_chapter(records: list) -> list:
    """Group consecutive page-records sharing the same (chapter_num,
    chapter_title) into one chapter group."""
    groups = []
    current = None
    for r in records:
        key = (r["chapter_num"], r["chapter_title"])
        if current is None or current["key"] != key:
            current = {"key": key, "pages": []}
            groups.append(current)
        current["pages"].append(r)
    return groups


def main():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    chapters = group_by_chapter(records)

    chunk_id = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for chapter in chapters:
            chapter_num, chapter_title = chapter["key"]
            combined = "".join(
                f"[[PAGE:{p['page']}]]{p['text']}\n\n" for p in chapter["pages"]
            )

            units = split_text_recursive(combined, SEPARATORS, MAX_TOKENS)
            raw_chunks = merge_with_overlap(units, MAX_TOKENS, OVERLAP_TOKENS)

            for raw_chunk in raw_chunks:
                pages, clean_text = extract_pages_and_strip(raw_chunk)
                if not clean_text:
                    continue
                record = {
                    "chunk_id": chunk_id,
                    "chapter_num": chapter_num,
                    "chapter_title": chapter_title,
                    "pages": pages,
                    "token_count": count_tokens(clean_text),
                    "text": clean_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_id += 1

    print(f"Wrote {chunk_id} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()