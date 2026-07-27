import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

ROOT        = Path(__file__).resolve().parent.parent
INPUT_PATH  = ROOT / "data" / "mahabharatam_chunks.jsonl"
OUTPUT_PATH = ROOT / "data" / "bm25_index.pkl"


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer for Telugu text.

    Why not a Telugu-specific tokenizer:
    BM25 doesn't need perfect linguistic tokenization — it just needs
    consistent tokenization at index time and query time. As long as we
    tokenize the same way for both, term matching works correctly.
    Telugu words are naturally space-separated in modern text, so
    whitespace splitting captures the important terms well.
    """
    import re
    # Split on whitespace and common punctuation, keep Telugu characters intact
    tokens = re.split(r'[\s\u0964\u0965,।\.!\?\"\'\(\)\[\]]+', text.lower())
    return [t for t in tokens if len(t) > 1]  # drop single chars


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {INPUT_PATH}...")
    chunks = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks. Building BM25 index...")

    # Tokenize each chunk's text
    corpus_tokens = [tokenize(c["text"]) for c in chunks]

    # Build BM25 index
    # BM25Okapi is the standard variant — Okapi BM25 with k1=1.5, b=0.75
    # k1 controls term frequency saturation (how much repeated terms matter)
    # b controls document length normalisation
    bm25 = BM25Okapi(corpus_tokens, k1=1.5, b=0.75)

    # Save index + chunk metadata needed for retrieval
    payload = {
        "bm25":     bm25,
        "chunks":   chunks,          # full chunk records for metadata lookup
        "tokenizer": tokenize,       # save the tokenizer function too
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"BM25 index built and saved to {OUTPUT_PATH}")
    print(f"Corpus size: {len(chunks)} chunks")
    print(f"Vocabulary size: {len(bm25.idf)} unique terms")


if __name__ == "__main__":
    main()