"""
embedding.py
-------------
Loads every chunk from chunking.py's output, generates BGE-M3 embeddings,
and writes everything to a single file ready for vector_store.py to ingest
into Chroma.

Why BGE-M3:
- Natively multilingual (100+ languages including Telugu)
- No translation step needed: a Telugu query encodes into the same vector
  space as Telugu document chunks, so cross-lingual similarity works
  without an intermediate English hop
- 8192 token context window (our 500-token chunks are well within this)
- Outputs dense vectors (1024-dim) — good enough for v1; hybrid
  dense+sparse available via FlagEmbedding if you want to improve recall later

Performance note:
- Full book will produce ~1500-2500 chunks depending on content density
- On CPU:  ~2-4 sec/chunk  -> expect 1-2 hours for the full corpus
- On GPU:  ~0.1-0.3 sec/chunk -> ~5-10 minutes
- Script batches chunks (BATCH_SIZE=32 by default) and checkpoints
  progress, so you can safely Ctrl+C and rerun — already-embedded chunks
  are skipped.

Input:  data/mahabharatam_chunks.jsonl  (from chunking.py)
Output: data/mahabharatam_embeddings.jsonl
        {
          "chunk_id": 0,
          "chapter_num": 1,
          "chapter_title": "భీష్మ ప్రతిజ్ఞ",
          "pages": [21, 22],
          "token_count": 108,
          "text": "...",
          "embedding": [0.021, -0.043, ...]   # 1024-dim float list
        }
"""

import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

# ---- config -------------------------------------------------------
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

INPUT_PATH  = DATA_DIR / "mahabharatam_chunks.jsonl"
OUTPUT_PATH = DATA_DIR / "mahabharatam_embeddings.jsonl"
MODEL_NAME  = "BAAI/bge-m3"
BATCH_SIZE  = 32    # lower to 8 if you get OOM on CPU; raise to 64 on GPU
# ---------------------------------------------------------------------


def already_embedded(output_path: str) -> set:
    """Return set of chunk_ids already written (for checkpoint/resume)."""
    done = set()
    p = Path(output_path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["chunk_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def load_chunks(input_path: str) -> list:
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    print("(First run downloads ~2.2 GB from Hugging Face — cached after that)")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")

    chunks = load_chunks(INPUT_PATH)
    done   = already_embedded(OUTPUT_PATH)

    pending = [c for c in chunks if c["chunk_id"] not in done]
    if not pending:
        print("All chunks already embedded. Nothing to do.")
        return

    if done:
        print(f"Resuming: {len(done)} chunks already done, {len(pending)} remaining.\n")
    else:
        print(f"Embedding {len(pending)} chunks (batch size {BATCH_SIZE})...\n")

    # BGE-M3 performs better with this query instruction prefix on the *query* side,
    # but for *documents* (what we embed here) no prefix is needed — plain text.
    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for batch_idx in range(total_batches):
            batch = pending[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            texts = [c["text"] for c in batch]

            t0 = time.time()
            vectors = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,   # cosine sim = dot product after L2 norm
                convert_to_numpy=True,
            )
            elapsed = time.time() - t0

            for chunk, vector in zip(batch, vectors):
                record = {**chunk, "embedding": vector.tolist()}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            pct = (batch_idx + 1) / total_batches * 100
            print(
                f"Batch {batch_idx+1}/{total_batches} ({pct:.0f}%) "
                f"| {len(batch)} chunks | {elapsed:.1f}s"
            )

    print(f"\nDone. Embeddings written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()