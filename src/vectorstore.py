"""
vector_store.py
----------------
Reads mahabharatam_embeddings.jsonl and upserts every chunk into a
persistent local Chroma collection.

Why Chroma (vs Pinecone / FAISS):
- Fully local — no API key, no network, no cost
- Persists to disk automatically (just point it at a folder)
- Stores metadata alongside vectors natively, so we can filter by
  chapter_num / chapter_title at query time without a separate DB
- Easy to swap for Pinecone later if you ever need cloud-scale

Collection design decisions:
- One collection for the whole book ("mahabharatam")
- chunk_id is the Chroma document ID → safe to rerun (upsert, not insert,
  so duplicates are overwritten, not doubled)
- Metadata stored per chunk: chapter_num, chapter_title, pages (as string),
  token_count — available for filtered retrieval or citation in answers
- Embeddings stored as-is (already L2-normalised from embedding.py),
  so cosine similarity = dot product — fast and accurate

Input:  data/mahabharatam_embeddings.jsonl  (from embedding.py)
Output: data/chroma_db/  (persistent Chroma directory)
"""

import json
from pathlib import Path

import chromadb

# ---- config -------------------------------------------------------
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_PATH   = DATA_DIR/"mahabharatam_embeddings.jsonl"
CHROMA_PATH  =  DATA_DIR/"chroma_db"
COLLECTION   = "mahabharatam"
BATCH_SIZE   = 100    # Chroma recommends batching large upserts
# ---------------------------------------------------------------------


def load_embeddings(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_metadata(record: dict) -> dict:
    """
    Chroma metadata values must be str | int | float | bool.
    - pages list -> comma-separated string (e.g. "21,22") so it's storable
    - None values -> empty string (Chroma rejects None)
    """
    return {
        "chapter_num":   record["chapter_num"] if record["chapter_num"] is not None else -1,
        "chapter_title": record["chapter_title"] or "",
        "pages":         ",".join(str(p) for p in record.get("pages", [])),
        "token_count":   record["token_count"],
    }


def main():
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {INPUT_PATH}...")
    records = load_embeddings(INPUT_PATH)
    print(f"Loaded {len(records)} chunks.\n")

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # match our normalised vectors
    )

    existing = collection.count()
    if existing > 0:
        print(f"Collection already has {existing} vectors — upserting (safe to rerun).\n")

    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        batch = records[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        collection.upsert(
            ids        = [str(r["chunk_id"]) for r in batch],
            embeddings = [r["embedding"]     for r in batch],
            documents  = [r["text"]          for r in batch],
            metadatas  = [build_metadata(r)  for r in batch],
        )

        pct = (batch_idx + 1) / total_batches * 100
        print(f"Batch {batch_idx+1}/{total_batches} ({pct:.0f}%) "
              f"| {len(batch)} chunks upserted")

    final_count = collection.count()
    print(f"\nDone. Collection '{COLLECTION}' now has {final_count} vectors.")
    print(f"Chroma DB persisted at: {CHROMA_PATH}/")


if __name__ == "__main__":
    main()