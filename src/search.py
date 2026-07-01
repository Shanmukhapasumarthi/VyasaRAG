"""
search.py
----------
Runtime retrieval layer — the bridge between a user's query and the
Telugu Mahabharata chunks stored in Chroma.

What this does (maps directly to your architecture diagram):
  [User Query (Lang A)]
        ↓
  [BGE-M3 Multilingual Embedding]   <- encodes query directly, no translation
        ↓ (vector)
  [Chroma Cross-lingual Vector DB]  <- matches query vector against Telugu chunk vectors
        ↓ (top-k chunks in Telugu)
  returned to app.py / main.py for LLM generation

Key design decisions:
1. NO query translation before retrieval.
   BGE-M3 was trained on parallel multilingual data, so an English/Hindi/Telugu
   query and a Telugu document chunk encoding the same *meaning* will naturally
   land close together in the 1024-dim vector space. Translation before retrieval
   adds latency, compounds errors, and is simply unnecessary with a properly
   multilingual embedding model.

2. Query instruction prefix for BGE-M3.
   BGE-M3 was fine-tuned with an instruction prefix on the query side
   ("Represent this sentence for searching relevant passages: <query>").
   Applying it on queries (NOT on documents — we skipped it in embedding.py
   intentionally) improves retrieval accuracy by ~2-4% on benchmarks.

3. Optional chapter filter.
   If the user specifies a chapter name, Chroma's metadata filter narrows
   retrieval to that chapter only — useful for "in the Bhishma chapter, what..."
   type questions.

4. Returns structured RetrievalResult objects, not raw Chroma dicts.
   Keeps app.py / main.py clean — they get typed objects, not nested raw JSON.

Input:  A query string (any language BGE-M3 supports)
        data/chroma_db/  (Chroma persistent store from vector_store.py)
Output: list[RetrievalResult] — top-k chunks with text + metadata + score
"""

from __future__ import annotations
from dataclasses import dataclass, field

import chromadb
from sentence_transformers import SentenceTransformer

# ---- config -------------------------------------------------------
# ---- config -------------------------------------------------------
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CHROMA_PATH  = DATA_DIR / "chroma_db"

COLLECTION_NAME = "mahabharatam"
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K           = 5      # number of chunks to retrieve per query
QUERY_PREFIX    = "Represent this sentence for searching relevant passages: "
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


@dataclass
class RetrievalResult:
    chunk_id:      int
    chapter_num:   int | None
    chapter_title: str
    pages:         list[int]
    token_count:   int
    text:          str
    score:         float          # cosine similarity (higher = more relevant)


class MahabharatamRetriever:
    """
    Initialise once (loads model + opens Chroma), then call .retrieve()
    repeatedly — avoids reloading the 2.2 GB model on every request.
    Designed to be instantiated as a singleton in app.py at startup.
    """

    def __init__(self):
        print("Loading BGE-M3 embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded.")

        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"Connected to Chroma collection '{COLLECTION_NAME}' "
              f"({self.collection.count()} vectors).\n")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        chapter_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Encode query and return top_k most relevant chunks.

        Args:
            query:          User query in any language (Telugu, English, Hindi …)
            top_k:          How many chunks to return
            chapter_filter: Optional chapter_title substring to restrict search.
                            e.g. "భీష్మ" returns only chunks from chapters
                            whose title contains that string.
        """
        # Apply query instruction prefix (improves BGE-M3 retrieval accuracy)
        prefixed = QUERY_PREFIX + query

        query_vector = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        # Fetch more candidates when filtering so we still get top_k after
        fetch_k = top_k * 3 if chapter_filter else top_k

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results = []
        for i in range(len(results["ids"][0])):
            meta  = results["metadatas"][0][i]

            # Post-filter by chapter title substring (version-proof vs Chroma $contains)
            if chapter_filter and chapter_filter not in meta.get("chapter_title", ""):
                continue

            dist  = results["distances"][0][i]
            score = 1.0 - dist   # Chroma cosine distance -> similarity

            pages_raw = meta.get("pages", "")
            pages = (
                [int(p) for p in pages_raw.split(",") if p]
                if pages_raw else []
            )

            retrieval_results.append(RetrievalResult(
                chunk_id      = int(results["ids"][0][i]),
                chapter_num   = meta.get("chapter_num") or None,
                chapter_title = meta.get("chapter_title", ""),
                pages         = pages,
                token_count   = meta.get("token_count", 0),
                text          = results["documents"][0][i],
                score         = round(score, 4),
            ))

            if len(retrieval_results) == top_k:
                break

        return retrieval_results


def format_context_for_llm(results: list[RetrievalResult]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.
    Each chunk is labelled with its chapter and page source so the LLM can
    cite accurately in its answer.
    """
    parts = []
    for r in results:
        chapter_label = r.chapter_title or "Prologue"
        pages_label   = f"pp. {', '.join(str(p) for p in r.pages)}" if r.pages else ""
        header        = f"[Chapter: {chapter_label} | {pages_label} | relevance: {r.score}]"
        parts.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(parts)


# ---- Quick smoke-test (run directly: python3 search.py) -----------
if __name__ == "__main__":
    retriever = MahabharatamRetriever()

    test_queries = [
        ("Telugu",  "భీష్ముడు ఎవరు?"),
        ("English", "Who is Bhishma and what vow did he take?"),
        ("Hindi",   "भीष्म ने क्या प्रतिज्ञा ली?"),
    ]

    for lang, query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query ({lang}): {query}")
        print("="*60)
        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results):
            print(f"\n  Rank {i+1} | score={r.score} | "
                  f"chapter='{r.chapter_title}' | pages={r.pages}")
            print(f"  {r.text[:120]}...")

        print("\n--- Formatted context for LLM ---")
        print(format_context_for_llm(results)[:400], "...")