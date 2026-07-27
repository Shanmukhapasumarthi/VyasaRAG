from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle
import re
import sys

import chromadb
from sentence_transformers import SentenceTransformer

ROOT            = Path(__file__).resolve().parent.parent
CHROMA_PATH     = str(ROOT / "data" / "chroma_db")
BM25_PATH       = ROOT / "data" / "bm25_index.pkl"
COLLECTION_NAME = "mahabharatam"
EMBEDDING_MODEL = "BAAI/bge-m3"
TOP_K           = 5
RRF_K           = 60       # RRF smoothing constant — 60 is the standard value
CANDIDATE_K     = 20       # fetch this many candidates from each system before fusion
QUERY_PREFIX    = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    """
    Query/corpus tokenizer for BM25.

    This MUST match the tokenizer used in bm25_index.py when the index
    was built — the query is tokenized with this and scored against a
    corpus tokenized the same way. If they differ, keyword retrieval
    silently degrades.

    `\\w+` with Python 3's default Unicode matching captures Telugu
    (and other Indic) word characters correctly.
    """
    return re.findall(r"\w+", text.lower())


@dataclass
class RetrievalResult:
    chunk_id:      int
    chapter_num:   int | None
    chapter_title: str
    pages:         list[int]
    token_count:   int
    text:          str
    score:         float
    semantic_rank: int | None = None
    bm25_rank:     int | None = None


class MahabharatamRetriever:
    """
    Hybrid retriever — loads BGE-M3, Chroma, and BM25 index once at startup.
    Call .retrieve() for every user query.
    """

    def __init__(self):
        print("Loading BGE-M3 embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded.")

        self.client     = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"Connected to Chroma collection '{COLLECTION_NAME}' "
              f"({self.collection.count()} vectors).")

        # Load BM25 index if available
        self.bm25       = None
        self.bm25_chunks = None
        self._tokenize  = None

        if BM25_PATH.exists():
            print("Loading BM25 index...")
            # The pickled tokenizer was saved as a reference to __main__.tokenize
            # (because bm25_index.py defined it in __main__ at build time).
            # Register our tokenize() under whatever module is currently __main__
            # so pickle can resolve that reference from ANY entry point
            # (search.py, translate_text.py, app.py, Streamlit, ...).
            sys.modules["__main__"].tokenize = tokenize
            with open(BM25_PATH, "rb") as f:
                payload = pickle.load(f)
            self.bm25        = payload["bm25"]
            self.bm25_chunks = payload["chunks"]
            self._tokenize   = payload["tokenizer"]
            print(f"BM25 index loaded ({len(self.bm25_chunks)} chunks).")
        else:
            print("⚠  BM25 index not found — falling back to semantic-only search.")
            print("   Run: python src/bm25_index.py")

        print()

    # ── Public API ──────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        chapter_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieval with Reciprocal Rank Fusion.
        Falls back to semantic-only if BM25 index is not built yet.
        """
        if self.bm25 is not None:
            return self._hybrid_retrieve(query, top_k, chapter_filter)
        else:
            return self._semantic_retrieve(query, top_k, chapter_filter)

    # ── Hybrid retrieval ─────────────────────────────────────────────

    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        chapter_filter: str | None,
    ) -> list[RetrievalResult]:

        # Step 1 — semantic candidates from Chroma
        semantic_hits = self._semantic_candidates(query, CANDIDATE_K)

        # Step 2 — BM25 keyword candidates
        bm25_hits = self._bm25_candidates(query, CANDIDATE_K)

        # Step 3 — Reciprocal Rank Fusion
        fused = self._rrf_fuse(semantic_hits, bm25_hits)

        # Step 4 — chapter filter (post-filter)
        if chapter_filter:
            fused = [r for r in fused
                     if chapter_filter in (r.chapter_title or "")]

        return fused[:top_k]

    def _semantic_candidates(self, query: str, k: int) -> list[RetrievalResult]:
        """Encode query with BGE-M3 and retrieve top-k from Chroma."""
        prefixed = QUERY_PREFIX + query
        vec = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings=[vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            meta      = results["metadatas"][0][i]
            pages_raw = meta.get("pages", "")
            pages     = [int(p) for p in pages_raw.split(",") if p] if pages_raw else []
            hits.append(RetrievalResult(
                chunk_id      = int(results["ids"][0][i]),
                chapter_num   = meta.get("chapter_num") or None,
                chapter_title = meta.get("chapter_title", ""),
                pages         = pages,
                token_count   = meta.get("token_count", 0),
                text          = results["documents"][0][i],
                score         = round(1.0 - results["distances"][0][i], 4),
                semantic_rank = i + 1,
            ))
        return hits

    def _bm25_candidates(self, query: str, k: int) -> list[RetrievalResult]:
        """Tokenize query and retrieve top-k from BM25 index."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get indices of top-k scores
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:k]

        hits = []
        for rank, idx in enumerate(top_indices):
            c = self.bm25_chunks[idx]
            pages_raw = c.get("pages", [])
            if isinstance(pages_raw, str):
                pages = [int(p) for p in pages_raw.split(",") if p]
            else:
                pages = pages_raw or []

            hits.append(RetrievalResult(
                chunk_id      = c["chunk_id"],
                chapter_num   = c.get("chapter_num"),
                chapter_title = c.get("chapter_title", ""),
                pages         = pages,
                token_count   = c.get("token_count", 0),
                text          = c["text"],
                score         = float(scores[idx]),
                bm25_rank     = rank + 1,
            ))
        return hits

    def _rrf_fuse(
        self,
        semantic_hits: list[RetrievalResult],
        bm25_hits: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion.

        For each chunk that appears in either list:
          rrf_score = 1/(k + semantic_rank) + 1/(k + bm25_rank)

        Chunks missing from one list get a penalty rank of CANDIDATE_K + 1
        (as if they were ranked last in that system).
        """
        # Build rank lookup dicts: chunk_id → rank (1-indexed)
        sem_rank = {r.chunk_id: r.semantic_rank for r in semantic_hits}
        bm25_rank = {r.chunk_id: r.bm25_rank    for r in bm25_hits}

        # Merge all unique chunk IDs
        all_ids = set(sem_rank.keys()) | set(bm25_rank.keys())

        # Build a lookup of full result objects
        result_map: dict[int, RetrievalResult] = {}
        for r in semantic_hits + bm25_hits:
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        penalty = CANDIDATE_K + 1   # rank assigned if absent from one system

        fused_scores: list[tuple[float, int]] = []
        for cid in all_ids:
            s_rank = sem_rank.get(cid,  penalty)
            b_rank = bm25_rank.get(cid, penalty)
            rrf    = 1.0 / (RRF_K + s_rank) + 1.0 / (RRF_K + b_rank)
            fused_scores.append((rrf, cid))

        fused_scores.sort(reverse=True)

        results = []
        for rrf_score, cid in fused_scores:
            r = result_map[cid]
            results.append(RetrievalResult(
                chunk_id      = r.chunk_id,
                chapter_num   = r.chapter_num,
                chapter_title = r.chapter_title,
                pages         = r.pages,
                token_count   = r.token_count,
                text          = r.text,
                score         = round(rrf_score, 6),
                semantic_rank = sem_rank.get(cid),
                bm25_rank     = bm25_rank.get(cid),
            ))
        return results

    # ── Semantic-only fallback ───────────────────────────────────────

    def _semantic_retrieve(
        self,
        query: str,
        top_k: int,
        chapter_filter: str | None,
    ) -> list[RetrievalResult]:
        fetch_k = top_k * 3 if chapter_filter else top_k
        hits    = self._semantic_candidates(query, fetch_k)

        if chapter_filter:
            hits = [r for r in hits
                    if chapter_filter in (r.chapter_title or "")]

        return hits[:top_k]


def format_context_for_llm(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a labelled context string for the LLM prompt."""
    parts = []
    for r in results:
        chapter_label = r.chapter_title or "Prologue"
        pages_label   = f"pp. {', '.join(str(p) for p in r.pages)}" if r.pages else ""
        sem_info      = f"sem_rank={r.semantic_rank}" if r.semantic_rank else ""
        bm25_info     = f"bm25_rank={r.bm25_rank}"   if r.bm25_rank    else ""
        rank_info     = " | ".join(filter(None, [sem_info, bm25_info]))
        header        = f"[Chapter: {chapter_label} | {pages_label} | {rank_info}]"
        parts.append(f"{header}\n{r.text}")
    return "\n\n---\n\n".join(parts)


# ── Smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = MahabharatamRetriever()

    queries = [
        ("Telugu",  "భీష్ముడు ఎవరు?"),
        ("English", "Who is Bhishma and what vow did he take?"),
        ("Hindi",   "भीष्म ने क्या प्रतिज्ञा ली?"),
    ]

    for lang, query in queries:
        print(f"\n{'='*60}")
        print(f"Query ({lang}): {query}")
        print("="*60)
        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results):
            print(f"  Rank {i+1} | chunk {r.chunk_id} | "
                  f"sem={r.semantic_rank} bm25={r.bm25_rank} | "
                  f"rrf={r.score:.4f} | ch: {r.chapter_title}")
            print(f"  {r.text[:80]}...")