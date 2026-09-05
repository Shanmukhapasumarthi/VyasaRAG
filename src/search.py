from __future__ import annotations
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import pickle
import re
import sys

os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

ROOT                    = Path(__file__).resolve().parent.parent
CHROMA_PATH             = str(ROOT / "data" / "chroma_db")
BM25_PATH               = ROOT / "data" / "bm25_index.pkl"
COLLECTION_NAME         = "mahabharatam"
EMBEDDING_MODEL         = "BAAI/bge-m3"
RERANKER_MODEL          = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_ENABLED        = True
EXPANSION_MODEL          = "llama-3.1-8b-instant"
FALLBACK_EXPANSION_MODEL = "openai/gpt-oss-20b"
QUERY_EXPANSION_ENABLED  = True
TOP_K                   = 5
RRF_K                   = 60       # RRF smoothing constant — 60 is the standard value
CANDIDATE_K             = 20       # fetch this many candidates from each system before fusion
QUERY_PREFIX            = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    r"""
    Query/corpus tokenizer for BM25.

    This MUST match the tokenizer used in bm25_index.py when the index
    was built — the query is tokenized with this and scored against a
    corpus tokenized the same way. If they differ, keyword retrieval
    silently degrades.

    `\w+` with Python 3's default Unicode matching captures Telugu
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
    Hybrid retriever with Query Expansion, Cross-Encoder reranking, and Query-Level Cache.
    Loads BGE-M3, Cross-Encoder, Groq client, Chroma, and BM25 index once at startup.
    Call .retrieve() for every user query.
    """

    def __init__(self):
        print("Loading BGE-M3 embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded.")

        self.reranker = None
        if RERANKER_ENABLED:
            print(f"Loading Cross-Encoder reranker ({RERANKER_MODEL})...")
            self.reranker = CrossEncoder(RERANKER_MODEL)
            print("Reranker loaded.")

        self.groq_client = None
        if QUERY_EXPANSION_ENABLED:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                try:
                    self.groq_client = Groq(api_key=groq_key)
                    print(f"Groq client loaded for query expansion ({EXPANSION_MODEL}).")
                except Exception as e:
                    print(f"⚠ Failed to initialize Groq client for query expansion: {e}")

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
            # Register tokenize() under whatever module is currently __main__
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

        # Initialize query-level in-memory cache and statistics counters
        self._cache: dict[str, list[RetrievalResult]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        print()

    # ── Cache Helpers ────────────────────────────────────────────────

    def _make_cache_key(self, query: str, top_k: int, chapter_filter: str | None) -> str:
        """MD5 hash of (query.strip().lower() + str(top_k) + str(chapter_filter))."""
        raw = query.strip().lower() + str(top_k) + str(chapter_filter)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get_cache_stats(self) -> dict[str, int]:
        """Expose cache metrics: hits, misses, size."""
        return {
            "cache_hits":   self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size":   len(self._cache),
        }

    # ── Public API ──────────────────────────────────────────────────

    def _expand_query(self, query: str) -> str:
        """
        Calls Groq (llama-3.1-8b-instant) to list 3-5 related search terms or alternative phrasings.
        Appends terms to original query before BM25 tokenization and BGE-M3 encoding.
        """
        if not QUERY_EXPANSION_ENABLED or not self.groq_client:
            return query

        prompt = (
            "Given this question about the Mahabharata, list 3-5 related search terms "
            "or alternative phrasings that would help retrieve relevant passages. "
            f"Return only the terms, comma-separated, no explanation.\n\nQuestion: {query}"
        )

        try:
            try:
                response = self.groq_client.chat.completions.create(
                    model=EXPANSION_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100,
                )
            except Exception as err:
                err_msg = str(err)
                if "404" in err_msg or "model_not_found" in err_msg:
                    response = self.groq_client.chat.completions.create(
                        model=FALLBACK_EXPANSION_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=100,
                    )
                else:
                    raise err

            terms = response.choices[0].message.content.strip()
            if terms:
                expanded = f"{query} {terms}"
                return expanded
        except Exception as e:
            print(f"⚠ Query expansion failed: {e}")

        return query

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        chapter_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieval with Reciprocal Rank Fusion, Query Expansion, Cross-Encoder Reranking, and Cache.
        Falls back to semantic-only if BM25 index is not built yet.
        """
        cache_key = self._make_cache_key(query, top_k, chapter_filter)
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1
        search_query = self._expand_query(query) if QUERY_EXPANSION_ENABLED else query

        if self.bm25 is not None:
            results = self._hybrid_retrieve(query, search_query, top_k, chapter_filter)
        else:
            results = self._semantic_retrieve(query, search_query, top_k, chapter_filter)

        self._cache[cache_key] = results
        return results

    # ── Hybrid retrieval ─────────────────────────────────────────────

    def _hybrid_retrieve(
        self,
        query: str,
        search_query: str | None = None,
        top_k: int = TOP_K,
        chapter_filter: str | None = None,
    ) -> list[RetrievalResult]:
        sq = search_query or query

        # Step 1 — semantic candidates from Chroma (using search_query)
        semantic_hits = self._semantic_candidates(sq, CANDIDATE_K)

        # Step 2 — BM25 keyword candidates (using search_query)
        bm25_hits = self._bm25_candidates(sq, CANDIDATE_K)

        # Step 3 — Reciprocal Rank Fusion
        fused = self._rrf_fuse(semantic_hits, bm25_hits)

        # Step 4 — chapter filter (post-filter)
        if chapter_filter:
            fused = [r for r in fused
                     if chapter_filter in (r.chapter_title or "")]

        # Step 5 — Cross-Encoder Reranking on top-20 candidates (using original query)
        candidates = fused[:CANDIDATE_K]
        return self._rerank_candidates(query, candidates, top_k)

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
        sem_rank = {r.chunk_id: r.semantic_rank for r in semantic_hits}
        bm25_rank = {r.chunk_id: r.bm25_rank    for r in bm25_hits}

        all_ids = set(sem_rank.keys()) | set(bm25_rank.keys())

        result_map: dict[int, RetrievalResult] = {}
        for r in semantic_hits + bm25_hits:
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        penalty = CANDIDATE_K + 1

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

    def _rerank_candidates(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Rerank candidate results using CrossEncoder model if RERANKER_ENABLED is True.
        """
        if not RERANKER_ENABLED or self.reranker is None or not candidates:
            return candidates[:top_k]

        pairs = [(query, r.text) for r in candidates]
        scores = self.reranker.predict(pairs)

        reranked = []
        for r, score in zip(candidates, scores):
            reranked.append(RetrievalResult(
                chunk_id      = r.chunk_id,
                chapter_num   = r.chapter_num,
                chapter_title = r.chapter_title,
                pages         = r.pages,
                token_count   = r.token_count,
                text          = r.text,
                score         = round(float(score), 4),
                semantic_rank = r.semantic_rank,
                bm25_rank     = r.bm25_rank,
            ))

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]

    # ── Semantic-only fallback ───────────────────────────────────────

    def _semantic_retrieve(
        self,
        query: str,
        search_query: str | None = None,
        top_k: int = TOP_K,
        chapter_filter: str | None = None,
    ) -> list[RetrievalResult]:
        sq = search_query or query
        fetch_k = CANDIDATE_K if (chapter_filter or RERANKER_ENABLED) else top_k
        hits    = self._semantic_candidates(sq, fetch_k)

        if chapter_filter:
            hits = [r for r in hits
                    if chapter_filter in (r.chapter_title or "")]

        return self._rerank_candidates(query, hits[:CANDIDATE_K], top_k)


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
                  f"score={r.score:.4f} | ch: {r.chapter_title}")
            print(f"  {r.text[:80]}...")

    print("\nCache Stats:", retriever.get_cache_stats())