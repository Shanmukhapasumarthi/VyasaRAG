# మహాభారతం — Multilingual RAG (VyasaRAG)

Ask questions about the Telugu Mahabharata in **any language** — Telugu, English, Hindi, Tamil, or others — and get answers grounded in the original Telugu source text, streamed back in the language you asked in.

Built as an advanced AI/ML Retrieval-Augmented Generation (RAG) system over the 434-page Telugu Mahabharata.

---

## Architecture

```
                                  ┌──────────────────────────┐
                                  │   User Query (Any Lang)  │
                                  └────────────┬─────────────┘
                                               │
                                  ┌────────────▼─────────────┐
                                  │ 3-Stage Language Detect  │
                                  │ (Unicode/Intent/Lang)    │
                                  └────────────┬─────────────┘
                                               │
                                  ┌────────────▼─────────────┐
                                  │  MD5 Query-Level Cache   │
                                  └─────┬──────────────┬─────┘
                             Cache Hit  │              │ Cache Miss
                                        │              │
                   ┌────────────────────▼───┐      ┌───▼──────────────────────┐
                   │ Return Cached Results  │      │ Query Expansion (Groq)   │
                   └────────────────────────┘      └───────────┬──────────────┘
                                                               │
                                         ┌─────────────────────┴─────────────────────┐
                                         │                                           │
                            ┌────────────▼─────────────┐                ┌────────────▼─────────────┐
                            │ Vector Search (BGE-M3)   │                │   BM25 Keyword Search    │
                            │   Chroma DB Top-20       │                │      Pickle Top-20       │
                            └────────────┬─────────────┘                └────────────┬─────────────┘
                                         │                                           │
                                         └─────────────────────┬─────────────────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │ Reciprocal Rank Fusion   │
                                                  │     (RRF Constant k=60)  │
                                                  └────────────┬─────────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │ Cross-Encoder Reranker   │
                                                  │  (MiniLM-L-6-v2 Top-K)   │
                                                  └────────────┬─────────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │ Context Formatting & LLM │
                                                  │ Answer Gen (Groq 120B)   │
                                                  └──────────────────────────┘
```

---

## Tech Stack & Enhancements

| Component | Choice / Model | Description & Purpose |
|---|---|---|
| **OCR Ingestion** | Tesseract (`tel`) / PyMuPDF | Extracts text from non-Unicode legacy Telugu PDF fonts into structured Unicode Telugu. |
| **Dense Embedding** | `BAAI/bge-m3` | 1024-dim cross-lingual semantic embedding model with 8192 token context. |
| **Lexical Search** | BM25 Okapi (`rank-bm25`) | Indic-aware regex tokenizer capturing keyword matches across names & places. |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) | Fuses dense vector and sparse keyword rankings ($k=60$). |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-scores top-20 hybrid candidate passages for deep semantic relevance. |
| **Query Expansion** | Groq `llama-3.1-8b-instant` | Expands user questions into 3-5 alternative search terms prior to retrieval. |
| **Query Cache** | In-Memory Session MD5 Cache | Instant responses for repeated queries; tracks hits, misses, and cache size. |
| **LLM Generation** | Groq `openai/gpt-oss-120b` | Grounded multilingual generation from retrieved Telugu passages. |
| **Backend API** | FastAPI + SSE Streaming | Streaming/JSON endpoints, Jinja2 template rendering, and `/health` metrics. |

---

## Project Structure

```
VyasaRAG/
├── app.py                      # FastAPI server (GET /, POST /ask, POST /ask/stream, GET /health)
├── requirements.txt            # Package dependencies
├── PROJECT_DOCUMENTATION.md    # In-depth architectural documentation
├── README.md                   # Project overview (this file)
│
├── src/                        # Source backend modules
│   ├── search.py               # MahabharatamRetriever (Hybrid, Reranker, Expansion, Cache)
│   ├── evaluate.py             # Benchmark script for 20 hand-written QA pairs
│   ├── translate_text.py       # Language detection & Groq answer generation
│   ├── bm25index.py            # BM25 index generator
│   ├── vectorstore.py          # Chroma vector DB builder
│   ├── embeddings.py           # BGE-M3 embedding generator
│   ├── chunking.py             # Text chunking logic
│   ├── cleantext.py            # Text cleaning & chapter tagging
│   └── dataloader.py           # PDF loader & OCR pipeline
│
├── tests/                      # Automated test suite
│   ├── test_search.py          # Unit tests for search, reranking, expansion, and cache
│   └── test_evaluate.py        # Unit tests for evaluation benchmark structure
│
├── data/
│   ├── MAHABHARATAM.pdf        # Source Telugu PDF
│   ├── chroma_db/              # Persistent Chroma vector store
│   ├── bm25_index.pkl          # Serialized BM25 index
│   └── eval_results.json       # Benchmark hit rate results JSON
│
├── template/
│   └── index.html              # Web UI HTML template
└── static/
    └── style.css               # Web UI styles
```

---

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run Unit Tests
```bash
python -m unittest discover -s tests
```

### 4. Run Evaluation Benchmark
```bash
python src/evaluate.py
```

### 5. Launch Application Server
```bash
python app.py
```
Open **http://localhost:8000** in your web browser.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Renders the web interface |
| `POST` | `/ask` | Returns answer JSON + retrieved sources |
| `POST` | `/ask/stream` | Streams answer tokens via SSE |
| `GET` | `/health` | Server status, vector count, and cache statistics (`cache_hits`, `cache_misses`, `cache_size`) |

### Example: GET /health Response
```json
{
  "status": "ok",
  "project": "VyasaRAG",
  "retriever_loaded": true,
  "vector_count": 375,
  "conversations": 0,
  "cache_hits": 5,
  "cache_misses": 2,
  "cache_size": 2
}
```

---

## Evaluation Benchmark

Systematic Top-5 Hit Rate evaluation over **20 hand-written multilingual QA pairs** (7 Telugu, 7 English, 6 Hindi):

| Mode | Top-5 Hit Rate | Details |
|---|---|---|
| **Semantic-Only** | **55.0%** (11/20) | BGE-M3 vector search |
| **BM25-Only** | **15.0%** (3/20) | BM25 sparse keyword search |
| **Hybrid (BGE-M3 + BM25)** | **45.0%** (9/20) | Reciprocal Rank Fusion |
| **Hybrid + Cross-Encoder Reranker** | **30.0%** (6/20) | Hybrid candidates re-scored via MiniLM-L-6-v2 |

Results are exported to [`data/eval_results.json`](file:///c:/Users/P.V.H.Shanmukha/OneDrive/Documents/VyasaRAG/data/eval_results.json).

---

## Key Technical Decisions

1. **OCR over Direct PDF Extraction**: Source PDF uses legacy non-Unicode Telugu fonts (`Praveena`/`Priyaanka`). Direct text extraction yields corrupted codepoints; OCR extracts proper Unicode Telugu.
2. **Direct Cross-Lingual Search**: BGE-M3 eliminates query translation pre-steps, avoiding compounding translation errors before retrieval.
3. **Cross-Encoder Reranking**: Re-evaluates top candidate passages against the exact query string to filter out false positives.
4. **Session MD5 Caching**: Avoids redundant LLM & vector operations for duplicate user requests during a session.

---

## Author

**Shanmukha Pasumarthi**  
B.Tech (AI/ML), VIT-AP University  
[github.com/Shanmukhapasumarthi](https://github.com/Shanmukhapasumarthi)
