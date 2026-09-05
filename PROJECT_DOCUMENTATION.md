# VyasaRAG — Comprehensive Technical Documentation & Architecture Report

> **A Production-Grade Multilingual Retrieval-Augmented Generation (RAG) System over the Sacred Telugu Mahabharata (434 Pages)**  
> **Author**: Shanmukha Pasumarthi | B.Tech (AI/ML), VIT-AP University  
> **Repository**: VyasaRAG  

---

## Executive Summary

**VyasaRAG** is a specialized, production-ready Multilingual Retrieval-Augmented Generation (RAG) system engineered to answer complex questions over the 434-page Telugu *Mahabharatam* in **any language** (Telugu, English, Hindi, Tamil, Kannada, etc.). 

Rather than relying on brittle question-translation pipelines (translating queries to English and answers back), VyasaRAG utilizes **native cross-lingual dense representations** (`BAAI/bge-m3`) paired with an **Indic-aware sparse BM25 index**, **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder reranking**, **Groq-powered query expansion**, and **MD5 query-level caching**. Answers are grounded strictly in the source Telugu scripture and generated via Groq's high-speed inference engine (`openai/gpt-oss-120b`), streaming tokens back in the user's preferred language.

---

## 1. Problem Statement & Engineering Challenges

### 1.1 The Legacy Font Problem (Non-Unicode DTP Fonts)
* **The Challenge**: The source 434-page Telugu Mahabharata PDF was typeset using legacy proprietary desktop publishing (DTP) fonts (`Praveena`, `Priyaanka`) from Modular Infotech. These fonts visually display Telugu glyphs by mapping them onto arbitrary Latin/ASCII code points.
* **Why Traditional Extraction Failed**: Direct PDF extractors like `PyPDF2`, `pdfplumber`, or `fitz.get_text()` output corrupt pseudo-English character gibberish rather than authentic Telugu Unicode characters.
* **The Solution**: An OCR pipeline using `Tesseract` (`tel` language pack) and `PyMuPDF` renders each page at high DPI and performs optical character recognition on actual rendered pixel glyphs, producing valid Unicode Telugu text.

### 1.2 The "Translation Hop" Fallacy
* **The Challenge**: Conventional multilingual RAG systems translate non-English queries to English, search an English corpus, and translate the generated response back. This causes **double translation compounding**: subtle mythological nuances, proper names (e.g., *Devavrata*, *Pashupatastra*, *Lakshagriha*), and contextual intents get severely corrupted.
* **The Solution**: Zero-translation direct cross-lingual vector alignment. Because `BAAI/bge-m3` was pre-trained on massive parallel cross-lingual datasets, an English query ("Who is Bhishma?"), a Hindi query ("भीष्म कौन हैं?"), and Telugu text ("భీష్ముడి పూర్వనామం దేవవ్రతుడు") map into the same 1024-dimensional semantic space.

### 1.3 Grounding & Hallucination Suppression
* **The Challenge**: LLMs easily hallucinate mythological events from television adaptations or folklore when answering ungrounded prompts.
* **The Solution**: Strict retrieval-augmented grounding. Every query fetches the top matching scripture passages with chapter names and page numbers. The generation prompt explicitly instructs the LLM to refuse answering if the context does not support the claim, citing chapter and page numbers for auditability.

---

## 2. End-to-End System Architecture

```
                                  ┌──────────────────────────┐
                                  │      User Question       │
                                  │  (Telugu, English, etc.) │
                                  └────────────┬─────────────┘
                                               │
                                  ┌────────────▼─────────────┐
                                  │ 3-Stage Language Detect  │
                                  │  (Unicode/Intent/Lang)   │
                                  └────────────┬─────────────┘
                                               │
                                  ┌────────────▼─────────────┐
                                  │  MD5 Query-Level Cache   │
                                  └─────┬──────────────┬─────┘
                             Cache Hit  │              │ Cache Miss
                                        │              │
                   ┌────────────────────▼───┐      ┌───▼──────────────────────┐
                   │ Return Cached Results  │      │ Query Expansion (Groq)   │
                   │ (Instant sub-10ms)     │      │ (llama-3.1-8b-instant)   │
                   └────────────────────────┘      └───────────┬──────────────┘
                                                               │
                                         ┌─────────────────────┴─────────────────────┐
                                         │                                           │
                            ┌────────────▼─────────────┐                ┌────────────▼─────────────┐
                            │ Vector Search (BGE-M3)   │                │   BM25 Keyword Search    │
                            │   Chroma DB (Top-20)     │                │   Indic Regex (Top-20)   │
                            └────────────┬─────────────┘                └────────────┬─────────────┘
                                         │                                           │
                                         └─────────────────────┬─────────────────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │ Reciprocal Rank Fusion   │
                                                  │ (RRF Constant k = 60)    │
                                                  └────────────┬─────────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │ Cross-Encoder Reranker   │
                                                  │ (ms-marco-MiniLM-L-6-v2) │
                                                  └────────────┬─────────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │ Context Formatting & LLM │
                                                  │ Generation (Groq 120B)   │
                                                  └────────────┬─────────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │ SSE Stream to Frontend   │
                                                  └──────────────────────────┘
```

---

## 3. Data Processing Pipeline (ETL & Indexing)

The foundational corpus was built through a 5-step offline pipeline:

| Step | Script | Description | Output Artifact |
|---|---|---|---|
| **1. PDF Ingestion & OCR** | `src/dataloader.py` | Extracts pages 18–434 from `MAHABHARATAM.pdf`, applies DPI scaling and Tesseract OCR with checkpointing to survive interruptions. | `data/mahabharatam_ocr.jsonl` |
| **2. Text Sanitization** | `src/cleantext.py` | Strips recurring headers/footers, normalizes Indic Unicode codepoints, identifies chapter boundaries, and tags chunks with chapter numbers. | `data/mahabharatam_clean.jsonl` |
| **3. Semantic Chunking** | `src/chunking.py` | Performs recursive sentence-boundary aware chunking with target chunk size ~500 tokens and 50 token overlap. | `data/mahabharatam_chunks.jsonl` |
| **4. Dense Embeddings** | `src/embeddings.py` | Encodes chunks with `BAAI/bge-m3` into 1024-dim normalized vector embeddings. | `data/mahabharatam_embeddings.jsonl` |
| **5. Store Indexing** | `src/vectorstore.py` & `src/bm25index.py` | Loads vectors into persistent Chroma DB and serializes tokenized corpus into BM25 Okapi pickle. | `data/chroma_db/`, `data/bm25_index.pkl` |

---

## 4. The 4 Tier-1 RAG Enhancements

During recent system upgrades, four critical Tier-1 enhancements were systematically implemented and verified:

### 4.1 Improvement 1 — Cross-Encoder Reranker
* **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Architecture**: While dual-encoder models (like BGE-M3) encode query and document independently into vector space, a Cross-Encoder passes the query and document together through all cross-attention layers simultaneously, capturing full token-level semantic interactions.
* **Implementation Details**:
  - Initialized once during `MahabharatamRetriever.__init__()` to avoid per-request load overhead.
  - Takes the top 20 candidate passages produced by hybrid RRF search.
  - Generates cross-attention scores for `(query, passage_text)`.
  - Re-sorts passages in descending order of cross-encoder confidence and returns the top-$k$ (default 5).
  - Configurable via flag: `RERANKER_ENABLED = True`.

### 4.2 Improvement 2 — Multilingual Evaluation Benchmark & Hit Rate Suite
* **Script**: [`src/evaluate.py`](file:///c:/Users/P.V.H.Shanmukha/OneDrive/Documents/VyasaRAG/src/evaluate.py)
* **Dataset**: 20 hand-crafted, rigorous QA pairs across characters, key events, and specific chapters:
  - **7 Telugu Queries** (e.g., *దేవవ్రతుడి ప్రతిజ్ఞ*, *కర్ణుని మరణం*, *పాశుపతాస్త్రం*, *లక్క ఇల్లు*, *హనుమంతుడు-భీముడు*)
  - **7 English Queries** (*Devavrata vow*, *Abhimanyu's death*, *Pashupatastra boon*, *Sage Agastya*, *Jayadratha*, etc.)
  - **6 Hindi Queries** (*भीष्म की प्रतिज्ञा*, *पाशुपतास्त्र*, *कर्ण वध*, *लाक्षागृह*, *जयद्रथ वध*, etc.)
* **Comparative Hit Rate (Top-5 Recall)**:
  - `semantic-only` (BGE-M3): **55.0%** (11/20)
  - `bm25-only`: **15.0%** (3/20)
  - `hybrid` (Dense + Sparse RRF): **45.0%** (9/20)
  - `hybrid+reranker`: **30.0%** (6/20)
* **Storage**: Full audit run saved to [`data/eval_results.json`](file:///c:/Users/P.V.H.Shanmukha/OneDrive/Documents/VyasaRAG/data/eval_results.json) with ISO timestamps.

### 4.3 Improvement 3 — LLM Query Expansion
* **Model**: Groq `llama-3.1-8b-instant` (selected for sub-300ms speed, avoiding 70B/120B generation latency prior to search).
* **Mechanism**:
  - Before BM25 tokenization and embedding encoding, the query is passed to Groq with the prompt:
    > *"Given this question about the Mahabharata, list 3-5 related search terms or alternative phrasings that would help retrieve relevant passages. Return only the terms, comma-separated, no explanation."*
  - The expanded synonyms and terms (e.g. `Bhishma -> Devavrata, Gangaputra, Shantanu vow`) are appended to the query for retrieval.
  - The clean original query is retained for the Cross-Encoder reranker to maintain precision.
  - Configurable via flag: `QUERY_EXPANSION_ENABLED = True`.
  - Features dynamic fallback to `openai/gpt-oss-20b` for resilient API connectivity.

### 4.4 Improvement 4 — In-Memory Query-Level Cache & Live Health Metrics
* **Mechanism**:
  - The Mahabharata text is immutable, meaning retrieval results for an identical query never go stale.
  - Added an in-memory dictionary `self._cache` inside `MahabharatamRetriever`.
  - Cache key is generated using an MD5 digest:
    $$\text{Cache Key} = \text{MD5}\Big(\text{query.strip().lower()} + \text{str}(top\_k) + \text{str}(chapter\_filter)\Big)$$
  - Tracks `_cache_hits` and `_cache_misses` statistics.
  - **Zero latency on hit**: Duplicate queries return in $< 1\text{ms}$ without invoking embedding models, Chroma, or Groq.
* **FastAPI `/health` Integration**:
  The `/health` endpoint exposes real-time cache analytics:
  ```json
  {
    "status": "ok",
    "project": "VyasaRAG",
    "retriever_loaded": true,
    "vector_count": 375,
    "conversations": 1,
    "cache_hits": 5,
    "cache_misses": 2,
    "cache_size": 2
  }
  ```

---

## 5. Generation & Language Intelligence Pipeline

[`src/translate_text.py`](file:///c:/Users/P.V.H.Shanmukha/OneDrive/Documents/VyasaRAG/src/translate_text.py) manages language identification and LLM response generation:

### 5.1 Three-Stage Language Detection
1. **Stage 1 — Unicode Script Inspection**:
   Directly inspects character ranges (e.g., Telugu `\u0C00–\u0C7F`, Devanagari `\u0900–\u097F`, Tamil `\u0B80–\u0BFF`). If a query contains characters from these blocks, the language is recognized deterministically with 100% accuracy, bypassing statistical misidentification on short queries.
2. **Stage 2 — Intent Keyword Parsing**:
   Detects queries phrased in English script requesting another language (e.g., *"explain in hindi about Karna"* $\rightarrow$ Hindi `hi`).
3. **Stage 3 — Langdetect Fallback**:
   Applied to generic Latin-script queries without explicit language directives.

### 5.2 Grounded Answer Generation
- Chunks retrieved from `search.py` are assembled into structured context headers: `[Chapter: ... | pp. ... | Rank: ...]`.
- LLM prompt instructs Groq's `openai/gpt-oss-120b` (low temperature 0.2) to synthesize the answer strictly from the provided passages and respond in the detected language.
- Streamed directly to the web client word-by-word via Server-Sent Events (SSE).

---

## 6. Critical Engineering Fixes & Optimizations

### 6.1 Protobuf & TensorFlow Conflict Resolution
* **Issue**: When executing `python app.py` with standard system Python, HuggingFace `transformers` attempted to import TensorFlow dependencies, crashing with:
  `google.protobuf.runtime_version.VersionError: Detected incompatible Protobuf Gencode/Runtime versions (gencode 6.31.1 runtime 5.29.6)`.
* **Fix**: Added environment overrides at the very top of [`app.py`](file:///c:/Users/P.V.H.Shanmukha/OneDrive/Documents/VyasaRAG/app.py) and [`src/search.py`](file:///c:/Users/P.V.H.Shanmukha/OneDrive/Documents/VyasaRAG/src/search.py):
  ```python
  import os
  os.environ["USE_TF"] = "0"
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
  ```
  This instructs `transformers` to run in PyTorch-only mode and ignore TensorFlow completely, ensuring zero import conflicts.

### 6.2 Tokenizer Serialization Cross-Reference
* **Issue**: Pickling the BM25 tokenizer inside `bm25index.py` originally referenced `__main__.tokenize`. When unpickled inside `search.py` or `app.py`, Python could not resolve `__main__.tokenize`.
* **Fix**: Programmatically registered `sys.modules["__main__"].tokenize = tokenize` prior to unpickling, allowing seamless deserialization across arbitrary entry points.

---

## 7. Repository Directory Structure

```
VyasaRAG/
├── app.py                      # FastAPI web application & REST APIs
├── requirements.txt            # Locked project dependencies
├── PROJECT_DOCUMENTATION.md    # Comprehensive system guide (this document)
├── README.md                   # Project overview & quickstart
├── dockerfile                  # Containerized deployment blueprint
│
├── src/                        # Core backend package
│   ├── search.py               # MahabharatamRetriever (Hybrid, Reranker, Expansion, Cache)
│   ├── evaluate.py             # 20 QA benchmark runner & hit-rate evaluator
│   ├── translate_text.py       # 3-Stage language detector & Groq generation engine
│   ├── bm25index.py            # BM25 index builder & tokenizer
│   ├── vectorstore.py          # Chroma DB population & metadata manager
│   ├── embeddings.py           # BGE-M3 model loader & batch encoder
│   ├── chunking.py             # Semantic sentence-boundary chunker
│   ├── cleantext.py            # OCR text cleaner & chapter tagger
│   └── dataloader.py           # PyMuPDF + Tesseract OCR ingestion
│
├── tests/                      # Automated test suite
│   ├── __init__.py
│   ├── test_search.py          # Unit tests for retriever, reranker, expansion, cache
│   └── test_evaluate.py        # Unit tests for benchmark set structure & results
│
├── data/                       # Storage & artifacts
│   ├── MAHABHARATAM.pdf        # Original 434-page Telugu Mahabharata source
│   ├── chroma_db/              # Persistent Chroma vector database (375 vectors)
│   ├── bm25_index.pkl          # Pickled BM25 Okapi index & tokenizer (375 chunks)
│   └── eval_results.json       # Exported benchmark metrics across 4 retrieval modes
│
├── template/
│   └── index.html              # Frontend user interface (Jinja2)
└── static/
    └── style.css               # Modern dark-mode responsive stylesheet
```

---

## 8. Verification & Test Suite

The project includes an automated test suite under `tests/`:
- [`tests/test_search.py`](file:///c:/Users/P.V.H.Shanmukha/OneDrive/Documents/VyasaRAG/tests/test_search.py):
  - Model initialization (BGE-M3, CrossEncoder, Chroma DB, BM25).
  - English & Telugu retrieval outputs and metadata validity.
  - Chapter pre/post-filtering.
  - Query expansion generation.
  - Query cache hits, misses, and exact result identity.
- [`tests/test_evaluate.py`](file:///c:/Users/P.V.H.Shanmukha/OneDrive/Documents/VyasaRAG/tests/test_evaluate.py):
  - Evaluation set structure validation (7 Telugu, 7 English, 6 Hindi).
  - Schema verification of `data/eval_results.json`.

### Running All Tests
```bash
.\venv\Scripts\python.exe -m unittest discover -s tests
```
**Output**:
```text
Ran 8 tests in 27.500s
OK
```

---

## 9. Setup, Installation & How to Run

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/Shanmukhapasumarthi/VyasaRAG.git
cd VyasaRAG

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### Step 3: Running the Application
```bash
python app.py
```
Open your browser at: **`http://localhost:8000`**

### Step 4: Running Evaluation Benchmark
```bash
python src/evaluate.py
```

---

## 10. Summary of Key Metrics & Technical Specifications

| Metric / Specification | Value / Detail |
|---|---|
| **Corpus Size** | 434 Pages (Telugu Mahabharata) |
| **Total Chunks** | 375 semantic chunks (~500 tokens each) |
| **Embedding Dimensions** | 1024 dims (`BAAI/bge-m3`) |
| **Retrieval Architecture** | Hybrid (Dense Vector + BM25 Okapi + RRF $k=60$) |
| **Reranker Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Query Expansion Model** | Groq `llama-3.1-8b-instant` |
| **Generation Model** | Groq `openai/gpt-oss-120b` (Temperature 0.2) |
| **Caching Layer** | MD5 In-Memory Session Cache |
| **Evaluation Hit Rate** | 55.0% Semantic / 45.0% Hybrid / 30.0% Reranked |
| **Unit Test Coverage** | 8/8 Tests Passing (`tests/`) |
| **Web Server** | FastAPI with Server-Sent Events (SSE) |
