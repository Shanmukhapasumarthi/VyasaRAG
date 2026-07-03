# మహాభారతం — Multilingual RAG

Ask questions about the Telugu Mahabharata in **any language** — Telugu, English, Hindi, Tamil, or others — and get answers grounded in the original Telugu source text, streamed back in the language you asked in.

Built as a final-year B.Tech capstone project in AI/ML engineering.

---

## Architecture

```
User Query (any language)
        ↓
BGE-M3 Multilingual Embedding   ← no translation needed
        ↓
Chroma Cross-lingual Vector DB  ← Telugu chunks stored here
        ↓
Top-K Telugu chunks retrieved
        ↓
Groq Llama-3.1-70B              ← reads Telugu, answers in query language
        ↓
Final Answer (same language as query)
```

**Key design choice:** the query is encoded directly by BGE-M3 without translating it to Telugu first. Because BGE-M3 was trained on parallel multilingual data, a question in English and a Telugu passage about the same topic land close together in the 1024-dim vector space. This means no translation errors compound before retrieval.

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| EASYOCR | Tesseract (`tel`) | Source PDF uses legacy non-Unicode Telugu fonts (Praveena/Priyaanka) — direct text extraction returns garbled output |
| Embedding | `BAAI/bge-m3` | Best-in-class cross-lingual dense retrieval, 10+ languages, 8192 token context |
| Vector DB | Chroma (local, persistent) | Zero infra, free, metadata filtering built-in |
| Chunking | Recursive semantic, ~500 tokens / 50 overlap | Preserves sentence boundaries; overlap prevents ideas split across chunk edges from being lost |
| LLM | Groq `openai/gpt-oss-120b` | Strong multilingual generation; 70B noticeably better than 8B for low-resource Telugu output |
| Backend | FastAPI + SSE streaming | Tokens stream to the frontend word-by-word — no long waits |
| Frontend | Vanilla JS + Jinja2 | No build step, no framework overhead |

---

## Project Structure

```
mahabharatam-rag/
├── .env.example            ← copy to .env, add your Groq API key
├── requirements.txt
│
├── data_loader.py          ← OCR the PDF → data/mahabharatam_ocr.jsonl
├── clean_text.py           ← strip footers, tag chapters → mahabharatam_clean.jsonl
├── chunking.py             ← recursive semantic split → mahabharatam_chunks.jsonl
├── embedding.py            ← BGE-M3 embed all chunks → mahabharatam_embeddings.jsonl
├── vector_store.py         ← upsert into Chroma → data/chroma_db/
├── search.py               ← runtime retrieval (MahabharatamRetriever)
├── translate_text.py       ← Groq generation + language detection
├── app.py                  ← FastAPI server (GET /, POST /ask, POST /ask/stream)
│
├── data/
│   └── document.pdf        ← place your Mahabharata PDF here
├── templates/
│   └── index.html
└── static/
    └── style.css
```

---

## Setup

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### API Key

```bash
cp .env.example .env
# Edit .env and add your Groq API key
# Get a free key at: https://console.groq.com
```

---

## Running the Pipeline

Run these **once** to build the vector database from the PDF:

```bash
# Step 1 — OCR the full PDF (~20 min, checkpointed)
python3 data_loader.py

# Step 2 — Clean OCR artifacts, tag chapters (~5 sec)
python3 clean_text.py

# Step 3 — Recursive semantic chunking (~2 min)
python3 chunking.py

# Step 4 — Generate BGE-M3 embeddings (~1-2 hrs CPU / ~10 min GPU, checkpointed)
python3 embedding.py

# Step 5 — Load into Chroma (~1 min)
python3 vector_store.py
```

> **Note:** Steps 1 and 4 checkpoint progress to disk. If interrupted, just rerun the same command — already-processed pages/chunks are skipped automatically.

After the pipeline completes, `data/chroma_db/` persists on disk. You never need to re-run these scripts unless you change the source PDF or chunking strategy.

---

## Starting the Server

```bash
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/ask` | Returns full answer + sources as JSON (non-streaming) |
| `POST` | `/ask/stream` | Streams answer tokens via Server-Sent Events |
| `GET` | `/health` | Returns server status and vector count |

### Example: POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is Bhishma and what vow did he take?"}'
```

```json
{
  "query": "Who is Bhishma and what vow did he take?",
  "detected_language": "English",
  "answer": "Bhishma, originally named Devavrata, was the son of King Shantanu...",
  "sources": [
    {
      "chunk_id": 9,
      "chapter_title": "భిష్మ ప్రతిజ్ఞ",
      "pages": [23, 24],
      "score": 0.91,
      "text": "..."
    }
  ]
}
```

---

## How It Works — Step by Step

### Why EASYOCR instead of direct text extraction?
The source PDF is typeset in **Praveena** and **Priyaanka** — legacy non-Unicode Telugu DTP fonts from Modular Infotech. These fonts map Telugu glyphs onto Latin/ASCII codepoints for visual rendering. Direct extraction (PyMuPDF, pdfplumber) returns the raw codepoints — readable-looking garbage, not Telugu Unicode. OCR reads the rendered glyphs as pixels, producing proper Unicode Telugu regardless of the font encoding.

### Why no translation step?
A common "multilingual RAG" approach translates the query to English, retrieves English chunks, then translates the answer back. This compounds translation errors twice and ignores the source language entirely. BGE-M3 was trained on parallel multilingual corpora — it encodes meaning, not just language-specific tokens. A Telugu passage and an English question about the same event naturally cluster together in the embedding space without any translation hop.

### Why 70B over 8B for generation?
Telugu is low-resource in most open LLMs' training data. The quality gap between 70B and 8B is most visible in the *generation* step — when the model needs to read Telugu context and produce fluent Telugu (or another language) output. 8B often produces stilted or grammatically inconsistent Telugu output. 70B handles this meaningfully better.

---

## Known Limitations

- **EasyOCR** grabs words from images, but it doesn't understand the meaning of the document. It treats a page like a flat picture, not a smart form.
- **Front matter exclusion:** Pages 0–17 (title, TOC, translator's foreword) are excluded from the index. Verify `CONTENT_START_PAGE = 18` in `clean_text.py` matches your specific edition.
- **Telugu generation quality:** Llama-3.1-70B handles Telugu well but is not a native-Telugu model. Answers in Telugu are fluent for most questions but may have minor grammatical imperfections.
- **CPU embedding time:** Generating BGE-M3 embeddings for ~2000 chunks takes 1-2 hours on CPU. Consider using a machine with a GPU for the embedding step.

---

## Future Improvements

- [ ] Cross-encoder reranker on top-k results before LLM context (improves precision)
- [ ] Chapter number correction lookup table (title → correct number)
- [ ] Streaming to mobile-friendly UI
- [ ] Evaluation set: 20–30 hand-written Telugu QA pairs for systematic quality measurement

---

## Author

**Shanmukha Pasumarthi**
B.Tech (AI/ML), VIT-AP University
[github.com/Shanmukhapasumarthi](https://github.com/Shanmukhapasumarthi)
