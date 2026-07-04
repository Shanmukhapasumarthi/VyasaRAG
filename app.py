"""
app.py
-------
FastAPI backend — wires MahabharatamRetriever + generate_answer into
HTTP endpoints consumed by the frontend (template/index.html).

Endpoints:
  GET  /              -> serves the HTML UI
  POST /ask           -> non-streaming: returns full answer as JSON
  POST /ask/stream    -> streaming SSE, tokens arrive word-by-word
  GET  /health        -> liveness check

Run:
  uvicorn app:app --reload --port 8000
  Then open: http://localhost:8000 or http://127.0.0.1:8000
"""

import os
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from dotenv import load_dotenv

from src.search import MahabharatamRetriever
from src.translate_text import generate_answer, detect_language, LANG_NAMES

load_dotenv()

# ── Paths anchored to project root (works from any working directory) ──
ROOT          = Path(__file__).resolve().parent
STATIC_DIR    = ROOT / "static"
TEMPLATE_DIR  = ROOT / "template"     # matches your actual folder name

# ── Config ─────────────────────────────────────────────────────────
TOP_K = 5

# ── Singleton retriever ────────────────────────────────────────────
retriever: MahabharatamRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load BGE-M3 model + open Chroma once at server startup."""
    global retriever
    print("Starting VyasaRAG...")
    print("Loading BGE-M3 + Chroma (this takes ~5s)...")
    retriever = MahabharatamRetriever()
    print("Retriever ready.")
    print("Open http://localhost:8000\n")
    yield


app = FastAPI(
    title="VyasaRAG",
    description="Multilingual Q&A over the Telugu Mahabharata",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — open for all localhost ports (covers 8000, 3000, 5500 etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ── Pydantic models ────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    chapter_filter: str | None = None
    top_k: int = TOP_K


class SourceChunk(BaseModel):
    chunk_id: int
    chapter_title: str
    pages: list[int]
    score: float
    text: str


class AskResponse(BaseModel):
    query: str
    detected_language: str
    answer: str
    sources: list[SourceChunk]
    source_chapters: list[str]          # deduplicated list of chapters retrieved from
    sources: list[SourceChunk]

# ── Routes ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main UI."""
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Non-streaming endpoint.
    Returns the complete answer + source chunks as JSON.
    Useful for programmatic access or API testing at /docs.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialised yet.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    chunks = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        chapter_filter=req.chapter_filter,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant passages found. Try rephrasing your question.",
        )

    answer = generate_answer(req.query, chunks, stream=False)
    lang   = detect_language(req.query)

    return AskResponse(
        query=req.query,
        detected_language=LANG_NAMES.get(lang, lang),
        answer=answer,
        sources=[
            SourceChunk(
                chunk_id=c.chunk_id,
                chapter_title=c.chapter_title,
                pages=c.pages,
                score=c.score,
                text=c.text[:300],
            )
            for c in chunks
        ],
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """
    Streaming endpoint via Server-Sent Events (SSE).

    Event types sent to the client:
      event: chunk   -> one token/partial text from the LLM
      event: sources -> JSON array of retrieved passages (sent after generation)
      event: done    -> stream complete
      event: error   -> something went wrong
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialised yet.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    chunks = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        chapter_filter=req.chapter_filter,
    )

    if not chunks:
        async def no_results():
            yield "event: error\ndata: No relevant passages found.\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    sources_payload = json.dumps(
        [
            {
                "chunk_id":      c.chunk_id,
                "chapter_title": c.chapter_title,
                "pages":         c.pages,
                "score":         c.score,
                "text":          c.text[:300],
            }
            for c in chunks
        ],
        ensure_ascii=False,
    )

    async def event_generator():
        try:
            for token in generate_answer(req.query, chunks, stream=True):
                safe = token.replace("\n", " ")
                yield f"event: chunk\ndata: {safe}\n\n"

            yield f"event: sources\ndata: {sources_payload}\n\n"
            yield "event: done\ndata: \n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":      "no-cache",
            "X-Accel-Buffering":  "no",
            "Connection":         "keep-alive",
        },
    )


@app.get("/health")
async def health():
    """Liveness check — visit http://localhost:8000/health to confirm server is up."""
    return {
        "status":           "ok",
        "project":          "VyasaRAG",
        "retriever_loaded": retriever is not None,
        "vector_count":     retriever.collection.count() if retriever else 0,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)