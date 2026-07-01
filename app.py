"""
app.py
-------
FastAPI backend — wires MahabharatamRetriever + generate_answer into
HTTP endpoints consumed by the frontend (templates/index.html).

Endpoints:
  GET  /              -> serves the HTML UI
  POST /ask           -> non-streaming: returns full answer as JSON
  POST /ask/stream    -> streaming: Server-Sent Events, tokens arrive
                         word-by-word so the UI feels responsive

Design decisions:
- Retriever is a module-level singleton, initialised once at startup.
  Loading BGE-M3 takes ~5s — doing it per-request would be unusable.
- /ask/stream uses SSE (text/event-stream) which works with a plain
  fetch() EventSource in vanilla JS — no websocket complexity needed.
- CORS is open for localhost only — tighten this if you deploy publicly.
- Request/response models are typed with Pydantic so FastAPI auto-generates
  docs at /docs — useful while you're building the frontend.

Run:
  uvicorn app:app --reload --port 8000
"""

import os
import json
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

# ---- config -------------------------------------------------------
TOP_K = 5
# ---------------------------------------------------------------------


# ── Singleton retriever, loaded once at startup ────────────────────
retriever: MahabharatamRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the BGE-M3 model and open Chroma once when the server starts."""
    global retriever
    print("Loading retriever (BGE-M3 + Chroma)...")
    retriever = MahabharatamRetriever()
    print("Retriever ready.\n")
    yield
    # Nothing to clean up — Chroma PersistentClient closes automatically


app = FastAPI(
    title="Mahabharatam RAG API",
    description="Multilingual Q&A over the Telugu Mahabharata",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Request / Response models ──────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    chapter_filter: str | None = None   # optional: restrict to a chapter
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


# ── Routes ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Non-streaming endpoint.
    Returns the full answer + source chunks once generation is complete.
    Use this for programmatic access or testing.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not ready")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    chunks = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        chapter_filter=req.chapter_filter,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant passages found. Try rephrasing your question."
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
                text=c.text[:300],   # truncate for response size
            )
            for c in chunks
        ],
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """
    Streaming endpoint using Server-Sent Events.
    Sends three event types to the client:
      - 'chunk'   : a token/partial text from the LLM
      - 'sources' : JSON array of retrieved passages (sent after generation)
      - 'done'    : signals the stream is complete
      - 'error'   : if something went wrong

    The frontend listens with fetch() + ReadableStream (not EventSource,
    because EventSource doesn't support POST) — see index.html.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not ready")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    chunks = retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        chapter_filter=req.chapter_filter,
    )

    if not chunks:
        async def no_results():
            yield "event: error\ndata: No relevant passages found.\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    sources_payload = json.dumps([
        {
            "chunk_id":      c.chunk_id,
            "chapter_title": c.chapter_title,
            "pages":         c.pages,
            "score":         c.score,
            "text":          c.text[:300],
        }
        for c in chunks
    ], ensure_ascii=False)

    async def event_generator():
        try:
            for token in generate_answer(req.query, chunks, stream=True):
                # SSE format: "event: <type>\ndata: <payload>\n\n"
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
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if deployed
        },
    )


@app.get("/health")
async def health():
    """Quick liveness check — useful for deployment."""
    return {
        "status": "ok",
        "retriever_loaded": retriever is not None,
        "vector_count": retriever.collection.count() if retriever else 0,
    }