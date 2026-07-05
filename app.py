"""
app.py
-------
FastAPI backend — wires MahabharatamRetriever + generate_answer into
HTTP endpoints consumed by the frontend (template/index.html).

Now supports a chat-style interface with multiple conversations,
follow-up questions that carry context from earlier turns, and
persistence of all conversations to a local data.json file (no
database required).

Endpoints:
  GET    /                          -> serves the HTML UI
  GET    /health                    -> liveness check
  GET    /conversations             -> list all conversations (summaries)
  GET    /conversations/{id}        -> full conversation incl. messages
  DELETE /conversations/{id}        -> delete a conversation
  POST   /ask                       -> non-streaming: returns full answer as JSON
  POST   /ask/stream                -> streaming SSE, tokens arrive word-by-word

Run:
  uvicorn app:app --reload --port 8000
  Then open: http://localhost:8000 or http://127.0.0.1:8000
"""

import os
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime, timezone
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
DATA_FILE     = ROOT / "data/data.json"

# ── Config ─────────────────────────────────────────────────────────
TOP_K = 5
HISTORY_TURNS_FOR_CONTEXT = 3   # how many prior user+assistant pairs to feed back in
MAX_TITLE_LEN = 60

# ── Singleton retriever ────────────────────────────────────────────
retriever: MahabharatamRetriever | None = None

# Guards concurrent writes to data.json (single-process dev server)
_data_lock = threading.Lock()


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
    version="2.0.0",
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
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ── data.json persistence helpers ───────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_data() -> dict:
    """Load the whole store. Returns {"conversations": {id: {...}}}."""
    with _data_lock:
        if not DATA_FILE.exists():
            return {"conversations": {}}
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable file — don't crash the app, start fresh.
            return {"conversations": {}}


def save_data(data: dict) -> None:
    with _data_lock:
        tmp_path = DATA_FILE.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(DATA_FILE)


def get_or_create_conversation(data: dict, conversation_id: str | None, seed_query: str) -> tuple[str, dict]:
    """Return (conversation_id, conversation_dict), creating one if needed."""
    if conversation_id and conversation_id in data["conversations"]:
        return conversation_id, data["conversations"][conversation_id]

    new_id = conversation_id or str(uuid.uuid4())
    title = seed_query.strip().replace("\n", " ")[:MAX_TITLE_LEN]
    conversation = {
        "id": new_id,
        "title": title or "New conversation",
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
    }
    data["conversations"][new_id] = conversation
    return new_id, conversation


def build_history_text(conversation: dict) -> str:
    """
    Plain-text transcript of the last few turns (prior to the current
    question). Used two ways downstream:
      - appended ahead of the current query for RETRIEVAL only, so a
        follow-up like "what about his brother?" still finds the right
        chunks even though it has no name in it.
      - passed separately into generate_answer() as `history`, so the
        LLM sees prior context WITHOUT it being mixed into the string
        that detect_language() inspects (see build_contextual_query()
        docstring below for why that separation matters).
    """
    past = conversation["messages"][-(HISTORY_TURNS_FOR_CONTEXT * 2):]
    if not past:
        return ""
    lines = []
    for m in past:
        role_label = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {m['content']}")
    return "\n".join(lines)


def build_contextual_query(history_text: str, query: str) -> str:
    """
    Retrieval-only helper: prefixes the current question with prior turns.

    IMPORTANT: this combined string is used for embedding/retrieval ONLY.
    It must never be passed as the `query` argument to generate_answer(),
    because that function calls detect_language(query) internally, and
    detect_language() returns as soon as it sees ANY character from a
    known script block. If an earlier assistant answer was in Telugu and
    the user's new follow-up is in English, mixing them would make a
    plain English follow-up get misdetected as Telugu. Keep the two
    query strings (retrieval vs. generation/detection) separate.
    """
    if not history_text:
        return query
    return f"{history_text}\nUser: {query}"


# ── Pydantic models ────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    chapter_filter: str | None = None
    top_k: int = TOP_K
    conversation_id: str | None = None   # None -> a new conversation is created


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
    conversation_id: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


# ── Routes ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main UI."""
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations():
    """Sidebar list — newest first."""
    data = load_data()
    summaries = [
        ConversationSummary(
            id=c["id"],
            title=c["title"],
            created_at=c["created_at"],
            updated_at=c["updated_at"],
            message_count=len(c["messages"]),
        )
        for c in data["conversations"].values()
    ]
    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    data = load_data()
    convo = data["conversations"].get(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return convo


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    data = load_data()
    if conversation_id in data["conversations"]:
        del data["conversations"][conversation_id]
        save_data(data)
        return {"status": "deleted", "id": conversation_id}
    raise HTTPException(status_code=404, detail="Conversation not found.")


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Non-streaming endpoint.
    Returns the complete answer + source chunks as JSON, and persists
    the exchange to data.json under the given (or a new) conversation.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialised yet.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    data = load_data()
    conversation_id, conversation = get_or_create_conversation(data, req.conversation_id, req.query)

    history_text     = build_history_text(conversation)
    contextual_query = build_contextual_query(history_text, req.query)  # retrieval only

    chunks = retriever.retrieve(
        query=contextual_query,
        top_k=req.top_k,
        chapter_filter=req.chapter_filter,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant passages found. Try rephrasing your question.",
        )

    # req.query (not contextual_query) so detect_language() only ever sees
    # the current question, never mixed-language history.
    answer = generate_answer(req.query, chunks, stream=False, history=history_text or None)
    lang   = detect_language(req.query)

    source_dicts = [
        {
            "chunk_id":      c.chunk_id,
            "chapter_title": c.chapter_title,
            "pages":         c.pages,
            "score":         c.score,
            "text":          c.text[:300],
        }
        for c in chunks
    ]

    conversation["messages"].append({"role": "user", "content": req.query, "timestamp": _now()})
    conversation["messages"].append({
        "role": "assistant", "content": answer, "sources": source_dicts, "timestamp": _now(),
    })
    conversation["updated_at"] = _now()
    save_data(data)

    return AskResponse(
        query=req.query,
        detected_language=LANG_NAMES.get(lang, lang),
        answer=answer,
        sources=[SourceChunk(**s) for s in source_dicts],
        conversation_id=conversation_id,
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """
    Streaming endpoint via Server-Sent Events (SSE).

    Event types sent to the client:
      event: meta    -> {"conversation_id": "..."} sent first so the client
                        can attach follow-up questions to this conversation
      event: chunk   -> one token/partial text from the LLM
      event: sources -> JSON array of retrieved passages (sent after generation)
      event: done    -> stream complete
      event: error   -> something went wrong
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialised yet.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    data = load_data()
    conversation_id, conversation = get_or_create_conversation(data, req.conversation_id, req.query)
    save_data(data)  # persist the (possibly new) conversation shell immediately

    history_text     = build_history_text(conversation)
    contextual_query = build_contextual_query(history_text, req.query)  # retrieval only

    chunks = retriever.retrieve(
        query=contextual_query,
        top_k=req.top_k,
        chapter_filter=req.chapter_filter,
    )

    if not chunks:
        async def no_results():
            yield f"event: meta\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"
            yield "event: error\ndata: No relevant passages found.\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    source_dicts = [
        {
            "chunk_id":      c.chunk_id,
            "chapter_title": c.chapter_title,
            "pages":         c.pages,
            "score":         c.score,
            "text":          c.text[:300],
        }
        for c in chunks
    ]
    sources_payload = json.dumps(source_dicts, ensure_ascii=False)

    # Save the user's turn right away so it survives even if generation fails.
    data = load_data()
    data["conversations"][conversation_id]["messages"].append({
        "role": "user", "content": req.query, "timestamp": _now(),
    })
    save_data(data)

    async def event_generator():
        full_answer = ""
        try:
            yield f"event: meta\ndata: {json.dumps({'conversation_id': conversation_id})}\n\n"

            # req.query (not contextual_query) so detect_language() only ever
            # sees the current question; `history` carries prior turns to the
            # LLM prompt separately, without corrupting language detection.
            for token in generate_answer(req.query, chunks, stream=True, history=history_text or None):
                full_answer += token
                safe = token.replace("\n", " ")
                yield f"event: chunk\ndata: {safe}\n\n"

            yield f"event: sources\ndata: {sources_payload}\n\n"
            yield "event: done\ndata: \n\n"

            # Persist the assistant's turn once streaming finished successfully.
            fresh = load_data()
            fresh["conversations"][conversation_id]["messages"].append({
                "role": "assistant",
                "content": full_answer,
                "sources": source_dicts,
                "timestamp": _now(),
            })
            fresh["conversations"][conversation_id]["updated_at"] = _now()
            save_data(fresh)

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
        "conversations":    len(load_data()["conversations"]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)