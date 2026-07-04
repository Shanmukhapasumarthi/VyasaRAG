"""
translate_text.py
------------------
LLM generation step — the final stage of the RAG pipeline.

Takes retrieved Telugu chunks + the user's original query, calls
Groq's Llama-3.1-70B, and returns an answer in whatever language
the user asked in (Lang A), grounded in the Telugu source text (Lang B).

This is exactly the bottom box in your architecture diagram:
  [Top-K chunks in Telugu (Lang B)]
        ↓
  [LLM Generator + Context Window]
        → Prompt instructs LLM:
          "Read chunks in Lang B,
           synthesise and write the answer in Lang A"
        ↓
  [Final Accurate Response (Lang A)]

Why Llama-3.1-70B over 8B:
  Telugu is low-resource in most open LLMs. The 70B model has meaningfully
  better multilingual understanding and generation than 8B — the quality
  gap is especially visible when the *answer* needs to be written in Telugu,
  not just retrieved from it. Use 70B for now; you can benchmark 8B later
  if latency becomes a concern.

Language detection:
  We detect the query language automatically so the prompt can explicitly
  instruct the LLM to answer in that language. This handles the cross-lingual
  flow without requiring the user to specify their language manually.
  NOTE: if the user's query explicitly requests a different output language
  (e.g. "explain this in German"), build_prompt() instructs the LLM to
  honor that explicit request over the auto-detected language — see
  INSTRUCTIONS point 3 below.

Streaming:
  Groq supports streaming — we expose both a streaming and non-streaming
  interface so app.py can stream tokens to the frontend for a better UX
  (answers appear word by word rather than after a multi-second wait).

Input:  query string + list[RetrievalResult] from search.py
Output: generated answer string (or stream of chunks)
"""

import os
import re
from groq import Groq
try:
    from search import RetrievalResult, format_context_for_llm, MahabharatamRetriever
except ImportError:
    from src.search import RetrievalResult, format_context_for_llm, MahabharatamRetriever
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL  = "openai/gpt-oss-120b"
MAX_TOKENS  = 1024
TEMPERATURE = 0.2  # low = factual/grounded, less creative hallucination

# Language name map for the prompt — keeps it readable for the LLM
LANG_NAMES = {
    "te": "Telugu",
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ur": "Urdu",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
}


def detect_language(text: str) -> str:
    """
    Three-stage language detection.

    Stage 1 — Unicode script block check:
      Every Telugu char is in U+0C00–U+0C7F, Hindi in U+0900–U+097F etc.
      If ANY character in the query belongs to a known script block,
      return that language immediately — no guessing needed.
      This is why "భీష్ముడు ఎవరు?" correctly returns 'te' even though
      langdetect sometimes misidentifies short Telugu queries as English.

    Stage 2 — Intent keyword check:
      Catches "explain in hindi about X" style queries written in English
      script where the user explicitly states the desired output language.

    Stage 3 — langdetect fallback:
      For plain Latin-script queries with no intent keyword.
    """
    # Stage 1 — Unicode block detection (most reliable for Indic scripts)
    SCRIPT_RANGES = [
        ('\u0C00', '\u0C7F', 'te'),   # Telugu
        ('\u0900', '\u097F', 'hi'),   # Devanagari — Hindi/Sanskrit
        ('\u0980', '\u09FF', 'bn'),   # Bengali
        ('\u0B80', '\u0BFF', 'ta'),   # Tamil
        ('\u0C80', '\u0CFF', 'kn'),   # Kannada
        ('\u0D00', '\u0D7F', 'ml'),   # Malayalam
        ('\u0A00', '\u0A7F', 'pa'),   # Gurmukhi — Punjabi
        ('\u0600', '\u06FF', 'ur'),   # Arabic/Urdu
        ('\u4E00', '\u9FFF', 'zh'),   # CJK — Chinese/Japanese
        ('\u3040', '\u30FF', 'ja'),   # Hiragana/Katakana — Japanese
    ]
    for char in text:
        for start, end, lang_code in SCRIPT_RANGES:
            if start <= char <= end:
                return lang_code

    # Stage 2 — explicit intent keywords in Latin script
    INTENT_PATTERNS = [
        (r"\bin\s+hindi\b",     "hi"),
        (r"\bin\s+telugu\b",    "te"),
        (r"\bin\s+english\b",   "en"),
        (r"\bin\s+tamil\b",     "ta"),
        (r"\bin\s+kannada\b",   "kn"),
        (r"\bin\s+malayalam\b", "ml"),
        (r"\bin\s+urdu\b",      "ur"),
        (r"\bin\s+french\b",    "fr"),
        (r"\bin\s+german\b",    "de"),
        (r"\bin\s+spanish\b",   "es"),
    ]
    for pattern, lang_code in INTENT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return lang_code

    # Stage 3 — langdetect fallback for plain Latin-script queries
    try:
        from langdetect import detect
        detected = detect(text)
        return detected.split("-")[0]
    except Exception:
        return "en"


def build_prompt(
    query: str,
    context: str,
    query_lang_code: str,
) -> str:
    """
    Build the RAG prompt that instructs the LLM to:
    - Read the Telugu source chunks
    - Synthesise an answer grounded ONLY in those chunks
    - Write the answer in the user's detected language, UNLESS the query
      itself explicitly asks for a different output language
    """
    answer_lang = LANG_NAMES.get(query_lang_code, "the same language as the question")

    return f"""You are a knowledgeable assistant helping users understand the Telugu Mahabharata.

The following passages are excerpts from the Telugu Mahabharata, retrieved based on the user's question.
These passages are written in Telugu.

--- RETRIEVED PASSAGES ---
{context}
--- END OF PASSAGES ---

USER QUESTION: {query}

INSTRUCTIONS:
1. Read and understand the Telugu passages above carefully.
2. Answer the user's question based ONLY on the information in those passages.
3. You MUST write your ENTIRE answer in {answer_lang}. Do NOT use any other language.
4. If the passages do not contain enough information, say so clearly in {answer_lang}.
5. Mention which chapter the information comes from when relevant.
6. Keep the answer concise and accurate.

ANSWER (in {answer_lang} only):"""


def generate_answer(
    query: str,
    retrieved_chunks: list[RetrievalResult],
    stream: bool = False,
):
    
    client  = Groq(api_key=os.environ["GROQ_API_KEY"])
    context = format_context_for_llm(retrieved_chunks)
    lang    = detect_language(query)
    prompt  = build_prompt(query, context, lang)
    messages = [{"role": "user", "content": prompt}]

    if stream:
        return _stream_answer(client, messages)
    else:
        return _complete_answer(client, messages)


def _complete_answer(client: Groq, messages: list) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
    )
    return response.choices[0].message.content.strip()


def _stream_answer(client: Groq, messages: list):
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Smoke test — run directly: python src/translate_text.py ────────
if __name__ == "__main__":
    # Quick detection test — no API calls needed
    from search import MahabharatamRetriever

    retriever = MahabharatamRetriever()
    test_cases = [
        ("English", "Who is Bhishma and what vow did he take?"),
        ("Telugu",  "భీష్ముడు ఎవరు? ఆయన ప్రతిజ్ఞ ఏమిటి?"),
        ("Hindi",   "भीष्म ने क्या प्रतिज्ञा ली?"),
    ]
    for lang, query in test_cases:
            print(f"\n{'='*60}")
            print(f"Query ({lang}): {query}")
            print("="*60)
    
            chunks  = retriever.retrieve(query, top_k=5)
            detected = detect_language(query)
            print(f"Detected language: {detected} "
                  f"({LANG_NAMES.get(detected, 'unknown')})")
            print(f"Retrieved {len(chunks)} chunks\n")
    
            print("Streaming answer:")
            for token in generate_answer(query, chunks, stream=True):
                print(token, end="", flush=True)
            print("\n")