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

Streaming:
  Groq supports streaming — we expose both a streaming and non-streaming
  interface so app.py can stream tokens to the frontend for a better UX
  (answers appear word by word rather than after a multi-second wait).

Input:  query string + list[RetrievalResult] from search.py
Output: generated answer string (or stream of chunks)
"""

import os
from groq import Groq
from search import RetrievalResult, format_context_for_llm
from search import MahabharatamRetriever
from dotenv import load_dotenv

load_dotenv()   # reads GROQ_API_KEY from .env

# ---- config -------------------------------------------------------
GROQ_MODEL   = "openai/gpt-oss-120b"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.2    # low = factual/grounded, less creative hallucination
# ---------------------------------------------------------------------

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
    Lightweight language detection using langdetect.
    Falls back to 'en' on failure.
    Install: pip install langdetect
    """
    try:
        from langdetect import detect
        return detect(text)
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
    - Write the answer in the user's language
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
3. Write your answer in {answer_lang}.
4. If the passages do not contain enough information to answer the question, say so clearly in {answer_lang} — do not guess or use outside knowledge.
5. When relevant, mention which chapter the information comes from.
6. Keep the answer concise and accurate.

ANSWER (in {answer_lang}):"""


def generate_answer(
    query: str,
    retrieved_chunks: list[RetrievalResult],
    stream: bool = False,
):
    """
    Generate a grounded multilingual answer using Groq + Llama-3.1-70B.

    Args:
        query:            User's question (any language)
        retrieved_chunks: Top-k results from search.py
        stream:           If True, returns a generator of text chunks
                          for real-time streaming to the frontend.
                          If False, returns the complete answer string.

    Returns:
        str if stream=False
        Generator[str] if stream=True
    """
    client  = Groq(api_key=os.environ["GROQ_API_KEY"])

    context        = format_context_for_llm(retrieved_chunks)
    query_lang     = detect_language(query)
    prompt         = build_prompt(query, context, query_lang)

    messages = [{"role": "user", "content": prompt}]

    if stream:
        return _stream_answer(client, messages)
    else:
        return _complete_answer(client, messages)


def _complete_answer(client: Groq, messages: list) -> str:
    response = client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = messages,
        max_tokens  = MAX_TOKENS,
        temperature = TEMPERATURE,
        stream      = False,
    )
    return response.choices[0].message.content.strip()


def _stream_answer(client: Groq, messages: list):
    """Yields text tokens one by one as Groq streams them."""
    stream = client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = messages,
        max_tokens  = MAX_TOKENS,
        temperature = TEMPERATURE,
        stream      = True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---- Quick smoke-test (run directly: python3 translate_text.py) ---
if __name__ == "__main__":
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