"""
Evaluation Set & Hit Rate Measurement for VyasaRAG.

Compares retrieval performance across four configurations:
  1. Semantic-only (BGE-M3)
  2. BM25-only
  3. Hybrid (BGE-M3 + BM25 + RRF)
  4. Hybrid + Cross-Encoder Reranker

Saves benchmark results to data/eval_results.json.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure project root is in Python path and standard paths are relative to root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.search as search_mod
from src.search import MahabharatamRetriever, RetrievalResult

DATA_DIR = ROOT / "data"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.json"

# ── Hand-written Evaluation Dataset (20 QA Pairs) ────────────────────
EVAL_SET = [
    # ── 7 Telugu Queries ─────────────────────────────────────────────
    {
        "id": 1,
        "lang": "Telugu",
        "category": "character",
        "query": "భీష్ముడి అసలు పేరు దేవవ్రతుడు అని దేవుని వరం గురించి తెలియజేయుము",
        "expected_chapter": "దేవవ్రతుడు",
    },
    {
        "id": 2,
        "lang": "Telugu",
        "category": "event",
        "query": "కర్ణుని మరణం యుద్ధరంగంలో ఎలా జరిగింది?",
        "expected_chapter": "కర్ణుని మరణం",
    },
    {
        "id": 3,
        "lang": "Telugu",
        "category": "chapter",
        "query": "పాశుపతాస్త్రం పొందిన ఘట్టం ఏ అధ్యాయంలో వివరించబడింది?",
        "expected_chapter": "పాశుపతం",
    },
    {
        "id": 4,
        "lang": "Telugu",
        "category": "event",
        "query": "పాండవులు లక్క ఇల్లు దహనం నుండి ఎలా బయటపడ్డారు?",
        "expected_chapter": "లక్క ఇల్లు",
    },
    {
        "id": 5,
        "lang": "Telugu",
        "category": "character",
        "query": "హనుమంతుడు భీముడిని కలుసుకున్న కథ మరియు విశేషాలు ఏమిటి?",
        "expected_chapter": "హనుమంతుడు",
    },
    {
        "id": 6,
        "lang": "Telugu",
        "category": "character",
        "query": "ద్రోణాచార్యుని విశేషాలు మరియు కౌరవ సేనానిగా ద్రోణుడు",
        "expected_chapter": "ద్రోణుడు",
    },
    {
        "id": 7,
        "lang": "Telugu",
        "category": "event",
        "query": "అభిమన్యుని మరణం మరియు పద్మవ్యూహం",
        "expected_chapter": "అభిమన్యు",
    },

    # ── 7 English Queries ────────────────────────────────────────────
    {
        "id": 8,
        "lang": "English",
        "category": "character",
        "query": "Who was Devavrata and what vow did he take to become Bhishma?",
        "expected_chapter": "దేవవ్రతుడు",
    },
    {
        "id": 9,
        "lang": "English",
        "category": "event",
        "query": "How did Abhimanyu die in the Kurukshetra war?",
        "expected_chapter": "అభిమన్యు",
    },
    {
        "id": 10,
        "lang": "English",
        "category": "chapter",
        "query": "Where is the Pashupatastra boon granted by Lord Shiva to Arjuna described?",
        "expected_chapter": "పాశుపతం",
    },
    {
        "id": 11,
        "lang": "English",
        "category": "character",
        "query": "Who is sage Agastya and what is the story of Agastya drinking ocean?",
        "expected_chapter": "అగస్యుడు",
    },
    {
        "id": 12,
        "lang": "English",
        "category": "event",
        "query": "How was Jayadratha killed by Arjuna in the Mahabharata war?",
        "expected_chapter": "జయద్రధుని",
    },
    {
        "id": 13,
        "lang": "English",
        "category": "character",
        "query": "What is the story of Shakuni coming to Hastinapura?",
        "expected_chapter": "శకుని",
    },
    {
        "id": 14,
        "lang": "English",
        "category": "event",
        "query": "What happened during the house of lac (lac house) incident?",
        "expected_chapter": "లక్క ఇల్లు",
    },

    # ── 6 Hindi Queries ──────────────────────────────────────────────
    {
        "id": 15,
        "lang": "Hindi",
        "category": "character",
        "query": "देवव्रत भीष्म की प्रतिज्ञा का क्या वृत्तांत है?",
        "expected_chapter": "దేవవ్రతుడు",
    },
    {
        "id": 16,
        "lang": "Hindi",
        "category": "chapter",
        "query": "अर्जुन को पाशुपतास्त्र प्राप्त होने की कथा किस अध्याय में है?",
        "expected_chapter": "పాశుపతం",
    },
    {
        "id": 17,
        "lang": "Hindi",
        "category": "event",
        "query": "महाभारत युद्ध में कर्ण की मृत्यु कैसे हुई?",
        "expected_chapter": "కర్ణుని మరణం",
    },
    {
        "id": 18,
        "lang": "Hindi",
        "category": "character",
        "query": "हनुमान और भीम की भेंट किस प्रकार हुई?",
        "expected_chapter": "హనుమంతుడు",
    },
    {
        "id": 19,
        "lang": "Hindi",
        "category": "event",
        "query": "लाक्षागृह (लक्का इल्लु) घटना की जानकारी",
        "expected_chapter": "లక్క ఇల్లు",
    },
    {
        "id": 20,
        "lang": "Hindi",
        "category": "event",
        "query": "जयद्रथ का वध अर्जुन द्वारा किस प्रकार किया गया?",
        "expected_chapter": "జయద్రధుని",
    },
]


def evaluate_query(
    retriever: MahabharatamRetriever,
    query: str,
    top_k: int,
    mode: str,
) -> list[RetrievalResult]:
    """Retrieve results according to specified evaluation mode."""
    if mode == "semantic-only":
        return retriever._semantic_candidates(query, top_k)
    elif mode == "bm25-only":
        if retriever.bm25 is None:
            return []
        return retriever._bm25_candidates(query, top_k)
    elif mode == "hybrid":
        old_flag = search_mod.RERANKER_ENABLED
        search_mod.RERANKER_ENABLED = False
        res = retriever.retrieve(query, top_k=top_k)
        search_mod.RERANKER_ENABLED = old_flag
        return res
    elif mode == "hybrid+reranker":
        old_flag = search_mod.RERANKER_ENABLED
        search_mod.RERANKER_ENABLED = True
        res = retriever.retrieve(query, top_k=top_k)
        search_mod.RERANKER_ENABLED = old_flag
        return res
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_eval_for_mode(
    retriever: MahabharatamRetriever,
    mode: str,
    eval_set: list[dict],
    top_k: int = 5,
) -> dict:
    print(f"\n{'='*70}")
    print(f"RUNNING EVALUATION: Mode = {mode.upper()} (Top-{top_k})")
    print(f"{'='*70}")

    hits = 0
    query_details = []

    for item in eval_set:
        qid = item["id"]
        lang = item["lang"]
        query = item["query"]
        expected = item["expected_chapter"]

        results = evaluate_query(retriever, query, top_k=top_k, mode=mode)
        top_titles = [r.chapter_title for r in results]

        # Substring match against expected_chapter
        hit = any(expected.lower() in (title or "").lower() for title in top_titles)
        if hit:
            hits += 1
            status = "HIT "
        else:
            status = "MISS"

        print(f"Q{qid:02d} [{lang:7s}] [{status}] Query: '{query[:45]}...'")
        if not hit:
            print(f"    Expected: '{expected}' | Got top: {top_titles[:2]}")

        query_details.append({
            "id": qid,
            "lang": lang,
            "category": item["category"],
            "query": query,
            "expected_chapter": expected,
            "hit": hit,
            "top_chapters": top_titles,
        })

    total = len(eval_set)
    hit_rate = (hits / total) * 100.0 if total > 0 else 0.0
    print(f"\nMode Summary [{mode}]: {hits}/{total} HITs = {hit_rate:.1f}%")

    return {
        "mode": mode,
        "hits": hits,
        "total": total,
        "hit_rate_pct": round(hit_rate, 2),
        "details": query_details,
    }


def main():
    # Force stdout unicode encoding for Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Initializing MahabharatamRetriever for evaluation benchmark...")
    retriever = MahabharatamRetriever()

    modes = ["semantic-only", "bm25-only", "hybrid", "hybrid+reranker"]
    summary_results = {}
    mode_outputs = []

    for mode in modes:
        res = run_eval_for_mode(retriever, mode, EVAL_SET, top_k=5)
        summary_results[mode] = f"{res['hits']}/{res['total']} = {res['hit_rate_pct']}%"
        mode_outputs.append(res)

    print("\n" + "=" * 70)
    print("OVERALL EVALUATION COMPARISON SUMMARY")
    print("=" * 70)
    for mode, score in summary_results.items():
        print(f"  {mode:20s} : {score}")
    print("=" * 70)

    # Save output to data/eval_results.json
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(EVAL_SET),
        "summary": summary_results,
        "modes": mode_outputs,
    }

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed evaluation results saved to: {EVAL_RESULTS_PATH}\n")


if __name__ == "__main__":
    main()
