import unittest
from src.search import (
    MahabharatamRetriever,
    RetrievalResult,
    RERANKER_ENABLED,
    QUERY_EXPANSION_ENABLED,
    EXPANSION_MODEL,
)


class TestMahabharatamRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Initializing MahabharatamRetriever for tests...")
        cls.retriever = MahabharatamRetriever()

    def test_initialization(self):
        self.assertIsNotNone(self.retriever.model)
        self.assertIsNotNone(self.retriever.collection)
        if RERANKER_ENABLED:
            self.assertIsNotNone(self.retriever.reranker)

    def test_retrieve_english_query(self):
        query = "Who is Bhishma and what vow did he take?"
        results = self.retriever.retrieve(query, top_k=5)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        for res in results:
            self.assertIsInstance(res, RetrievalResult)
            self.assertIsInstance(res.score, float)
            self.assertTrue(len(res.text) > 0)

    def test_retrieve_telugu_query(self):
        query = "భీష్ముడు ఎవరు?"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)

    def test_chapter_filter(self):
        results = self.retriever.retrieve("Arjuna", top_k=5, chapter_filter="సంభవ")
        for res in results:
            if res.chapter_title:
                self.assertIn("సంభవ", res.chapter_title)

    def test_query_expansion(self):
        query = "Who is Karna?"
        expanded = self.retriever._expand_query(query)
        self.assertIsInstance(expanded, str)
        self.assertTrue(expanded.startswith(query))

    def test_query_cache(self):
        query = "Unique Cache Query Test - Dronacharya"
        initial_stats = self.retriever.get_cache_stats()

        # Call 1 — Miss
        res1 = self.retriever.retrieve(query, top_k=3)
        mid_stats = self.retriever.get_cache_stats()
        self.assertEqual(mid_stats["cache_misses"], initial_stats["cache_misses"] + 1)
        self.assertEqual(mid_stats["cache_hits"], initial_stats["cache_hits"])

        # Call 2 — Hit
        res2 = self.retriever.retrieve(query, top_k=3)
        final_stats = self.retriever.get_cache_stats()
        self.assertEqual(final_stats["cache_hits"], mid_stats["cache_hits"] + 1)
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main()
