import unittest
import json
from pathlib import Path

from src.evaluate import EVAL_SET
from src.search import MahabharatamRetriever


class TestEvaluationScript(unittest.TestCase):
    def test_eval_set_structure(self):
        """Verify that the evaluation set contains 20 QA pairs split as 7 Telugu, 7 English, 6 Hindi."""
        self.assertEqual(len(EVAL_SET), 20)
        telugu_count = sum(1 for q in EVAL_SET if q["lang"] == "Telugu")
        english_count = sum(1 for q in EVAL_SET if q["lang"] == "English")
        hindi_count = sum(1 for q in EVAL_SET if q["lang"] == "Hindi")

        self.assertEqual(telugu_count, 7)
        self.assertEqual(english_count, 7)
        self.assertEqual(hindi_count, 6)

        for q in EVAL_SET:
            self.assertIn("id", q)
            self.assertIn("lang", q)
            self.assertIn("category", q)
            self.assertIn("query", q)
            self.assertIn("expected_chapter", q)

    def test_eval_results_file(self):
        """Verify that data/eval_results.json exists and has expected keys."""
        root = Path(__file__).resolve().parent.parent
        results_file = root / "data" / "eval_results.json"
        self.assertTrue(results_file.exists())

        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("timestamp", data)
        self.assertIn("summary", data)
        self.assertEqual(data["total_queries"], 20)
        self.assertIn("semantic-only", data["summary"])
        self.assertIn("bm25-only", data["summary"])
        self.assertIn("hybrid", data["summary"])
        self.assertIn("hybrid+reranker", data["summary"])


if __name__ == "__main__":
    unittest.main()
