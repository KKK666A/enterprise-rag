"""Regression tests for combining the two retrieval routes with RRF."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_hybrid_search():
    """Load the module with lightweight stand-ins for optional services."""
    stubs = {
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
        "jieba": types.SimpleNamespace(lcut=lambda text: text.split()),
        "pymilvus": types.SimpleNamespace(MilvusClient=object),
        "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
        "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)

    try:
        path = Path(__file__).parents[1] / "hybrid_search.py"
        spec = importlib.util.spec_from_file_location(
            "hybrid_search_cross_route_test", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class CrossRouteFusionTests(unittest.TestCase):
    def test_shared_document_accumulates_both_rank_contributions(self):
        hybrid_search = load_hybrid_search()
        keyword_hits = [
            {
                "id": 7,
                "score": 2.5,
                "text": "Synthetic leave policy excerpt.",
                "source": "handbook.txt",
            }
        ]
        vector_hits = [
            {
                "id": 9,
                "score": 0.91,
                "text": "Synthetic expense policy excerpt.",
                "source": "expenses.txt",
            },
            {
                "id": 7,
                "score": 0.83,
                "text": "Synthetic leave policy excerpt.",
                "source": "handbook.txt",
            },
        ]

        results = hybrid_search.rrf_fusion(keyword_hits, vector_hits)

        shared = next(row for row in results if row["id"] == 7)
        self.assertEqual(shared["bm25_rank"], 1)
        self.assertEqual(shared["vector_rank"], 2)
        self.assertEqual(shared["bm25_score"], 2.5)
        self.assertEqual(shared["vector_score"], 0.83)
        self.assertAlmostEqual(
            shared["rrf_score"],
            1 / (hybrid_search.RRF_K + 1)
            + 1 / (hybrid_search.RRF_K + 2),
        )
        self.assertEqual(results[0]["id"], 7)


if __name__ == "__main__":
    unittest.main()
