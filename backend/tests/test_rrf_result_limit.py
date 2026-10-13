"""Regression coverage for the public result limit of RRF fusion."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


def load_hybrid_search():
    """Load the module without requiring its optional service dependencies."""
    stubs = {
        "jieba": types.SimpleNamespace(lcut=lambda text: text.split()),
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
        "pymilvus": types.SimpleNamespace(MilvusClient=object),
        "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
        "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)

    try:
        spec = importlib.util.spec_from_file_location(
            "hybrid_search_result_limit_test",
            BACKEND_DIR / "hybrid_search.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


hybrid_search = load_hybrid_search()


class RrfResultLimitTests(unittest.TestCase):
    def test_fusion_returns_only_top_k_candidates(self):
        keyword_hits = [
            {"id": index, "score": 1.0 / index, "text": f"policy {index}", "source": "handbook.txt"}
            for index in range(1, 6)
        ]
        vector_hits = [
            {"id": index, "score": 1.0 - index / 100, "text": f"policy {index}", "source": "handbook.txt"}
            for index in range(1, 6)
        ]

        results = hybrid_search.rrf_fusion(keyword_hits, vector_hits)

        self.assertEqual(len(results), hybrid_search.TOP_K)
        self.assertEqual([row["id"] for row in results], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
