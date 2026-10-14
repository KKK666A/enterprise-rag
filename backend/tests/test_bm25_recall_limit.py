"""Regression test for the per-route BM25 recall limit."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "hybrid_search.py"


class FakeBM25:
    def get_scores(self, _query_tokens):
        # Later documents deliberately rank higher so the assertion also checks
        # that truncation happens after scoring and sorting.
        return list(range(12))


def load_hybrid_search():
    dependencies = {
        "jieba": types.SimpleNamespace(lcut=lambda text: text.split()),
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *_args, **_kwargs: None),
        "pymilvus": types.SimpleNamespace(MilvusClient=object),
        "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
        "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
    }
    spec = importlib.util.spec_from_file_location(
        "hybrid_search_recall_limit_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


class BM25RecallLimitTest(unittest.TestCase):
    def test_returns_only_the_ten_highest_ranked_candidates(self):
        hybrid_search = load_hybrid_search()
        documents = [
            {"id": index, "text": f"synthetic policy {index}", "source": "sample.txt"}
            for index in range(12)
        ]
        tokenized_documents = [["policy"] for _ in documents]

        results = hybrid_search.bm25_search(
            "policy", documents, tokenized_documents, FakeBM25()
        )

        self.assertEqual(len(results), hybrid_search.RECALL_K)
        self.assertEqual(
            [result["id"] for result in results],
            list(range(11, 1, -1)),
        )


if __name__ == "__main__":
    unittest.main()
