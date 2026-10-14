"""Regression coverage for deterministic BM25 result ordering."""

import importlib
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


class _FakeBm25:
    def get_scores(self, _query_tokens):
        return [0.5, 0.5, 0.5]


class Bm25TieOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_modules = {
            name: sys.modules.get(name)
            for name in (
                "dotenv",
                "embedding_model",
                "jieba",
                "pymilvus",
                "rank_bm25",
            )
        }

        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None
        embedding_model = types.ModuleType("embedding_model")
        embedding_model.EmbedddingModel = object
        jieba = types.ModuleType("jieba")
        jieba.lcut = lambda text: text.split()
        pymilvus = types.ModuleType("pymilvus")
        pymilvus.MilvusClient = object
        rank_bm25 = types.ModuleType("rank_bm25")
        rank_bm25.BM25Okapi = object

        sys.modules.update(
            {
                "dotenv": dotenv,
                "embedding_model": embedding_model,
                "jieba": jieba,
                "pymilvus": pymilvus,
                "rank_bm25": rank_bm25,
            }
        )
        sys.path.insert(0, str(BACKEND_DIR))
        sys.modules.pop("hybrid_search", None)
        cls.hybrid_search = importlib.import_module("hybrid_search")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("hybrid_search", None)
        sys.path.remove(str(BACKEND_DIR))
        for name, module in cls._original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_equal_scores_are_ordered_by_document_id(self):
        documents = [
            {"id": 30, "text": "policy gamma", "source": "gamma.txt"},
            {"id": 10, "text": "policy alpha", "source": "alpha.txt"},
            {"id": 20, "text": "policy beta", "source": "beta.txt"},
        ]
        tokenized_documents = [
            ["policy", "gamma"],
            ["policy", "alpha"],
            ["policy", "beta"],
        ]

        results = self.hybrid_search.bm25_search(
            "policy",
            documents,
            tokenized_documents,
            _FakeBm25(),
        )

        self.assertEqual([row["id"] for row in results], [10, 20, 30])
        self.assertEqual(
            [row["source"] for row in results],
            ["alpha.txt", "beta.txt", "gamma.txt"],
        )


if __name__ == "__main__":
    unittest.main()
