"""Regression coverage for legacy rows without source metadata."""

import importlib
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


class _FakeBm25:
    def get_scores(self, _tokens):
        return [0.75]


class Bm25MissingSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_modules = {
            name: sys.modules.get(name)
            for name in ("jieba", "dotenv", "pymilvus", "rank_bm25", "embedding_model")
        }

        jieba = types.ModuleType("jieba")
        jieba.lcut = lambda text: text.split()
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None
        pymilvus = types.ModuleType("pymilvus")
        pymilvus.MilvusClient = object
        rank_bm25 = types.ModuleType("rank_bm25")
        rank_bm25.BM25Okapi = object
        embedding_model = types.ModuleType("embedding_model")
        embedding_model.EmbedddingModel = object

        sys.modules.update(
            {
                "jieba": jieba,
                "dotenv": dotenv,
                "pymilvus": pymilvus,
                "rank_bm25": rank_bm25,
                "embedding_model": embedding_model,
            }
        )
        sys.path.insert(0, str(BACKEND_DIR))
        sys.modules.pop("hybrid_search", None)
        cls.hybrid_search = importlib.import_module("hybrid_search")

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(BACKEND_DIR))
        sys.modules.pop("hybrid_search", None)
        for name, module in cls._original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_matching_legacy_document_uses_empty_source(self):
        documents = [{"id": 7, "text": "synthetic policy"}]

        results = self.hybrid_search.bm25_search(
            "policy",
            documents,
            [["synthetic", "policy"]],
            _FakeBm25(),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 7)
        self.assertEqual(results[0]["source"], "")
        self.assertEqual(results[0]["score"], 0.75)


if __name__ == "__main__":
    unittest.main()
