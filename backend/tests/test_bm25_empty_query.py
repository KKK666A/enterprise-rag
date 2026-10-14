"""Regression tests for BM25 queries with no searchable tokens."""

import importlib
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


class _UnexpectedBm25Call:
    def get_scores(self, _tokens):
        raise AssertionError("BM25 scoring should be skipped for an empty token list")


class Bm25EmptyQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(BACKEND_DIR))
        cls._saved_modules = {
            name: sys.modules.get(name)
            for name in ("jieba", "dotenv", "pymilvus", "rank_bm25", "embedding_model")
        }

        jieba = types.ModuleType("jieba")
        jieba.lcut = lambda text: list(text)
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *_args, **_kwargs: None
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
        sys.modules.pop("hybrid_search", None)
        cls.hybrid_search = importlib.import_module("hybrid_search")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("hybrid_search", None)
        for name, module in cls._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.path.remove(str(BACKEND_DIR))

    def test_punctuation_only_query_skips_bm25_scoring(self):
        documents = [{"id": 1, "text": "Synthetic policy text", "source": "policy.txt"}]

        results = self.hybrid_search.bm25_search(
            "...?!",
            documents,
            [["synthetic", "policy"]],
            _UnexpectedBm25Call(),
        )

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
