"""Offline tests for hybrid-search token normalization."""

import importlib
import sys
import types
import unittest
from unittest import mock
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


def _load_hybrid_search():
    """Import the module without requiring optional service dependencies."""
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

    stubs = {
        "jieba": jieba,
        "dotenv": dotenv,
        "pymilvus": pymilvus,
        "rank_bm25": rank_bm25,
        "embedding_model": embedding_model,
    }
    with mock.patch.dict(sys.modules, stubs):
        sys.modules.pop("hybrid_search", None)
        return importlib.import_module("hybrid_search")


class TokenizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hybrid_search = _load_hybrid_search()

    def test_normalizes_case_and_removes_stop_words_and_punctuation(self):
        stop_word = next(iter(self.hybrid_search.STOP_WORDS))

        tokens = self.hybrid_search.tokenize(
            f"POLICY {stop_word} !!! 2026"
        )

        self.assertEqual(tokens, ["policy", "2026"])

    def test_returns_empty_list_when_no_searchable_tokens_remain(self):
        self.assertEqual(self.hybrid_search.tokenize("... !!! ---"), [])


if __name__ == "__main__":
    unittest.main()
