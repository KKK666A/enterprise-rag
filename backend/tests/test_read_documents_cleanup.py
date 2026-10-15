"""Regression coverage for successful Milvus iterator cleanup."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]


class ReadDocumentsCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_modules = {
            name: sys.modules.get(name)
            for name in ("dotenv", "embedding_model", "jieba", "pymilvus", "rank_bm25")
        }

        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = Mock()
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
        sys.path.remove(str(BACKEND_DIR))
        sys.modules.pop("hybrid_search", None)
        for name, module in cls._original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_successful_paginated_read_closes_iterator(self):
        iterator = Mock()
        iterator.next.side_effect = [
            [{"id": 8, "text": "Second synthetic chunk", "source": "guide.txt"}],
            [{"id": 3, "text": "First synthetic chunk", "source": "policy.txt"}],
            [],
        ]
        client = Mock()
        client.query_iterator.return_value = iterator

        documents = self.hybrid_search.read_documents(client)

        self.assertEqual([document["id"] for document in documents], [3, 8])
        self.assertEqual(iterator.next.call_count, 3)
        iterator.close.assert_called_once_with()
        client.query_iterator.assert_called_once_with(
            collection_name=self.hybrid_search.COLLECTION_NAME,
            filter="",
            output_fields=["id", "text", "source"],
            batch_size=500,
            consistency_level="Strong",
        )


if __name__ == "__main__":
    unittest.main()
