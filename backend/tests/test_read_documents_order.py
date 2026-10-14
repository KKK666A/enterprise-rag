import importlib
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


class FakeIterator:
    def __init__(self):
        self.batches = [
            [
                {"id": 10, "text": "Travel policy", "source": "policy.txt"},
                {"id": 2, "text": "Leave policy", "source": "handbook.txt"},
            ],
            [{"id": 7, "text": "Expense policy", "source": "expenses.txt"}],
            [],
        ]
        self.closed = False

    def next(self):
        return self.batches.pop(0)

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self):
        self.iterator = FakeIterator()
        self.query_iterator_kwargs = None

    def query_iterator(self, **kwargs):
        self.query_iterator_kwargs = kwargs
        return self.iterator


class ReadDocumentsOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(BACKEND_DIR))

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

        cls.original_modules = {
            name: sys.modules.get(name)
            for name in ("jieba", "dotenv", "pymilvus", "rank_bm25", "embedding_model")
        }
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
        for name, module in cls.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.path.remove(str(BACKEND_DIR))

    def test_returns_paginated_documents_in_id_order(self):
        client = FakeClient()

        documents = self.hybrid_search.read_documents(client)

        self.assertEqual([document["id"] for document in documents], [2, 7, 10])
        self.assertTrue(client.iterator.closed)
        self.assertEqual(
            client.query_iterator_kwargs,
            {
                "collection_name": self.hybrid_search.COLLECTION_NAME,
                "filter": "",
                "output_fields": ["id", "text", "source"],
                "batch_size": 500,
                "consistency_level": "Strong",
            },
        )


if __name__ == "__main__":
    unittest.main()
