"""Offline contract test for the vector-search adapter."""

import importlib
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


class VectorSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_modules = {
            name: sys.modules.get(name)
            for name in ("dotenv", "embedding_model", "jieba", "pymilvus", "rank_bm25")
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
        for name, module in cls.original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_vector_search_uses_query_embedding_and_normalizes_hits(self):
        class Model:
            def encode_query(self, question):
                self.question = question
                return [0.25, 0.75]

        class Client:
            def search(self, **kwargs):
                self.kwargs = kwargs
                return [[
                    {
                        "id": 17,
                        "distance": 0.8125,
                        "entity": {
                            "text": "Synthetic leave policy excerpt.",
                            "source": "synthetic-handbook.txt",
                        },
                    },
                    {
                        "id": 23,
                        "distance": 0.5,
                        "entity": {"text": "Synthetic fallback excerpt."},
                    },
                ]]

        model = Model()
        client = Client()

        results = self.hybrid_search.vector_search(
            "How much leave is available?", client, model
        )

        self.assertEqual(model.question, "How much leave is available?")
        self.assertEqual(
            client.kwargs,
            {
                "collection_name": "rag_test_v1",
                "data": [[0.25, 0.75]],
                "anns_field": "vector",
                "limit": 10,
                "search_params": {"metric_type": "COSINE"},
                "output_fields": ["text", "source"],
                "consistency_level": "Strong",
            },
        )
        self.assertEqual(
            results,
            [
                {
                    "id": 17,
                    "score": 0.8125,
                    "text": "Synthetic leave policy excerpt.",
                    "source": "synthetic-handbook.txt",
                },
                {
                    "id": 23,
                    "score": 0.5,
                    "text": "Synthetic fallback excerpt.",
                    "source": "",
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
