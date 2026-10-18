"""Regression coverage for vector hits created before source metadata existed."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


class _Model:
    def encode_query(self, question):
        self.question = question
        return [0.25, 0.75]


class _Client:
    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return [[
            {
                "id": 17,
                "distance": 0.875,
                "entity": {"text": "Synthetic legacy policy excerpt."},
            }
        ]]


class VectorSearchMissingSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stubs = {
            "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
            "jieba": types.SimpleNamespace(lcut=lambda text: text.split()),
            "pymilvus": types.SimpleNamespace(MilvusClient=object),
            "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
            "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
        }
        cls.previous_modules = {
            name: sys.modules.get(name) for name in stubs
        }
        sys.modules.update(stubs)

        spec = importlib.util.spec_from_file_location(
            "hybrid_search_vector_missing_source_test",
            BACKEND_DIR / "hybrid_search.py",
        )
        cls.hybrid_search = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.hybrid_search)

    @classmethod
    def tearDownClass(cls):
        for name, module in cls.previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_vector_hit_without_source_uses_empty_source(self):
        client = _Client()
        model = _Model()

        results = self.hybrid_search.vector_search(
            "synthetic policy question", client, model
        )

        self.assertEqual(model.question, "synthetic policy question")
        self.assertEqual(
            results,
            [{
                "id": 17,
                "score": 0.875,
                "text": "Synthetic legacy policy excerpt.",
                "source": "",
            }],
        )
        self.assertEqual(client.search_kwargs["output_fields"], ["text", "source"])
        self.assertEqual(client.search_kwargs["data"], [[0.25, 0.75]])


if __name__ == "__main__":
    unittest.main()
