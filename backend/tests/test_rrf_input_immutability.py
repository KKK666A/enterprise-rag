import copy
import importlib
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


class RrfInputImmutabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(BACKEND_DIR))

        cls.original_modules = {
            name: sys.modules.get(name)
            for name in (
                "dotenv",
                "embedding_model",
                "jieba",
                "pymilvus",
                "rank_bm25",
            )
        }

        stubs = {
            "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
            "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
            "jieba": types.SimpleNamespace(lcut=lambda text: text.split()),
            "pymilvus": types.SimpleNamespace(MilvusClient=object),
            "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
        }
        sys.modules.update(stubs)
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

    def test_fusion_does_not_mutate_route_hits(self):
        keyword_hits = [
            {"id": 7, "score": 1.25, "text": "Synthetic policy text", "source": "policy.txt"}
        ]
        vector_hits = [
            {"id": 7, "score": 0.91, "text": "Synthetic policy text", "source": "policy.txt"},
            {"id": 9, "score": 0.75, "text": "Synthetic handbook text", "source": "handbook.txt"},
        ]
        original_keyword_hits = copy.deepcopy(keyword_hits)
        original_vector_hits = copy.deepcopy(vector_hits)

        results = self.hybrid_search.rrf_fusion(keyword_hits, vector_hits)

        self.assertEqual(keyword_hits, original_keyword_hits)
        self.assertEqual(vector_hits, original_vector_hits)
        self.assertIsNot(results[0], keyword_hits[0])
        self.assertIsNot(results[0], vector_hits[0])


if __name__ == "__main__":
    unittest.main()
