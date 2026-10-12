import importlib
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# hybrid_search imports its runtime integrations at module load time.  Minimal
# stubs keep this unit test offline and focused on the pure fusion function.
sys.modules.setdefault("jieba", types.SimpleNamespace(lcut=lambda text: [text]))
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault("pymilvus", types.SimpleNamespace(MilvusClient=object))
sys.modules.setdefault(
    "rank_bm25",
    types.SimpleNamespace(BM25Okapi=object),
)
sys.modules.setdefault(
    "embedding_model",
    types.SimpleNamespace(EmbedddingModel=object),
)

hybrid_search = importlib.import_module("hybrid_search")


class RrfRouteMetadataTests(unittest.TestCase):
    def test_single_route_hits_leave_other_route_metadata_empty(self):
        keyword_hits = [
            {
                "id": 11,
                "score": 1.25,
                "text": "Synthetic leave policy excerpt.",
                "source": "synthetic-policy.txt",
            }
        ]
        vector_hits = [
            {
                "id": 22,
                "score": 0.75,
                "text": "Synthetic expense policy excerpt.",
                "source": "synthetic-expenses.txt",
            }
        ]

        results = hybrid_search.rrf_fusion(keyword_hits, vector_hits)
        by_id = {result["id"]: result for result in results}

        self.assertEqual(by_id[11]["bm25_rank"], 1)
        self.assertEqual(by_id[11]["bm25_score"], 1.25)
        self.assertIsNone(by_id[11]["vector_rank"])
        self.assertIsNone(by_id[11]["vector_score"])

        self.assertEqual(by_id[22]["vector_rank"], 1)
        self.assertEqual(by_id[22]["vector_score"], 0.75)
        self.assertIsNone(by_id[22]["bm25_rank"])
        self.assertIsNone(by_id[22]["bm25_score"])


if __name__ == "__main__":
    unittest.main()
