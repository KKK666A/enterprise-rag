"""Offline regression tests for hybrid-search result fusion."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class RrfFusionTests(unittest.TestCase):
    def test_equal_scores_are_ordered_by_document_id(self):
        """Input order must not make tied fusion results nondeterministic."""
        dependency_stubs = {
            "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
            "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
            "jieba": types.SimpleNamespace(lcut=lambda text: list(text)),
            "pymilvus": types.SimpleNamespace(MilvusClient=object),
            "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
        }
        backend_dir = str(Path(__file__).resolve().parents[1])

        with patch.dict(sys.modules, dependency_stubs), patch.object(
            sys, "path", [backend_dir, *sys.path]
        ):
            sys.modules.pop("hybrid_search", None)
            hybrid_search = importlib.import_module("hybrid_search")

        hits = [
            {"id": 2, "score": 0.8, "text": "合成文本乙", "source": "sample.txt"},
            {"id": 1, "score": 0.7, "text": "合成文本甲", "source": "sample.txt"},
        ]
        results = hybrid_search.rrf_fusion(hits, list(reversed(hits)))

        self.assertEqual([row["id"] for row in results], [1, 2])


if __name__ == "__main__":
    unittest.main()
