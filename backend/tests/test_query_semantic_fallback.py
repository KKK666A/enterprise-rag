"""Regression coverage for semantic-only query results."""

import importlib
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


def _load_app_module():
    """Import app without requiring the optional service dependencies."""
    class FakeFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()

        def get(self, *args, **kwargs):
            return lambda function: function

        post = get

    class FakeBaseModel:
        pass

    class FakeHTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.FastAPI = FakeFastAPI
    fastapi_stub.File = lambda *args, **kwargs: None
    fastapi_stub.HTTPException = FakeHTTPException
    fastapi_stub.Request = object
    fastapi_stub.UploadFile = object
    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.RedirectResponse = object
    pydantic_stub = types.ModuleType("pydantic")
    pydantic_stub.BaseModel = FakeBaseModel
    pydantic_stub.Field = lambda *args, **kwargs: None
    hybrid_search_stub = types.ModuleType("hybrid_search")
    hybrid_search_stub.COLLECTION_NAME = "test_collection"
    hybrid_search_stub.tokenize = Mock()
    hybrid_search_stub.read_documents = Mock()
    hybrid_search_stub.bm25_search = Mock()
    hybrid_search_stub.vector_search = Mock()
    hybrid_search_stub.rrf_fusion = lambda keyword_hits, vector_hits: [
        {
            **vector_hits[0],
            "rrf_score": 1 / 61,
            "bm25_rank": None,
            "vector_rank": 1,
            "bm25_score": None,
            "vector_score": vector_hits[0]["score"],
        }
    ]

    dependency_stubs = {
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
        "fastapi": fastapi_stub,
        "fastapi.responses": responses_stub,
        "pydantic": pydantic_stub,
        "pymilvus": types.SimpleNamespace(MilvusClient=object),
        "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
        "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
        "hybrid_search": hybrid_search_stub,
    }

    with patch.dict(sys.modules, dependency_stubs):
        sys.modules.pop("app", None)
        return importlib.import_module("app")


class QuerySemanticFallbackTests(unittest.TestCase):
    def test_vector_results_are_returned_when_bm25_has_no_hits(self):
        app_module = _load_app_module()
        service = app_module.RagService.__new__(app_module.RagService)
        service.lock = threading.Lock()
        service.ready = True
        service.documents = [
            {"id": 7, "text": "Synthetic remote-work policy", "source": "policy.txt"}
        ]
        service.tokenized_documents = [["remote", "work"]]
        service.bm25 = object()
        service.client = object()
        service.model = object()

        vector_hit = {
            "id": 7,
            "score": 0.91,
            "text": "Synthetic remote-work policy",
            "source": "policy.txt",
        }
        vector_search = Mock(return_value=[vector_hit])

        with (
            patch.object(app_module, "bm25_search", return_value=[]) as bm25_search,
            patch.object(app_module, "vector_search", vector_search),
        ):
            result = service.query("working away from the office")

        bm25_search.assert_called_once()
        vector_search.assert_called_once_with(
            "working away from the office", service.client, service.model
        )
        self.assertEqual(result["question"], "working away from the office")
        self.assertIsNone(result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["id"], 7)
        self.assertIsNone(result["sources"][0]["bm25_rank"])
        self.assertEqual(result["sources"][0]["vector_rank"], 1)
        self.assertEqual(result["sources"][0]["vector_score"], 0.91)


if __name__ == "__main__":
    unittest.main()
