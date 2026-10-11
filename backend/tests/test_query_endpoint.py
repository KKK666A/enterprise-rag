"""Offline contract tests for the query endpoint."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeFastAPI:
    def __init__(self, **_kwargs):
        self.state = SimpleNamespace()

    def get(self, *_args, **_kwargs):
        return lambda function: function

    post = get


def load_app_module():
    """Import app.py without requiring its optional service dependencies."""
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FakeFastAPI
    fastapi.File = lambda *args, **kwargs: None
    fastapi.HTTPException = HTTPException
    fastapi.Request = object
    fastapi.UploadFile = object

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = object

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = object
    pydantic.Field = lambda *args, **kwargs: None

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = object

    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = object

    embedding_model = types.ModuleType("embedding_model")
    embedding_model.EmbedddingModel = object

    hybrid_search = types.ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "test_collection"
    for name in (
        "tokenize",
        "read_documents",
        "bm25_search",
        "vector_search",
        "rrf_fusion",
    ):
        setattr(hybrid_search, name, Mock(name=name))

    modules = {
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "pydantic": pydantic,
        "dotenv": dotenv,
        "pymilvus": pymilvus,
        "rank_bm25": rank_bm25,
        "embedding_model": embedding_model,
        "hybrid_search": hybrid_search,
    }

    with patch.dict(sys.modules, modules):
        sys.path.insert(0, str(BACKEND_DIR))
        try:
            sys.modules.pop("app", None)
            return importlib.import_module("app")
        finally:
            sys.path.remove(str(BACKEND_DIR))


class QueryEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = load_app_module()

    def test_query_trims_surrounding_whitespace_before_search(self):
        service = Mock()
        service.query.return_value = {"sources": []}
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(rag=service)))

        result = self.app_module.query_document(
            SimpleNamespace(question="  synthetic policy question  "), request
        )

        self.assertEqual({"sources": []}, result)
        service.query.assert_called_once_with("synthetic policy question")

    def test_query_rejects_whitespace_only_question_without_searching(self):
        service = Mock()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(rag=service)))

        with self.assertRaises(HTTPException) as raised:
            self.app_module.query_document(SimpleNamespace(question=" \t\n "), request)

        self.assertEqual(400, raised.exception.status_code)
        service.query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
