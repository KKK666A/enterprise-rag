import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FastAPI:
    def __init__(self, **_kwargs):
        pass

    def get(self, *_args, **_kwargs):
        return lambda function: function

    def post(self, *_args, **_kwargs):
        return lambda function: function


def import_app_module():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FastAPI
    fastapi.File = lambda *args, **kwargs: None
    fastapi.HTTPException = HTTPException
    fastapi.Request = object
    fastapi.UploadFile = object

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = object

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
        setattr(hybrid_search, name, lambda *args, **kwargs: None)

    modules = {
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "dotenv": dotenv,
        "pymilvus": pymilvus,
        "rank_bm25": rank_bm25,
        "embedding_model": embedding_model,
        "hybrid_search": hybrid_search,
    }

    with patch.dict(sys.modules, modules), patch.object(
        sys, "path", [str(BACKEND_DIR), *sys.path]
    ):
        sys.modules.pop("app", None)
        return importlib.import_module("app")


class QueryServiceErrorTests(unittest.TestCase):
    def test_query_failure_returns_generic_bad_gateway_detail(self):
        app_module = import_app_module()
        secret = "provider rejected api_key=synthetic-secret"

        class FailingService:
            def query(self, _question):
                raise RuntimeError(secret)

        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(rag=FailingService())
            )
        )
        body = SimpleNamespace(question="  synthetic question  ")

        with self.assertRaises(HTTPException) as raised:
            app_module.query_document(body, request)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertNotIn(secret, raised.exception.detail)
        self.assertNotIn("synthetic-secret", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
