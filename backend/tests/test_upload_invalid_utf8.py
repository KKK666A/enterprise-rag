"""Regression coverage for rejecting non-UTF-8 document uploads."""

import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeFastAPI:
    def __init__(self, *args, **kwargs):
        self.state = types.SimpleNamespace()

    def get(self, *args, **kwargs):
        return lambda function: function

    def post(self, *args, **kwargs):
        return lambda function: function


def dependency_stubs():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FakeFastAPI
    fastapi.File = lambda *args, **kwargs: None
    fastapi.HTTPException = HTTPException
    fastapi.Request = object
    fastapi.UploadFile = object

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = object

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
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
    hybrid_search.COLLECTION_NAME = "synthetic_collection"
    for name in ("tokenize", "read_documents", "bm25_search", "vector_search", "rrf_fusion"):
        setattr(hybrid_search, name, lambda *args, **kwargs: None)

    return {
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "pydantic": pydantic,
        "dotenv": dotenv,
        "pymilvus": pymilvus,
        "rank_bm25": rank_bm25,
        "embedding_model": embedding_model,
        "hybrid_search": hybrid_search,
    }


def load_app_module():
    app_path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("app_invalid_utf8_test", app_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependency_stubs()):
        spec.loader.exec_module(module)
    return module


class InvalidUtf8UploadTests(unittest.TestCase):
    def test_rejects_invalid_utf8_before_import_and_closes_stream(self):
        app = load_app_module()
        stream = io.BytesIO(b"synthetic text\xff")
        upload = types.SimpleNamespace(filename="synthetic.txt", file=stream)
        service = types.SimpleNamespace(import_document=lambda *args: self.fail("import called"))
        request = types.SimpleNamespace(app=types.SimpleNamespace(
            state=types.SimpleNamespace(rag=service)
        ))

        with self.assertRaises(app.HTTPException) as raised:
            app.upload_document(request, upload)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("UTF-8", raised.exception.detail)
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
