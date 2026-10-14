"""Regression test for queries against an empty knowledge base."""

import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class _Field:
    def __init__(self, **_kwargs):
        pass


def _load_app_module():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = lambda **_kwargs: types.SimpleNamespace(
        get=lambda *_args, **_kwargs: lambda function: function,
        post=lambda *_args, **_kwargs: lambda function: function,
    )
    fastapi.File = lambda *_args, **_kwargs: None
    fastapi.HTTPException = type("HTTPException", (Exception,), {})
    fastapi.Request = object
    fastapi.UploadFile = object

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = object
    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = object
    pydantic.Field = lambda **kwargs: _Field(**kwargs)
    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = object
    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = object
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *_args, **_kwargs: None
    embedding_model = types.ModuleType("embedding_model")
    embedding_model.EmbedddingModel = object
    hybrid_search = types.ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "synthetic_collection"
    hybrid_search.tokenize = lambda text: text.split()
    hybrid_search.read_documents = lambda _client: []
    hybrid_search.bm25_search = lambda *_args: []
    hybrid_search.vector_search = lambda *_args: []
    hybrid_search.rrf_fusion = lambda *_args: []

    modules = {
        "dotenv": dotenv,
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "pydantic": pydantic,
        "pymilvus": pymilvus,
        "rank_bm25": rank_bm25,
        "embedding_model": embedding_model,
        "hybrid_search": hybrid_search,
    }

    with patch.dict(sys.modules, modules), patch.object(
        sys, "path", [str(BACKEND_DIR), *sys.path]
    ):
        spec = importlib.util.spec_from_file_location(
            "app_under_empty_query_test", BACKEND_DIR / "app.py"
        )
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)

    return app


class EmptyKnowledgeBaseQueryTests(unittest.TestCase):
    def test_query_returns_before_external_search_for_empty_index(self):
        app = _load_app_module()
        service = app.RagService.__new__(app.RagService)
        service.lock = threading.Lock()
        service.ready = True
        service.documents = []

        result = service.query("synthetic policy question")

        self.assertEqual(result["question"], "synthetic policy question")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertIn("知识库为空", result["message"])


if __name__ == "__main__":
    unittest.main()
