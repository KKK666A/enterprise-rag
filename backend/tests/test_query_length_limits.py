import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class _FieldInfo:
    def __init__(self, **kwargs):
        self.metadata = kwargs


class _FastAPI:
    def __init__(self, **kwargs):
        self.state = types.SimpleNamespace()

    def get(self, *args, **kwargs):
        return lambda function: function

    post = get


class QueryLengthLimitTests(unittest.TestCase):
    def test_query_body_declares_supported_length_range(self):
        fastapi = types.ModuleType("fastapi")
        fastapi.FastAPI = _FastAPI
        fastapi.File = lambda *args, **kwargs: None
        fastapi.HTTPException = type("HTTPException", (Exception,), {})
        fastapi.Request = object
        fastapi.UploadFile = object

        responses = types.ModuleType("fastapi.responses")
        responses.RedirectResponse = object

        pydantic = types.ModuleType("pydantic")
        pydantic.BaseModel = object
        pydantic.Field = lambda **kwargs: _FieldInfo(**kwargs)

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
            "dotenv": dotenv,
            "fastapi": fastapi,
            "fastapi.responses": responses,
            "pydantic": pydantic,
            "pymilvus": pymilvus,
            "rank_bm25": rank_bm25,
            "embedding_model": embedding_model,
            "hybrid_search": hybrid_search,
        }

        spec = importlib.util.spec_from_file_location(
            "app_query_length_test", BACKEND_DIR / "app.py"
        )
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)

        field = module.QueryBody.question
        self.assertEqual(
            field.metadata,
            {"min_length": 1, "max_length": 200},
        )


if __name__ == "__main__":
    unittest.main()
