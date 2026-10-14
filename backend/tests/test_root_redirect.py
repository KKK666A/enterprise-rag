import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class _FastAPI:
    def __init__(self, **_kwargs):
        pass

    def get(self, *_args, **_kwargs):
        return lambda function: function

    post = get


class _RedirectResponse:
    def __init__(self, url, status_code=307):
        self.headers = {"location": url}
        self.status_code = status_code


def _dependency_modules():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FastAPI
    fastapi.File = lambda *args, **kwargs: None
    fastapi.HTTPException = type("HTTPException", (Exception,), {})
    fastapi.Request = type("Request", (), {})
    fastapi.UploadFile = type("UploadFile", (), {})

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = _RedirectResponse

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    pydantic.Field = lambda *args, **kwargs: None

    hybrid_search = types.ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "synthetic_collection"
    for name in ("tokenize", "read_documents", "bm25_search", "vector_search", "rrf_fusion"):
        setattr(hybrid_search, name, lambda *args, **kwargs: None)

    modules = {
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
        "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "hybrid_search": hybrid_search,
        "pydantic": pydantic,
        "pymilvus": types.SimpleNamespace(MilvusClient=object),
        "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
    }
    return modules


class RootRedirectTests(unittest.TestCase):
    def test_root_redirects_to_interactive_docs(self):
        spec = importlib.util.spec_from_file_location("app_under_test", BACKEND_DIR / "app.py")
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, _dependency_modules()):
            spec.loader.exec_module(module)

        response = module.index()

        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/docs")


if __name__ == "__main__":
    unittest.main()
