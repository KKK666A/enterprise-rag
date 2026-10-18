import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def load_app_module():
    """Load the application with lightweight stand-ins for optional services."""
    class FakeFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()

        def get(self, *args, **kwargs):
            return lambda function: function

        post = get

    class FakeHTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FakeFastAPI
    fastapi.File = lambda *args, **kwargs: None
    fastapi.HTTPException = FakeHTTPException
    fastapi.Request = type("Request", (), {})
    fastapi.UploadFile = type("UploadFile", (), {})

    modules = {
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
        "embedding_model": types.SimpleNamespace(EmbedddingModel=Mock),
        "fastapi": fastapi,
        "fastapi.responses": types.SimpleNamespace(RedirectResponse=Mock),
        "pymilvus": types.SimpleNamespace(MilvusClient=Mock),
        "pydantic": types.SimpleNamespace(
            BaseModel=type("BaseModel", (), {}),
            Field=lambda *args, **kwargs: None,
        ),
        "rank_bm25": types.SimpleNamespace(BM25Okapi=Mock),
    }
    hybrid_search = types.ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "synthetic_documents"
    for name in (
        "tokenize",
        "read_documents",
        "bm25_search",
        "vector_search",
        "rrf_fusion",
    ):
        setattr(hybrid_search, name, Mock(name=name))
    modules["hybrid_search"] = hybrid_search

    module_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("app_query_route_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class QueryRouteFusionTest(unittest.TestCase):
    def test_query_fuses_lexical_and_vector_hits(self):
        app = load_app_module()
        service = app.RagService.__new__(app.RagService)
        service.lock = threading.Lock()
        service.ready = True
        service.documents = [{"id": 1, "text": "Synthetic leave policy"}]
        service.tokenized_documents = [["synthetic", "leave", "policy"]]
        service.bm25 = object()
        service.client = object()
        service.model = object()

        lexical_hits = [{"id": 1, "score": 0.5}]
        vector_hits = [{"id": 2, "score": 0.9}]
        fused_hits = [{"id": 2, "rrf_score": 0.03}]

        with (
            patch.object(app, "bm25_search", return_value=lexical_hits) as lexical,
            patch.object(app, "vector_search", return_value=vector_hits) as vector,
            patch.object(app, "rrf_fusion", return_value=fused_hits) as fusion,
        ):
            result = service.query("What is the leave policy?")

        lexical.assert_called_once_with(
            "What is the leave policy?",
            service.documents,
            service.tokenized_documents,
            service.bm25,
        )
        vector.assert_called_once_with(
            "What is the leave policy?", service.client, service.model
        )
        fusion.assert_called_once_with(lexical_hits, vector_hits)
        self.assertEqual(
            result,
            {
                "question": "What is the leave policy?",
                "answer": None,
                "sources": fused_hits,
            },
        )


if __name__ == "__main__":
    unittest.main()
