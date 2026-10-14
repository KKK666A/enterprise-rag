import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]


def load_app_module():
    """Load app.py with small dependency doubles for an offline unit test."""
    modules = {
        "dotenv": types.ModuleType("dotenv"),
        "fastapi": types.ModuleType("fastapi"),
        "fastapi.responses": types.ModuleType("fastapi.responses"),
        "pydantic": types.ModuleType("pydantic"),
        "pymilvus": types.ModuleType("pymilvus"),
        "rank_bm25": types.ModuleType("rank_bm25"),
        "embedding_model": types.ModuleType("embedding_model"),
        "hybrid_search": types.ModuleType("hybrid_search"),
    }

    modules["dotenv"].load_dotenv = Mock()

    class FastAPI:
        def __init__(self, **_kwargs):
            pass

        def get(self, *_args, **_kwargs):
            return lambda function: function

        post = get

    modules["fastapi"].FastAPI = FastAPI
    modules["fastapi"].File = lambda *args, **kwargs: None
    modules["fastapi"].HTTPException = type("HTTPException", (Exception,), {})
    modules["fastapi"].Request = object
    modules["fastapi"].UploadFile = object
    modules["fastapi.responses"].RedirectResponse = object
    modules["pydantic"].BaseModel = object
    modules["pydantic"].Field = lambda *args, **kwargs: None
    modules["pymilvus"].MilvusClient = object
    modules["rank_bm25"].BM25Okapi = Mock()
    modules["embedding_model"].EmbedddingModel = object

    hybrid_search = modules["hybrid_search"]
    hybrid_search.COLLECTION_NAME = "synthetic_collection"
    hybrid_search.tokenize = Mock(return_value=[])
    hybrid_search.read_documents = Mock(
        return_value=[
            {"id": 1, "text": "!!!", "source": "synthetic.txt"},
            {"id": 2, "text": "……", "source": "synthetic.txt"},
        ]
    )
    hybrid_search.bm25_search = Mock()
    hybrid_search.vector_search = Mock()
    hybrid_search.rrf_fusion = Mock()

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "app_under_test", BACKEND_DIR / "app.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, modules
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class RefreshIndexTokenlessDocumentsTest(unittest.TestCase):
    def test_refresh_succeeds_when_every_document_has_no_search_terms(self):
        app_module, modules = load_app_module()
        service = app_module.RagService.__new__(app_module.RagService)
        service.client = object()
        service.ready = False

        service.refresh_index()

        self.assertEqual(
            service.documents,
            modules["hybrid_search"].read_documents.return_value,
        )
        self.assertEqual(service.tokenized_documents, [[], []])
        self.assertIsNone(service.bm25)
        self.assertTrue(service.ready)
        modules["rank_bm25"].BM25Okapi.assert_not_called()


if __name__ == "__main__":
    unittest.main()
