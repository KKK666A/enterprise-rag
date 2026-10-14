import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return lambda function: function

    def post(self, *args, **kwargs):
        return lambda function: function


def load_app_module():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FastAPI
    fastapi.File = lambda *args, **kwargs: None
    fastapi.HTTPException = HTTPException
    fastapi.Request = object
    fastapi.UploadFile = object

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = object

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    pydantic.Field = lambda *args, **kwargs: None

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = object

    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = object

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None

    embedding_model = types.ModuleType("embedding_model")
    embedding_model.EmbedddingModel = object

    hybrid_search = types.ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "synthetic_collection"
    for name in (
        "tokenize",
        "read_documents",
        "bm25_search",
        "vector_search",
        "rrf_fusion",
    ):
        setattr(hybrid_search, name, Mock(name=name))

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

    path = Path(__file__).parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("app_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class DuplicateDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = load_app_module()

    def test_duplicate_source_is_rejected_before_embedding_or_upsert(self):
        service = self.app.RagService.__new__(self.app.RagService)
        service.lock = threading.Lock()
        service.ready = True
        service.documents = [
            {"id": 7, "source": "handbook.txt", "text": "Synthetic policy."}
        ]
        service.model = Mock()
        service.client = Mock()

        with self.assertRaises(self.app.HTTPException) as raised:
            service.import_document("handbook.txt", ["Replacement policy."])

        self.assertEqual(raised.exception.status_code, 409)
        service.model.encode.assert_not_called()
        service.client.upsert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
