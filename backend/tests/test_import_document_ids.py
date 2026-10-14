import importlib
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]


def _stub_dependencies():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = Mock()
    sys.modules["dotenv"] = dotenv

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = Mock(return_value=types.SimpleNamespace(get=Mock(return_value=lambda f: f), post=Mock(return_value=lambda f: f)))
    fastapi.File = Mock()

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    fastapi.HTTPException = HTTPException
    fastapi.Request = object
    fastapi.UploadFile = object
    sys.modules["fastapi"] = fastapi

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = object
    sys.modules["fastapi.responses"] = responses

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = object
    pydantic.Field = Mock()
    sys.modules["pydantic"] = pydantic

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = object
    sys.modules["pymilvus"] = pymilvus

    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = object
    sys.modules["rank_bm25"] = rank_bm25

    embedding_model = types.ModuleType("embedding_model")
    embedding_model.EmbedddingModel = object
    sys.modules["embedding_model"] = embedding_model

    hybrid_search = types.ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "test_collection"
    for name in ("tokenize", "read_documents", "bm25_search", "vector_search", "rrf_fusion"):
        setattr(hybrid_search, name, Mock())
    sys.modules["hybrid_search"] = hybrid_search


class ImportDocumentIdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(BACKEND_DIR))
        _stub_dependencies()
        cls.app_module = importlib.import_module("app")

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(BACKEND_DIR))
        sys.modules.pop("app", None)

    def test_assigns_consecutive_ids_after_highest_existing_id(self):
        service = self.app_module.RagService.__new__(self.app_module.RagService)
        service.lock = threading.Lock()
        service.ready = True
        service.documents = [
            {"id": 2, "text": "Synthetic policy A", "source": "a.txt"},
            {"id": 9, "text": "Synthetic policy B", "source": "b.txt"},
        ]
        service.model = Mock()
        service.model.encode.return_value = [[0.1], [0.2]]
        service.client = Mock()
        service.refresh_index = Mock()

        service.import_document(
            "synthetic.txt",
            ["Synthetic first chunk", "Synthetic second chunk"],
        )

        rows = service.client.upsert.call_args.kwargs["data"]
        self.assertEqual(
            service.client.upsert.call_args.kwargs["collection_name"],
            "test_collection",
        )
        self.assertEqual([row["id"] for row in rows], [10, 11])
        self.assertEqual([row["source"] for row in rows], ["synthetic.txt"] * 2)


if __name__ == "__main__":
    unittest.main()
