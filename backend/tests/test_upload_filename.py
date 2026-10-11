import importlib
import io
import sys
import types
import unittest
from unittest import mock
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def load_app_module():
    fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def __init__(self, **_kwargs):
            pass

        def get(self, *_args, **_kwargs):
            return lambda function: function

        post = get

    fastapi.FastAPI = FastAPI
    fastapi.File = lambda *_args, **_kwargs: None
    fastapi.HTTPException = HTTPException
    fastapi.Request = type("Request", (), {})
    fastapi.UploadFile = type("UploadFile", (), {})

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = type("RedirectResponse", (), {})

    embedding_model = types.ModuleType("embedding_model")
    embedding_model.EmbedddingModel = type("EmbedddingModel", (), {})

    hybrid_search = types.ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "test_collection"
    for name in ("tokenize", "read_documents", "bm25_search", "vector_search", "rrf_fusion"):
        setattr(hybrid_search, name, lambda *_args, **_kwargs: [])

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = type("MilvusClient", (), {})
    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = type("BM25Okapi", (), {})

    modules = {
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "embedding_model": embedding_model,
        "hybrid_search": hybrid_search,
        "pymilvus": pymilvus,
        "rank_bm25": rank_bm25,
    }
    with mock.patch.dict(sys.modules, modules):
        sys.modules.pop("app", None)
        return importlib.import_module("app")


class UploadFilenameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = load_app_module()

    def test_client_path_is_removed_before_source_is_stored(self):
        captured = {}

        class Service:
            def import_document(self, filename, chunks):
                captured["filename"] = filename
                captured["chunks"] = chunks
                return {"filename": filename, "chunks": len(chunks)}

        stream = io.BytesIO("Synthetic handbook text.".encode())
        upload = types.SimpleNamespace(
            filename=r"C:\fakepath\handbook.txt",
            file=stream,
        )
        request = types.SimpleNamespace(
            app=types.SimpleNamespace(
                state=types.SimpleNamespace(rag=Service())
            )
        )

        result = self.app.upload_document(request, upload)

        self.assertEqual("handbook.txt", captured["filename"])
        self.assertEqual(["Synthetic handbook text."], captured["chunks"])
        self.assertEqual("handbook.txt", result["filename"])
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
