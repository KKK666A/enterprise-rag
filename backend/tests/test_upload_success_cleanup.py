import importlib
import io
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


class _FastAPI:
    def __init__(self, *args, **kwargs):
        self.state = types.SimpleNamespace()

    def get(self, *args, **kwargs):
        return lambda function: function

    post = get


class _Field:
    def __init__(self, *args, **kwargs):
        pass


class _HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail


class _Model:
    pass


class _TrackedFile(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.was_closed = False

    def close(self):
        self.was_closed = True
        super().close()


class _Service:
    def __init__(self):
        self.received = None

    def import_document(self, filename, chunks):
        self.received = (filename, chunks)
        return {"filename": filename, "chunks": len(chunks)}


def _load_app_module():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FastAPI
    fastapi.File = lambda *args, **kwargs: None
    fastapi.HTTPException = _HTTPException
    fastapi.Request = object
    fastapi.UploadFile = object

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = object

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = _Model
    pydantic.Field = _Field

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

    stubs = {
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "pydantic": pydantic,
        "dotenv": dotenv,
        "pymilvus": pymilvus,
        "rank_bm25": rank_bm25,
        "embedding_model": embedding_model,
        "hybrid_search": hybrid_search,
    }

    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        sys.modules.pop("app", None)
        return importlib.import_module("app")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class UploadSuccessCleanupTests(unittest.TestCase):
    def test_successful_upload_closes_temporary_file(self):
        app = _load_app_module()
        service = _Service()
        request = types.SimpleNamespace(
            app=types.SimpleNamespace(state=types.SimpleNamespace(rag=service))
        )
        tracked = _TrackedFile(b"Synthetic policy paragraph.")
        upload = types.SimpleNamespace(filename="policy.txt", file=tracked)

        result = app.upload_document(request, upload)

        self.assertEqual(result, {"filename": "policy.txt", "chunks": 1})
        self.assertEqual(service.received, ("policy.txt", ["Synthetic policy paragraph."]))
        self.assertTrue(tracked.was_closed)


if __name__ == "__main__":
    unittest.main()
