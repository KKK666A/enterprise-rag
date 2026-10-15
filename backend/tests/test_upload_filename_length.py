import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class FakeFastAPI:
    def __init__(self, **_kwargs):
        self.state = types.SimpleNamespace()

    def get(self, *_args, **_kwargs):
        return lambda function: function

    def post(self, *_args, **_kwargs):
        return lambda function: function


class FakeField:
    def __init__(self, data):
        self._data = data
        self.closed = False

    def read(self, _size):
        return self._data

    def close(self):
        self.closed = True


class FakeUpload:
    def __init__(self, filename, data=b"synthetic policy text"):
        self.filename = filename
        self.file = FakeField(data)


class FakeService:
    def __init__(self):
        self.calls = []

    def import_document(self, filename, chunks):
        self.calls.append((filename, chunks))
        return {"filename": filename, "chunks": len(chunks)}


def load_app_module():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FakeFastAPI
    fastapi.File = lambda *_args, **_kwargs: None
    fastapi.HTTPException = HTTPException
    fastapi.Request = object
    fastapi.UploadFile = object

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = object

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = object
    pydantic.Field = lambda **_kwargs: None

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *_args, **_kwargs: None

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = object

    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = object

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

    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "app_filename_length_test", APP_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class UploadFilenameLengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = load_app_module()

    def test_filename_at_200_character_limit_is_accepted(self):
        filename = f"{'a' * 196}.txt"
        upload = FakeUpload(filename)
        service = FakeService()
        request = types.SimpleNamespace(
            app=types.SimpleNamespace(
                state=types.SimpleNamespace(rag=service)
            )
        )

        result = self.app_module.upload_document(request, upload)

        self.assertEqual(result["filename"], filename)
        self.assertEqual(service.calls, [(filename, ["synthetic policy text"])])
        self.assertTrue(upload.file.closed)

    def test_filename_over_200_character_limit_is_rejected(self):
        filename = f"{'a' * 197}.txt"
        upload = FakeUpload(filename)
        service = FakeService()
        request = types.SimpleNamespace(
            app=types.SimpleNamespace(
                state=types.SimpleNamespace(rag=service)
            )
        )

        with self.assertRaises(HTTPException) as raised:
            self.app_module.upload_document(request, upload)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(service.calls, [])
        self.assertTrue(upload.file.closed)


if __name__ == "__main__":
    unittest.main()
