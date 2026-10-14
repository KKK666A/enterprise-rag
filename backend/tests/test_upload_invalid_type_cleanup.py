import importlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeFastAPI:
    def __init__(self, *args, **kwargs):
        self.state = types.SimpleNamespace()

    def get(self, *args, **kwargs):
        return lambda function: function

    def post(self, *args, **kwargs):
        return lambda function: function


class TrackingFile(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.was_closed = False

    def close(self):
        self.was_closed = True
        super().close()


class UploadInvalidTypeCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(backend))

        fastapi = types.ModuleType("fastapi")
        fastapi.FastAPI = FakeFastAPI
        fastapi.File = lambda *args, **kwargs: None
        fastapi.HTTPException = HTTPException
        fastapi.Request = object
        fastapi.UploadFile = object

        responses = types.ModuleType("fastapi.responses")
        responses.RedirectResponse = object

        pydantic = types.ModuleType("pydantic")
        pydantic.BaseModel = object
        pydantic.Field = lambda *args, **kwargs: None

        with patch.dict(
            sys.modules,
            {
                "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
                "fastapi": fastapi,
                "fastapi.responses": responses,
                "pydantic": pydantic,
                "pymilvus": types.SimpleNamespace(MilvusClient=object),
                "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
                "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
                "hybrid_search": types.SimpleNamespace(
                    COLLECTION_NAME="synthetic_collection",
                    tokenize=lambda text: [],
                    read_documents=lambda client: [],
                    bm25_search=lambda *args: [],
                    vector_search=lambda *args: [],
                    rrf_fusion=lambda *args: [],
                ),
            },
        ):
            cls.app_module = importlib.import_module("app")

    def test_rejected_non_txt_upload_closes_file(self):
        stream = TrackingFile(b"synthetic document")
        upload = types.SimpleNamespace(filename="sample.pdf", file=stream)

        with self.assertRaises(HTTPException) as raised:
            self.app_module.upload_document(
                types.SimpleNamespace(app=types.SimpleNamespace(state=None)),
                upload,
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertTrue(stream.was_closed)


if __name__ == "__main__":
    unittest.main()
