import importlib
import io
import sys
import types
import unittest
from pathlib import Path


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeFastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return lambda function: function

    def post(self, *args, **kwargs):
        return lambda function: function


def import_app_without_service_dependencies():
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
    sys.modules.update(stubs)

    backend_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_dir))
    try:
        sys.modules.pop("app", None)
        return importlib.import_module("app")
    finally:
        sys.path.remove(str(backend_dir))
        for name in stubs:
            sys.modules.pop(name, None)


class UploadSizeLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = import_app_without_service_dependencies()

    def test_oversized_upload_is_rejected_and_closed_before_service_access(self):
        upload = types.SimpleNamespace(
            filename="synthetic-policy.txt",
            file=io.BytesIO(b"x" * (self.app.MAX_FILE_BYTES + 1)),
        )

        with self.assertRaises(HTTPException) as raised:
            self.app.upload_document(object(), upload)

        self.assertEqual(raised.exception.status_code, 413)
        self.assertTrue(upload.file.closed)


if __name__ == "__main__":
    unittest.main()
