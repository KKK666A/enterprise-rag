import importlib.util
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


class _FastAPI:
    def __init__(self, **_kwargs):
        pass

    def get(self, *_args, **_kwargs):
        return lambda function: function

    post = get


def _install_import_stubs():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *_args, **_kwargs: None

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FastAPI
    fastapi.File = lambda *args, **kwargs: None
    fastapi.HTTPException = HTTPException
    fastapi.Request = object
    fastapi.UploadFile = object

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = object

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = object
    pydantic.Field = lambda **_kwargs: None

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = object

    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = object

    embedding_model = types.ModuleType("embedding_model")
    embedding_model.EmbedddingModel = object

    hybrid_search = types.ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "documents"
    hybrid_search.tokenize = lambda text: text.split()
    hybrid_search.read_documents = lambda _client: []
    hybrid_search.bm25_search = lambda *_args: []
    hybrid_search.vector_search = lambda *_args: []
    hybrid_search.rrf_fusion = lambda *_args: []

    return {
        "dotenv": dotenv,
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "pydantic": pydantic,
        "pymilvus": pymilvus,
        "rank_bm25": rank_bm25,
        "embedding_model": embedding_model,
        "hybrid_search": hybrid_search,
    }


class UploadUtf8BomTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_modules = {
            name: sys.modules.get(name) for name in _install_import_stubs()
        }
        sys.modules.update(_install_import_stubs())

        app_path = Path(__file__).parents[1] / "app.py"
        spec = importlib.util.spec_from_file_location("app_under_test", app_path)
        cls.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("app_under_test", None)
        for name, module in cls._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_upload_strips_utf8_bom_before_import(self):
        imported = {}

        class Service:
            def import_document(self, filename, chunks):
                imported["filename"] = filename
                imported["chunks"] = chunks
                return {"filename": filename, "chunks": len(chunks)}

        raw_file = io.BytesIO(b"\xef\xbb\xbfSynthetic policy text")
        upload = types.SimpleNamespace(filename="policy.txt", file=raw_file)
        request = types.SimpleNamespace(
            app=types.SimpleNamespace(
                state=types.SimpleNamespace(rag=Service())
            )
        )

        result = self.app.upload_document(request, upload)

        self.assertEqual(result, {"filename": "policy.txt", "chunks": 1})
        self.assertEqual(imported["chunks"], ["Synthetic policy text"])
        self.assertTrue(raw_file.closed)


if __name__ == "__main__":
    unittest.main()
