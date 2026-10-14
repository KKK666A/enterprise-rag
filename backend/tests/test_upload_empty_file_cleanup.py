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


class FastAPI:
    def __init__(self, **_kwargs):
        pass

    def get(self, *_args, **_kwargs):
        return lambda function: function

    post = get


class BaseModel:
    pass


class TrackingBytesIO(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.was_closed = False

    def close(self):
        self.was_closed = True
        super().close()


def load_app_module():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = FastAPI
    fastapi.File = lambda *_args, **_kwargs: None
    fastapi.HTTPException = HTTPException
    fastapi.Request = type("Request", (), {})
    fastapi.UploadFile = type("UploadFile", (), {})

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = type("RedirectResponse", (), {})

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = BaseModel
    pydantic.Field = lambda **_kwargs: None

    hybrid_search = types.ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "synthetic_collection"
    for name in (
        "tokenize",
        "read_documents",
        "bm25_search",
        "vector_search",
        "rrf_fusion",
    ):
        setattr(hybrid_search, name, lambda *_args, **_kwargs: [])

    modules = {
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *_args, **_kwargs: None),
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "pydantic": pydantic,
        "pymilvus": types.SimpleNamespace(MilvusClient=object),
        "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
        "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
        "hybrid_search": hybrid_search,
    }

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        app_path = Path(__file__).parents[1] / "app.py"
        spec = importlib.util.spec_from_file_location("app_under_test", app_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class EmptyUploadCleanupTests(unittest.TestCase):
    def test_empty_upload_is_rejected_and_file_is_closed(self):
        app = load_app_module()
        stream = TrackingBytesIO(b"")
        upload = types.SimpleNamespace(filename="empty.txt", file=stream)
        request = types.SimpleNamespace(app=types.SimpleNamespace(state=None))

        with self.assertRaises(HTTPException) as raised:
            app.upload_document(request, upload)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, "文件内容为空")
        self.assertTrue(stream.was_closed)


if __name__ == "__main__":
    unittest.main()
