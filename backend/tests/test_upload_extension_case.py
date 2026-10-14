"""Regression test for case-insensitive TXT upload validation."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _FastAPI:
    def __init__(self, **_kwargs):
        pass

    def get(self, *_args, **_kwargs):
        return lambda function: function

    post = get


class _Field:
    def __call__(self, default=None, **_kwargs):
        return default


class _Upload:
    def __init__(self, name, content):
        import io

        self.filename = name
        self.file = io.BytesIO(content)


class UploadExtensionCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "dotenv", "fastapi", "fastapi.responses", "pydantic",
                "pymilvus", "rank_bm25", "embedding_model", "hybrid_search",
            )
        }

        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *_args, **_kwargs: None

        fastapi = types.ModuleType("fastapi")
        fastapi.FastAPI = _FastAPI
        fastapi.File = lambda *_args, **_kwargs: None
        fastapi.HTTPException = type("HTTPException", (Exception,), {})
        fastapi.Request = type("Request", (), {})
        fastapi.UploadFile = type("UploadFile", (), {})

        responses = types.ModuleType("fastapi.responses")
        responses.RedirectResponse = lambda **kwargs: kwargs

        pydantic = types.ModuleType("pydantic")
        pydantic.BaseModel = type("BaseModel", (), {})
        pydantic.Field = _Field()

        pymilvus = types.ModuleType("pymilvus")
        pymilvus.MilvusClient = type("MilvusClient", (), {})

        rank_bm25 = types.ModuleType("rank_bm25")
        rank_bm25.BM25Okapi = type("BM25Okapi", (), {})

        embedding_model = types.ModuleType("embedding_model")
        embedding_model.EmbedddingModel = type("EmbedddingModel", (), {})

        hybrid_search = types.ModuleType("hybrid_search")
        hybrid_search.COLLECTION_NAME = "synthetic_collection"
        for name in ("tokenize", "read_documents", "bm25_search", "vector_search", "rrf_fusion"):
            setattr(hybrid_search, name, lambda *_args, **_kwargs: [])

        sys.modules.update({
            "dotenv": dotenv,
            "fastapi": fastapi,
            "fastapi.responses": responses,
            "pydantic": pydantic,
            "pymilvus": pymilvus,
            "rank_bm25": rank_bm25,
            "embedding_model": embedding_model,
            "hybrid_search": hybrid_search,
        })

        app_path = Path(__file__).parents[1] / "app.py"
        spec = importlib.util.spec_from_file_location("app_under_test", app_path)
        cls.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app)

    @classmethod
    def tearDownClass(cls):
        for name, module in cls._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_uppercase_txt_extension_is_accepted(self):
        imported = []
        service = types.SimpleNamespace(
            import_document=lambda filename, chunks: imported.append((filename, chunks)) or {"ok": True}
        )
        request = types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace(rag=service)))
        upload = _Upload("SYNTHETIC-POLICY.TXT", b"Synthetic leave policy")

        result = self.app.upload_document(request, upload)

        self.assertEqual({"ok": True}, result)
        self.assertEqual([("SYNTHETIC-POLICY.TXT", ["Synthetic leave policy"])], imported)


if __name__ == "__main__":
    unittest.main()
