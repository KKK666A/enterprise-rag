import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class _FastAPI:
    def __init__(self, *args, **kwargs):
        self.state = types.SimpleNamespace()

    def get(self, *args, **kwargs):
        return lambda function: function

    post = get


def _load_app_module():
    modules = {
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
        "fastapi": types.SimpleNamespace(
            FastAPI=_FastAPI,
            File=lambda *args, **kwargs: None,
            HTTPException=type("HTTPException", (Exception,), {}),
            Request=object,
            UploadFile=object,
        ),
        "fastapi.responses": types.SimpleNamespace(RedirectResponse=object),
        "pydantic": types.SimpleNamespace(
            BaseModel=object,
            Field=lambda *args, **kwargs: None,
        ),
        "pymilvus": types.SimpleNamespace(MilvusClient=object),
        "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
        "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
        "hybrid_search": types.SimpleNamespace(
            COLLECTION_NAME="synthetic_collection",
            tokenize=lambda text: text.split(),
            read_documents=lambda client: [],
            bm25_search=lambda *args: [],
            vector_search=lambda *args: [],
            rrf_fusion=lambda *args: [],
        ),
    }

    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "app_under_test", BACKEND_DIR / "app.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class RefreshIndexAtomicityTests(unittest.TestCase):
    def test_failed_rebuild_preserves_previous_index(self):
        app = _load_app_module()
        service = app.RagService.__new__(app.RagService)
        old_documents = [{"id": 1, "text": "existing synthetic policy"}]
        old_tokens = [["existing", "synthetic", "policy"]]
        old_bm25 = object()
        service.client = object()
        service.documents = old_documents
        service.tokenized_documents = old_tokens
        service.bm25 = old_bm25
        service.ready = True

        new_documents = [{"id": 2, "text": "replacement synthetic policy"}]

        with (
            patch.object(app, "read_documents", return_value=new_documents),
            patch.object(app, "tokenize", side_effect=lambda text: text.split()),
            patch.object(app, "BM25Okapi", side_effect=RuntimeError("rebuild failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "rebuild failed"):
                service.refresh_index()

        self.assertIs(service.documents, old_documents)
        self.assertIs(service.tokenized_documents, old_tokens)
        self.assertIs(service.bm25, old_bm25)
        self.assertTrue(service.ready)


if __name__ == "__main__":
    unittest.main()
