import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


APP_PATH = Path(__file__).parents[1] / "app.py"


def load_app_module():
    fastapi = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, **_kwargs):
            pass

        def get(self, *_args, **_kwargs):
            return lambda function: function

        post = get

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    fastapi.FastAPI = FastAPI
    fastapi.File = lambda *_args, **_kwargs: None
    fastapi.HTTPException = HTTPException
    fastapi.Request = type("Request", (), {})
    fastapi.UploadFile = type("UploadFile", (), {})

    responses = types.ModuleType("fastapi.responses")
    responses.RedirectResponse = type("RedirectResponse", (), {})

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    pydantic.Field = lambda **_kwargs: None

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *_args, **_kwargs: None

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = type("MilvusClient", (), {})

    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = type("BM25Okapi", (), {})

    embedding_model = types.ModuleType("embedding_model")
    embedding_model.EmbedddingModel = type("EmbedddingModel", (), {})

    hybrid_search = types.ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "synthetic_collection"
    hybrid_search.tokenize = lambda text: text.split()
    hybrid_search.read_documents = lambda _client: []
    hybrid_search.bm25_search = lambda *_args: []
    hybrid_search.vector_search = lambda *_args: []
    hybrid_search.rrf_fusion = lambda *_args: []

    modules = {
        "dotenv": dotenv,
        "embedding_model": embedding_model,
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "hybrid_search": hybrid_search,
        "pydantic": pydantic,
        "pymilvus": pymilvus,
        "rank_bm25": rank_bm25,
    }

    spec = importlib.util.spec_from_file_location("app_for_batch_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class RecordingModel:
    def __init__(self):
        self.batches = []

    def encode(self, texts):
        self.batches.append(list(texts))
        return [[float(text.removeprefix("chunk-"))] for text in texts]


class RecordingClient:
    def __init__(self):
        self.upserts = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


class ImportDocumentBatchingTest(unittest.TestCase):
    def test_embeddings_are_requested_in_bounded_batches(self):
        app = load_app_module()
        service = app.RagService.__new__(app.RagService)
        service.lock = threading.Lock()
        service.ready = True
        service.documents = []
        service.model = RecordingModel()
        service.client = RecordingClient()
        service.refresh_index = lambda: service.documents.extend(
            service.client.upserts[0]["data"]
        )
        chunks = [f"chunk-{index}" for index in range(23)]

        result = service.import_document("synthetic.txt", chunks)

        self.assertEqual([10, 10, 3], [len(batch) for batch in service.model.batches])
        self.assertEqual(chunks, [text for batch in service.model.batches for text in batch])
        rows = service.client.upserts[0]["data"]
        self.assertEqual(list(range(1, 24)), [row["id"] for row in rows])
        self.assertEqual(23, result["chunks"])


if __name__ == "__main__":
    unittest.main()
