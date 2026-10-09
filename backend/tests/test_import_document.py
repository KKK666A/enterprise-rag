import ast
import threading
import unittest
from pathlib import Path


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def load_rag_service():
    app_path = Path(__file__).parents[1] / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    required_names = {"EMBEDDING_BATCH_SIZE"}
    selected_nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "RagService"
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in required_names
                for target in node.targets
            )
        )
    ]
    namespace = {
        "BM25Okapi": object,
        "COLLECTION_NAME": "test_collection",
        "EmbedddingModel": object,
        "HTTPException": HTTPException,
        "MilvusClient": object,
        "bm25_search": None,
        "read_documents": None,
        "rrf_fusion": None,
        "threading": threading,
        "tokenize": None,
        "vector_search": None,
    }
    exec(compile(ast.Module(selected_nodes, type_ignores=[]), app_path, "exec"), namespace)
    return namespace["RagService"]


RagService = load_rag_service()


class ShortEmbeddingModel:
    def encode(self, texts):
        return [[0.1, 0.2]] if texts else []


class RecordingClient:
    def __init__(self):
        self.upsert_calls = []

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)


class ImportDocumentTests(unittest.TestCase):
    def test_embedding_count_mismatch_does_not_write_partial_document(self):
        service = object.__new__(RagService)
        service.lock = threading.Lock()
        service.ready = True
        service.documents = []
        service.model = ShortEmbeddingModel()
        service.client = RecordingClient()

        with self.assertRaisesRegex(
            RuntimeError,
            "文本块数量和向量数量不一致",
        ):
            service.import_document(
                "synthetic-policy.txt",
                ["Synthetic first section.", "Synthetic second section."],
            )

        self.assertEqual(service.client.upsert_calls, [])
        self.assertTrue(service.ready)


if __name__ == "__main__":
    unittest.main()
