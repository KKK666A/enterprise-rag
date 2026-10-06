import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class FakeMilvusClient:
    instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.created = []
        self.closed = False
        type(self).instance = self

    def has_collection(self, **kwargs):
        return True

    def describe_collection(self, **kwargs):
        return {
            "fields": [
                {"name": "id", "params": {}},
                {"name": "vector", "params": {"dim": 768}},
            ]
        }

    def create_collection(self, **kwargs):
        self.created.append(kwargs)

    def close(self):
        self.closed = True


def load_init_collection():
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    pymilvus = ModuleType("pymilvus")
    pymilvus.MilvusClient = FakeMilvusClient
    hybrid_search = ModuleType("hybrid_search")
    hybrid_search.COLLECTION_NAME = "rag_test_v1"

    module_path = BACKEND_DIR / "init_collection.py"
    spec = importlib.util.spec_from_file_location("init_collection_under_test", module_path)
    module = importlib.util.module_from_spec(spec)

    with patch.dict(
        sys.modules,
        {
            "dotenv": dotenv,
            "pymilvus": pymilvus,
            "hybrid_search": hybrid_search,
        },
    ):
        spec.loader.exec_module(module)

    return module


class InitCollectionTests(unittest.TestCase):
    def test_existing_collection_with_matching_dimension_is_not_recreated(self):
        module = load_init_collection()

        with patch.dict(
            os.environ,
            {"EMBEDDING_DIMENSIONS": "768"},
            clear=True,
        ):
            module.main()

        client = FakeMilvusClient.instance
        self.assertEqual(client.created, [])
        self.assertTrue(client.closed)

    def test_dimension_mismatch_fails_without_recreating_collection(self):
        module = load_init_collection()

        with patch.dict(
            os.environ,
            {"EMBEDDING_DIMENSIONS": "1152"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "dimension differs"):
                module.main()

        client = FakeMilvusClient.instance
        self.assertEqual(client.created, [])
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
