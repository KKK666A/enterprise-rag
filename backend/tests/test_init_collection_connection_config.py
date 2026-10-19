import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class RecordingMilvusClient:
    constructor_arguments = None

    def __init__(self, **kwargs):
        type(self).constructor_arguments = kwargs

    def has_collection(self, **_kwargs):
        return True

    def describe_collection(self, **_kwargs):
        return {
            "fields": [
                {"name": "vector", "params": {"dim": "384"}},
            ]
        }

    def close(self):
        pass


class InitCollectionConnectionConfigTest(unittest.TestCase):
    def test_main_forwards_custom_url_and_token_to_milvus(self):
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *_args, **_kwargs: None

        pymilvus = types.ModuleType("pymilvus")
        pymilvus.MilvusClient = RecordingMilvusClient

        hybrid_search = types.ModuleType("hybrid_search")
        hybrid_search.COLLECTION_NAME = "rag_test_v1"

        module_spec = importlib.util.spec_from_file_location(
            "init_collection_under_test",
            BACKEND_DIR / "init_collection.py",
        )
        module = importlib.util.module_from_spec(module_spec)

        environment = {
            "MILVUS_URL": "https://milvus.example.invalid:19530",
            "MILVUS_TOKEN": "synthetic-test-token",
            "EMBEDDING_DIMENSIONS": "384",
        }
        dependencies = {
            "dotenv": dotenv,
            "pymilvus": pymilvus,
            "hybrid_search": hybrid_search,
        }

        with patch.dict(sys.modules, dependencies), patch.dict(
            os.environ, environment, clear=False
        ):
            module_spec.loader.exec_module(module)
            module.main()

        self.assertEqual(
            RecordingMilvusClient.constructor_arguments,
            {
                "uri": "https://milvus.example.invalid:19530",
                "token": "synthetic-test-token",
            },
        )


if __name__ == "__main__":
    unittest.main()
