"""Offline coverage for creation of the empty Milvus collection."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class InitCollectionCreationTests(unittest.TestCase):
    def test_main_creates_retrieval_schema_and_closes_client(self):
        client = Mock()
        client.has_collection.return_value = False

        dotenv = ModuleType("dotenv")
        dotenv.load_dotenv = Mock()
        pymilvus = ModuleType("pymilvus")
        pymilvus.MilvusClient = Mock(return_value=client)
        hybrid_search = ModuleType("hybrid_search")
        hybrid_search.COLLECTION_NAME = "rag_test_v1"

        replacements = {
            "dotenv": dotenv,
            "pymilvus": pymilvus,
            "hybrid_search": hybrid_search,
        }
        original_modules = {
            name: sys.modules.get(name) for name in replacements
        }

        try:
            sys.modules.update(replacements)
            spec = importlib.util.spec_from_file_location(
                "init_collection_under_test",
                BACKEND_DIR / "init_collection.py",
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            with patch.dict(os.environ, {"EMBEDDING_DIMENSIONS": ""}):
                module.main()
        finally:
            for name, original in original_modules.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

        client.create_collection.assert_called_once_with(
            collection_name="rag_test_v1",
            dimension=1152,
            primary_field_name="id",
            auto_id=False,
            vector_field_name="vector",
            metric_type="COSINE",
            enable_dynamic_field=True,
            consistency_level="Strong",
        )
        client.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
