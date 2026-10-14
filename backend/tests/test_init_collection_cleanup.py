"""Regression test for cleanup when collection initialization fails."""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class FailingMilvusClient:
    instance = None

    def __init__(self, **kwargs):
        self.closed = False
        type(self).instance = self

    def has_collection(self, **kwargs):
        return False

    def create_collection(self, **kwargs):
        raise RuntimeError("synthetic create failure")

    def close(self):
        self.closed = True


class InitCollectionCleanupTest(unittest.TestCase):
    def test_client_closes_when_collection_creation_fails(self):
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None
        pymilvus = types.ModuleType("pymilvus")
        pymilvus.MilvusClient = FailingMilvusClient
        hybrid_search = types.ModuleType("hybrid_search")
        hybrid_search.COLLECTION_NAME = "synthetic_collection"

        with patch.dict(
            sys.modules,
            {
                "dotenv": dotenv,
                "pymilvus": pymilvus,
                "hybrid_search": hybrid_search,
            },
        ), patch.dict(os.environ, {}, clear=True), patch.object(
            sys, "path", [str(BACKEND_DIR), *sys.path]
        ):
            sys.modules.pop("init_collection", None)
            module = importlib.import_module("init_collection")

            with self.assertRaisesRegex(RuntimeError, "synthetic create failure"):
                module.main()

        self.assertIsNotNone(FailingMilvusClient.instance)
        self.assertTrue(FailingMilvusClient.instance.closed)


if __name__ == "__main__":
    unittest.main()
