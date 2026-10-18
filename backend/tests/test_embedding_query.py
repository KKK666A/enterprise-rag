"""Tests for the single-query embedding adapter."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# Keep this unit test independent of optional service dependencies.
sys.modules.setdefault("dashscope", types.ModuleType("dashscope"))
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_stub)

embedding_model = importlib.import_module("embedding_model")


class EncodeQueryTests(unittest.TestCase):
    def test_wraps_query_as_batch_and_returns_its_vector(self):
        model = object.__new__(embedding_model.EmbedddingModel)
        expected_vector = [0.25, -0.5, 0.75]
        model.encode = Mock(return_value=[expected_vector])

        result = model.encode_query("synthetic retention policy question")

        self.assertIs(result, expected_vector)
        model.encode.assert_called_once_with(
            ["synthetic retention policy question"]
        )


if __name__ == "__main__":
    unittest.main()
