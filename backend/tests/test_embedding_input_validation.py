import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock


BACKEND_DIR = Path(__file__).resolve().parents[1]


class EmbeddingInputValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedding_call = Mock()

        dashscope = ModuleType("dashscope")
        dashscope.MultiModalEmbedding = type(
            "MultiModalEmbedding",
            (),
            {"call": cls.embedding_call},
        )

        dotenv = ModuleType("dotenv")
        dotenv.load_dotenv = Mock()

        sys.modules["dashscope"] = dashscope
        sys.modules["dotenv"] = dotenv
        sys.path.insert(0, str(BACKEND_DIR))
        sys.modules.pop("embedding_model", None)
        cls.module = importlib.import_module("embedding_model")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("embedding_model", None)
        sys.modules.pop("dashscope", None)
        sys.modules.pop("dotenv", None)
        sys.path.remove(str(BACKEND_DIR))

    def setUp(self):
        self.embedding_call.reset_mock()
        self.model = self.module.EmbedddingModel.__new__(
            self.module.EmbedddingModel
        )
        self.model.api_key = "synthetic-api-key"
        self.model.model = "synthetic-model"
        self.model.dimension = 3

    def test_empty_batch_is_rejected_without_api_call(self):
        with self.assertRaises(ValueError):
            self.model.encode([])

        self.embedding_call.assert_not_called()

    def test_whitespace_item_is_rejected_without_api_call(self):
        with self.assertRaises(ValueError):
            self.model.encode(["synthetic text", " \t\n "])

        self.embedding_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
