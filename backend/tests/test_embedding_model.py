import importlib
import sys
import unittest
from http import HTTPStatus
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class EmbeddingModelResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashscope = ModuleType("dashscope")
        cls.dashscope.MultiModalEmbedding = SimpleNamespace(call=Mock())

        dotenv = ModuleType("dotenv")
        dotenv.load_dotenv = Mock()

        with patch.dict(
            sys.modules,
            {"dashscope": cls.dashscope, "dotenv": dotenv},
        ), patch.object(sys, "path", [str(BACKEND_DIR), *sys.path]):
            cls.module = importlib.import_module("embedding_model")

    def setUp(self):
        self.model = self.module.EmbedddingModel.__new__(
            self.module.EmbedddingModel
        )
        self.model.api_key = "synthetic-key"
        self.model.model = "synthetic-model"
        self.model.dimension = 2
        self.dashscope.MultiModalEmbedding.call.reset_mock()

    def test_encode_normalizes_inputs_and_restores_response_order(self):
        self.dashscope.MultiModalEmbedding.call.return_value = SimpleNamespace(
            status_code=HTTPStatus.OK,
            output={
                "embeddings": [
                    {"index": 1, "embedding": [3.0, 4.0]},
                    {"index": 0, "embedding": [1.0, 2.0]},
                ]
            },
        )

        vectors = self.model.encode([" first ", "second\n"])

        self.assertEqual(vectors, [[1.0, 2.0], [3.0, 4.0]])
        self.dashscope.MultiModalEmbedding.call.assert_called_once_with(
            api_key="synthetic-key",
            model="synthetic-model",
            input=[{"text": "first"}, {"text": "second"}],
            dimension=2,
        )

    def test_encode_rejects_missing_response_index(self):
        self.dashscope.MultiModalEmbedding.call.return_value = SimpleNamespace(
            status_code=HTTPStatus.OK,
            output={
                "embeddings": [
                    {"index": 0, "embedding": [1.0, 2.0]},
                    {"index": 2, "embedding": [3.0, 4.0]},
                ]
            },
        )

        with self.assertRaises(RuntimeError):
            self.model.encode(["first", "second"])


if __name__ == "__main__":
    unittest.main()
