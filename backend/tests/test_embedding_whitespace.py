"""Offline regression tests for embedding request normalization."""

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class EmbeddingWhitespaceTests(unittest.TestCase):
    def test_encode_sends_trimmed_text_to_provider(self):
        calls = []

        class FakeMultiModalEmbedding:
            @staticmethod
            def call(**kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(
                    status_code=200,
                    output={
                        "embeddings": [
                            {"index": 0, "embedding": [0.1, 0.2]},
                            {"index": 1, "embedding": [0.3, 0.4]},
                        ]
                    },
                )

        fake_dashscope = types.ModuleType("dashscope")
        fake_dashscope.MultiModalEmbedding = FakeMultiModalEmbedding
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda *args, **kwargs: None

        spec = importlib.util.spec_from_file_location(
            "embedding_model_under_test",
            BACKEND_DIR / "embedding_model.py",
        )
        module = importlib.util.module_from_spec(spec)

        environment = {
            "EMBEDDING_API_KEY": "synthetic-key",
            "EMBEDDING_MODEL": "synthetic-model",
            "EMBEDDING_DIMENSIONS": "2",
        }
        with patch.dict(
            sys.modules,
            {"dashscope": fake_dashscope, "dotenv": fake_dotenv},
        ), patch.dict(os.environ, environment, clear=True):
            spec.loader.exec_module(module)
            model = module.EmbedddingModel()
            vectors = model.encode(["  alpha  ", "\n beta\t"])

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(
            calls[0]["input"],
            [{"text": "alpha"}, {"text": "beta"}],
        )


if __name__ == "__main__":
    unittest.main()
