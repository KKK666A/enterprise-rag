import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]


class EmbeddingBaseUrlTests(unittest.TestCase):
    def test_custom_base_url_has_trailing_slashes_removed(self):
        dashscope = types.ModuleType("dashscope")
        dashscope.MultiModalEmbedding = object()
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: None

        spec = importlib.util.spec_from_file_location(
            "embedding_model_under_test",
            BACKEND_DIR / "embedding_model.py",
        )
        module = importlib.util.module_from_spec(spec)

        environment = {
            "EMBEDDING_API_KEY": "synthetic-key",
            "EMBEDDING_MODEL": "synthetic-model",
            "EMBEDDING_BASE_URL": "https://embedding.example.test/api/v1///",
        }
        with patch.dict(sys.modules, {"dashscope": dashscope, "dotenv": dotenv}), \
                patch.dict(os.environ, environment, clear=True):
            spec.loader.exec_module(module)
            module.EmbedddingModel()

        self.assertEqual(
            dashscope.base_http_api_url,
            "https://embedding.example.test/api/v1",
        )


if __name__ == "__main__":
    unittest.main()
