import importlib
import os
import sys
import types
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

fake_dashscope = types.ModuleType("dashscope")
fake_dashscope.MultiModalEmbedding = types.SimpleNamespace(call=None)
fake_dotenv = types.ModuleType("dotenv")
fake_dotenv.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dashscope", fake_dashscope)
sys.modules.setdefault("dotenv", fake_dotenv)

embedding_model = importlib.import_module("embedding_model")


class EmbeddingErrorHandlingTests(unittest.TestCase):
    def test_upstream_failure_does_not_expose_response_message_or_api_key(self):
        response = types.SimpleNamespace(
            status_code=HTTPStatus.UNAUTHORIZED,
            code="InvalidApiKey",
            request_id="synthetic-request-id",
            message="Authorization failed for sk-synthetic-secret",
        )

        with patch.dict(
            os.environ,
            {
                "EMBEDDING_API_KEY": "sk-synthetic-secret",
                "EMBEDDING_MODEL": "synthetic-embedding-model",
            },
            clear=True,
        ), patch.object(
            embedding_model.dashscope.MultiModalEmbedding,
            "call",
            return_value=response,
        ):
            model = embedding_model.EmbedddingModel()

            with self.assertRaises(RuntimeError) as raised:
                model.encode(["synthetic policy text"])

        error = str(raised.exception)
        self.assertIn("status=401", error)
        self.assertIn("code=InvalidApiKey", error)
        self.assertIn("request_id=synthetic-request-id", error)
        self.assertNotIn(response.message, error)
        self.assertNotIn("sk-synthetic-secret", error)


if __name__ == "__main__":
    unittest.main()
