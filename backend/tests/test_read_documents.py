import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def load_hybrid_search():
    """Load the module with lightweight stand-ins for optional services."""
    modules = {
        "dotenv": types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
        "embedding_model": types.SimpleNamespace(EmbedddingModel=object),
        "jieba": types.SimpleNamespace(lcut=lambda text: list(text)),
        "pymilvus": types.SimpleNamespace(MilvusClient=object),
        "rank_bm25": types.SimpleNamespace(BM25Okapi=object),
    }
    module_path = Path(__file__).parents[1] / "hybrid_search.py"
    spec = importlib.util.spec_from_file_location(
        "hybrid_search_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)

    return module


class ReadDocumentsTests(unittest.TestCase):
    def test_closes_query_iterator_when_reading_a_batch_fails(self):
        hybrid_search = load_hybrid_search()
        iterator = Mock()
        iterator.next.side_effect = RuntimeError("synthetic read failure")
        client = Mock()
        client.query_iterator.return_value = iterator

        with self.assertRaisesRegex(RuntimeError, "synthetic read failure"):
            hybrid_search.read_documents(client)

        iterator.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
