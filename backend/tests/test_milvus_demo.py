from pathlib import Path
import sys
import tempfile
import types
import unittest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# The loader itself only uses pathlib; keep this unit test independent of the
# optional services imported by the executable demo module.
pymilvus = types.ModuleType("pymilvus")
pymilvus.MilvusClient = object
sys.modules.setdefault("pymilvus", pymilvus)

dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv)

embedding_model = types.ModuleType("embedding_model")
embedding_model.EmbedddingModel = object
sys.modules.setdefault("embedding_model", embedding_model)
sys.modules.setdefault("text", types.ModuleType("text"))

from milvus_demo import load_documents


class LoadDocumentsTests(unittest.TestCase):
    def test_ignores_blank_paragraphs_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_path = Path(directory) / "employee-handbook.txt"
            document_path.write_text(
                "\ufeffAnnual leave is five days.\r\n\r\n"
                "  \r\n\r\nExpense claims need approval.  \r\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_documents(document_path),
                [
                    {
                        "id": 1,
                        "source": "employee-handbook.txt",
                        "text": "Annual leave is five days.",
                    },
                    {
                        "id": 2,
                        "source": "employee-handbook.txt",
                        "text": "Expense claims need approval.",
                    },
                ],
            )

    def test_rejects_a_file_without_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document_path = Path(directory) / "empty.txt"
            document_path.write_text(" \n\n\t", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "文件没有可用于向量化的文本",
            ):
                load_documents(document_path)


if __name__ == "__main__":
    unittest.main()
