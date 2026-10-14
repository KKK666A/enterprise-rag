import ast
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_split_text():
    """Load the pure text splitter without importing service dependencies."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    wanted_names = {"CHUNK_SIZE", "CHUNK_OVERLAP"}
    nodes = []

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted_names
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "split_text":
            nodes.append(node)

    namespace = {"re": __import__("re")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace["split_text"]


class SplitTextParagraphTests(unittest.TestCase):
    def test_normalizes_line_endings_and_ignores_blank_paragraphs(self):
        split_text = load_split_text()
        content = "  First synthetic paragraph.  \r\n\t\r\n\r\nSecond paragraph.\r\n\r\n  "

        self.assertEqual(
            split_text(content),
            ["First synthetic paragraph.", "Second paragraph."],
        )


if __name__ == "__main__":
    unittest.main()
