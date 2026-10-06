import ast
import re
import unittest
from pathlib import Path


def load_split_text():
    app_path = Path(__file__).parents[1] / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    required_names = {"CHUNK_SIZE", "CHUNK_OVERLAP"}
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in required_names
                for target in node.targets
            )
        )
        or isinstance(node, ast.FunctionDef) and node.name == "split_text"
    ]
    namespace = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(app_path), "exec"), namespace)
    return namespace["split_text"], namespace["CHUNK_SIZE"], namespace["CHUNK_OVERLAP"]


split_text, CHUNK_SIZE, CHUNK_OVERLAP = load_split_text()


class SplitTextTests(unittest.TestCase):
    def test_long_paragraph_preserves_configured_overlap(self):
        content = "".join(str(index % 10) for index in range(CHUNK_SIZE + 25))

        chunks = split_text(content)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], content[:CHUNK_SIZE])
        self.assertEqual(chunks[1], content[CHUNK_SIZE - CHUNK_OVERLAP :])
        self.assertEqual(chunks[0][-CHUNK_OVERLAP:], chunks[1][:CHUNK_OVERLAP])


if __name__ == "__main__":
    unittest.main()
