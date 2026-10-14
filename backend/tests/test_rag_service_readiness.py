"""Regression tests for the lazy in-memory index recovery guard."""

import ast
import unittest
from pathlib import Path


APP_PATH = Path(__file__).parents[1] / "app.py"


def load_ensure_ready():
    """Load the real method without importing optional service dependencies."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    service = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RagService"
    )
    method = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef) and node.name == "ensure_ready"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace["ensure_ready"]


ENSURE_READY = load_ensure_ready()


class SyntheticService:
    ensure_ready = ENSURE_READY

    def __init__(self, ready):
        self.ready = ready
        self.refresh_calls = 0

    def refresh_index(self):
        self.refresh_calls += 1
        self.ready = True


class RagServiceReadinessTests(unittest.TestCase):
    def test_ready_service_does_not_rebuild_the_index(self):
        service = SyntheticService(ready=True)

        service.ensure_ready()

        self.assertEqual(service.refresh_calls, 0)

    def test_unready_service_rebuilds_the_index_once(self):
        service = SyntheticService(ready=False)

        service.ensure_ready()
        service.ensure_ready()

        self.assertEqual(service.refresh_calls, 1)
        self.assertTrue(service.ready)


if __name__ == "__main__":
    unittest.main()
