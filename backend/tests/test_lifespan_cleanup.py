import ast
import unittest
from contextlib import asynccontextmanager
from pathlib import Path


APP_PATH = Path(__file__).parents[1] / "app.py"


def load_lifespan(service_factory, load_dotenv):
    """Load only the lifespan function so this test needs no service packages."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "__file__": str(APP_PATH),
        "asynccontextmanager": asynccontextmanager,
        "load_dotenv": load_dotenv,
        "Path": Path,
        "RagService": service_factory,
    }
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace["lifespan"]


class LifespanCleanupTest(unittest.IsolatedAsyncioTestCase):
    async def test_service_is_closed_when_application_raises(self):
        class FakeService:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        service = FakeService()
        lifespan = load_lifespan(lambda: service, lambda *args, **kwargs: None)
        app = type("FakeApp", (), {"state": type("State", (), {})()})()

        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            async with lifespan(app):
                self.assertIs(app.state.rag, service)
                raise RuntimeError("synthetic failure")

        self.assertTrue(service.closed)


if __name__ == "__main__":
    unittest.main()
