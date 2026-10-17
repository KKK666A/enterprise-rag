"""Regression test for upload stream cleanup when reading the body fails."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


APP_PATH = Path(__file__).parents[1] / "app.py"


def _load_upload_document():
    """Load only the endpoint function so this test needs no service packages."""
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "upload_document"
    )

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    namespace = {
        "File": lambda *args, **kwargs: None,
        "HTTPException": HTTPException,
        "MAX_CHUNKS": 200,
        "MAX_FILE_BYTES": 200 * 1024,
        "Request": object,
        "UploadFile": object,
        "split_text": lambda content: [content],
    }
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace["upload_document"]


class FailingStream:
    def __init__(self):
        self.closed = False

    def read(self, _size):
        raise OSError("synthetic read failure")

    def close(self):
        self.closed = True


class UploadReadFailureCleanupTest(unittest.TestCase):
    def test_stream_is_closed_when_read_fails(self):
        upload_document = _load_upload_document()
        stream = FailingStream()
        upload = SimpleNamespace(filename="synthetic.txt", file=stream)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

        with self.assertRaisesRegex(OSError, "synthetic read failure"):
            upload_document(request, upload)

        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
