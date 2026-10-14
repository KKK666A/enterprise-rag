"""Regression checks for the repository's local credential-file contract."""

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SecretHygieneTests(unittest.TestCase):
    def test_local_environment_files_are_ignored(self):
        candidates = (".env", "backend/.env", "frontend/.env.local")

        for candidate in candidates:
            with self.subTest(candidate=candidate):
                result = subprocess.run(
                    ["git", "check-ignore", "--quiet", "--no-index", candidate],
                    cwd=ROOT,
                    check=False,
                )
                self.assertEqual(0, result.returncode)

    def test_environment_example_contains_no_credentials(self):
        values = {}
        for line in (ROOT / "backend" / ".env.example").read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                values[key] = value

        self.assertEqual("", values["EMBEDDING_API_KEY"])
        self.assertEqual("", values["MILVUS_TOKEN"])


if __name__ == "__main__":
    unittest.main()
