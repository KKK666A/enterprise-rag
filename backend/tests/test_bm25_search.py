from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from hybrid_search import bm25_search


class StubBm25:
    def get_scores(self, query_tokens):
        assert query_tokens == ["policy"]
        return [-0.5, 100.0, 0.0]


def test_bm25_search_keeps_lexical_matches_with_non_positive_scores():
    documents = [
        {"id": 11, "text": "Policy alpha", "source": "alpha.txt"},
        {"id": 12, "text": "Unrelated", "source": "other.txt"},
        {"id": 13, "text": "Policy beta", "source": "beta.txt"},
    ]
    tokenized_documents = [
        ["policy", "alpha"],
        ["unrelated"],
        ["policy", "beta"],
    ]

    results = bm25_search(
        "POLICY!",
        documents,
        tokenized_documents,
        StubBm25(),
    )

    assert [result["id"] for result in results] == [13, 11]
    assert [result["score"] for result in results] == [0.0, -0.5]
