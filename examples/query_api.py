"""Query a running enterprise-rag backend using only the Python standard library.

Start the backend, then run:

    python examples/query_api.py --question "How many days of annual leave are available?"
"""

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the enterprise-rag API")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="backend base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--question",
        default="How many days of annual leave are available?",
        help="synthetic question to submit",
    )
    args = parser.parse_args()

    request = Request(
        f"{args.base_url.rstrip('/')}/api/query",
        data=json.dumps({"question": args.question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except HTTPError as error:
        try:
            detail = json.load(error).get("detail", error.reason)
        except (json.JSONDecodeError, AttributeError):
            detail = error.reason
        parser.exit(1, f"API request failed ({error.code}): {detail}\n")
    except URLError as error:
        parser.exit(1, f"Could not reach the backend: {error.reason}\n")

    print(f"Question: {result['question']}")
    sources = result.get("sources", [])
    if not sources:
        print(result.get("message", "No matching text was found."))
        return 0

    for rank, source in enumerate(sources, start=1):
        print(f"\n{rank}. {source.get('source') or '(unknown source)'}")
        print(source["text"])
        print(f"RRF score: {source['rrf_score']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
