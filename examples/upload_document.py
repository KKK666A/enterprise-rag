"""Upload a synthetic UTF-8 document to the local enterprise-rag API."""

import argparse
import json
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SYNTHETIC_DOCUMENT = """Example Travel Policy

Employees submit fictional travel requests through the Example Portal.
This sample contains no real employee, customer, or company information.
"""


def multipart_body(filename: str, text: str) -> tuple[bytes, str]:
    """Build the single-file multipart payload expected by /api/documents."""
    boundary = f"enterprise-rag-{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        f"{text}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Backend origin (default: %(default)s)",
    )
    args = parser.parse_args()

    body, content_type = multipart_body(
        "synthetic-travel-policy.txt",
        SYNTHETIC_DOCUMENT,
    )
    request = Request(
        f"{args.base_url.rstrip('/')}/api/documents",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Upload failed ({error.code}): {detail}") from error
    except URLError as error:
        raise SystemExit(f"Could not reach the backend: {error.reason}") from error

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
