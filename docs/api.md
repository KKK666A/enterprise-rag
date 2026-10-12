# HTTP API contract

The backend exposes two JSON-facing operations under `/api`. The frontend uses
the same relative paths through the Vite development proxy. This service is a
retrieval demo: a query returns matching source chunks, not a generated answer.

## Import a document

`POST /api/documents` accepts `multipart/form-data` with one field named
`file`.

- The file name must end in `.txt` (case-insensitive) and be at most 200
  characters. Any directory portion supplied by a client is discarded.
- The file must be UTF-8; an optional UTF-8 byte-order mark is accepted.
- The maximum file size is 200 KiB (`204800` bytes).
- Empty documents are rejected. A document may produce at most 200 chunks.
- A file whose sanitized name already exists is rejected rather than
  overwritten.

A successful import returns HTTP 200 with the stored file name, the number of
new chunks, the total number of chunks, and a status message:

```json
{
  "filename": "synthetic-policy.txt",
  "chunks": 2,
  "total_chunks": 7,
  "message": "导入成功，BM25 索引已刷新"
}
```

The exact `message` text is intended for display and may be localized; clients
should use the other fields for program logic.

Expected error statuses are:

| Status | Meaning |
| --- | --- |
| `400` | Invalid extension, name, encoding, or empty content |
| `409` | A document with the same sanitized file name already exists |
| `413` | The byte or chunk limit was exceeded |
| `422` | The multipart `file` field is missing |
| `502` | The embedding service or Milvus operation failed |

## Query documents

`POST /api/query` accepts JSON with a single `question` string. Request
validation requires the original string to be 1 to 200 characters. The backend
then trims it; a value made entirely of whitespace is rejected with HTTP 400.
A missing, non-string, empty, or over-length value is rejected by request
validation with HTTP 422.

Example request body:

```json
{
  "question": "What is the synthetic travel approval process?"
}
```

When the knowledge base contains documents, the response has this shape:

```json
{
  "question": "What is the synthetic travel approval process?",
  "answer": null,
  "sources": [
    {
      "id": 12,
      "text": "Synthetic travel requests require manager approval.",
      "source": "synthetic-policy.txt",
      "rrf_score": 0.03278688524590164,
      "bm25_rank": 1,
      "vector_rank": 1,
      "bm25_score": 0.73,
      "vector_score": 0.91
    }
  ]
}
```

`sources` contains at most three chunks, ordered by reciprocal rank fusion
(RRF). A chunk found by only one retrieval route has `null` rank and score
fields for the other route. Scores are ranking signals, not confidence
percentages.

An empty knowledge base returns HTTP 200, an empty `sources` array, and a
display-oriented `message`. Query infrastructure failures return HTTP 502.
