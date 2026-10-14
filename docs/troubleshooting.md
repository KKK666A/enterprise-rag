# Troubleshooting

This guide describes failure modes reported by the current backend. It does not
assume that a failed import was rolled back: the upload path is not a database
transaction.

## The API does not start

The service validates Milvus before accepting requests. Check the backend
terminal for one of these conditions:

- `rag_test_v1` is missing. Start Milvus, then run
  `python backend/init_collection.py`. The initializer creates an empty
  collection and does not call the embedding service.
- The collection vector dimension differs from `EMBEDDING_DIMENSIONS`. Use a
  collection built for the configured model, or restore the model and dimension
  that produced the existing vectors. Matching dimensions alone does not make
  vectors from different models compatible.
- Milvus cannot be reached. Confirm that `MILVUS_URL` points to a running,
  client-compatible Milvus instance and that `MILVUS_TOKEN` is correct when the
  server requires authentication.

The backend closes the Milvus client if startup validation fails, so correct the
underlying configuration and restart the process.

## An upload returns 502

Document chunks are embedded before the first database write. After that, the
backend performs one Milvus upsert and rebuilds its in-memory BM25 index. A 502
therefore has two materially different cases:

1. If embedding failed, no rows were written by that request.
2. If the upsert or subsequent index refresh failed, rows may already exist.

Inspect `rag_test_v1` before retrying the same file. The application rejects a
source filename already present in its refreshed document view, but it does not
provide rollback, deletion, or overwrite endpoints. Do not repeatedly retry an
ambiguous failure without first checking the stored rows.

## Queries work in one process but not another

BM25 state is built in memory when each backend process starts and after that
process completes an upload. Updates are not broadcast between workers. Run the
demo with a single worker; otherwise, separate workers can hold different
lexical indexes even though they share Milvus.

## A query returns evidence instead of an answer

This is expected. `POST /api/query` returns ranked source chunks and always sets
`answer` to `null`; the frontend displays only the highest-ranked chunk. The
repository implements retrieval, not generated answers.
