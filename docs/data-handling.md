# Data handling and privacy boundaries

This prototype processes document text outside the browser. Before using it
with non-synthetic data, review the data flow and the policies of the services
configured in `backend/.env`.

## Where data goes

| Input | Destination | Stored by this application |
| --- | --- | --- |
| Uploaded `.txt` contents | FastAPI, the configured embedding service, and Milvus | Each text chunk is stored in Milvus with its source filename and embedding vector. |
| Uploaded filename | FastAPI and Milvus | Stored as the `source` value for every chunk. |
| Query text | FastAPI and the configured embedding service | Not explicitly persisted by this repository. Milvus receives only the resulting query vector for search. |

The browser sends uploads and queries to the backend through `/api/documents`
and `/api/query`. Document chunks are sent to the embedding service in batches
of 10. A query is also sent to that service to produce its search vector.
Setting a custom `EMBEDDING_BASE_URL` changes that external destination; it
does not make embedding local unless the configured endpoint itself is local.

## Persistence and deletion

Milvus is the durable store for chunk text, source filenames, and vectors. The
backend also keeps document text and tokenized text in process memory for its
BM25 index until the process exits or the index is refreshed.

The application currently has no document deletion endpoint, retention policy,
or per-user/tenant isolation. Removing a local upload does not remove its
chunks from Milvus. Operators must manage collection retention and deletion
directly in Milvus and should not run `backend/milvus_demo.py` against a
collection containing data they need to preserve, because that script upserts
records beginning at ID 1.

## Operational checklist

- Use synthetic or explicitly authorized documents for evaluation.
- Treat filenames as potentially sensitive metadata; rename files before
  upload when their names reveal confidential information.
- Confirm the embedding endpoint's region, retention, and logging policy.
- Restrict access to both the API and Milvus. This repository does not provide
  authentication or authorization.
- Back up or delete Milvus data according to your own retention policy.
- Rotate `EMBEDDING_API_KEY` if it is exposed. The key is read only by the
  backend and must not be placed in frontend code or committed in `.env`.

API error responses intentionally replace embedding and Milvus exception text
with a generic message. This reduces accidental disclosure to the browser, but
it is not a substitute for access controls, infrastructure log review, or the
external provider's privacy controls.
