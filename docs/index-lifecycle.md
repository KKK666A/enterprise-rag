# Index lifecycle and consistency

The backend combines two indexes over the same text chunks:

- Milvus stores each chunk and its embedding for vector search.
- An in-process BM25 index is rebuilt from the chunks stored in Milvus.

This design keeps Milvus as the persisted source of truth. The BM25 index is
process-local and is not persisted separately.

## Startup

During application startup, `RagService`:

1. connects to Milvus and checks that the `rag_test_v1` collection exists;
2. verifies that the collection's vector dimension matches the configured
   embedding model;
3. loads the collection; and
4. reads every stored chunk, ordered by ID, to build the BM25 index.

Startup fails if any of these steps fail. Run the collection initialization
script before starting the API.

## Document import

An accepted UTF-8 text file is split into chunks before it reaches the
service. Imports and queries share a process-local lock, so a query cannot run
while an import is changing the indexes.

For an import, the service checks for an existing source filename, assigns IDs
after the largest currently loaded ID, creates all embeddings in batches, and
then upserts the chunks into Milvus. After the upsert it reloads all documents
from Milvus and replaces the in-memory BM25 index as one unit.

If a failure happens after the Milvus write begins, Milvus may contain data
that the current BM25 index has not loaded. The service marks the in-memory
state as not ready before writing, so the next import or query first attempts
to rebuild BM25 from Milvus. Before retrying an uncertain import, inspect the
collection: duplicate source filenames are rejected once the persisted chunks
have been reloaded.

## Query

Each query runs BM25 keyword retrieval and Milvus vector retrieval, then uses
reciprocal rank fusion (RRF) to combine their rankings. The API returns at most
three source chunks. It does not generate an answer from those chunks; the
`answer` field is currently `null`.

The lock and recovery behavior apply only within one backend process. Running
multiple backend workers would give each worker its own lock and BM25 snapshot,
so this demo should be operated as a single process unless cross-process index
coordination is added.
