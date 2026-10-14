# Deployment and process model

The backend is designed to run as a single process. `RagService` keeps the
BM25 corpus and index in process memory, and its lock serializes uploads and
queries only within that process.

Do not start the current application with multiple Uvicorn workers. An upload
refreshes only the worker that handled it, so another worker can continue to
serve an older BM25 index even though the new chunks are already present in
Milvus. Milvus strong consistency applies to database reads; it does not
synchronize these per-process in-memory indexes.

For this demo, run one backend process (the README development command already
does this). If the service is restarted, startup rebuilds the BM25 index from
the documents stored in Milvus before requests are served.

A future multi-worker deployment would need a shared or externally managed
lexical index, or an explicit cross-process refresh mechanism. Until then,
scale the current service as a single backend instance rather than adding
workers behind a load balancer.
