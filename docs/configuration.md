# Backend configuration

Copy `backend/.env.example` to `backend/.env` before running the API, collection
initializer, or hybrid-search CLI. The backend reads these settings:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `EMBEDDING_API_KEY` | Yes | None | Authenticates requests to the embedding service. |
| `EMBEDDING_MODEL` | Yes | None | Selects the embedding model. |
| `EMBEDDING_BASE_URL` | No | `https://dashscope.aliyuncs.com/api/v1` | Overrides the embedding service endpoint. |
| `EMBEDDING_DIMENSIONS` | No | `1152` | Sets both the requested vector size and the Milvus collection dimension. |
| `MILVUS_URL` | No | `http://127.0.0.1:19530` | Selects the Milvus server. |
| `MILVUS_TOKEN` | No | Empty | Authenticates to Milvus when the server requires it. |

Keep `EMBEDDING_DIMENSIONS` consistent with the selected model and the existing
`rag_test_v1` collection. `backend/init_collection.py` deliberately refuses to
alter an existing collection when its vector dimension differs; the API and
hybrid-search CLI also reject that mismatch at startup. To change dimensions,
use a separately managed collection or remove and recreate the demo collection
only after preserving any data you need.

Do not commit `backend/.env`; it is ignored by Git. The tracked
`backend/.env.example` contains placeholders only.
