# Milvus storage schema

The backend stores one Milvus row per text chunk in the `rag_test_v1`
collection. The collection is created by `backend/init_collection.py` with
dynamic fields enabled.

| Field | Type | Purpose |
| --- | --- | --- |
| `id` | `INT64` primary key | Stable chunk identifier assigned by the backend |
| `vector` | Float vector | Embedding used for cosine-similarity search |
| `text` | Dynamic string field | Original chunk text returned in search results |
| `source` | Dynamic string field | Sanitized basename of the uploaded `.txt` file |

The vector dimension comes from `EMBEDDING_DIMENSIONS` and defaults to `1152`.
Both collection initialization and API startup reject a dimension mismatch;
changing the configured dimension does not migrate an existing collection.

## Row creation

An upload is decoded as UTF-8, split into non-empty chunks, and embedded before
rows are written. New IDs start at one greater than the largest ID currently in
the collection. For example, a synthetic file named `handbook.txt` split into
two chunks may produce rows shaped like:

```json
{"id": 41, "text": "Synthetic policy excerpt A.", "source": "handbook.txt", "vector": [0.01, -0.02]}
{"id": 42, "text": "Synthetic policy excerpt B.", "source": "handbook.txt", "vector": [0.03, 0.04]}
```

The vectors above are abbreviated examples; stored vectors must have exactly
the configured dimension. The service rejects another upload with the same
`source` filename. It does not currently expose row deletion, document
replacement, schema migration, or tenant metadata.

Treat `id`, `text`, and `source` as part of the application's storage contract
when inspecting or maintaining the collection manually. Adding unrelated
dynamic fields will not make them available through the current API or search
responses.
