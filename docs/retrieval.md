# Retrieval and ranking

The service uses hybrid retrieval: it searches the same document chunks with
BM25 and vector similarity, then combines the two ranked lists with reciprocal
rank fusion (RRF). It does not generate an answer from the retrieved text;
`POST /api/query` returns the ranked chunks in `sources` and leaves `answer` as
`null`.

## Keyword route

Documents and questions share the same tokenizer. Text is lowercased, segmented
with `jieba`, and stripped of a small built-in set of Chinese stop words.
Punctuation-only tokens are discarded.

BM25 candidates must share at least one token with the question. This explicit
token-overlap check matters for small collections, where a legitimate BM25
match can have a zero or negative score. Candidates are ordered by descending
BM25 score, with the numeric chunk ID as the tie-breaker.

## Vector route

The question is embedded with the configured embedding model and searched
against the collection's `vector` field using cosine similarity. Milvus is
queried with strong consistency, matching the consistency used when the local
BM25 index reads the collection.

## Fusion and result fields

Each route contributes at most 10 chunks. For a chunk at one-based rank `r`,
that route adds `1 / (60 + r)` to its `rrf_score`. The raw BM25 and cosine
scores are not added together because their scales are unrelated. The fused
list is ordered by descending `rrf_score`, then by numeric chunk ID, and is
limited to three chunks.

Every returned source includes:

- `id`, `text`, and `source` from the stored chunk;
- `rrf_score`, the score used for final ordering;
- `bm25_rank` and `bm25_score` when the keyword route found the chunk;
- `vector_rank` and `vector_score` when the vector route found the chunk.

A rank or route-specific score is `null` when that route did not return the
chunk. Consumers should use `rrf_score` for the final order and treat the raw
route scores as diagnostics only.
