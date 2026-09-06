import os
import re
from pathlib import Path

import jieba
from dotenv import load_dotenv
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi

from embedding_model import EmbedddingModel


COLLECTION_NAME = "rag_test_v1"

# 每路最多召回 10 条，融合后返回 3 条。
RECALL_K = 10
TOP_K = 3
RRF_K = 60

# 初始使用少量停用词，之后根据检索效果调整。
STOP_WORDS = {"的", "了", "呢", "吗", "啊", "请问"}


def tokenize(text: str) -> list[str]:
    """文档和问题共用同一个分词函数。"""
    return [
        word
        for word in jieba.lcut(text.lower())
        if word not in STOP_WORDS
        and re.search(r"[\w\u4e00-\u9fff]", word)
    ]


def read_documents(client: MilvusClient) -> list[dict]:
    """分批读取全部文本块，不读取向量。"""
    iterator = client.query_iterator(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["id", "text", "source"],
        batch_size=500,
        consistency_level="Strong",
    )

    documents = []

    try:
        while True:
            batch = iterator.next()

            if not batch:
                break

            documents.extend(batch)
    finally:
        iterator.close()

    # 固定顺序，便于重复测试。
    return sorted(documents, key=lambda doc: doc["id"])


def bm25_search(
    question: str,
    documents: list[dict],
    tokenized_documents: list[list[str]],
    bm25: BM25Okapi,
) -> list[dict]:
    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    query_words = set(query_tokens)

    # 只保留存在关键词交集的文档。
    # 小语料下 BM25 分数可能为零或负数，
    # 因此不简单地使用 score > 0 判断是否命中。
    candidates = [
        index
        for index, tokens in enumerate(tokenized_documents)
        if query_words.intersection(tokens)
    ]

    candidates.sort(
        key=lambda index: (-float(scores[index]), documents[index]["id"])
    )

    return [
        {
            "id": documents[index]["id"],
            "score": float(scores[index]),
            "text": documents[index]["text"],
            "source": documents[index].get("source", ""),
        }
        for index in candidates[:RECALL_K]
    ]


def vector_search(
    question: str,
    client: MilvusClient,
    model: EmbedddingModel,
) -> list[dict]:
    query_vector = model.encode_query(question)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        limit=RECALL_K,
        search_params={"metric_type": "COSINE"},
        output_fields=["text", "source"],
        consistency_level="Strong",
    )

    return [
        {
            "id": hit["id"],
            "score": float(hit["distance"]),
            "text": hit["entity"]["text"],
            "source": hit["entity"].get("source", ""),
        }
        for hit in results[0]
    ]


def rrf_fusion(
    keyword_hits: list[dict],
    vector_hits: list[dict],
) -> list[dict]:
    """根据两路排名融合，不直接相加原始分数。"""
    merged = {}

    for route, hits in (
        ("bm25", keyword_hits),
        ("vector", vector_hits),
    ):
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit["id"]

            if doc_id not in merged:
                merged[doc_id] = {
                    "id": doc_id,
                    "text": hit["text"],
                    "source": hit["source"],
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "vector_rank": None,
                    "bm25_score": None,
                    "vector_score": None,
                }

            row = merged[doc_id]
            row["rrf_score"] += 1.0 / (RRF_K + rank)
            row[f"{route}_rank"] = rank
            row[f"{route}_score"] = hit["score"]

    return sorted(
        merged.values(),
        key=lambda row: (-row["rrf_score"], row["id"]),
    )[:TOP_K]


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"), override=True)

    client = MilvusClient(
        uri=os.getenv("MILVUS_URL") or "http://127.0.0.1:19530",
        token=os.getenv("MILVUS_TOKEN") or "",
    )

    try:
        if not client.has_collection(collection_name=COLLECTION_NAME):
            raise ValueError("找不到 rag_test_v1，请先运行入库脚本")

        client.load_collection(collection_name=COLLECTION_NAME)

        documents = read_documents(client)

        if not documents:
            raise ValueError("集合为空，请先写入文档")

        tokenized_documents = [
            tokenize(doc["text"]) for doc in documents
        ]

        if not any(tokenized_documents):
            raise ValueError("文档分词结果全部为空")

        # 启动时构建一次 BM25 索引，供多次提问复用。
        bm25 = BM25Okapi(tokenized_documents)
        model = EmbedddingModel()

        info = client.describe_collection(
            collection_name=COLLECTION_NAME
        )
        vector_field = next(
            field for field in info["fields"]
            if field["name"] == "vector"
        )

        if int(vector_field["params"]["dim"]) != model.dimension:
            raise ValueError("当前模型配置维度与数据库向量维度不一致")

        print(f"已加载 {len(documents)} 个文本块")
        print("输入问题开始检索，输入 exit 退出")

        while True:
            question = input("\n问题：").strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            keyword_hits = bm25_search(
                question, documents, tokenized_documents, bm25
            )
            vector_hits = vector_search(question, client, model)

            results = rrf_fusion(keyword_hits, vector_hits)

            for rank, row in enumerate(results, start=1):
                print(f"\n第 {rank} 条，ID={row['id']}")
                print(f"RRF 分数：{row['rrf_score']:.6f}")
                print(
                    f"BM25 排名：{row['bm25_rank']}，"
                    f"原始分数：{row['bm25_score']}"
                )
                print(
                    f"向量排名：{row['vector_rank']}，"
                    f"余弦相似度：{row['vector_score']}"
                )
                print("来源：", row["source"])
                print("内容：", row["text"])

    finally:
        client.close()


if __name__ == "__main__":
    main()