import os
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import MilvusClient

import embedding_model
import text
from embedding_model import EmbedddingModel


def load_documents(file_path: Path) -> list[dict]:
    # 1. 读取整个文件
    content = file_path.read_text(encoding="utf-8-sig")

    # 2. 统一换行符，然后按空行切成段落
    paragraphs = content.replace("\r\n", "\n").split("\n\n")

    documents = []

    # 3. 去掉空白段落，给每个文本块记录来源
    for paragraph in paragraphs:
        text = paragraph.strip()

        if not text:
            continue

        documents.append(
            {
                "id": len(documents) + 1,
                "source": file_path.name,
                "text": text,
            }
        )

    if not documents:
        raise ValueError("文件没有可用于向量化的文本")

    return documents
def main():
    load_dotenv(override=True)

    client = MilvusClient(
        uri=os.getenv("MILVUS_URL") or "http://127.0.0.1:19530",
    )

    try :
        print("Milvus 连接成功",client.list_collections())
        embedding_model = EmbedddingModel()

        file_path = Path(__file__).parent / "data.txt"
        documents = load_documents(file_path)
        texts = [document["text"] for document in documents]
        vectors = embedding_model.encode(texts)

        if len(vectors) != len(texts):
            raise ValueError("文档和向量数不一样")

        dimension = len(vectors[0])
        print("文档数量：", len(texts))
        print("向量维度：", dimension)

        collection_name="rag_test_v1"

        if not client.has_collection(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                primary_field_name="id",
                auto_id=False,
                vector_field_name="vector",
                metric_type="COSINE",
                enable_dynamic_field=True,
                consistency_level="Strong",
            )
            print("已创建集合：", collection_name)
        else:
            # 已有集合的向量维度必须与当前模型一致
            info = client.describe_collection(
                collection_name=collection_name
            )

            vector_field = next(
                field
                for field in info["fields"]
                if field["name"] == "vector"
            )

            if int(vector_field["params"]["dim"]) != dimension:
                raise ValueError("已有集合维度不一致，请使用新的集合名称")

        rows = []

        for document, vector in zip(documents, vectors):
                rows.append(
                    {
                        "id": document["id"],
                        "vector": vector,
                        "text": document["text"],
                        "source": document["source"],
                    }
                )

        result = client.upsert(
                collection_name=collection_name,
                data=rows,
            )
        client.load_collection(collection_name=collection_name)

        saved_rows = client.query(
            collection_name=collection_name,
            filter="id > 0",
            output_fields=["id", "text", "source"],
            limit=100,
            consistency_level="Strong",
        )

        print("本次查询读到的记录数：", len(saved_rows))

        for row in sorted(saved_rows, key=lambda item: item["id"]):
            print(row["id"], row["source"], row["text"])






    finally:
        client.close()


if __name__ == "__main__":
        main()