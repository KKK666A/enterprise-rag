from embedding_model import EmbedddingModel


def main() -> None:
    embedding_model =  EmbedddingModel()

    texts = [
        "员工每年享有五天带薪年假。",
        "费用报销需要经过部门负责人审批。",
    ]

    vectors = embedding_model.encode(texts)

    print("向量模型调用成功")
    print("文本数量：", len(texts))
    print("向量数量：", len(vectors))
    print("向量维度：", len(vectors[0]))
    print("第一个向量前五项：", vectors[0][:5])

    query_vector = embedding_model.encode_query("员工有多少天年假？")

    print("问题向量维度：", len(query_vector))


if __name__ == "__main__":
    main()