import os
from http import HTTPStatus
from pathlib import Path

import dashscope
from dotenv import load_dotenv


class EmbedddingModel:
    def __init__(self):
        load_dotenv(Path(__file__).with_name(".env"), override=True)
        api_key = os.getenv("EMBEDDING_API_KEY")
        base_url = os.getenv("EMBEDDING_BASE_URL") or "https://dashscope.aliyuncs.com/api/v1"
        model=os.getenv("EMBEDDING_MODEL")
        if not api_key:
            raise ValueError("没有配置 EMBEDDING_API_KEY")
        if not model:
            raise ValueError("没有配置 EMBEDDING_MODEL")

        self.model = model
        self.api_key = api_key
        self.dimension = int(os.getenv("EMBEDDING_DIMENSIONS") or "1152")
        dashscope.base_http_api_url = base_url.rstrip("/")
    def encode(self,texts:list[str]) ->list[list[float]]:
        """
               将多段文本转换为向量。

               参数：
                   texts: 已经切分好的文本列表

               返回：
                   二维列表，每一项对应一段文本的向量
               """
        if not texts:
            raise ValueError("texts 不能为空")
        cleaned_texts = [text.strip() for text in texts]
        if any(not text for text in cleaned_texts):
            raise ValueError("texts 中不能包含空白文本")

        response = dashscope.MultiModalEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=[{"text": text} for text in cleaned_texts],
            dimension=self.dimension,
        )
        if response.status_code != HTTPStatus.OK:
            # 不打印完整服务端错误信息，避免某些鉴权错误回显密钥。
            raise RuntimeError(
                f"向量调用失败：status={response.status_code}, "
                f"code={response.code}, request_id={response.request_id}"
            )
        ordered_data = sorted(
            response.output["embeddings"],
            key=lambda item: item["index"],
        )
        if [item["index"] for item in ordered_data] != list(range(len(cleaned_texts))):
            raise RuntimeError("返回向量的数量或索引与输入文本不匹配")
        vectors = [item["embedding"] for item in ordered_data]
        if any(len(vector) != self.dimension for vector in vectors):
            raise RuntimeError("返回向量的维度与配置不一致")
        return vectors

    def encode_query(self, query: str) -> list[float]:
        """向量化单个用户问题。"""
        return self.encode([query])[0]



