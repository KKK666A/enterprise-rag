import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi

from embedding_model import EmbedddingModel
from hybrid_search import (
    COLLECTION_NAME,
    tokenize,
    read_documents,
    bm25_search,
    vector_search,
    rrf_fusion,
)


# 小型学习 Demo 的限制
MAX_FILE_BYTES = 200 * 1024
MAX_CHUNKS = 200
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
EMBEDDING_BATCH_SIZE = 10


def split_text(content: str) -> list[str]:
    """按空行分段；长段落再按字符数切分，保留少量重叠。"""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", content)

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        start = 0

        while start < len(paragraph):
            end = min(start + CHUNK_SIZE, len(paragraph))
            chunk = paragraph[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == len(paragraph):
                break

            start = end - CHUNK_OVERLAP

    return chunks


class RagService:
    """管理数据库连接、模型和本地 BM25 索引。"""

    def __init__(self):
        self.model = EmbedddingModel()

        self.client = MilvusClient(
            uri=os.getenv("MILVUS_URL") or "http://127.0.0.1:19530",
            token=os.getenv("MILVUS_TOKEN") or "",
        )

        # 单进程 Demo：上传和查询串行处理，避免索引更新期间查询。
        self.lock = threading.Lock()

        self.documents = []
        self.tokenized_documents = []
        self.bm25 = None
        self.ready = False

        try:
            if not self.client.has_collection(
                collection_name=COLLECTION_NAME
            ):
                raise RuntimeError(
                    f"集合 {COLLECTION_NAME} 不存在，请先运行建库脚本"
                )

            info = self.client.describe_collection(
                collection_name=COLLECTION_NAME
            )

            vector_field = next(
                field
                for field in info["fields"]
                if field["name"] == "vector"
            )

            if int(vector_field["params"]["dim"]) != self.model.dimension:
                raise RuntimeError("数据库向量维度与模型配置不一致")

            self.client.load_collection(
                collection_name=COLLECTION_NAME
            )

            self.refresh_index()

        except Exception:
            self.client.close()
            raise

    def refresh_index(self):
        """从数据库重新构建 BM25，构建成功后整体替换。"""
        documents = read_documents(self.client)

        tokenized_documents = [
            tokenize(doc["text"]) for doc in documents
        ]

        bm25 = (
            BM25Okapi(tokenized_documents)
            if any(tokenized_documents)
            else None
        )

        self.documents = documents
        self.tokenized_documents = tokenized_documents
        self.bm25 = bm25
        self.ready = True

    def ensure_ready(self):
        # 如果上次写入或刷新中途失败，下次请求先重新读取数据库。
        if not self.ready:
            self.refresh_index()

    def import_document(self, filename: str, chunks: list[str]) -> dict:
        with self.lock:
            self.ensure_ready()

            # 本版明确拒绝同名文件，避免无意覆盖已有文档。
            if any(
                doc.get("source") == filename
                for doc in self.documents
            ):
                raise HTTPException(
                    status_code=409,
                    detail="同名文档已经存在。本版暂不支持覆盖更新。",
                )

            # 沿用当前 INT64 主键，接着已有最大 ID 分配。
            next_id = max(
                (int(doc["id"]) for doc in self.documents),
                default=0,
            ) + 1

            # 先生成全部向量，再开始写入。
            vectors = []

            for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
                vectors.extend(self.model.encode(batch))

            if len(vectors) != len(chunks):
                raise RuntimeError("文本块数量和向量数量不一致")

            rows = [
                {
                    "id": next_id + index,
                    "text": chunk,
                    "source": filename,
                    "vector": vector,
                }
                for index, (chunk, vector) in enumerate(
                    zip(chunks, vectors)
                )
            ]

            # 写入开始后，直到 BM25 刷新完成才恢复 ready。
            self.ready = False

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                data=rows,
            )

            self.refresh_index()

            return {
                "filename": filename,
                "chunks": len(rows),
                "total_chunks": len(self.documents),
                "message": "导入成功，BM25 索引已刷新",
            }

    def query(self, question: str) -> dict:
        with self.lock:
            self.ensure_ready()

            if not self.documents:
                return {
                    "question": question,
                    "answer": None,
                    "sources": [],
                    "message": "知识库为空，请先上传文档",
                }

            keyword_hits = (
                bm25_search(
                    question,
                    self.documents,
                    self.tokenized_documents,
                    self.bm25,
                )
                if self.bm25 is not None
                else []
            )

            vector_hits = vector_search(
                question,
                self.client,
                self.model,
            )

            results = rrf_fusion(keyword_hits, vector_hits)

            return {
                "question": question,
                "answer": None,
                "sources": results,
            }

    def close(self):
        self.client.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时初始化，退出时关闭连接。"""
    load_dotenv(
        Path(__file__).with_name(".env"),
        override=True,
    )

    service = RagService()
    app.state.rag = service

    try:
        yield
    finally:
        service.close()


app = FastAPI(
    title="企业知识库检索 Demo",
    lifespan=lifespan,
)


class QueryBody(BaseModel):
    question: str = Field(min_length=1, max_length=200)


@app.get("/", include_in_schema=False)
def index():
    # 第一版直接用 FastAPI 自带页面测试。
    return RedirectResponse(url="/docs")


@app.post("/api/documents")
def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    # 文件名只作为来源信息，不拼接为磁盘写入路径。
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]

    try:
        if not filename or not filename.lower().endswith(".txt"):
            raise HTTPException(
                status_code=400,
                detail="请上传 .txt 文件",
            )

        if len(filename) > 200:
            raise HTTPException(
                status_code=400,
                detail="文件名过长",
            )

        raw = file.file.read(MAX_FILE_BYTES + 1)

        if len(raw) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="学习版文件大小不能超过 200KB",
            )

        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="文件编码不是 UTF-8，请另存为 UTF-8 后上传",
            )

        chunks = split_text(content)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="文件内容为空",
            )

        if len(chunks) > MAX_CHUNKS:
            raise HTTPException(
                status_code=413,
                detail="学习版单次最多导入 200 个文本块",
            )

        service = request.app.state.rag

        try:
            return service.import_document(filename, chunks)
        except HTTPException:
            raise
        except Exception:
            # 不把第三方接口可能包含敏感信息的错误原文返回前端。
            raise HTTPException(
                status_code=502,
                detail=(
                    "导入失败，请检查模型服务和 Milvus。"
                    "若失败发生在写入阶段，可能已有部分数据写入，"
                    "请先检查数据库再重试。"
                ),
            )

    finally:
        file.file.close()


@app.post("/api/query")
def query_document(body: QueryBody, request: Request):
    question = body.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="问题不能全部为空白",
        )

    try:
        return request.app.state.rag.query(question)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="查询失败，请检查模型服务和 Milvus",
        )