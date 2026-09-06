# Enterprise RAG · 企业知识库检索 Demo

Vue 3 + Element Plus 前端，FastAPI 后端，阿里云 DashScope 向量模型和本机 Milvus。支持 TXT 上传、切分、向量入库，以及 BM25 + 向量召回后的 RRF 融合排序。

**当前是检索 Demo，不包含大模型生成回答。** 前端只展示排名第一的原文；RRF 和向量分数不是正确率或置信度。

## 目录

```text
enterprise-rag/
├── backend/
│   ├── app.py                 # FastAPI 上传与查询
│   ├── embedding_model.py     # DashScope 向量模型
│   ├── hybrid_search.py       # BM25、向量检索、RRF
│   ├── init_collection.py     # 创建空集合，不调用模型
│   ├── milvus_demo.py         # 原始学习入库脚本（有覆盖风险）
│   ├── text.py                # 模型调用练习
│   ├── data.txt               # 虚构员工手册样例
│   ├── .env.example
│   └── requirements.txt
└── frontend/
    ├── src/App.vue            # 页面与交互
    ├── src/api.js             # 请求与错误处理
    ├── vite.config.js         # 开发反向代理
    └── package-lock.json
```

## 本地启动（Windows PowerShell）

准备 Python 3.11+（原项目环境为 3.13）、Node.js 22.12+ 或 24，以及已经启动的 Milvus。当前依赖沿用原项目 PyMilvus 3.0.x；请使用与客户端兼容的 Milvus 服务端。本仓库不包含 Docker 数据卷，也不自动安装或启动数据库。

### 1. 配置后端

在仓库根目录执行：

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `backend/.env` 填入自己的密钥，确认模型、地域 URL、维度与实际开通服务一致。不要提交 `.env`。已有集合的模型和维度必须保持一致；更换模型需重新构建索引。

### 2. 初始化集合并启动 API

Milvus 必须先在 `127.0.0.1:19530` 可连接：

```powershell
.\.venv\Scripts\python.exe init_collection.py
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

初始化脚本只创建不存在的 `rag_test_v1` 空集合；已有集合只检查向量维度，不删除或重写数据。API 文档：http://127.0.0.1:8000/docs 。首次启动可使用空集合，再通过前端上传样例。

### 3. 启动前端

另开终端，从仓库根目录执行：

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

访问 http://127.0.0.1:5173 。使用 `npm.cmd` 可避免 PowerShell 拦截 `npm.ps1`，无需修改系统执行策略。端口占用时先检查是否已有服务，勿重复启动。

前端 `/api/*` 由 Vite 原样代理至 `http://127.0.0.1:8000/api/*`。更换后端端口时修改 `frontend/vite.config.js` 的 target。

## 试用与接口

1. 上传 `backend/data.txt`（虚构资料），等待切分、向量化、写入及 BM25 刷新成功。
2. 查询“年假需要提前多久申请？”，应能检索到相关原文；具体排序取决于模型及语料。

| 接口 | 请求 | 返回 |
| --- | --- | --- |
| POST /api/documents | multipart/form-data，字段 file | filename、chunks、total_chunks、message |
| POST /api/query | JSON：{"question":"年假如何申请？"} | question、answer:null、sources |

上传仅支持 UTF-8 TXT，最大 200KB、200 个文本块；同名文档返回 409，不支持覆盖。问题最多 200 字符。后端返回融合后的多条 sources，前端只显示 sources[0]。失败不自动重试：若写入中途失败，应先检查数据库是否已有部分数据。

**勿对已有数据直接运行旧版 `milvus_demo.py`**：它从 ID=1 开始 upsert，可能覆盖相同 ID。该脚本仅作为学习材料保留；正常流程用 `init_collection.py` + 网页上传。

## 安全与边界

- 文档和问题会发送到配置的外部向量服务，可能产生费用；请勿上传未授权敏感资料。
- 未实现登录鉴权、用户隔离、生产级并发写入、删除更新、任务队列或限流。不要直接暴露到公网。
- 后端仅用单进程锁，并在内存维护 BM25；请单 worker 运行，不使用多 worker。
- 改动向量模型时，即使维度相同，也不能直接混用旧向量。
- 默认 `.gitignore` 排除密钥、依赖、虚拟环境、构建产物和数据卷。不要使用 `git add -f` 强行提交敏感文件。
- 若密钥曾出现在截图、日志或历史提交中，应在服务商控制台撤销并换新；删除文件不等于消除已泄露的密钥。
- 本包不含数据库记录、原项目 .env、Git 历史或实际密钥，仅包含源码和虚构样例。

## 构建与 GitHub

```powershell
cd frontend
npm.cmd run build
```

产物为 frontend/dist。Vite 代理只在开发时生效，生产部署需另配 Nginx 等反向代理，并补齐上述安全能力。Element Plus 全量引入可能产生包体积警告。

在 GitHub 创建空仓库后，在本项目根目录手动执行（将占位地址换成自己的仓库）：

```powershell
git init
git add .
git status
git diff --cached --stat
# 确认没有 .env、密钥、真实文档后再提交
git commit -m "Initial RAG demo"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/YOUR_REPO.git
git push -u origin main
```

尚未指定开源许可证，因此本包没有自动添加 LICENSE；公开前请自行决定许可，并核对引用来源与第三方许可。
