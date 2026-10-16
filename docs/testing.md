# 本地验证

提交改动前，可先从仓库根目录运行不需要密钥、Milvus 或外部模型服务的检查。

## 后端语法检查

```powershell
python -m compileall -q backend
```

该命令只编译 Python 源文件，不会启动 FastAPI、连接 Milvus 或调用向量模型。它适合发现语法错误，但不验证检索结果或服务集成。

## 前端生产构建

首次运行先安装锁文件指定的依赖，然后执行 Vite 构建：

```powershell
cd frontend
npm.cmd ci
npm.cmd run build
```

构建产物位于 `frontend/dist`，且已被 Git 忽略。`npm.cmd run build` 验证 Vue 单文件组件能够被生产配置打包；它不会启动后端，也不会发送 API 请求。

## 服务集成验证

上传和查询流程不是离线检查：后端启动时会创建向量模型客户端并连接已有的 `rag_test_v1` 集合，上传或查询还会调用配置的外部向量服务。只有在测试环境已配置自有密钥和兼容的 Milvus 时，才运行这类验证；不要在自动化检查中使用真实企业文档。

可使用仓库中的虚构 `backend/data.txt` 手动验证上传，再查询“年假需要提前多久申请？”。排序会受到所配置模型影响，因此应检查返回结构和来源文本，不要把某个具体融合分数作为固定断言。
