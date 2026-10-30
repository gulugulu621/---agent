# 企业知识问答 Agent

合成企业制度问答原型。LangGraph 编排模型选动作、工具执行、观察反馈的循环；本地字符及双字词 BM25 检索，按 ID 读取原文。FastAPI 和 CLI 共用同一 Agent。资料仅为合成示例，不代表真实公司制度。

## 安装运行

Python 3.11+：`python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`

真实模型：设置 `OPENAI_API_KEY`，可选 `MODEL_NAME`、`OPENAI_BASE_URL`（须支持 OpenAI 兼容的 JSON 输出）。`python -m src.cli '差旅报销期限多久'`

离线流程演示：`python -m src.cli '差旅报销期限多久' --mock`。Mock 只验证工具和状态流转，不可作为模型动态决策效果的证明。

API：`uvicorn src.api:app --reload`，然后 `curl -s http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"question":"报销需要哪些材料","history":[]}'`。离线 API 可设置 `MOCK_MODEL=1`。

测试：`python -m unittest discover -s tests -v`。输出包含 `answer`、`citations`（文档 ID、片段 ID、原文）、`missing`、`status` 和 `trace`。状态：answered / conflict / insufficient / tool_error / incomplete。

部署前请用真实模型运行并保存逐场景输出；日志中的模型动作必须来自真实调用。每次改动 `data/documents.json` 后，新进程会重建索引。
