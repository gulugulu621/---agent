# 方案

本地合成 JSON 按 doc_id、version_date、chunk_id 管理。词法 BM25 检索返回候选摘要；read 按 ID 返回原文。LangGraph 的 decide 节点用模型输出动作，tool 节点执行并将观察反馈给模型，直到 finish 或步骤上限。回答引用必须在已读取片段中逐字匹配。FastAPI 提供接口，CLI 便于复现。

选用 LangGraph 的 StateGraph、条件边和编译执行；模型通过 langchain-openai 调用。选择本地索引是为了让少量合成文档易替换、易复现；词法检索会漏掉语义相近但字面完全不同的问题。模型负责补充检索和冲突判断，引用校验提供确定性约束；原文包含引文却不一定语义支持回答，仍需人工/测试核查。
