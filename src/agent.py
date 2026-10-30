"""Model-driven LangGraph loop. Mock mode is explicitly a workflow demonstration."""
import json
import os
from pathlib import Path
from typing import Any, TypedDict

from .retrieval import Corpus

DATA = Path(__file__).resolve().parents[1] / "data" / "documents.json"


class State(TypedDict, total=False):
    question: str
    history: list[dict]
    observations: list[dict]
    reads: list[dict]
    trace: list[dict]
    next_action: dict
    output: dict
    steps: int
    max_steps: int


SYSTEM = """你是企业知识问答 Agent。根据用户问题、历史和工具观察选择下一步动作。
每次只输出 JSON 对象。动作类型 search(query), read(doc_id,chunk_id), finish(answer,citations,missing,status)。
status 仅可为 answered、conflict、insufficient、tool_error。
必须先检索，再按 ID 读取支持答案的原文；引用只能是实际读取的原文，格式为 {doc_id,chunk_id,quote}，quote 必须是原文连续子串。
若检索到有关联但不同结论的制度，应读取双方，再检索适用范围/替代关系；仅凭日期较新不能推断优先级。仍无法判定则 status=conflict，列出双方并请用户确认适用制度。
若资料不足可换说法补充检索，再说明缺口；上下文追问结合历史补全检索词。工具失败可调整动作或结束，不可编造结果。
不要输出内部思维过程，finish 中只给简短依据和必要的资料缺口。"""


class LivePolicy:
    def __init__(self):
        from langchain_openai import ChatOpenAI
        self.model = ChatOpenAI(model=os.environ.get("MODEL_NAME", "gpt-4o-mini"), api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL") or None, temperature=0)

    def decide(self, state: State) -> dict:
        from langchain_core.messages import SystemMessage, HumanMessage
        payload = {k: state.get(k) for k in ("question", "history", "observations", "reads", "steps", "max_steps")}
        response = self.model.bind(response_format={"type": "json_object"}).invoke([SystemMessage(content=SYSTEM), HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
        return json.loads(response.content)


class MockPolicy:
    """Deterministic offline fixture for plumbing tests; never evidence of LLM decisions."""
    def decide(self, state: State) -> dict:
        observations, reads = state.get("observations", []), state.get("reads", [])
        question = state["question"]
        if not observations:
            history = " ".join(x.get("content", "") for x in state.get("history", [])[-2:])
            return {"type": "search", "query": history + " " + question}
        searches = [o for o in observations if o["action"] == "search"]
        hits = [h for o in searches if o.get("ok") for h in o["result"]]
        seen = {(r["doc_id"], r["chunk_id"]) for r in reads}
        if observations[-1].get("ok") is False:
            return {"type": "finish", "answer": "检索或读取工具失败，暂无法核实。", "citations": [], "missing": "工具不可用", "status": "tool_error"}
        if not hits:
            return {"type": "finish", "answer": "现有资料中未找到可靠答案。", "citations": [], "missing": "缺少相关制度或产品资料", "status": "insufficient"}
        relevant = [h for h in hits if h["doc_id"] in ({"D01", "D02"} if "期限" in question or "多久" in question else {hits[0]["doc_id"]})]
        for h in relevant:
            if (h["doc_id"], h["chunk_id"]) not in seen:
                return {"type": "read", "doc_id": h["doc_id"], "chunk_id": h["chunk_id"]}
        if len(relevant) > 1 and len(searches) == 1:
            return {"type": "search", "query": "差旅报销期限 适用范围 替代关系 优先级"}
        evidence = [r for r in reads if (r["doc_id"], r["chunk_id"]) in {(h["doc_id"], h["chunk_id"]) for h in relevant}]
        cites = [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "quote": r["text"]} for r in evidence]
        if len(relevant) > 1:
            return {"type": "finish", "answer": "资料存在冲突：D01 为30天，D02 为15天。请确认适用制度。", "citations": cites, "missing": "适用范围及制度优先级", "status": "conflict"}
        return {"type": "finish", "answer": evidence[0]["text"], "citations": cites, "missing": "", "status": "answered"}


def validate_finish(action: dict, reads: list[dict]) -> dict:
    permitted = {(r["doc_id"], r["chunk_id"]): r["text"] for r in reads}
    citations = action.get("citations", [])
    if not isinstance(citations, list) or any(not isinstance(c, dict) or not isinstance(c.get("quote"), str) or not c["quote"] or c["quote"] not in permitted.get((c.get("doc_id"), c.get("chunk_id")), "") for c in citations):
        return {"answer": "引用校验失败，无法可靠回答。", "citations": [], "missing": "引用缺少已读取原文依据", "status": "insufficient"}
    status = action.get("status")
    if status not in {"answered", "conflict", "insufficient", "tool_error"} or (status in {"answered", "conflict"} and not citations):
        return {"answer": "证据不足，无法可靠回答。", "citations": [], "missing": "有效引用", "status": "insufficient"}
    return {k: action.get(k, "") for k in ("answer", "citations", "missing", "status")}


def build_graph(corpus: Corpus, policy: Any):
    from langgraph.graph import START, END, StateGraph

    def decide(state: State) -> State:
        if state.get("steps", 0) >= state.get("max_steps", 8):
            return {"output": {"answer": "达到步骤上限，任务未完成。", "citations": [], "missing": "未能在限定步骤内完成核查", "status": "incomplete"}}
        try:
            action = policy.decide(state)
            if not isinstance(action, dict) or action.get("type") not in {"search", "read", "finish"}:
                raise ValueError("invalid model action")
        except Exception as exc:
            return {"output": {"answer": "模型决策失败，任务未完成。", "citations": [], "missing": type(exc).__name__, "status": "incomplete"}}
        trace = list(state.get("trace", []))
        trace.append({"step": state.get("steps", 0) + 1, "action": action.get("type"), "parameters": {k: v for k, v in action.items() if k in ("query", "doc_id", "chunk_id")}})
        if action["type"] == "finish":
            return {"next_action": action, "trace": trace, "steps": state.get("steps", 0) + 1, "output": validate_finish(action, state.get("reads", []))}
        return {"next_action": action, "trace": trace, "steps": state.get("steps", 0) + 1}

    def tool(state: State) -> State:
        action = state["next_action"]
        try:
            if action["type"] == "search":
                result = corpus.search(str(action["query"]))
            else:
                result = corpus.read(str(action["doc_id"]), str(action["chunk_id"]))
            observation = {"action": action["type"], "ok": True, "result": result}
        except Exception as exc:
            observation = {"action": action["type"], "ok": False, "error": type(exc).__name__ + ": " + str(exc)}
        trace = list(state["trace"])
        trace[-1] = {**trace[-1], "observation": observation}
        update: State = {"observations": list(state.get("observations", [])) + [observation], "trace": trace}
        if observation["ok"] and action["type"] == "read":
            update["reads"] = list(state.get("reads", [])) + [result]
        return update

    graph = StateGraph(State)
    graph.add_node("decide", decide)
    graph.add_node("tool", tool)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", lambda s: "done" if "output" in s else "tool", {"done": END, "tool": "tool"})
    graph.add_edge("tool", "decide")
    return graph.compile()


def ask(question: str, history: list[dict] | None = None, *, corpus: Corpus | None = None, policy: Any | None = None, max_steps: int = 8) -> dict:
    if not question.strip():
        raise ValueError("question must not be empty")
    corpus = corpus or Corpus(DATA)
    policy = policy or LivePolicy()
    state = build_graph(corpus, policy).invoke({"question": question, "history": history or [], "observations": [], "reads": [], "trace": [], "steps": 0, "max_steps": max_steps}, config={"recursion_limit": 2 * max_steps + 5})
    return {**state["output"], "trace": state["trace"]}
