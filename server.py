"""
Web 服务：把 main.py 里的 Agent 通过 SSE 接口暴露给网页
- GET /chat?query=xxx  → SSE 流，事件类型：
    thought      思考文案（增量文本）
    tool         工具调用（名称+参数）
    observation  工具返回结果
    answer_start 正文开始标记
    answer       最终回答（增量文本）
    done         结束
- GET /                → 返回 web/index.html 聊天页面

启动：uvicorn server:app --reload  然后浏览器打开 http://127.0.0.1:8000
"""
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

# 复用 main.py 里已创建好的 agent_executor（导入时不会执行交互循环）
load_dotenv()
from main import agent_executor

app = FastAPI(title="AI Agent Streaming Demo")

THOUGHT_PREFIX = "Thought:"
MARK_ACTION = "Action:"
MARK_ANSWER = "Final Answer:"


def trim_partial_mark(text: str) -> str:
    """若文本尾部是 Action: / Final Answer: 的前缀碎片，先扣下不发，
    避免思考区闪出 'Action'、'Final Answer' 这类半个标记"""
    for mark in (MARK_ACTION, MARK_ANSWER):
        for i in range(len(mark) - 1, 0, -1):
            if text.endswith(mark[:i]):
                return text[:-i]
    return text


def parse_action_input(buffer: str):
    """从 LLM 输出中解析 'Action Input: {...}' 的工具参数"""
    idx = buffer.rfind("Action Input:")
    if idx == -1:
        return None
    raw = buffer[idx + len("Action Input:"):].strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def sse(data: dict) -> str:
    """格式化一条 SSE 消息"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/chat")
async def chat(query: str):
    async def event_gen():
        # 状态机：与终端版 ReActStreamHandler 同样的"该显示什么"逻辑
        buffer = ""      # 当前这次 LLM 调用累计收到的原文
        shown_len = 0    # 思考部分已下发的字符数
        phase = "thought"  # thought -> answer
        llm_running = False

        async for event in agent_executor.astream_events(
            {"input": query}, version="v2"
        ):
            kind = event["event"]

            if kind == "on_chat_model_start":
                # 每轮思考开始：重置状态
                buffer, shown_len, phase = "", 0, "thought"
                llm_running = True

            elif kind == "on_chat_model_stream" and llm_running:
                chunk = event["data"]["chunk"]
                token = chunk.content if isinstance(chunk.content, str) else ""
                if not token:
                    continue
                buffer += token

                if MARK_ANSWER in buffer:
                    # 检测到最终答案标记 → 切换为正文模式
                    if phase != "answer":
                        phase = "answer"
                        shown_len = 0
                        yield sse({"type": "answer_start"})
                    visible = buffer.split(MARK_ANSWER, 1)[1]
                    yield sse({"type": "answer", "text": visible})
                else:
                    # 思考阶段：Action 之后的内容是给解析器的，不下发
                    visible = buffer.split(MARK_ACTION, 1)[0]
                    # 过滤开头的 "Thought:" 前缀（前缀本身逐 token 到达，
                    # 前缀未完整到达前先不下发，避免页面闪出 "Thought" 字样）
                    if buffer.startswith("Thought"):
                        visible = visible[len(THOUGHT_PREFIX):] if len(buffer) >= len(THOUGHT_PREFIX) else ""
                    visible = trim_partial_mark(visible)
                    new_text = visible[shown_len:]
                    if new_text:
                        yield sse({"type": "thought", "text": new_text})
                        shown_len = len(visible)

            elif kind == "on_chat_model_end":
                llm_running = False

            elif kind == "on_tool_start":
                yield sse({
                    "type": "tool",
                    "name": event["name"],
                    "input": parse_action_input(buffer),  # AgentExecutor 事件的 input 为空，从原文解析
                })

            elif kind == "on_tool_end":
                output = event["data"].get("output")
                text = getattr(output, "content", None) or str(output)
                yield sse({"type": "observation", "text": str(text)[:500]})

        yield sse({"type": "done"})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 托管前端页面：访问 http://127.0.0.1:8000 即可
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "web"), html=True), name="web")
