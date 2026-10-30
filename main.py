import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

# --------------------------
# 1. 加载环境配置
# --------------------------
load_dotenv()  # 自动读取同目录下.env文件的配置

# 初始化大模型
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.3,  # 数值越低输出越严谨、越少创意
    streaming=True,  # 开启 token 级流式输出（回调在下方注入）
)

# --------------------------
# 2. 定义Agent可用工具
# --------------------------
@tool
def calculator(expression: str) -> str:
    """
    执行数学表达式计算，支持加减乘除、括号、幂运算等。
    参数 expression: 字符串格式的数学表达式，例如 "1234 * 567 + 890"
    """
    try:
        # 仅演示使用，生产环境请替换为安全计算库
        result = eval(expression)
        return f"计算成功，结果：{result}"
    except Exception as e:
        return f"计算失败，错误信息：{str(e)}"

# 网页搜索工具：支持查询实时信息、最新资讯
web_search = TavilySearchResults(
    max_results=3,
    description="调用搜索引擎查询实时信息、最新资讯、行业动态、未知知识。仅当问题涉及实时、最新或模型不知道的信息时使用，输入为搜索关键词。"
)


# 所有工具打包成列表，后续加新工具只需在这里添加
tools = [calculator, web_search]

# --------------------------
# 3. ReAct 提示词模板
# --------------------------
react_prompt = PromptTemplate.from_template("""
你是一个擅长分步推理的智能助手，可以调用工具解决复杂问题。

【可用工具说明】
{tools}

【工具名称列表】
{tool_names}

【执行规则】
1. 先思考问题是否需要调用工具，不需要就直接给出最终答案
2. 需要调用工具时，严格按照下面格式输出，不要有多余内容：
Thought: 你对当前问题的推理分析，说明为什么要调用工具
Action: 要调用的工具名称，必须是上面列表里的准确名称
Action Input: {{"参数名": "参数值"}}，严格JSON格式
Observation: 等待工具返回结果
3. 拿到工具返回的观察结果后，继续推理，判断是否还需要更多工具调用
4. 确认可以给出答案时，按下面格式输出：
Thought: 最终的推理总结
Final Answer: 你的完整最终回答

【用户问题】
{input}

【历史执行记录】
{agent_scratchpad}
""")

# --------------------------
# 4. 创建Agent与执行器
# --------------------------
# 创建ReAct模式Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

# 创建执行器，管理Agent的运行循环
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,  # 流式输出由回调负责，verbose 会重复打印
    max_iterations=5,  # 最大执行步数，防止死循环
    handle_parsing_errors=True  # 自动处理格式解析错误
)


# --------------------------
# 5. token 级流式回调：像聊天界面一样"思考实时滚屏 + 正文逐字输出"
# --------------------------
# ANSI 颜色码：思考用灰色（像聊天界面的折叠思考区），工具/观察用彩色区分
DIM = "\033[90m"     # 灰色 → 思考文案
CYAN = "\033[96m"    # 青色 → 工具调用
BLUE = "\033[94m"    # 蓝色 → 观察结果
RESET = "\033[0m"    # 恢复默认颜色


def enable_ansi_on_windows():
    """让 Windows 老版控制台也支持 ANSI 转义码"""
    if os.name == "nt":
        try:
            import colorama
            colorama.just_fix_windows_console()
        except ImportError:
            os.system("")  # 无 colorama 时的兜底：触发控制台启用转义序列


class ReActStreamHandler(BaseCallbackHandler):
    """
    监听大模型的每一个 token：
    - 思考阶段（Thought 部分）：一边生成一边实时打印
    - 出现 "Action:" 后停止显示（那是给框架解析用的机械格式）
    - 出现 "Final Answer:" 后切换到正文模式，逐字流式输出最终回答
    """
    MARK_ACTION = "Action:"
    MARK_ANSWER = "Final Answer:"

    def __init__(self):
        self._reset()

    def _reset(self):
        self.buffer = ""          # 当前这次 LLM 调用累计收到的原文
        self.shown_len = 0        # 已经打印出去的字符数
        self.answer_started = False  # 是否已进入正文模式

    def on_llm_start(self, *args, **kwargs):
        # 每轮新的 LLM 调用（一次思考）开始时重置状态，并让思考文案显示为灰色
        self._reset()
        print(DIM, end="", flush=True)

    def on_llm_new_token(self, token: str, **kwargs):
        self.buffer += token

        if self.MARK_ANSWER in self.buffer:
            # 检测到最终答案标记 → 退出灰色，切换为正文流式
            if not self.answer_started:
                self.answer_started = True
                print(RESET + "\n✅ 最终回复：", flush=True)
                self.shown_len = 0
            visible = self.buffer.split(self.MARK_ANSWER, 1)[1]
        else:
            # 思考阶段：只显示 Thought 部分，Action 及之后的内容是给解析器看的
            visible = self.buffer.split(self.MARK_ACTION, 1)[0]
            if visible.startswith("Thought:"):
                visible = visible[len("Thought:"):]

        new_text = visible[self.shown_len:]
        if new_text:
            print(new_text, end="", flush=True)
            self.shown_len = len(visible)


stream_handler = ReActStreamHandler()
llm.callbacks = [stream_handler]  # 注入到模型，所有调用都会经过它


def chat(user_input: str):
    """
    stream() 负责按轮次产出结构（工具调用/观察结果），
    token 级的思考文案和最终答案由 ReActStreamHandler 实时打印。
    """
    for chunk in agent_executor.stream({"input": user_input}):
        if "actions" in chunk:
            for action in chunk["actions"]:
                print(f"\n{CYAN}🔧 调用工具：{action.tool} | 参数：{action.tool_input}{RESET}", flush=True)
        elif "steps" in chunk:
            for step in chunk["steps"]:
                obs = str(step.observation)
                preview = obs[:200] + ("..." if len(obs) > 200 else "")
                print(f"{BLUE}👁 观察结果：{preview}{RESET}", flush=True)
        elif "output" in chunk:
            print(RESET)  # 正文已由回调实时输出，这里恢复颜色并换行收尾


# --------------------------
# 6. 交互入口
# --------------------------
if __name__ == "__main__":
    enable_ansi_on_windows()  # Windows 控制台启用 ANSI 颜色
    print("=" * 50)
    print("  AI Agent 已启动 | 输入 exit 退出程序")
    print("=" * 50)
    
    while True:
        user_input = input("\n请输入你的问题：")
        if user_input.strip().lower() in ["exit", "退出", "q"]:
            print("Agent已退出，再见")
            break
        
        # 运行Agent（流式）
        try:
            chat(user_input)
        except Exception as e:
            print(f"\n❌ 执行出错：{str(e)}")
