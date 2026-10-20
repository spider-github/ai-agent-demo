import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

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

# 所有工具打包成列表，后续加新工具只需在这里添加
tools = [calculator]

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
    verbose=True,  # 开启后会打印完整的思考-行动-观察过程，方便调试
    max_iterations=5,  # 最大执行步数，防止死循环
    handle_parsing_errors=True  # 自动处理格式解析错误
)

# --------------------------
# 5. 交互入口
# --------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("  AI Agent 已启动 | 输入 exit 退出程序")
    print("=" * 50)
    
    while True:
        user_input = input("\n请输入你的问题：")
        if user_input.strip().lower() in ["exit", "退出", "q"]:
            print("Agent已退出，再见")
            break
        
        # 运行Agent
        try:
            result = agent_executor.invoke({"input": user_input})
            print("\n✅ 最终回复：")
            print(result["output"])
        except Exception as e:
            print(f"\n❌ 执行出错：{str(e)}")
