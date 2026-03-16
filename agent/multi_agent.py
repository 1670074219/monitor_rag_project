import importlib
import os
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agent.tools import (
    get_video_by_time,
    get_videos_by_location,
    get_videos_by_semantic,
    track_person_globally,
)


class RouteDecision(BaseModel):
    agent_name: Literal["security", "general"] = Field(
        ...,
        description="路由目标，只能是 security 或 general",
    )


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "MiniMax-M2.5"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-sp-7ad1c3d44076499287a8ec02fee42615"),
        temperature=0.1,
    )


def build_general_agent(llm: ChatOpenAI):
    try:
        module = importlib.import_module("langchain_community.tools")
        DuckDuckGoSearchRun = getattr(module, "DuckDuckGoSearchRun")
    except Exception as e:
        raise ImportError(
            "未安装 langchain-community，请先安装: pip install langchain-community duckduckgo-search"
        ) from e

    return create_react_agent(
        model=llm,
        tools=[DuckDuckGoSearchRun()],
        prompt="你是监控室贴心的 AI 助理。负责天气、搜索和闲聊。回答轻松简短。",
    )


def build_security_agent(llm: ChatOpenAI):
    security_sop = """你是一个顶尖的安防监控轨迹分析专家。你的职责是根据用户的模糊自然语言指令，精准调用工具组合，完成监控追踪。
必须遵守以下【SOP】：
### Phase 1: 锁定目标 (Target Localization)
优先调用语义或地点工具。综合返回结果，最终锁定目标人物首次出现的唯一标识：video_id 和 person_index。
### Phase 2: 全局跨镜追踪 (Global Tracking)
拿到 target_info 后，如果用户要求看后续情况，必须调用 track_person_globally 工具，传入推算出的精确时间范围。
### Phase 3: 行为与意图研判 (Behavior Judgment)
根据 traces 列表中的 camera_id 和 points，判定其是否去过特定区域。
最后生成一份包含“起始确认”、“活动轨迹”和“关键研判”的《安保排查报告》。"""

    return create_react_agent(
        model=llm,
        tools=[
            get_video_by_time,
            get_videos_by_location,
            get_videos_by_semantic,
            track_person_globally,
        ],
        prompt=security_sop,
    )


def route_query(llm: ChatOpenAI, user_query: str) -> RouteDecision:
    router_prompt = """你是 Supervisor Router，只负责意图识别与任务分发，绝对不调用任何工具。
路由规则：
1. 涉及找人、查监控、轨迹追踪、安防排查、摄像头分析，路由到 security。
2. 涉及天气、网页搜索、新闻资讯、日常闲聊，路由到 general。
【极其重要】你必须且只能输出合法的纯 JSON 字符串！绝对不要在输出中包含```json标记，也不要输出任何多余的解释性文本。
返回格式严格为：{"agent_name": "security"} 或 {"agent_name": "general"}。"""

    try:
        router = llm.with_structured_output(RouteDecision)
        return router.invoke(
            [
                SystemMessage(content=router_prompt),
                HumanMessage(content=user_query),
            ]
        )
    except Exception:
        return RouteDecision(agent_name="general")


def run_once(user_query: str) -> dict:
    llm = build_llm()
    decision = route_query(llm, user_query)

    print(f"🚦 路由到: {decision.agent_name}")

    if decision.agent_name == "security":
        chosen_agent = build_security_agent(llm)
    else:
        chosen_agent = build_general_agent(llm)

    final_answer = ""
    for event in chosen_agent.stream(
        {"messages": [HumanMessage(content=user_query)]},
        stream_mode="values",
    ):
        message = event["messages"][-1]
        if message.type == "ai":
            if getattr(message, "tool_calls", None):
                call = message.tool_calls[0]
                print(f"🤖 Agent 思考: 调用工具 {call['name']}，参数 {call['args']}")
            else:
                final_answer = message.content
                print(f"\n🤖 Agent 最终回答:\n{final_answer}")
        elif message.type == "tool":
            print(f"📦 工具返回结果: {message.content}")

    return {
        "route": decision.agent_name,
        "answer": final_answer,
    }


if __name__ == "__main__":
    query = input("请输入你的问题: ").strip()
    if not query:
        print("请输入非空问题。")
        raise SystemExit(1)

    run_once(query)
