import os
import requests
from typing import Dict
from .state import State

# 假设在 docker-compose 网络中，每个容器可以通过服务名访问
# 所有 Agent 使用同一个 hostname，仅端口不同（8100 起递增）
AGENT_HOST = os.getenv("AGENT_HOST", "localhost")
AGENT_SCHEME = os.getenv("AGENT_SCHEME", "http")

AGENT_URLS = {
    "input_guard": os.getenv("AGENT_INPUT_GUARD_URL", f"{AGENT_SCHEME}://{AGENT_HOST}:8100/api/invoke/input_guard"),
    "intent_profile": os.getenv("AGENT_INTENT_PROFILE_URL", f"{AGENT_SCHEME}://{AGENT_HOST}:8101/api/invoke/intent_profile"),
    "search": os.getenv("AGENT_SEARCH_URL", f"{AGENT_SCHEME}://{AGENT_HOST}:8102/api/invoke/search"),
    "planner": os.getenv("AGENT_PLANNER_URL", f"{AGENT_SCHEME}://{AGENT_HOST}:8103/api/invoke/planner"),
    "debate": os.getenv("AGENT_DEBATE_URL", f"{AGENT_SCHEME}://{AGENT_HOST}:8104/api/invoke/debate"),
    "explain": os.getenv("AGENT_EXPLAIN_URL", f"{AGENT_SCHEME}://{AGENT_HOST}:8105/api/invoke/explain"),
    "output_guard": os.getenv("AGENT_OUTPUT_GUARD_URL", f"{AGENT_SCHEME}://{AGENT_HOST}:8106/api/invoke/output_guard"),
    "replanner": os.getenv("AGENT_REPLANNER_URL", f"{AGENT_SCHEME}://{AGENT_HOST}:8107/api/invoke/replanner"),
}

def call_remote_agent(agent_name: str, state: State) -> Dict:
    """
    【通用底层函数】向指定的 Docker 容器发送当前 State，并接收更新后的 State 片段。
    """
    url = AGENT_URLS.get(agent_name)
    if not url:
        raise ValueError(f"Unknown agent: {agent_name}")

    print(f"--- 正在调用远程容器: [{agent_name}] ---")
    
    try:
        # 将整个 state 作为 JSON 发送给对应的容器
        # 设置 timeout 防止某个 Agent 卡死导致整个 Graph 阻塞
        response = requests.post(url, json={"state": state}, timeout=60.0)
        
        # 如果返回非 200 状态码，抛出异常
        response.raise_for_status()
        
        # 容器应该返回一个字典，包含需要更新的 state 字段
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"!!! 调用容器 [{agent_name}] 失败: {e}")
        # 如果调用失败，返回一个错误信息，并将控制权交还给 orchestrator
        return {
            "error_message": f"{agent_name} 服务不可用: {str(e)}",
            "next_node": "orchestrator" # 强制回到中控进行错误处理
        }

# ==========================================
# 极其简洁的节点封装 (Thin Wrappers)
# ==========================================

def input_guard_node(state: State) -> Dict:
    return call_remote_agent("input_guard", state)

def intent_profile_node(state: State) -> Dict:
    return call_remote_agent("intent_profile", state)

def search_node(state: State) -> Dict:
    return call_remote_agent("search", state)

def planner_node(state: State) -> Dict:
    return call_remote_agent("planner", state)

def debate_node(state: State) -> Dict:
    return call_remote_agent("debate", state)

def explain_node(state: State) -> Dict:
    return call_remote_agent("explain", state)

def output_guard_node(state: State) -> Dict:
    return call_remote_agent("output_guard", state)

def replanner_node(state: State) -> Dict:
    return call_remote_agent("replanner", state)
    # 路由判定函数
def orchestrator_routing(state: State):
    # 获取 state 中记录的下一个节点，如果没有，安全起见默认结束
    return state.get("next_node", "END")

# ==========================================
# Orchestrator 核心调度器 (本地执行，无需调远端)
# ==========================================

def orchestrator_node(state: State) -> Dict:
    """
    这是你的中央大脑，它不请求外部 API，只在内存中根据当前状态进行极速路由判断。
    """
    # 0. 错误捕获：如果某个远程 Agent 挂了
    if state.get("error_message"):
        print(f"Orchestrator 拦截到错误: {state['error_message']}")
        return {"next_node": "END"} # 或路由到专门的 error_handler 节点

    # 1. 刚开始跑，没有用户画像，去 Agent 1
    if not state.get("user_profile"):
        return {"next_node": "intent_profile"}
        
    # 2. 有了画像，但没检索数据，去 Agent 2
    if state.get("user_profile") and not state.get("search_results"):
        return {"next_node": "search"}
        
    # 3. 检索完了，还没做计划，去 Agent 3
    if state.get("search_results") and not state.get("itinerary_options"):
        return {"next_node": "planner"}
        
    # 4. 做完计划了，去辩论 Agent 4
    if state.get("itinerary_options") and not state.get("is_valid") and state.get("debate_count", 0) < 3:
        return {"next_node": "debate"}
        
    # 5. 辩论通过（或满3次），去解释 Agent 6
    if (state.get("is_valid") or state.get("debate_count", 0) >= 3) and not state.get("explanation"):
        return {"next_node": "explain"}
        
    # 6. 生成了解释，去出口安全门禁 Agent 7
    if state.get("explanation") and not state.get("final_output"):
        return {"next_node": "output_guard"}
        
    # 7. 全流程完毕
    return {"next_node": "END"}