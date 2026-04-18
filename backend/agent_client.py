"""
backend/agent_client.py — Agent call abstraction

AGENT_MODE=local  → direct Python function call (for individual testing)
AGENT_MODE=http   → HTTP POST to agent containers (for full team deployment)

Toggle in .env:
    AGENT_MODE=local
    PLANNER_URL=http://planner:8001
    EXPLAINABILITY_URL=http://explainability:8002
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

AGENT_MODE         = os.getenv("AGENT_MODE", "local")
PLANNER_URL        = os.getenv("PLANNER_URL",        "http://planner:8001")
EXPLAINABILITY_URL = os.getenv("EXPLAINABILITY_URL", "http://explainability:8002")
RESEARCH_URL       = os.getenv("RESEARCH_URL",       "http://research:8003")

# Add project root to path so agents/ and tools/ are importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def call_research(state: dict) -> dict:
    """
    Call Agent2 (Research) — returns flights, hotels, attractions, restaurants.
    local mode : direct function call
    http  mode : POST {RESEARCH_URL}/run
    """
    if AGENT_MODE == "http":
        try:
            resp = requests.post(f"{RESEARCH_URL}/run", json=state, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": f"Research agent HTTP error: {e}"}
    else:
        from agents.specialists.research_agent import research_agent
        from agents.agent_tools import get_tools_for_agent
        tools = get_tools_for_agent("research_agent")
        return research_agent(state, tools=tools)


def call_planner(state: dict) -> dict:
    """
    Call Agent3 (Planner) — generates 3 itinerary options from research data.
    local mode : direct function call
    http  mode : POST {PLANNER_URL}/run
    """
    if AGENT_MODE == "http":
        try:
            resp = requests.post(f"{PLANNER_URL}/run", json=state, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": f"Planner agent HTTP error: {e}"}
    else:
        from agents.specialists.planner_agent import planner_agent
        from agents.agent_tools import get_tools_for_agent
        tools = get_tools_for_agent("planner_agent")
        return planner_agent(state, tools=tools)


def call_planner_revise(state: dict, critique: str, current_result: dict) -> dict:
    """
    Call Agent3 (Planner) in revision mode — reuses cached inventory, 1 LLM call only.
    Used by Agent4 debate loop.
    """
    if AGENT_MODE == "http":
        try:
            payload = {"state": state, "critique": critique, "current_result": current_result}
            resp = requests.post(f"{PLANNER_URL}/revise", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": f"Planner revise HTTP error: {e}"}
    else:
        from agents.specialists.planner_agent import revise_itinerary
        return revise_itinerary(state, critique, current_result)


# def run_debate_loop(state: dict) -> dict:
    

def call_explainability(state: dict) -> dict:
    """
    Call Agent6 (Explainability).
    local mode : direct function call
    http  mode : POST {EXPLAINABILITY_URL}/run
    """
    if AGENT_MODE == "http":
        try:
            resp = requests.post(f"{EXPLAINABILITY_URL}/run", json=state, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": f"Explainability agent HTTP error: {e}"}
    else:
        from agents.specialists.explainability_agent import explainability_agent
        return explainability_agent(state)
