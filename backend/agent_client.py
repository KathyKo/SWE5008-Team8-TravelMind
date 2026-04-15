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
        from agents.specialists.research_agent_1 import research_agent_1
        from agents.agent_tools import get_tools_for_agent
        tools = get_tools_for_agent("research_agent")
        return research_agent_1(state, tools=tools)


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
        from agents.specialists.planner_agent_1 import planner_agent_1
        from agents.agent_tools import get_tools_for_agent
        tools = get_tools_for_agent("planner_agent")
        return planner_agent_1(state, tools=tools)


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
        from agents.specialists.planner_agent_1 import revise_itinerary_1
        return revise_itinerary_1(state, critique, current_result)


def run_debate_loop(state: dict) -> dict:
    """
    Full Agent3 ↔ Agent4 debate loop orchestrator.

    Flow:
      Round 0: Agent3 generates initial itineraries (Phase 1+2+3)
      Agent4 critiques → if acceptable: done
      Round 1-3: Agent3 revises (Phase 2+3 only) → Agent4 re-critiques → repeat

    Returns the final Agent3 result (with revised itineraries if debate improved them).
    """
    # Step 1: Initial generation from Agent3
    print("[Debate] Round 0 — Agent3 generating initial itineraries...")
    planner_result = call_planner(state)
    if "error" in planner_result:
        return planner_result

    current_result = planner_result

    # Step 2: Try to import Agent4 — if not available, skip debate
    try:
        from agent4_critic.main import run_agent4, _build_state_from_dict
        from agent4_critic.nodes.input_parser import _convert_agent3_itineraries
    except ImportError:
        print("[Debate] Agent4 not available — returning Agent3 result without debate")
        return current_result

    # Step 3: Build Agent4 input from Agent3 output
    # _build_state_from_dict expects Agent3's full output format:
    #   { user_profile: {origin, destination, ...}, itineraries: {"A": [...], ...}, ... }
    # Agent4 expects duration as an int — extract number from "7 days" / "7" / 7
    import re as _re
    _dur_raw = str(state.get("duration", "0"))
    _dur_match = _re.search(r"\d+", _dur_raw)
    _dur_int = int(_dur_match.group()) if _dur_match else 0

    agent4_input = {
        "user_profile": {
            "origin":             state.get("origin", ""),
            "destination":        state.get("destination", ""),
            "dates":              state.get("dates", ""),
            "duration":           _dur_int,
            "budget":             state.get("budget", ""),
            "preferences":        state.get("preferences", ""),
            "outbound_time_pref": state.get("outbound_time_pref", ""),
            "return_time_pref":   state.get("return_time_pref", ""),
        },
        "itineraries":            current_result.get("itineraries", {}),
        "option_meta":            current_result.get("option_meta", {}),
        "research":               current_result.get("research", {}),
        "tool_log":               current_result.get("tool_log", []),
        "chain_of_thought":       current_result.get("chain_of_thought", ""),
        "flight_options_outbound": current_result.get("flight_options_outbound", []),
        "flight_options_return":  current_result.get("flight_options_return", []),
        "hotel_options":          current_result.get("hotel_options", []),
    }
    profile, transport, itineraries, extra = _build_state_from_dict(agent4_input)

    # Step 4: Define callback — Agent4 calls this to get Agent3 revisions
    def agent3_callback(critique: str, round_num: int):
        nonlocal current_result
        print(f"[Debate] Round {round_num} — Agent3 revising based on critique...")
        revised = call_planner_revise(state, critique, current_result)
        if "error" in revised:
            print(f"[Debate] Agent3 revision failed: {revised['error']}")
            return None
        current_result = revised
        return _convert_agent3_itineraries(
            revised.get("itineraries", {}),
            revised.get("option_meta", {}),
        )

    # Step 5: Run Agent4 with callback (max 3 rounds handled internally by Agent4)
    print("[Debate] Starting Agent4 critique...")
    agent4_result = run_agent4(
        profile, transport, itineraries,
        agent3_callback=agent3_callback,
        extra_payload=extra,
        verbose=True,
    )

    # Step 6: Attach Agent4 verdict to the final result
    verdict = agent4_result.get("final_verdict")
    current_result["debate_verdict"] = {
        "accepted":         verdict.accepted if verdict else False,
        "final_round":      verdict.final_round if verdict else 0,
        "reason":           verdict.reason if verdict else "",
        "remaining_issues": [i.model_dump() for i in (verdict.remaining_issues if verdict else [])],
    }
    current_result["debate_history"] = [
        m.model_dump() for m in agent4_result.get("debate_history", [])
    ]

    print(f"[Debate] Complete — accepted: {current_result['debate_verdict']['accepted']}")
    return current_result


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
