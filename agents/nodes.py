import os
import requests
from typing import Dict
from .state import State

# Assume in the docker-compose network, each container can be accessed by the service name
# All Agents use the same hostname, only the port is different (starting from 8100)
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
    Send the current State to the specified Docker container and receive the updated State fragment.
    """
    url = AGENT_URLS.get(agent_name)
    if not url:
        raise ValueError(f"Unknown agent: {agent_name}")

    print(f"--- Calling remote container: [{agent_name}] ---")

    try:
        response = requests.post(url, json={"state": state}, timeout=60.0)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"!!! Calling container [{agent_name}] failed: {e}")
        return {
            "error_message": f"{agent_name} service unavailable: {str(e)}",
        }


# ============================================================
#  Thin Node Wrappers — Thin wrapper for each agent node
# ============================================================

def input_guard_node(state: State) -> Dict:
    return call_remote_agent("input_guard", state)


def intent_profile_node(state: State) -> Dict:
    return call_remote_agent("intent_profile", state)


def search_node(state: State) -> Dict:
    return call_remote_agent("search", state)


def planner_node(state: State) -> Dict:
    """
    Call the remote planner container.
    Whether it's the first planning or re-planning after debate failure,
    reset is_valid=None to let orchestrator re-route to debate for review.
    """
    result = call_remote_agent("planner", state)
    result["is_valid"] = None
    return result


def debate_node(state: State) -> Dict:
    """
    Call the remote debate container (internal debate between debate LLM and planner LLM).
    Ensure debate_count increases, so orchestrator can determine if the maximum number of loops has been reached.
    """
    result = call_remote_agent("debate", state)
    result.setdefault("debate_count", state.get("debate_count", 0) + 1)
    return result


def replanner_node(state: State) -> Dict:
    """
    Call the remote replanner container (only triggered by user feedback).
    Reset the state of all downstream stages, so the workflow can start from debate to review the new plan.
    """
    result = call_remote_agent("replanner", state)
    result["user_feedback"] = None
    result["is_valid"] = None
    result["explanation"] = None
    result["explain_data"] = None
    result["output_guard_decision"] = None
    result.setdefault("replan_attempts", state.get("replan_attempts", 0) + 1)
    return result


def explain_node(state: State) -> Dict:
    return call_remote_agent("explain", state)


def output_guard_node(state: State) -> Dict:
    return call_remote_agent("output_guard", state)


# ============================================================
#  Orchestrator — Central routing brain (pure memory judgment, no remote call)
# ============================================================
#
#  Normal workflow:
#    START → input_guard → orchestrator
#      → [threat_blocked?] ──True──→ END
#      → intent_profile → orchestrator
#      → search → orchestrator
#      → planner → orchestrator          ←──┐
#      → debate → orchestrator                │
#        ├─ is_valid=True ──────→ explain      │
#        ├─ is_valid=False & count < max ──────┘  (return to planner with critique to regenerate)
#        └─ is_valid=False & count ≥ max → explain (强制推进)
#      → explain → orchestrator
#      → output_guard → END
#
#  User feedback path(replanner):
#    user_feedback Set as: → replanner → orchestrator
#      → debate → … (re-run debate→explain→output_guard)
# ============================================================

def orchestrator_node(state: State) -> Dict:
    """
    Central orchestrator: determine next_node based on the fields written by each agent in the state.
    Return {"next_node": "<agent_name>"} to write into state,
    and read by orchestrator_routing to pass to LangGraph conditional routing.
    Read by orchestrator_routing and passed to LangGraph conditional routing.
    """

    # ── 0. Global error intercept ──────────────────────────────────
    if state.get("error_message"):
        print(f"[Orchestrator] error intercepted: {state['error_message']}")
        return {"next_node": "END"}

    # ── 1. Input Guard (security gate) ──────────────────────────
    if state.get("threat_blocked") is True:
        print("[Orchestrator] input blocked by input_guard, workflow terminated.")
        return {"next_node": "END"}

    # ── 2. User feedback → Replanner (priority higher than normal workflow) ───
    #    User submitted a modification feedback at H.I.T. Checkpoint
    if state.get("user_feedback"):
        print("[Orchestrator] → replanner (user submitted a modification feedback)")
        return {"next_node": "replanner"}

    # ── 3. Intent Profile (user intent profile) ──────────────────
    if not state.get("intent_profile_output"):
        print("[Orchestrator] → intent_profile (extract user intent)")
        return {"next_node": "intent_profile"}

    # ── 4. Research / Search (information retrieval) ───────────────────
    if not state.get("search_results") and not state.get("research"):
        print("[Orchestrator] → search (information retrieval)")
        return {"next_node": "search"}

    # ── 5. Planner (trip planning) ────────────────────────────
    #    First planning: itineraries does not exist
    #    Re-planning after debate failure: planner_node will reset is_valid=None when returning
    if not state.get("itineraries") and not state.get("final_itineraries"):
        print("[Orchestrator] → planner (generate itinerary)")
        return {"next_node": "planner"}

    # ── 6. Debate ↔ Planner loop ─────────────────────────
    is_valid = state.get("is_valid")        # None / True / False
    debate_count = state.get("debate_count", 0)
    max_debate_rounds = 3

    # 6a. Not reviewed yet or planner/replanner just generated a new plan → debate
    if is_valid is None:
        print("[Orchestrator] → debate (trip quality review)")
        return {"next_node": "debate"}

    # 6b. debate failed and not reached the maximum number of rounds → return with critique to planner for improvement
    if is_valid is False and debate_count < max_debate_rounds:
        print(f"[Orchestrator] → planner (debate failed, round {debate_count}/{max_debate_rounds} improvement)")
        return {"next_node": "planner"}

    # 6c. debate passed (is_valid=True) or reached the maximum number of rounds → force push

    # ── 7. Explainability (explainability) ─────────────────────
    if not state.get("explanation") and not state.get("explain_data"):
        print("[Orchestrator] → explain (generate decision explanation)")
        return {"next_node": "explain"}

    # ── 8. Output Guard (output security check) ───────────────────
    if not state.get("output_guard_decision"):
        print("[Orchestrator] → output_guard (output security check)")
        return {"next_node": "output_guard"}

    # ── 9. All workflow completed ───────────────────────────────────
    print("[Orchestrator] workflow completed.")
    return {"next_node": "END"}


def orchestrator_routing(state: State) -> str:
    """
    LangGraph add_conditional_edges routing function.
    Must return str (node name), not Dict.
    Only read the next_node field written by orchestrator_node.
    """
    return state.get("next_node", "END")
