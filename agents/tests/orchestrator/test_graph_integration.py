"""
Orchestrator GRAPH integration tests.

Build the real LangGraph graph and invoke it end-to-end with mocked
remote agents. Verifies the full START → ... → END routing chain.
"""

from unittest.mock import patch

import pytest


# ── Helper: stateful mock dispatcher ────────────────────────

class MockAgentDispatcher:
    """
    Callable mock for call_remote_agent.
    Returns pre-configured responses per agent, in order.
    """

    def __init__(self, responses: dict[str, list[dict]]):
        self.responses = responses
        self.call_counts: dict[str, int] = {}
        self.call_order: list[str] = []

    def __call__(self, agent_name: str, state: dict) -> dict:
        count = self.call_counts.get(agent_name, 0)
        self.call_counts[agent_name] = count + 1
        self.call_order.append(agent_name)

        agent_responses = self.responses.get(agent_name, [{}])
        idx = min(count, len(agent_responses) - 1)
        return agent_responses[idx].copy()


def _initial_state():
    return {
        "messages": [{"role": "user", "content": "Plan a 5-day trip to Tokyo from Singapore, budget SGD 3000"}],
        "origin": None,
        "destination": None,
        "budget": None,
        "dates": None,
        "preferences": None,
        "duration": None,
        "user_profile": None,
        "travelers": None,
        "outbound_time_pref": None,
        "return_time_pref": None,
        "session_id": None,
        "intent_profile_output": None,
        "is_complete": False,
        "threat_blocked": None,
        "search_results": None,
        "research": None,
        "itineraries": None,
        "final_itineraries": None,
        "is_valid": None,
        "debate_count": 0,
        "critique": None,
        "approval_threshold": 75.0,
        "user_feedback": None,
        "replan_attempts": 0,
        "max_replan_attempts": 2,
        "explanation": None,
        "explain_data": None,
        "output_guard_decision": None,
        "next_node": None,
        "error_message": None,
        "final_output": None,
    }


# ── Graph builder (lazy import to avoid top-level langgraph load) ──

def _build_and_invoke(dispatcher, **state_overrides):
    """Build graph, patch remote calls, invoke, return (result, dispatcher)."""
    with patch("agents.nodes.call_remote_agent", side_effect=dispatcher):
        from agents.graph import build_travel_graph
        graph = build_travel_graph()
        state = _initial_state()
        state.update(state_overrides)
        result = graph.invoke(state, config={"recursion_limit": 80})
    return result


# ============================================================
#  Test 1: Happy Path — 全流程顺利通过
# ============================================================

def test_graph_happy_path():
    """
    START → input_guard → intent_profile → search → planner → debate(pass) → explain → output_guard → END
    """
    d = MockAgentDispatcher({
        "input_guard": [{"threat_blocked": False, "sanitised_input": "Plan trip", "input_guard_decision": "pass"}],
        "intent_profile": [{"intent_profile_output": {"destination": "Tokyo"}, "origin": "SG", "destination": "Tokyo", "dates": "2026-06-01 to 2026-06-05", "budget": "SGD 3000", "duration": "5 days", "is_complete": True}],
        "search": [{"search_results": {"hotels": [{"name": "A"}]}, "research": {"summary": "done"}}],
        "planner": [{"itineraries": [{"name": "Plan A"}], "final_itineraries": [{"name": "Plan A"}]}],
        "debate": [{"is_valid": True, "composite_score": 85.0, "debate_output": {"verdict": "approved"}}],
        "explain": [{"explanation": {"overall": "Great"}, "explain_data": {"ok": True}}],
        "output_guard": [{"output_flagged": False, "output_guard_decision": "pass"}],
    })

    result = _build_and_invoke(d)

    assert d.call_order == [
        "input_guard", "intent_profile", "search",
        "planner", "debate", "explain", "output_guard",
    ], f"Unexpected routing: {d.call_order}"
    assert result["output_guard_decision"] == "pass"
    assert result["is_valid"] is True


# ============================================================
#  Test 2: Threat Blocked — input_guard 拦截
# ============================================================

def test_graph_threat_blocked():
    d = MockAgentDispatcher({
        "input_guard": [{"threat_blocked": True, "threat_type": "injection", "input_guard_decision": "blocked"}],
    })

    result = _build_and_invoke(d)

    assert d.call_order == ["input_guard"]
    assert result["threat_blocked"] is True


# ============================================================
#  Test 3: Debate Fail → Planner Retry → Debate Pass
# ============================================================

def test_graph_debate_retry():
    d = MockAgentDispatcher({
        "input_guard": [{"threat_blocked": False, "input_guard_decision": "pass"}],
        "intent_profile": [{"intent_profile_output": {"dest": "Tokyo"}, "destination": "Tokyo", "origin": "SG", "dates": "2026-06-01", "budget": "3000", "duration": "5d"}],
        "search": [{"search_results": {"data": True}, "research": {"ok": True}}],
        "planner": [
            {"itineraries": [{"v": 1}], "final_itineraries": [{"v": 1}]},
            {"itineraries": [{"v": 2}], "final_itineraries": [{"v": 2}]},
        ],
        "debate": [
            {"is_valid": False, "composite_score": 50.0, "critique": "Budget allocation poor"},
            {"is_valid": True, "composite_score": 82.0, "critique": None},
        ],
        "explain": [{"explanation": {"ok": True}, "explain_data": {"v": 2}}],
        "output_guard": [{"output_flagged": False, "output_guard_decision": "pass"}],
    })

    result = _build_and_invoke(d)

    assert d.call_order == [
        "input_guard", "intent_profile", "search",
        "planner", "debate",   # round 1 — FAIL
        "planner", "debate",   # round 2 — PASS
        "explain", "output_guard",
    ], f"Unexpected routing: {d.call_order}"
    assert d.call_counts["planner"] == 2
    assert d.call_counts["debate"] == 2


# ============================================================
#  Test 4: Debate 连续失败达到上限 → 强制推进
# ============================================================

def test_graph_debate_max_rounds():
    d = MockAgentDispatcher({
        "input_guard": [{"threat_blocked": False, "input_guard_decision": "pass"}],
        "intent_profile": [{"intent_profile_output": {"d": "T"}, "destination": "T", "origin": "S", "dates": "2026", "budget": "3k", "duration": "5d"}],
        "search": [{"search_results": {"x": 1}, "research": {"x": 1}}],
        "planner": [
            {"itineraries": [{"v": i}], "final_itineraries": [{"v": i}]}
            for i in range(1, 5)
        ],
        "debate": [
            {"is_valid": False, "composite_score": 40.0, "critique": f"Issue {i}"}
            for i in range(1, 5)
        ],
        "explain": [{"explanation": {"forced": True}, "explain_data": {"forced": True}}],
        "output_guard": [{"output_flagged": False, "output_guard_decision": "pass"}],
    })

    result = _build_and_invoke(d)

    assert d.call_counts["debate"] == 3
    assert d.call_counts["planner"] == 3
    assert result["is_valid"] is False
    assert result["output_guard_decision"] == "pass"


# ============================================================
#  Test 5: Agent 服务不可用 → error → END
# ============================================================

def test_graph_agent_error():
    def failing(agent_name, state):
        if agent_name == "input_guard":
            return {"threat_blocked": False, "input_guard_decision": "pass"}
        if agent_name == "intent_profile":
            return {"error_message": "intent_profile 服务不可用: Connection refused"}
        return {}

    with patch("agents.nodes.call_remote_agent", side_effect=failing):
        from agents.graph import build_travel_graph
        graph = build_travel_graph()
        result = graph.invoke(_initial_state(), config={"recursion_limit": 50})

    assert result.get("error_message") is not None
    assert "intent_profile" in result["error_message"]
