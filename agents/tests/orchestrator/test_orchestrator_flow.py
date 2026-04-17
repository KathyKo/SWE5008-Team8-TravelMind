"""
Orchestrator routing unit tests.

Directly test orchestrator_node, orchestrator_routing, and the thin node
wrappers (planner_node, debate_node, replanner_node) without building
the LangGraph graph — avoids langgraph/uuid_utils import issues under coverage.
"""

from unittest.mock import patch, MagicMock

import pytest


# ============================================================
#  orchestrator_node — 核心路由逻辑
# ============================================================

class TestOrchestratorNode:
    """Test the central routing brain."""

    @staticmethod
    def _make_state(**overrides):
        base = {
            "messages": [],
            "threat_blocked": None,
            "user_feedback": None,
            "intent_profile_output": None,
            "search_results": None,
            "research": None,
            "itineraries": None,
            "final_itineraries": None,
            "is_valid": None,
            "debate_count": 0,
            "replan_attempts": 0,
            "max_replan_attempts": 2,
            "explanation": None,
            "explain_data": None,
            "output_guard_decision": None,
            "error_message": None,
        }
        base.update(overrides)
        return base

    def _route(self, **overrides) -> str:
        from agents.nodes import orchestrator_node
        state = self._make_state(**overrides)
        result = orchestrator_node(state)
        return result["next_node"]

    # ── Error handling ───────────────────────────────────

    def test_error_message_ends_flow(self):
        assert self._route(error_message="something broke") == "END"

    # ── Input Guard ──────────────────────────────────────

    def test_threat_blocked_ends_flow(self):
        assert self._route(threat_blocked=True) == "END"

    def test_threat_not_blocked_continues(self):
        assert self._route(threat_blocked=False) == "intent_profile"

    def test_threat_none_continues(self):
        assert self._route(threat_blocked=None) == "intent_profile"

    # ── User feedback → Replanner ────────────────────────

    def test_user_feedback_triggers_replanner(self):
        assert self._route(
            threat_blocked=False,
            user_feedback="请把第二天换成购物行程",
        ) == "replanner"

    def test_user_feedback_takes_priority_over_normal_flow(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"done": True},
            user_feedback="change hotel",
        ) == "replanner"

    # ── Intent Profile ───────────────────────────────────

    def test_no_profile_routes_to_intent(self):
        assert self._route(threat_blocked=False) == "intent_profile"

    def test_has_profile_skips_intent(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"destination": "Tokyo"},
        ) == "search"

    # ── Search / Research ────────────────────────────────

    def test_no_search_routes_to_search(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
        ) == "search"

    def test_has_search_results_skips(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"hotels": []},
        ) == "planner"

    def test_has_research_skips(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            research={"summary": "done"},
        ) == "planner"

    # ── Planner ──────────────────────────────────────────

    def test_no_itineraries_routes_to_planner(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
        ) == "planner"

    def test_has_itineraries_routes_to_debate(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"name": "Plan A"}],
            is_valid=None,
        ) == "debate"

    def test_has_final_itineraries_routes_to_debate(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            final_itineraries=[{"name": "Plan A"}],
            is_valid=None,
        ) == "debate"

    # ── Debate ↔ Planner loop ────────────────────────────

    def test_is_valid_none_routes_to_debate(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"plan": True}],
            is_valid=None,
        ) == "debate"

    def test_debate_fail_round1_routes_to_planner(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"plan": True}],
            is_valid=False,
            debate_count=1,
        ) == "planner"

    def test_debate_fail_round2_routes_to_planner(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"plan": True}],
            is_valid=False,
            debate_count=2,
        ) == "planner"

    def test_debate_fail_round3_forces_explain(self):
        """Max rounds (3) reached → force proceed to explain."""
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"plan": True}],
            is_valid=False,
            debate_count=3,
        ) == "explain"

    def test_debate_pass_routes_to_explain(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"plan": True}],
            is_valid=True,
            debate_count=1,
        ) == "explain"

    # ── Explainability ───────────────────────────────────

    def test_no_explanation_routes_to_explain(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"plan": True}],
            is_valid=True,
            debate_count=1,
        ) == "explain"

    def test_has_explanation_routes_to_output_guard(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"plan": True}],
            is_valid=True,
            debate_count=1,
            explanation={"summary": "good"},
        ) == "output_guard"

    def test_has_explain_data_routes_to_output_guard(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"plan": True}],
            is_valid=True,
            debate_count=1,
            explain_data={"scores": {}},
        ) == "output_guard"

    # ── Output Guard ─────────────────────────────────────

    def test_no_output_guard_routes_to_output_guard(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"plan": True}],
            is_valid=True,
            explanation={"ok": True},
        ) == "output_guard"

    def test_has_output_guard_ends_flow(self):
        assert self._route(
            threat_blocked=False,
            intent_profile_output={"ok": True},
            search_results={"data": True},
            itineraries=[{"plan": True}],
            is_valid=True,
            explanation={"ok": True},
            output_guard_decision="pass",
        ) == "END"


# ============================================================
#  orchestrator_routing — 读取 next_node 返回 str
# ============================================================

class TestOrchestratorRouting:

    def test_returns_next_node(self):
        from agents.nodes import orchestrator_routing
        assert orchestrator_routing({"next_node": "debate"}) == "debate"

    def test_defaults_to_end(self):
        from agents.nodes import orchestrator_routing
        assert orchestrator_routing({}) == "END"

    def test_returns_end_explicitly(self):
        from agents.nodes import orchestrator_routing
        assert orchestrator_routing({"next_node": "END"}) == "END"


# ============================================================
#  Thin wrappers — planner_node, debate_node, replanner_node
# ============================================================

class TestPlannerNodeWrapper:

    @patch("agents.nodes.call_remote_agent")
    def test_resets_is_valid(self, mock_call):
        from agents.nodes import planner_node
        mock_call.return_value = {
            "itineraries": [{"name": "Plan A"}],
            "final_itineraries": [{"name": "Plan A"}],
        }
        result = planner_node({"is_valid": True})
        assert result["is_valid"] is None

    @patch("agents.nodes.call_remote_agent")
    def test_preserves_agent_output(self, mock_call):
        from agents.nodes import planner_node
        mock_call.return_value = {"itineraries": [{"x": 1}]}
        result = planner_node({})
        assert result["itineraries"] == [{"x": 1}]
        mock_call.assert_called_once_with("planner", {})


class TestDebateNodeWrapper:

    @patch("agents.nodes.call_remote_agent")
    def test_increments_debate_count_from_zero(self, mock_call):
        from agents.nodes import debate_node
        mock_call.return_value = {"is_valid": True, "composite_score": 85.0}
        result = debate_node({"debate_count": 0})
        assert result["debate_count"] == 1

    @patch("agents.nodes.call_remote_agent")
    def test_increments_debate_count_from_existing(self, mock_call):
        from agents.nodes import debate_node
        mock_call.return_value = {"is_valid": False}
        result = debate_node({"debate_count": 2})
        assert result["debate_count"] == 3

    @patch("agents.nodes.call_remote_agent")
    def test_respects_agent_debate_count(self, mock_call):
        """If the remote agent already sets debate_count, don't override."""
        from agents.nodes import debate_node
        mock_call.return_value = {"is_valid": True, "debate_count": 99}
        result = debate_node({"debate_count": 1})
        assert result["debate_count"] == 99


class TestReplannerNodeWrapper:

    @patch("agents.nodes.call_remote_agent")
    def test_resets_downstream_state(self, mock_call):
        from agents.nodes import replanner_node
        mock_call.return_value = {"itineraries": [{"name": "v2"}]}
        result = replanner_node({
            "user_feedback": "change hotel",
            "is_valid": True,
            "explanation": {"old": True},
            "explain_data": {"old": True},
            "output_guard_decision": "pass",
            "replan_attempts": 0,
        })
        assert result["user_feedback"] is None
        assert result["is_valid"] is None
        assert result["explanation"] is None
        assert result["explain_data"] is None
        assert result["output_guard_decision"] is None

    @patch("agents.nodes.call_remote_agent")
    def test_increments_replan_attempts(self, mock_call):
        from agents.nodes import replanner_node
        mock_call.return_value = {}
        result = replanner_node({"replan_attempts": 1})
        assert result["replan_attempts"] == 2

    @patch("agents.nodes.call_remote_agent")
    def test_preserves_agent_output(self, mock_call):
        from agents.nodes import replanner_node
        mock_call.return_value = {"itineraries": [{"new": True}], "replanner_output": {"v": 2}}
        result = replanner_node({"replan_attempts": 0})
        assert result["itineraries"] == [{"new": True}]
        assert result["replanner_output"] == {"v": 2}


# ============================================================
#  call_remote_agent — HTTP 调用层
# ============================================================

class TestCallRemoteAgent:

    @patch("agents.nodes.requests.post")
    def test_successful_call(self, mock_post):
        from agents.nodes import call_remote_agent
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"threat_blocked": False},
        )
        result = call_remote_agent("input_guard", {})
        assert result == {"threat_blocked": False}

    @patch("agents.nodes.requests.post")
    def test_connection_error_returns_error_message(self, mock_post):
        from agents.nodes import call_remote_agent
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")
        result = call_remote_agent("input_guard", {})
        assert "error_message" in result
        assert "input_guard" in result["error_message"]

    @patch("agents.nodes.requests.post")
    def test_timeout_returns_error_message(self, mock_post):
        from agents.nodes import call_remote_agent
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        result = call_remote_agent("input_guard", {})
        assert "error_message" in result

    def test_unknown_agent_raises(self):
        from agents.nodes import call_remote_agent
        with pytest.raises(ValueError, match="Unknown agent"):
            call_remote_agent("nonexistent_agent", {})


# ============================================================
#  Simple wrapper nodes — 确认正确调用 call_remote_agent
# ============================================================

class TestSimpleWrappers:

    @patch("agents.nodes.call_remote_agent")
    def test_input_guard_node(self, mock_call):
        from agents.nodes import input_guard_node
        mock_call.return_value = {"threat_blocked": False}
        result = input_guard_node({"messages": []})
        mock_call.assert_called_once_with("input_guard", {"messages": []})
        assert result == {"threat_blocked": False}

    @patch("agents.nodes.call_remote_agent")
    def test_intent_profile_node(self, mock_call):
        from agents.nodes import intent_profile_node
        mock_call.return_value = {"intent_profile_output": {}}
        result = intent_profile_node({})
        mock_call.assert_called_once_with("intent_profile", {})

    @patch("agents.nodes.call_remote_agent")
    def test_search_node(self, mock_call):
        from agents.nodes import search_node
        mock_call.return_value = {"search_results": {}}
        result = search_node({})
        mock_call.assert_called_once_with("search", {})

    @patch("agents.nodes.call_remote_agent")
    def test_explain_node(self, mock_call):
        from agents.nodes import explain_node
        mock_call.return_value = {"explanation": {}}
        result = explain_node({})
        mock_call.assert_called_once_with("explain", {})

    @patch("agents.nodes.call_remote_agent")
    def test_output_guard_node(self, mock_call):
        from agents.nodes import output_guard_node
        mock_call.return_value = {"output_flagged": False}
        result = output_guard_node({})
        mock_call.assert_called_once_with("output_guard", {})
