from copy import deepcopy

import pytest

from .conftest import debate


def test_debate_and_judge_llm_constructor(monkeypatch):
    captured = {}

    def _fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(debate, "ChatOpenAI", _fake_chat_openai)
    debate._debate_llm(temperature=0.7)
    assert captured["model"] == debate.DEBATE_MODEL
    assert captured["temperature"] == 0.7
    debate._judge_llm(temperature=0.3)
    assert captured["model"] == debate.JUDGE_MODEL
    assert captured["temperature"] == 0.3


def test_build_payload_helpers(monkeypatch, base_state):
    class _FakeChain:
        def invoke(self, data):
            assert isinstance(data, dict)
            return {
                "round_decision": "continue",
                "winner_option": "A",
                "dimension_scores": {"A": {"bias_fairness": 80, "logistics": 80, "preference_alignment": 80, "option_diversity": 80}},
            }

    class _FakePrompt:
        def __or__(self, other):
            return self

        def invoke(self, data):
            return _FakeChain().invoke(data)

    monkeypatch.setattr(debate.ChatPromptTemplate, "from_messages", lambda *_args, **_kwargs: _FakePrompt())
    monkeypatch.setattr(debate, "_debate_llm", lambda **_kwargs: object())
    monkeypatch.setattr(debate, "_judge_llm", lambda **_kwargs: object())
    monkeypatch.setattr(debate, "JsonOutputParser", lambda: object())

    payload = debate._plan_payload_from_state(base_state)
    round_payload = debate._build_round_critique_payload(payload, [], 1)
    assert "dimension_scores" in round_payload
    judge_payload = debate._build_judge_payload(payload, [])
    assert "dimension_scores" in judge_payload
    selected = debate._build_selected_option_check_payload(
        selected_option="A",
        selected_plan=payload["itineraries"]["A"],
        trip_context={"origin": "Singapore"},
    )
    assert selected["winner_option"] == "A"


def test_extract_duration_days_handles_int_and_text():
    assert debate._extract_duration_days(4) == 4
    assert debate._extract_duration_days("7 days") == 7
    assert debate._extract_duration_days("unknown") is None
    assert debate._extract_duration_days(None) is None


def test_safe_round_dim_scores_sanitizes_and_computes_composite():
    payload = {
        "dimension_scores": {
            "A": {
                "bias_fairness": "101",
                "logistics": -4,
                "preference_alignment": "bad",
                "option_diversity": 80,
            },
            "B": "invalid",
        }
    }
    result = debate._safe_round_dim_scores(payload)
    assert result["A"]["bias_fairness"] == 100.0
    assert result["A"]["logistics"] == 0.0
    assert result["A"]["preference_alignment"] == 70.0
    assert result["A"]["composite"] == pytest.approx(62.5)
    assert "B" not in result


def test_winner_by_scores_returns_highest_composite():
    scores = {"A": {"composite": 78.5}, "B": {"composite": 88.2}, "C": {"composite": 80.0}}
    assert debate._winner_by_scores(scores) == "B"


def test_plan_payload_from_state_uses_fallback_fields(base_state):
    state = deepcopy(base_state)
    state.pop("itineraries")
    state["validated_itineraries"] = {"A": []}
    state["session_id"] = "sess-1"
    payload = debate._plan_payload_from_state(state)
    assert payload["plan_id"] == "plan-1"
    assert payload["itineraries"] == {"A": []}
    assert payload["debate_history"] == []


def test_round_and_planner_helpers():
    history = [{"sender": "agent4_critic"}, {"sender": "agent3_response"}, {"sender": "agent4_critic"}]
    assert debate._round_from_history(history) == 3
    planner_state = debate._planner_state_from_plan_payload({"origin": "SG", "destination": "JP"})
    assert planner_state["origin"] == "SG"
    assert planner_state["outbound_time_pref"] == ""


def test_merge_revised_plan_updates_only_allowed_keys(base_state):
    revised = {"itineraries": {"A": []}, "tool_log": ["x"], "unknown": "ignore"}
    merged = debate._merge_revised_plan(base_state, revised)
    assert merged["itineraries"] == {"A": []}
    assert merged["tool_log"] == ["x"]
    assert "unknown" not in merged


def test_evaluate_selected_option_fairness_not_found(base_state):
    result = debate.evaluate_selected_option_fairness(base_state, "Z")
    assert "not found" in result["error"]


def test_evaluate_selected_option_fairness_success_defaults(monkeypatch, base_state):
    monkeypatch.setattr(
        debate,
        "_build_selected_option_check_payload",
        lambda **kwargs: {
            "filter_bubble_detected": 1,
            "demographic_bias_detected": 0,
            "personalization_confidence": "LOW",
            "cold_start_note": " ",
            "highlights": "bad-type",
        },
    )
    result = debate.evaluate_selected_option_fairness(base_state, "A")
    assert result["selected_option"] == "A"
    assert result["filter_bubble_detected"] is True
    assert result["demographic_bias_detected"] is False
    assert result["personalization_confidence"] == "low"
    assert "Cold-start" in result["cold_start_note"]
    assert result["highlights"] == []
    assert result["filter_bubble_detail"]
    assert result["demographic_bias_detail"]


def test_evaluate_selected_option_fairness_exception(monkeypatch, base_state):
    def _boom(**kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(debate, "_build_selected_option_check_payload", _boom)
    result = debate.evaluate_selected_option_fairness(base_state, "A")
    assert "fairness check failed" in result["error"]


def test_persist_plan_debate_returns_when_no_plan_id(monkeypatch):
    called = {"v": False}

    def _session():
        called["v"] = True
        raise AssertionError("should not create session")

    monkeypatch.setattr(debate, "SessionLocal", _session)
    debate._persist_plan_debate(plan_id=None, plan_payload={}, debate_history=[], debate_verdict=None)
    assert called["v"] is False


def test_persist_plan_debate_updates_and_closes(monkeypatch):
    tracker = {"closed": False, "updated": False}

    class _FakeDB:
        def close(self):
            tracker["closed"] = True

    def _update(db, plan_id, revised):
        tracker["updated"] = True
        assert plan_id == "p1"
        assert "debate_history" in revised

    monkeypatch.setattr(debate, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(debate, "update_plan_result", _update)
    debate._persist_plan_debate(
        plan_id="p1",
        plan_payload={"itineraries": {"A": []}},
        debate_history=[{"sender": "x"}],
        debate_verdict={"accepted": True},
    )
    assert tracker == {"closed": True, "updated": True}


def test_debate_agent_missing_itineraries():
    result = debate.debate_agent({"plan_id": "p1", "debate_count": 2})
    assert result["is_valid"] is False
    assert result["debate_count"] == 2
    assert "Missing planner output" in result["debate_output"]["error"]


def test_debate_agent_rounds_already_completed(base_state):
    state = deepcopy(base_state)
    state["debate_history"] = [{"sender": "agent4_critic"}] * 4
    state["debate_verdict"] = {"winner_option": "B"}
    result = debate.debate_agent(state)
    assert result["is_valid"] is True
    assert result["debate_count"] == debate.MAX_DEBATE_ROUNDS
    assert result["debate_output"]["current_round_summary"] == "Debate rounds already completed."


def test_debate_agent_immediate_win(monkeypatch, base_state):
    monkeypatch.setattr(
        debate,
        "_build_round_critique_payload",
        lambda *args: {
            "round_decision": "decide",
            "winner_option": "B",
            "winner_reason": "clear winner",
            "critique_summary": "B dominates",
            "dimension_scores": {"B": {"composite": 90}},
        },
    )
    persisted = {"called": False}
    monkeypatch.setattr(
        debate,
        "_persist_plan_debate",
        lambda **kwargs: persisted.__setitem__("called", True),
    )
    result = debate.debate_agent(base_state)
    assert result["is_valid"] is True
    assert result["debate_verdict"]["winner_option"] == "B"
    assert result["debate_verdict"]["via_judge"] is False
    assert persisted["called"] is True


def test_debate_agent_continue_and_merge_revision(monkeypatch, base_state):
    monkeypatch.setattr(
        debate,
        "_build_round_critique_payload",
        lambda *args: {
            "round_decision": "continue",
            "winner_option": None,
            "critique_summary": "need refinement",
            "dimension_scores": {},
        },
    )
    monkeypatch.setattr(
        debate,
        "revise_itinerary",
        lambda planner_state, critique_summary, plan_payload: {
            "itineraries": {"A": [{"day": "Day 1", "items": []}]},
            "tool_log": ["revised"],
            "planner_chain_of_thought": "updated",
        },
    )
    monkeypatch.setattr(debate, "_persist_plan_debate", lambda **kwargs: None)
    result = debate.debate_agent(base_state)
    assert result["is_valid"] is False
    assert result["debate_verdict"] is None
    assert result["tool_log"] == ["revised"]
    assert any(x["sender"] == "agent3_response" for x in result["debate_history"])


def test_debate_agent_continue_with_revision_error(monkeypatch, base_state):
    monkeypatch.setattr(
        debate,
        "_build_round_critique_payload",
        lambda *args: {
            "round_decision": "continue",
            "winner_option": None,
            "critique_summary": "retry",
            "dimension_scores": {},
        },
    )
    monkeypatch.setattr(debate, "revise_itinerary", lambda *args: {"error": "planner failed"})
    monkeypatch.setattr(debate, "_persist_plan_debate", lambda **kwargs: None)
    result = debate.debate_agent(base_state)
    assert result["is_valid"] is False
    assert result["debate_history"][-1]["content"]["action"] == "keep_due_to_error"


def test_debate_agent_judge_path(monkeypatch, base_state):
    state = deepcopy(base_state)
    state["debate_history"] = [{"sender": "agent4_critic"}, {"sender": "agent3_response"}, {"sender": "agent4_critic"}, {"sender": "agent3_response"}]
    monkeypatch.setattr(
        debate,
        "_build_round_critique_payload",
        lambda *args: {
            "round_decision": "continue",
            "winner_option": None,
            "critique_summary": "max round",
            "dimension_scores": {"A": {"composite": 80.0}},
        },
    )
    monkeypatch.setattr(
        debate,
        "_build_judge_payload",
        lambda *args: {
            "winner_option": "C",
            "winner_reason": "judge pick",
            "dimension_scores": {"C": {"composite": 91}},
        },
    )
    monkeypatch.setattr(debate, "_persist_plan_debate", lambda **kwargs: None)
    result = debate.debate_agent(state)
    assert result["is_valid"] is True
    assert result["debate_verdict"]["via_judge"] is True
    assert result["debate_verdict"]["winner_option"] == "C"


def test_debate_agent_judge_fallback(monkeypatch, base_state):
    state = deepcopy(base_state)
    state["debate_history"] = [{"sender": "agent4_critic"}, {"sender": "agent3_response"}, {"sender": "agent4_critic"}, {"sender": "agent3_response"}]
    monkeypatch.setattr(
        debate,
        "_build_round_critique_payload",
        lambda *args: {
            "round_decision": "continue",
            "winner_option": None,
            "critique_summary": "fallback round",
            "dimension_scores": {"B": {"composite": 81.0}, "A": {"composite": 70.0}},
        },
    )
    monkeypatch.setattr(debate, "_build_judge_payload", lambda *args: (_ for _ in ()).throw(RuntimeError("judge err")))
    monkeypatch.setattr(debate, "_persist_plan_debate", lambda **kwargs: None)
    result = debate.debate_agent(state)
    assert result["debate_verdict"]["winner_option"] == "B"
    assert result["debate_verdict"]["via_judge"] is True


def test_debate_agent_round_critique_exception_still_continues(monkeypatch, base_state):
    monkeypatch.setattr(debate, "_build_round_critique_payload", lambda *args: (_ for _ in ()).throw(RuntimeError("bad llm")))
    monkeypatch.setattr(debate, "revise_itinerary", lambda *args: {"error": "planner failed"})
    monkeypatch.setattr(debate, "_persist_plan_debate", lambda **kwargs: None)
    result = debate.debate_agent(base_state)
    assert result["is_valid"] is False
    assert "Round critique failed" in result["debate_history"][0]["content"]["critique_summary"]
