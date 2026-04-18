"""
debate_agent 单元测试：覆盖 agents/specialists/debate_agent.py 的主要分支与辅助函数。

在仓库根目录执行:
    pytest agents/tests/debate/test_debate_agent.py -v \\
        --cov=agents.specialists.debate_agent --cov-report=term-missing
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

da = importlib.import_module("agents.specialists.debate_agent")  # noqa: E402


def _dim_scores(a: float = 70, b: float = 71, c: float = 72, d: float = 73) -> dict:
    return {
        "bias_fairness": a,
        "logistics": b,
        "preference_alignment": c,
        "option_diversity": d,
    }


def _minimal_plan_payload(**overrides):
    base = {
        "origin": "SIN",
        "destination": "TYO",
        "dates": "2026-06-01 to 2026-06-07",
        "duration": "7 days",
        "budget": "3000",
        "preferences": "food",
        "itineraries": {"A": [{"day": 1}], "B": [{"day": 1}], "C": [{"day": 1}]},
        "option_meta": {"A": {"k": 1}, "B": {"k": 2}, "C": {"k": 3}},
        "flight_options_outbound": [{"f": 1}],
        "flight_options_return": [{"f": 2}],
        "hotel_options": [{"h": 1}],
        "debate_history": [],
        "debate_verdict": None,
        "tool_log": [],
        "planner_decision_trace": {},
        "chain_of_thought": "",
    }
    base.update(overrides)
    return base


# ── 辅助函数 ───────────────────────────────────────────────────────────────────


def test_extract_duration_days():
    assert da._extract_duration_days(None) is None
    assert da._extract_duration_days(5) == 5
    assert da._extract_duration_days("14 nights") == 14
    assert da._extract_duration_days("no digits") is None


def test_safe_round_dim_scores_clamp_and_errors():
    raw = {
        "dimension_scores": {
            "A": "not a dict",
            "B": {
                "bias_fairness": -10,
                "logistics": 200,
                "preference_alignment": "bad",
                "option_diversity": 50,
            },
        }
    }
    out = da._safe_round_dim_scores(raw)
    assert "A" not in out
    b = out["B"]
    assert b["bias_fairness"] == 0.0
    assert b["logistics"] == 100.0
    assert b["preference_alignment"] == 70.0
    assert b["option_diversity"] == 50.0
    assert b["composite"] == round(sum(b[d] for d in da.DIMENSIONS) / len(da.DIMENSIONS), 2)

    empty = da._safe_round_dim_scores({})
    assert empty == {}


def test_winner_by_scores():
    assert da._winner_by_scores({}) is None
    scores = {
        "A": {"composite": 10.0},
        "B": {"composite": 88.0},
        "C": {"composite": 20.0},
    }
    assert da._winner_by_scores(scores) == "B"
    assert da._winner_by_scores({"X": {}}) == "X"


def test_round_from_history():
    assert da._round_from_history([]) == 1
    hist = [
        {"sender": "agent4_critic", "round": 1},
        {"sender": "other", "round": 1},
        {"sender": "agent4_critic", "round": 2},
    ]
    assert da._round_from_history(hist) == 3


def test_planner_state_from_plan_payload():
    p = _minimal_plan_payload(origin="O", destination="D")
    st = da._planner_state_from_plan_payload(p)
    assert st["origin"] == "O"
    assert st["outbound_time_pref"] == ""
    assert st["duration"] == "7 days"


def test_merge_revised_plan():
    base = _minimal_plan_payload(itineraries={"A": [1]})
    revised = {
        "itineraries": {"A": [2]},
        "option_meta": {"A": {"x": 1}},
        "flight_options_outbound": [9],
        "flight_options_return": [8],
        "hotel_options": [7],
        "tool_log": [{"t": 1}],
        "planner_decision_trace": {"z": 1},
        "chain_of_thought": "cot",
        "ignored": True,
    }
    merged = da._merge_revised_plan(base, revised)
    assert merged["itineraries"]["A"] == [2]
    assert merged["chain_of_thought"] == "cot"
    assert "ignored" not in merged


def _chain_filter_then_order(first_result):
    m = MagicMock()
    m.filter.return_value.order_by.return_value.first.return_value = first_result
    return m


def _chain_order_only(first_result):
    m = MagicMock()
    m.order_by.return_value.first.return_value = first_result
    return m


def test_resolve_active_plan_id_priority():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
        "pid-1",
    )
    with patch.object(da, "SessionLocal", return_value=mock_db):
        assert da._resolve_active_plan_id() == "pid-1"
    mock_db.close.assert_called_once()


def test_resolve_active_plan_id_fallback_latest():
    mock_db = MagicMock()
    mock_db.query.side_effect = [
        _chain_filter_then_order(None),
        _chain_order_only(("pid-latest",)),
    ]
    with patch.object(da, "SessionLocal", return_value=mock_db):
        assert da._resolve_active_plan_id() == "pid-latest"
    mock_db.close.assert_called_once()


def test_resolve_active_plan_id_no_rows():
    mock_db = MagicMock()
    mock_db.query.side_effect = [
        _chain_filter_then_order(None),
        _chain_order_only(None),
    ]
    with patch.object(da, "SessionLocal", return_value=mock_db):
        assert da._resolve_active_plan_id() is None
    mock_db.close.assert_called_once()


def test_persist_plan_debate_calls_update():
    mock_db = MagicMock()
    with patch.object(da, "SessionLocal", return_value=mock_db), patch.object(
        da, "update_plan_result"
    ) as up:
        da._persist_plan_debate(
            plan_id="p9",
            plan_payload=_minimal_plan_payload(),
            debate_history=[{"r": 1}],
            debate_verdict={"ok": True},
        )
    up.assert_called_once()
    args = up.call_args[0]
    assert args[1] == "p9"
    revised = args[2]
    assert revised["debate_history"] == [{"r": 1}]
    assert revised["debate_verdict"] == {"ok": True}
    mock_db.close.assert_called()


def _patch_langchain_chain_invoke(return_value: dict):
    """ChatPromptTemplate.from_messages(...) | llm | parser  -> mock with .invoke"""
    final = MagicMock()
    final.invoke.return_value = return_value
    mid = MagicMock()
    mid.__or__ = MagicMock(return_value=final)
    template = MagicMock()
    template.__or__ = MagicMock(return_value=mid)
    return patch.object(
        da.ChatPromptTemplate,
        "from_messages",
        return_value=template,
    )


def test_build_round_critique_payload():
    raw = {
        "round_decision": "continue",
        "winner_option": None,
        "critique_summary": "ok",
        "dimension_scores": {"A": _dim_scores()},
    }
    with patch.object(da, "_debate_llm", return_value=MagicMock()), _patch_langchain_chain_invoke(
        dict(raw)
    ):
        out = da._build_round_critique_payload(_minimal_plan_payload(), [], 1)
    assert out["round_decision"] == "continue"
    assert "A" in out["dimension_scores"]
    assert "composite" in out["dimension_scores"]["A"]


def test_build_judge_payload():
    raw = {
        "winner_option": "B",
        "winner_reason": "balanced",
        "dimension_scores": {"B": _dim_scores(80, 80, 80, 80)},
    }
    with patch.object(da, "_judge_llm", return_value=MagicMock()), _patch_langchain_chain_invoke(
        dict(raw)
    ):
        out = da._build_judge_payload(_minimal_plan_payload(), [])
    assert out["winner_option"] == "B"
    assert out["dimension_scores"]["B"]["composite"] == 80.0


# ── debate_agent 主流程 ────────────────────────────────────────────────────────


def test_debate_agent_no_plan():
    with patch.object(da, "_resolve_active_plan_id", return_value=None):
        r = da.debate_agent({})
    assert r["is_valid"] is False
    assert "No plan found" in r["debate_output"]["error"]


def test_debate_agent_plan_missing_in_db():
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="missing"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=None):
        r = da.debate_agent({})
    assert r["plan_id"] == "missing"
    assert r["is_valid"] is False
    mock_db.close.assert_called()


def test_debate_agent_rounds_already_done_no_verdict():
    hist = [{"sender": "agent4_critic", "round": i} for i in range(1, 4)]
    payload = _minimal_plan_payload(debate_history=hist, debate_verdict=None)
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="p1"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=payload):
        r = da.debate_agent({})
    assert r["debate_count"] == da.MAX_DEBATE_ROUNDS
    assert r["is_valid"] is False
    assert "already completed" in r["debate_output"]["current_round_summary"]


def test_debate_agent_rounds_already_done_with_verdict():
    verdict = {"winner_option": "A"}
    hist = [{"sender": "agent4_critic", "round": i} for i in range(1, 4)]
    payload = _minimal_plan_payload(debate_history=hist, debate_verdict=verdict)
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="p1"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=payload):
        r = da.debate_agent({})
    assert r["is_valid"] is True
    assert r["debate_output"]["debate_verdict"] == verdict


def test_debate_agent_critique_chain_raises():
    payload = _minimal_plan_payload()
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="p1"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=payload), patch.object(
        da, "_build_round_critique_payload", side_effect=RuntimeError("boom")
    ), patch.object(da, "_persist_plan_debate") as pers, patch.object(
        da, "revise_itinerary", return_value={"planner_chain_of_thought": "ok", "itineraries": {}}
    ):
        r = da.debate_agent({})
    assert r["is_valid"] is False
    assert "Round critique failed" in r["debate_output"]["debate_history"][-2]["content"]["critique_summary"]
    pers.assert_called()
    assert r["debate_output"]["debate_history"][-1]["sender"] == "agent3_response"


def test_debate_agent_immediate_win():
    payload = _minimal_plan_payload()
    critique = {
        "round_decision": "DECIDE",
        "winner_option": "B",
        "winner_reason": "clear win",
        "critique_summary": "B wins",
        "dimension_scores": {"B": _dim_scores(90, 90, 90, 90)},
    }
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="p1"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=payload), patch.object(
        da, "_build_round_critique_payload", return_value=critique
    ), patch.object(da, "_persist_plan_debate") as pers:
        r = da.debate_agent({})
    assert r["is_valid"] is True
    assert r["debate_output"]["debate_verdict"]["winner_option"] == "B"
    assert r["debate_output"]["debate_verdict"]["via_judge"] is False
    pers.assert_called_once()


def test_debate_agent_continue_merge_and_revise_error():
    """第二轮：评审继续，Planner 返回 error 分支。"""
    hist = [{"sender": "agent4_critic", "round": 1}]
    payload = _minimal_plan_payload(debate_history=hist)
    critique = {
        "round_decision": "continue",
        "winner_option": "Z",
        "critique_summary": "need work",
        "dimension_scores": {"A": _dim_scores()},
    }
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="p1"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=payload), patch.object(
        da, "_build_round_critique_payload", return_value=critique
    ), patch.object(da, "_persist_plan_debate") as pers, patch.object(
        da, "revise_itinerary", return_value={"error": "planner failed"}
    ):
        r = da.debate_agent({})
    assert r["is_valid"] is False
    planner_msg = r["debate_output"]["debate_history"][-1]["content"]
    assert planner_msg["action"] == "keep_due_to_error"
    assert "planner failed" in planner_msg["message"]
    pers.assert_called_once()


def test_debate_agent_continue_revise_success():
    hist = [{"sender": "agent4_critic", "round": 1}]
    payload = _minimal_plan_payload(debate_history=hist)
    critique = {
        "round_decision": "continue",
        "winner_option": None,
        "critique_summary": "revise please",
        "dimension_scores": {"A": _dim_scores()},
    }
    revised = {
        "itineraries": {"A": [{"day": 99}]},
        "option_meta": {"A": {"new": True}},
        "flight_options_outbound": [],
        "flight_options_return": [],
        "hotel_options": [],
        "tool_log": [],
        "planner_decision_trace": {},
        "chain_of_thought": "x",
    }
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="p1"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=payload), patch.object(
        da, "_build_round_critique_payload", return_value=critique
    ), patch.object(da, "_persist_plan_debate") as pers, patch.object(
        da, "revise_itinerary", return_value=revised
    ):
        r = da.debate_agent({})
    assert r["is_valid"] is False
    pers.assert_called_once()
    _, kwargs = pers.call_args
    assert kwargs["plan_payload"]["itineraries"]["A"][0]["day"] == 99


def test_debate_agent_max_rounds_judge_ok():
    hist = [
        {"sender": "agent4_critic", "round": 1},
        {"sender": "agent3_response", "round": 1},
        {"sender": "agent4_critic", "round": 2},
        {"sender": "agent3_response", "round": 2},
    ]
    payload = _minimal_plan_payload(debate_history=hist)
    last_scores = {"C": _dim_scores(10, 10, 10, 10)}
    critique = {
        "round_decision": "continue",
        "winner_option": None,
        "critique_summary": "final tension",
        "dimension_scores": last_scores,
    }
    judge = {
        "winner_option": "C",
        "winner_reason": "judge says C",
        "dimension_scores": {"C": _dim_scores(60, 60, 60, 60)},
    }
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="p1"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=payload), patch.object(
        da, "_build_round_critique_payload", return_value=critique
    ), patch.object(da, "_build_judge_payload", return_value=judge), patch.object(
        da, "_persist_plan_debate"
    ) as pers:
        r = da.debate_agent({})
    assert r["is_valid"] is True
    assert r["debate_output"]["debate_verdict"]["via_judge"] is True
    assert r["debate_output"]["debate_verdict"]["winner_option"] == "C"
    pers.assert_called_once()


def test_debate_agent_max_rounds_judge_fallback():
    hist = [
        {"sender": "agent4_critic", "round": 1},
        {"sender": "agent3_response", "round": 1},
        {"sender": "agent4_critic", "round": 2},
        {"sender": "agent3_response", "round": 2},
    ]
    payload = _minimal_plan_payload(debate_history=hist)
    last_scores = {
        "A": {"composite": 1.0},
        "B": {"composite": 99.0},
    }
    critique = {
        "round_decision": "continue",
        "winner_option": None,
        "critique_summary": "max",
        "dimension_scores": last_scores,
    }
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="p1"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=payload), patch.object(
        da, "_build_round_critique_payload", return_value=critique
    ), patch.object(da, "_build_judge_payload", side_effect=ValueError("judge down")), patch.object(
        da, "_persist_plan_debate"
    ) as pers:
        r = da.debate_agent({})
    assert r["debate_output"]["debate_verdict"]["winner_option"] == "B"
    assert "fallback" in r["debate_output"]["debate_verdict"]["winner_reason"].lower()
    pers.assert_called_once()


def test_debate_agent_max_rounds_judge_fallback_default_a():
    hist = [
        {"sender": "agent4_critic", "round": 1},
        {"sender": "agent3_response", "round": 1},
        {"sender": "agent4_critic", "round": 2},
        {"sender": "agent3_response", "round": 2},
    ]
    payload = _minimal_plan_payload(debate_history=hist)
    critique = {
        "round_decision": "continue",
        "winner_option": None,
        "critique_summary": "max",
        "dimension_scores": {},
    }
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="p1"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=payload), patch.object(
        da, "_build_round_critique_payload", return_value=critique
    ), patch.object(da, "_build_judge_payload", side_effect=RuntimeError("down")), patch.object(
        da, "_persist_plan_debate"
    ):
        r = da.debate_agent({})
    assert r["debate_output"]["debate_verdict"]["winner_option"] == "A"


def test_debate_agent_round3_no_continue_revise():
    """第三轮且 round_decision 为 decide 但 winner 不在 itineraries 中 -> 走 judge。"""
    hist = [
        {"sender": "agent4_critic", "round": 1},
        {"sender": "agent3_response", "round": 1},
        {"sender": "agent4_critic", "round": 2},
        {"sender": "agent3_response", "round": 2},
    ]
    payload = _minimal_plan_payload(debate_history=hist)
    critique = {
        "round_decision": "decide",
        "winner_option": "INVALID",
        "critique_summary": "claimed win but invalid key",
        "dimension_scores": {"A": _dim_scores()},
    }
    judge = {"winner_option": "A", "winner_reason": "forced", "dimension_scores": {}}
    mock_db = MagicMock()
    with patch.object(da, "_resolve_active_plan_id", return_value="p1"), patch.object(
        da, "SessionLocal", return_value=mock_db
    ), patch.object(da, "load_plan", return_value=payload), patch.object(
        da, "_build_round_critique_payload", return_value=critique
    ), patch.object(da, "_build_judge_payload", return_value=judge), patch.object(
        da, "_persist_plan_debate"
    ), patch.object(da, "revise_itinerary") as rev:
        r = da.debate_agent({})
    rev.assert_not_called()
    assert r["debate_output"]["debate_verdict"]["winner_option"] == "A"


def test_debate_llm_and_judge_llm_construct():
    with patch.object(da, "ChatOpenAI") as m:
        da._debate_llm(temperature=0.3)
        da._judge_llm(temperature=0.05)
    assert m.call_count == 2
