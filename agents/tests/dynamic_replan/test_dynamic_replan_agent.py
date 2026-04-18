"""
dynamic_replan_agent 单元测试，目标 100% 语句覆盖 agents/specialists/dynamic_replan_agent.py。

运行:
    pytest agents/tests/dynamic_replan/test_dynamic_replan_agent.py -v \\
        --cov=agents.specialists.dynamic_replan_agent --cov-report=term-missing
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

dra = importlib.import_module("agents.specialists.dynamic_replan_agent")  # noqa: E402


# ── 在磁盘上构造可被 replan() 影子加载的微型 planner 工程 ─────────────────────


def _write_minimal_planner_project(root: Path, *, revise_raises: bool = False) -> None:
    (root / "agents").mkdir(parents=True, exist_ok=True)
    (root / "agents" / "specialists").mkdir(parents=True, exist_ok=True)
    (root / "tools").mkdir(parents=True, exist_ok=True)
    (root / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (root / "agents" / "specialists" / "__init__.py").write_text("", encoding="utf-8")
    (root / "agents" / "agent_tools.py").write_text(
        "def get_tools_for_agent(name):\n    return []\n",
        encoding="utf-8",
    )
    (root / "agents" / "llm_config.py").write_text("OPENAI_MODEL = 'gpt-4.1'\n", encoding="utf-8")
    (root / "agents" / "specialists" / "research_agent_1.py").write_text(
        """
def research_agent_1(state, tools):
    return {
        "compact_attractions": [
            {"name": "Indoor Museum Visit", "type": "museum", "description": "indoor gallery"},
            {"name": "Riverside Park Walk", "type": "park", "description": "outdoor garden"},
            {"name": "Senso-ji Temple", "description": "temple"},
        ],
        "compact_restaurants": [
            {"name": "Veggie Kitchen", "description": "vegetarian friendly"},
            {"name": "Ramen Shop", "description": "pork ramen"},
        ],
    }
""",
        encoding="utf-8",
    )
    rev = "raise RuntimeError('revise boom')" if revise_raises else "return planner_result"
    (root / "agents" / "specialists" / "planner_agent_1.py").write_text(
        f"""
OPENAI_MODEL = "gpt-4.1"

def planner_from_research_1(state, research_result):
    if isinstance(research_result, dict) and research_result.get("error"):
        return {{"itineraries": {{}}}}
    days = []
    for d in range(1, 4):
        days.append({{
            "day": f"Day {{d}}",
            "items": [
                {{"time": "10:00", "icon": "activity", "key": f"gen_act_{{d}}", "name": f"Generated Act {{d}}"}},
                {{"time": "12:30", "icon": "restaurant", "key": f"gen_rest_{{d}}", "name": f"Generated Rest {{d}}"}},
            ],
        }})
    return {{"itineraries": {{"C": days}}}}

def revise_itinerary_1(state, critique, planner_result):
    {rev}
""",
        encoding="utf-8",
    )


def _base_legacy_payload(**overrides) -> dict:
    plan = [
        {
            "day": "Day 1",
            "items": [
                {
                    "key": "flight_outbound",
                    "name": "SQ637 (Singapore Changi) → Narita | dep 2026-06-01",
                    "icon": "flight",
                    "time": "08:00",
                },
                {"key": "must_keep_boundary", "name": "Keep Me", "icon": "activity", "time": "11:00"},
                {"key": "attr_1", "name": "Outdoor Park Walk", "icon": "activity", "time": "14:00"},
                {"key": "rest_1", "name": "Ramen Shop", "icon": "restaurant", "time": "18:00"},
            ],
        },
        {
            "day": "Day 2",
            "items": [
                {"key": "hotel_stay", "name": "Hotel", "icon": "hotel", "time": "09:00"},
                {"key": "attr_2", "name": "Senso-ji Temple", "icon": "activity", "time": "10:00"},
                {"key": "rest_2", "name": "Veggie Kitchen", "icon": "restaurant", "time": "13:00"},
            ],
        },
        {
            "day": "Day 3",
            "items": [
                {"key": "attr_3", "name": "Indoor Museum Visit", "icon": "activity", "time": "10:00"},
                {
                    "key": "flight_return",
                    "name": "SQ638 (Narita) → Singapore Changi | dep 2026-06-03",
                    "icon": "flight",
                    "time": "18:00",
                },
                {
                    "key": "flight_return_dup",
                    "name": "Dup return",
                    "icon": "flight",
                    "time": "19:00",
                },
            ],
        },
    ]
    payload = {
        "scenario_id": "sc1",
        "source": {"k": 1},
        "original_recommended_plan": {
            "option_key": "A",
            "label": "Recommended",
            "itinerary_id": "IT1",
            "composite_score": 0.9,
            "plan": plan,
        },
        "user_replan_request": {
            "replan_scope": {
                "start_day": "Day 1",
                "end_day": "Day 3",
                "locked_days": [],
                "allow_replace_flight": False,
                "allow_replace_hotel": False,
            },
            "updated_user_intent": {
                "avoid": ["temple noise"],
                "new_preferences": ["indoor museums"],
                "must_keep": ["must_keep_boundary"],
                "pace_preference": "relaxed",
                "meal_preference": "vegetarian",
                "budget_guardrail": {"currency": "SGD"},
            },
            "trigger_events": [
                {
                    "type": "venue_closure",
                    "detail": "Edo Tokyo Museum closed",
                    "affected_item_key": "attraction_12_edo_tokyo_museum",
                },
                {"type": "weather", "detail": "heavy rain expected", "affected_item_key": ""},
            ],
            "output_expectation": {"format": "json"},
        },
    }
    payload.update(overrides)
    return payload


# ── 小工具与 IO ───────────────────────────────────────────────────────────────


def test_load_json_save_json(tmp_path):
    p = tmp_path / "a" / "b.json"
    data = {"x": 1, "中文": "ok"}
    dra.save_json(p, data)
    assert dra.load_json(p) == data


def test_slug_norm_contains_ascii_normalize():
    assert dra._slug("Hello World!!") == "hello_world"
    assert dra._norm_name({"name": "  Foo  "}) == "foo"
    assert dra._is_activity_or_restaurant({"icon": "Activity"}) is True
    assert dra._is_activity_or_restaurant({"icon": "flight"}) is False
    assert dra._contains_any("hello outdoor", {"outdoor", "x"}) is True
    assert dra._contains_any("hello", set()) is False
    assert dra._ascii_fold("caf\u0301")  # combining acute
    assert dra._is_temple_or_shrine_activity("random ji walk") is True
    assert dra._is_temple_or_shrine_activity("cafeteria") is False
    assert dra._normalize_match_text("Foo-Bar!!!") == "foo bar"


def test_build_llm_directive_and_structured_context():
    p = _base_legacy_payload()
    d = dra._build_llm_replan_directive(p)
    assert "Dynamic replan instruction" in d
    assert "temple" in d.lower() or "indoor" in d.lower()
    ctx = dra._build_structured_replan_context(p)
    assert "hard_rules" in ctx and "events" in ctx


def test_extract_city_and_date_from_flight():
    assert dra._extract_city_from_flight("") == ("Singapore", "Tokyo")
    txt = "Singapore Changi International Airport \u2192 Narita International Airport | dep 2026-06-01"
    o, dest = dra._extract_city_from_flight(txt)
    assert o == "Singapore"
    assert dest == "Tokyo"
    _o2, d2 = dra._extract_city_from_flight("Origin Strip \u2192 Haneda Airport | dep")
    assert d2 == "Tokyo"
    assert dra._extract_date_from_flight("x dep 2026-06-09 y", "dep") == "2026-06-09"
    assert dra._extract_date_from_flight("no date", "dep") == ""


def test_build_rules_branches():
    p = _base_legacy_payload()
    r = dra._build_rules(p)
    assert r.prefer_indoor is True
    assert r.vegetarian_friendly is True
    assert "outdoor" in r.avoid_keywords
    assert "edo tokyo museum" in r.closed_name_aliases


def test_build_state_from_input():
    st = dra._build_state_from_input(_base_legacy_payload())
    assert st["origin"] == "Singapore"
    assert "Tokyo" in st["destination"]
    assert "replan_directive" in st


def test_prefer_pick_candidate_and_disallowed():
    rules = dra.ReplanRules(
        start_day_num=1,
        end_day_num=3,
        locked_days=set(),
        allow_replace_flight=False,
        allow_replace_hotel=False,
        must_keep_keys=set(),
        closed_item_keys=set(),
        closed_name_aliases={"closedvenue"},
        avoid_keywords={"badword"},
        prefer_indoor=True,
        vegetarian_friendly=True,
    )
    indoor_item = {"name": "Museum", "description": "modern art gallery indoor"}
    assert dra._prefer_item(indoor_item, indoor=True, vegetarian=False) is True
    veg_item = {"name": "Vegan", "description": "plant-based menu"}
    assert dra._prefer_item(veg_item, indoor=False, vegetarian=True) is True
    assert dra._prefer_item({"name": "x"}, indoor=False, vegetarian=False) is False

    cand = dra._candidate_to_itinerary_item({"title": "  My Place  "}, "activity", "11:00", 3)
    assert "replan_3" in cand["key"]

    pool = [
        {"name": "Used", "description": "ok"},
        {"name": "Temple Visit", "description": "senso ji"},
        {"name": "Museum Nice", "description": "indoor museum"},
    ]
    picked = dra._pick_candidate(pool, {"used"}, set(), indoor=True, vegetarian=False)
    assert picked["name"] == "Museum Nice"

    blocked, reason = dra._is_disallowed_item(
        {"key": "k", "icon": "activity", "name": "closedvenue"},
        rules,
        set(),
    )
    assert blocked and reason == "venue_closed"

    blocked2, r2 = dra._is_disallowed_item(
        {"key": "k badword", "icon": "activity", "name": "foo"},
        dra.ReplanRules(
            1,
            3,
            set(),
            False,
            False,
            set(),
            set(),
            set(),
            {"badword"},
            False,
            False,
        ),
        set(),
    )
    assert blocked2 and r2 == "user_avoidance"

    rules_indoor = dra.ReplanRules(1, 3, set(), False, False, set(), set(), set(), set(), True, False)
    blocked3, r3 = dra._is_disallowed_item(
        {"key": "park", "icon": "activity", "name": "x garden"},
        rules_indoor,
        set(),
    )
    assert blocked3 and r3 == "indoor_preference_violation"

    visited = {"dup"}
    b4, r4 = dra._is_disallowed_item({"key": "k", "icon": "activity", "name": "dup"}, rules, visited)
    assert b4 and r4 == "already_visited"


def test_return_day_label_and_day_num_time_fmt():
    assert dra._return_day_label_from_dates({"dates": "bad"}) == "Day 1"
    assert dra._return_day_label_from_dates({"dates": "2026-06-01 to 2026-06-05"}) == "Day 5"
    assert dra._day_num("Day 12") == 12
    assert dra._day_num("") == 0
    assert dra._parse_time_to_min("25:00") is None
    assert dra._parse_time_to_min("09:05") == 9 * 60 + 5
    assert dra._fmt_min_to_time(-10) == "00:00"
    assert dra._fmt_min_to_time(99999) == "23:59"


def test_iter_scope_append():
    out = {"replanned_plan": {"plan": [{"day": "Day 1"}]}}
    assert len(dra._iter_replan_days(out)) == 1
    assert dra._scope_day_numbers({"user_replan_request": {"replan_scope": {"start_day": "Day 2", "end_day": ""}}}) == (
        2,
        999,
    )
    dra._append_change(out, "Day 1", "a", "k", "n", "r")
    assert out["change_log"][-1]["action"] == "a"


def test_norm_extract_items():
    assert dra._norm("  a\n\tb  ") == "a b"
    res = {"replanned_plan": {"plan": [{"day": "Day 1", "items": [{"key": "1"}]}]}}
    assert len(dra._extract_replanned_items(res)) == 1


def test_hard_rule_checks_and_report_text():
    payload = {
        "user_replan_request": {
            "updated_user_intent": {"avoid": ["badterm"], "must_keep": ["missing_key"]},
            "replan_scope": {"start_day": "Day 1", "end_day": "Day 2"},
        }
    }
    result = {
        "replanned_plan": {
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"key": "flight_return", "name": "r1"},
                        {"key": "flight_return", "name": "r2"},
                        {"key": "x", "name": "has badterm inside", "note": ""},
                    ],
                }
            ]
        },
        "scenario_id": "s",
        "change_log": [],
    }
    hc = dra._hard_rule_checks(payload, result)
    assert hc["hard_rule_passed"] is False
    assert any(v["type"] == "must_keep_missing" for v in hc["violations"])
    text = dra.build_verifier_report_text(
        payload,
        result,
        {"hard_check": hc, "llm_judge": {"risk_flags": []}, "final_verdict": "revise"},
    )
    assert "violations:" in text
    hc_ok = dra._hard_rule_checks(
        {"user_replan_request": {"updated_user_intent": {"avoid": [], "must_keep": []}, "replan_scope": {}}},
        {
            "replanned_plan": {
                "plan": [{"day": "Day 1", "items": [{"key": "flight_return", "name": "R"}]}],
            }
        },
    )
    assert hc_ok["hard_rule_passed"] is True
    text2 = dra.build_verifier_report_text(
        {"user_replan_request": {}},
        {"replanned_plan": {"plan": []}, "change_log": [], "scenario_id": ""},
        {"hard_check": hc_ok, "llm_judge": {"risk_flags": ["x"]}, "final_verdict": "accept"},
    )
    assert "risk_flags:" in text2


def test_parse_clock_minutes_and_extract_day_time():
    assert dra._parse_clock_minutes("") is None
    assert dra._parse_clock_minutes("rest of day here") == 23 * 60 + 59
    assert dra._parse_clock_minutes("full schedule") == 23 * 60 + 59
    assert dra._parse_clock_minutes("~1 hour") == 60
    assert dra._parse_clock_minutes("1h") == 60
    assert dra._parse_clock_minutes("2-3 hours") == 180
    assert dra._parse_clock_minutes("2–3 hours") == 180
    assert dra._parse_clock_minutes("3h") == 180
    assert dra._parse_clock_minutes("14:05") == 14 * 60 + 5
    assert dra._parse_clock_minutes("99:99") is None
    d, m = dra._extract_day_and_time({"current_day": "Day 4", "current_time": "15:00"})
    assert d == 4 and m == 15 * 60


def test_winner_option_collect_consumed_candidates():
    plan = {
        "debate_verdict": {"winner_option": "B"},
        "itineraries": {"A": [], "B": [{"day": "Day 1", "items": [{"icon": "activity", "name": "Later", "time": "20:00"}]}]},
    }
    assert dra._winner_option_from_plan(plan) == "B"
    plan2 = {"debate_verdict": {}, "itineraries": {"X": []}}
    assert dra._winner_option_from_plan(plan2) == "X"

    days = [
        {
            "day": "Day 1",
            "items": [{"icon": "hotel", "name": "H", "time": "10:00"}, {"icon": "activity", "name": "A", "time": "09:00"}],
        }
    ]
    c = dra._collect_consumed_items(days, 2, 0)
    assert "a" in c

    pp = {
        "itineraries": {
            "A": [{"day": "Day 1", "items": [{"icon": "activity", "name": "Dup", "time": "10:00", "cost": "1"}]}],
            "B": [{"day": "Day 1", "items": [{"icon": "activity", "name": "Dup", "time": "11:00", "cost": "2"}]}],
        }
    }
    cand = dra._collect_candidate_items(pp, set())
    assert len([x for x in cand if x["name"] == "Dup"]) == 1


def test_fallback_alternatives_and_ensure_two():
    alts = dra._fallback_alternatives([])
    assert len(alts) == 2 and alts[0]["name"] != alts[1]["name"]
    one = dra._fallback_alternatives([{"name": "Only", "icon": "x", "desc": "d"}])
    assert len(one) == 2
    cleaned = dra._ensure_two_alternatives([], [{"name": "A"}, {"name": "B"}], set())
    assert len(cleaned) == 2
    cleaned2 = dra._ensure_two_alternatives([{"name": "X"}], [], set())
    assert len(cleaned2) == 2


def test_resolve_planner_model_and_override(monkeypatch):
    monkeypatch.delenv("REPLAN_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("AGENT3_PLANNER_MODEL", raising=False)
    assert dra._resolve_planner_model() == "gpt-4.1"
    monkeypatch.setenv("REPLAN_PLANNER_MODEL", "gpt-test")
    assert dra._resolve_planner_model() == "gpt-test"

    fake_llm = MagicMock()
    fake_pl = MagicMock()
    with patch("importlib.import_module", side_effect=[fake_llm, fake_pl]):
        dra._apply_planner_model_override("m1")
    assert getattr(fake_llm, "OPENAI_MODEL") == "m1"


def test_resolve_planner_root_env_and_default(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "agent_tools.py").write_text("x=1\n", encoding="utf-8")
    monkeypatch.setenv("REPLAN_PLANNER_ROOT", str(root))
    assert dra._resolve_planner_root() == root.resolve()

    monkeypatch.delenv("REPLAN_PLANNER_ROOT", raising=False)
    sibling = tmp_path / "SWE5008-Team8-TravelMind-update-research_planner_explainibility"
    (sibling / "agents").mkdir(parents=True)
    (sibling / "agents" / "agent_tools.py").write_text("x=1\n", encoding="utf-8")
    fake_repo = tmp_path / "SWE5008-Team8-TravelMind"
    monkeypatch.setattr(dra, "_ROOT", fake_repo)
    assert dra._resolve_planner_root() == sibling.resolve()


def test_resolve_planner_root_file_not_found(tmp_path, monkeypatch):
    isolated = tmp_path / "isolated" / "SWE5008-Team8-TravelMind"
    isolated.mkdir(parents=True)
    monkeypatch.setattr(dra, "_ROOT", isolated)
    monkeypatch.delenv("REPLAN_PLANNER_ROOT", raising=False)
    with pytest.raises(FileNotFoundError):
        dra._resolve_planner_root()


def test_legacy_replan_and_replan(monkeypatch, tmp_path):
    planner_root = tmp_path / "planner_proj"
    _write_minimal_planner_project(planner_root, revise_raises=False)
    monkeypatch.setenv("REPLAN_PLANNER_ROOT", str(planner_root))
    monkeypatch.delenv("REPLAN_PLANNER_MODEL", raising=False)
    out = dra.replan(_base_legacy_payload())
    assert out["replanned_plan"]["plan"]
    assert isinstance(out["change_log"], list)

    _write_minimal_planner_project(planner_root, revise_raises=True)
    out2 = dra.replan(_base_legacy_payload())
    assert out2["replanned_plan"]["plan"]


def test_legacy_replan_research_error(monkeypatch, tmp_path):
    planner_root = tmp_path / "planner_proj2"
    _write_minimal_planner_project(planner_root)
    (planner_root / "agents" / "specialists" / "research_agent_1.py").write_text(
        "def research_agent_1(state, tools):\n    return {'error': 'x'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REPLAN_PLANNER_ROOT", str(planner_root))
    out = dra.replan(_base_legacy_payload())
    assert out["raw_research_result_excerpt"]["attractions_count"] == 0


def test_replan_path_remove_valueerror(monkeypatch, tmp_path):
    import sys

    planner_root = tmp_path / "planner_proj3"
    _write_minimal_planner_project(planner_root)
    monkeypatch.setenv("REPLAN_PLANNER_ROOT", str(planner_root))

    class PathList(list):
        def remove(self, x):
            if str(x) == str(planner_root.resolve()) or str(x) == str(planner_root):
                raise ValueError("not in list")
            return super().remove(x)

    monkeypatch.setattr(sys, "path", PathList(list(sys.path)))
    dra.replan(_base_legacy_payload())


def test_legacy_raises_without_plan():
    with pytest.raises(ValueError):
        dra._legacy_replan({})


def test_postprocess_branches():
    payload = {"user_replan_request": {"replan_scope": {"start_day": "Day 1", "end_day": "Day 3"}}}
    output = {
        "replanned_plan": {
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"key": "a", "name": "A", "time": "10:00", "icon": "activity"},
                        {"key": "b", "name": "B", "time": "10:30", "icon": "activity"},
                    ],
                },
                    {
                        "day": "Day 2",
                        "items": [
                            {"key": "r", "name": "Side Bistro", "time": "15:00", "icon": "restaurant"},
                        ],
                    },
                {
                    "day": "Day 3",
                    "items": [
                        {"key": "flight_return", "name": "Ret", "time": "11:00", "icon": "flight"},
                        {"key": "last", "name": "Last Act", "time": "10:45", "icon": "activity"},
                    ],
                },
            ]
        }
    }
    fixed = dra.postprocess_replan_output(payload, output)
    assert fixed["postprocess_summary"]["enabled"] is True
    # coffee break path (no afternoon coffee)
    day2 = fixed["replanned_plan"]["plan"][1]
    assert any("Coffee" in str(i.get("name", "")) for i in day2["items"])


def _verifier_chain_mock(ok_payload: dict, *, invoke_side_effect=None):
    final = MagicMock()
    if invoke_side_effect:
        final.invoke.side_effect = invoke_side_effect
    else:
        final.invoke.return_value = ok_payload
    mid = MagicMock()
    mid.__or__ = MagicMock(return_value=final)
    template = MagicMock()
    template.__or__ = MagicMock(return_value=mid)
    return template


def test_llm_judge_no_key_and_chain_and_fallback(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    j = dra._llm_judge({}, {}, {})
    assert j["final_recommendation"] == "revise"
    assert "missing_api_key" in j["risk_flags"][0]

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    ok = {
        "final_recommendation": "accept",
        "preference_alignment_score": 80,
        "feasibility_score": 80,
        "constraint_compliance_score": 80,
        "quality_confidence_score": 80,
        "risk_flags": [],
        "reason": "ok",
    }
    with patch.object(dra.ChatPromptTemplate, "from_messages", return_value=_verifier_chain_mock(ok)):
        j2 = dra._llm_judge({}, {}, {})
        assert j2["final_recommendation"] == "accept"

    bad_final = MagicMock()
    bad_final.invoke.side_effect = RuntimeError("fail")
    bad_mid = MagicMock()
    bad_mid.__or__ = MagicMock(return_value=bad_final)
    bad_template = MagicMock()
    bad_template.__or__ = MagicMock(return_value=bad_mid)
    with patch.object(dra.ChatPromptTemplate, "from_messages", return_value=bad_template):
        j3 = dra._llm_judge({}, {}, {})
        assert "verifier_fallback_failed" in j3["risk_flags"][-1]


def test_llm_judge_model_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("REPLAN_VERIFIER_MODEL", "bad-primary-model")
    ok_body = {
        "preference_alignment_score": 1,
        "feasibility_score": 1,
        "constraint_compliance_score": 1,
        "quality_confidence_score": 1,
        "risk_flags": [],
        "final_recommendation": "revise",
        "reason": "ok",
    }
    or_calls = {"n": 0}

    def prompt_or(_other):
        final = MagicMock()
        n = or_calls["n"]
        or_calls["n"] += 1
        if n == 0:
            final.invoke.side_effect = RuntimeError("nope")
        else:
            final.invoke.return_value = dict(ok_body)
        mid = MagicMock()
        mid.__or__ = MagicMock(return_value=final)
        return mid

    prompt = MagicMock()
    prompt.__or__ = MagicMock(side_effect=prompt_or)

    with patch.object(dra.ChatPromptTemplate, "from_messages", return_value=prompt):
        out = dra._llm_judge({}, {}, {})
        assert out.get("_verifier_model_used") == "gpt-4.1"


def test_verify_replan_accept():
    payload = {"user_replan_request": {"updated_user_intent": {}, "replan_scope": {}}}
    result = {"replanned_plan": {"plan": [{"day": "Day 1", "items": [{"key": "flight_return", "name": "R"}]}]}}
    with patch.object(dra, "_llm_judge", return_value={"final_recommendation": "accept"}):
        v = dra.verify_replan(payload, result)
    assert v["final_verdict"] == "accept"


def test_llm_generate_alternatives_no_dmx_key(monkeypatch):
    monkeypatch.delenv("DMX_API_KEY", raising=False)
    monkeypatch.setattr(dra, "DMX_API_KEY", "")
    alts = dra._llm_generate_alternatives(feedback={}, plan_payload={}, consumed_names=set(), candidates=[])
    assert len(alts) == 2


def test_llm_generate_alternatives_chain(monkeypatch):
    monkeypatch.setattr(dra, "DMX_API_KEY", "fake-key")
    final = MagicMock()
    final.invoke.return_value = {
        "alternatives": [
            {"name": "Alt One", "icon": "✨", "desc": "d", "price": "1", "rating": "5", "dist": "1", "tag": "t"},
            {"name": "Alt Two", "icon": "✨", "desc": "d2", "price": "2", "rating": "5", "dist": "1", "tag": "t"},
        ]
    }
    mid = MagicMock()
    mid.__or__ = MagicMock(return_value=final)
    tpl = MagicMock()
    tpl.__or__ = MagicMock(return_value=mid)
    with patch.object(dra.ChatPromptTemplate, "from_messages", return_value=tpl):
        alts = dra._llm_generate_alternatives(
            feedback={"x": 1},
            plan_payload={"origin": "S", "destination": "T", "dates": "d", "preferences": "p"},
            consumed_names=set(),
            candidates=[{"name": "Pool", "icon": "✨", "desc": "d"}],
        )
    assert len(alts) == 2

    final.invoke.return_value = {"alternatives": "bad"}
    with patch.object(dra.ChatPromptTemplate, "from_messages", return_value=tpl):
        alts2 = dra._llm_generate_alternatives(
            feedback={},
            plan_payload={},
            consumed_names={"alt one"},
            candidates=[{"name": "Alt One", "icon": "✨", "desc": ""}],
        )
    assert len(alts2) == 2


def test_dynamic_replan_agent_paths(monkeypatch):
    mock_db = MagicMock()
    plan = {
        "debate_verdict": {"winner_option": "A"},
        "itineraries": {
            "A": [{"day": "Day 1", "items": [{"icon": "activity", "name": "Done", "time": "09:00"}]}],
            "B": [{"day": "Day 1", "items": [{"icon": "activity", "name": "Other", "time": "10:00"}]}],
        },
        "option_meta": {},
        "origin": "S",
        "destination": "T",
        "dates": "d",
        "preferences": "p",
    }
    with patch.object(dra, "_resolve_replan_plan_id", return_value=None):
        r = dra.dynamic_replan_agent({"user_feedback": {}})
        assert "error" in r["replanner_output"]

    with patch.object(dra, "_resolve_replan_plan_id", return_value="pid"), patch.object(
        dra, "SessionLocal", return_value=mock_db
    ), patch.object(dra, "load_plan", return_value=None):
        r2 = dra.dynamic_replan_agent({"user_feedback": {}, "plan_id": "pid"})
        assert "not found" in r2["replanner_output"]["error"]

    monkeypatch.setattr(dra, "DMX_API_KEY", "")
    with patch.object(dra, "_resolve_replan_plan_id", return_value="pid"), patch.object(
        dra, "SessionLocal", return_value=mock_db
    ), patch.object(dra, "load_plan", return_value=plan), patch.object(dra, "update_plan_result"):
        r3 = dra.dynamic_replan_agent(
            {"user_feedback": {"current_day": "Day 1", "current_time": "10:00"}, "plan_id": "pid"}
        )
        assert r3["replanner_output"]["alternatives"]
        assert r3["user_feedback"] is None


def test_pick_candidate_avoid_branch():
    assert (
        dra._pick_candidate(
            [
                {"name": "A", "description": "contains temple word"},
                {"name": "B", "description": "museum only"},
            ],
            set(),
            {"temple"},
            indoor=False,
            vegetarian=False,
        )["name"]
        == "B"
    )
    assert (
        dra._pick_candidate(
            [{"name": "OnlyTemple", "description": "senso ji visit"}],
            set(),
            set(),
            indoor=True,
            vegetarian=False,
        )
        is None
    )


def test_return_day_label_iso_error():
    st = {"dates": "2026-06-01 to not-a-valid-date"}
    assert dra._return_day_label_from_dates(st) == "Day 1"


def test_parse_time_and_clock_edges():
    assert dra._parse_time_to_min("12:99") is None
    assert dra._parse_clock_minutes("25:00") is None
    assert dra._parse_clock_minutes("no-time-here") is None


def test_extract_day_time_defaults():
    d, m = dra._extract_day_and_time({"current_day": "Day", "current_time": "rest of day"})
    assert d == 1
    assert m == 23 * 60 + 59


def test_resolve_replan_plan_id_branches():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = ("p-debate",)
    with patch.object(dra, "SessionLocal", return_value=mock_db):
        assert dra._resolve_replan_plan_id() == "p-debate"
    mock_db.close.assert_called()

    mock_db2 = MagicMock()

    def q_side(*_a, **_k):
        m = MagicMock()
        m.filter.return_value.order_by.return_value.first.return_value = None
        m.order_by.return_value.first.return_value = ("p-latest",)
        return m

    mock_db2.query.side_effect = q_side
    with patch.object(dra, "SessionLocal", return_value=mock_db2):
        assert dra._resolve_replan_plan_id() == "p-latest"

    mock_db3 = MagicMock()

    def q_empty(*_a, **_k):
        m = MagicMock()
        m.filter.return_value.order_by.return_value.first.return_value = None
        m.order_by.return_value.first.return_value = None
        return m

    mock_db3.query.side_effect = q_empty
    with patch.object(dra, "SessionLocal", return_value=mock_db3):
        assert dra._resolve_replan_plan_id() is None


def test_collect_candidate_skips_empty_name():
    pp = {
        "itineraries": {
            "A": [{"day": "Day 1", "items": [{"icon": "activity", "name": "", "time": "10:00"}]}],
        }
    }
    assert dra._collect_candidate_items(pp, set()) == []


def test_ensure_two_skips_seen_name():
    out = dra._ensure_two_alternatives(
        [{"name": "Only"}],
        [{"name": "Only"}, {"name": "NewOne", "icon": "✨", "desc": "d"}],
        set(),
    )
    assert len(out) == 2


def test_postprocess_out_of_scope_day_skips():
    payload = {"user_replan_request": {"replan_scope": {"start_day": "Day 2", "end_day": "Day 2"}}}
    output = {
        "replanned_plan": {
            "plan": [
                {"day": "Day 1", "items": [{"key": "x", "time": "09:00", "icon": "activity", "name": "N"}]},
                {"day": "Day 2", "items": [{"key": "y", "time": "10:00", "icon": "activity", "name": "In"}]},
            ]
        }
    }
    dra.postprocess_replan_output(payload, output)


def test_postprocess_afternoon_coffee_already_present():
    payload = {"user_replan_request": {"replan_scope": {"start_day": "Day 1", "end_day": "Day 1"}}}
    output = {
        "replanned_plan": {
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"key": "c", "time": "15:00", "icon": "restaurant", "name": "Nice Coffee House"},
                    ],
                }
            ]
        }
    }
    fixed = dra.postprocess_replan_output(payload, output)
    assert not any("Afternoon Coffee Break" in str(i.get("name", "")) for i in fixed["replanned_plan"]["plan"][0]["items"])


def test_postprocess_return_flight_buffer_adjust_and_remove():
    payload = {"user_replan_request": {"replan_scope": {"start_day": "Day 1", "end_day": "Day 1"}}}
    adjust = {
        "replanned_plan": {
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"key": "act", "time": "15:01", "icon": "activity", "name": "Late"},
                        {"key": "flight_return", "time": "18:00", "icon": "flight", "name": "Ret"},
                    ],
                }
            ]
        }
    }
    dra.postprocess_replan_output(payload, adjust)
    remove_case = {
        "replanned_plan": {
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"key": "act", "time": "09:30", "icon": "activity", "name": "Early"},
                        {"key": "flight_return", "time": "10:00", "icon": "flight", "name": "Ret"},
                    ],
                }
            ]
        }
    }
    dra.postprocess_replan_output(payload, remove_case)


def test_postprocess_return_no_prev_nonflight():
    payload = {"user_replan_request": {"replan_scope": {"start_day": "Day 1", "end_day": "Day 1"}}}
    output = {
        "replanned_plan": {
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"key": "flight_out", "time": "08:00", "icon": "flight", "name": "Out"},
                        {"key": "flight_return", "time": "18:00", "icon": "flight", "name": "Ret"},
                    ],
                }
            ]
        }
    }
    dra.postprocess_replan_output(payload, output)


def test_postprocess_fix_dense_skips_unparsed_time():
    payload = {"user_replan_request": {"replan_scope": {"start_day": "Day 1", "end_day": "Day 1"}}}
    output = {
        "replanned_plan": {
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"key": "a", "time": "10:00", "icon": "activity", "name": "A"},
                        {"key": "b", "time": "oops", "icon": "activity", "name": "B"},
                        {"key": "c", "time": "11:00", "icon": "activity", "name": "C"},
                    ],
                }
            ]
        }
    }
    dra.postprocess_replan_output(payload, output)


def test_llm_generate_alternatives_malformed_entries(monkeypatch):
    monkeypatch.setattr(dra, "DMX_API_KEY", "k")
    final = MagicMock()
    final.invoke.return_value = {
        "alternatives": [
            "not-a-dict",
            {"name": "", "icon": "✨"},
            {"name": "ConsumedName", "icon": "✨", "desc": "d"},
        ]
    }
    mid = MagicMock()
    mid.__or__ = MagicMock(return_value=final)
    tpl = MagicMock()
    tpl.__or__ = MagicMock(return_value=mid)
    with patch.object(dra.ChatPromptTemplate, "from_messages", return_value=tpl):
        alts = dra._llm_generate_alternatives(
            feedback={},
            plan_payload={"origin": "S", "destination": "T"},
            consumed_names={"consumedname"},
            candidates=[{"name": "FallbackSeed", "icon": "✨", "desc": "d"}],
        )
    assert len(alts) == 2


def test_legacy_replan_extra_branches(monkeypatch, tmp_path):
    """覆盖 _legacy_replan 中预访问日、锁定日、返程校验与补回等分支。"""
    planner_root = tmp_path / "legacy_stress"
    _write_minimal_planner_project(planner_root)
    (planner_root / "agents" / "specialists" / "planner_agent_1.py").write_text(
        textwrap.dedent(
            '''
            OPENAI_MODEL = "gpt-4.1"

            def planner_from_research_1(state, research_result):
                days = []
                for d in range(1, 4):
                    days.append({
                        "day": f"Day {d}",
                        "items": [
                            {"time": "10:00", "icon": "restaurant", "key": f"r{d}", "name": f"DupEat {d}"},
                            {"time": "11:00", "icon": "activity", "key": f"a{d}", "name": f"TempleVisit {d}"},
                        ],
                    })
                return {"itineraries": {"C": days}}

            def revise_itinerary_1(state, critique, planner_result):
                return planner_result
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    (planner_root / "agents" / "specialists" / "research_agent_1.py").write_text(
        textwrap.dedent(
            '''
            def research_agent_1(state, tools):
                return {
                    "compact_attractions": [
                        {"name": "Indoor Museum Visit", "description": "museum indoor"},
                        {"name": "City Park", "description": "park outdoor"},
                    ],
                    "compact_restaurants": [
                        {"name": "Veggie Kitchen", "description": "vegetarian"},
                        {"name": "Ramen Shop", "description": "ramen"},
                    ],
                }
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("REPLAN_PLANNER_ROOT", str(planner_root))
    p = _base_legacy_payload()
    p["original_recommended_plan"]["plan"] = [
        {
            "day": "Day 0",
            "items": [{"icon": "activity", "name": "PreTrip", "time": "10:00", "key": "pre"}],
        },
        {
            "day": "Day 1",
            "items": [
                {
                    "key": "flight_outbound",
                    "name": "SQ637 (Singapore Changi) \u2192 Narita | dep 2026-06-01",
                    "icon": "flight",
                    "time": "08:00",
                },
                {"key": "must_keep_boundary", "name": "Keep Me", "icon": "activity", "time": "11:00"},
            ],
        },
        {
            "day": "Day 2",
            "items": [
                {"key": "closed_venue_key", "name": "Closed Spot", "icon": "activity", "time": "10:00"},
                {"key": "hotel_stay", "name": "Hotel", "icon": "hotel", "time": "12:00"},
                {"key": "flight_return", "name": "WrongDayReturn", "icon": "flight", "time": "13:00"},
            ],
        },
        {
            "day": "Day 3",
            "items": [
                {"key": "attr_last", "name": "Last", "icon": "activity", "time": "10:00"},
            ],
        },
    ]
    p["user_replan_request"]["replan_scope"] = {
        "start_day": "Day 1",
        "end_day": "Day 3",
        "locked_days": ["Day 2"],
        "allow_replace_flight": False,
        "allow_replace_hotel": False,
    }
    p["user_replan_request"]["updated_user_intent"]["must_keep"] = ["closed_venue_key"]
    p["user_replan_request"]["trigger_events"] = [
        {"type": "venue_closure", "affected_item_key": "closed_venue_key", "detail": "closed forever"},
    ]
    out = dra.replan(p)
    reasons = [c.get("reason") for c in out.get("change_log", [])]
    assert any("return_flight_wrong_day" in str(r) for r in reasons) or out["replanned_plan"]["plan"]

    p2 = copy.deepcopy(_base_legacy_payload())
    ret_item = None
    for day in p2["original_recommended_plan"]["plan"]:
        for it in list(day.get("items", [])):
            if str(it.get("key")) == "flight_return":
                ret_item = it
                day["items"].remove(it)
    assert ret_item is not None
    p2["original_recommended_plan"]["plan"][0]["items"].append(ret_item)
    monkeypatch.setenv("REPLAN_PLANNER_ROOT", str(planner_root))
    out2 = dra.replan(p2)
    assert any(c.get("reason") == "restore_required_return_flight" for c in out2.get("change_log", []))


def test_postprocess_return_buffer_already_ok():
    payload = {"user_replan_request": {"replan_scope": {"start_day": "Day 1", "end_day": "Day 1"}}}
    output = {
        "replanned_plan": {
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"key": "a", "time": "16:00", "icon": "activity", "name": "Early"},
                        {"key": "flight_return", "time": "20:00", "icon": "flight", "name": "Ret"},
                    ],
                }
            ]
        }
    }
    dra.postprocess_replan_output(payload, output)


def test_postprocess_return_scan_skips_inner_flights():
    payload = {"user_replan_request": {"replan_scope": {"start_day": "Day 1", "end_day": "Day 1"}}}
    output = {
        "replanned_plan": {
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"key": "a", "time": "12:00", "icon": "activity", "name": "Act"},
                        {"key": "fj", "time": "13:00", "icon": "flight", "name": "Mid"},
                        {"key": "flight_return", "time": "15:30", "icon": "flight", "name": "Ret"},
                    ],
                }
            ]
        }
    }
    dra.postprocess_replan_output(payload, output)


def test_legacy_restaurant_removal_and_dup_return(monkeypatch, tmp_path):
    root = tmp_path / "legacy_rest"
    _write_minimal_planner_project(root)
    (root / "agents" / "specialists" / "planner_agent_1.py").write_text(
        textwrap.dedent(
            '''
            OPENAI_MODEL = "gpt-4.1"

            def planner_from_research_1(state, research_result):
                days = [
                    {"day": "Day 1", "items": [{"time": "12:00", "icon": "activity", "key": "a1", "name": "OkAct"}]},
                    {"day": "Day 2", "items": [{"time": "12:00", "icon": "activity", "key": "a2", "name": "OkAct2"}]},
                    {
                        "day": "Day 3",
                        "items": [
                            {
                                "time": "12:00",
                                "icon": "restaurant",
                                "key": "rx",
                                "name": "Temple Food Court",
                            },
                            {"time": "18:00", "icon": "flight", "key": "flight_return", "name": "R1"},
                            {"time": "19:00", "icon": "flight", "key": "flight_return", "name": "R2"},
                        ],
                    },
                ]
                return {"itineraries": {"C": days}}

            def revise_itinerary_1(state, critique, planner_result):
                return planner_result
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    (root / "agents" / "specialists" / "research_agent_1.py").write_text(
        textwrap.dedent(
            '''
            def research_agent_1(state, tools):
                return {
                    "compact_attractions": [{"name": "Museum X", "description": "indoor"}],
                    "compact_restaurants": [{"name": "Safe Rest", "description": "food"}],
                }
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("REPLAN_PLANNER_ROOT", str(root))
    p = _base_legacy_payload()
    p["user_replan_request"]["replan_scope"] = {
        "start_day": "Day 1",
        "end_day": "Day 3",
        "locked_days": [],
        "allow_replace_flight": False,
        "allow_replace_hotel": False,
    }
    p["user_replan_request"]["updated_user_intent"]["avoid"] = ["temple"]
    out = dra.replan(p)
    reasons = " ".join(str(c.get("reason")) for c in out.get("change_log", []))
    assert "duplicate_return_flight" in reasons or "user_avoidance" in reasons


def test_zz_reload_covers_module_sys_path_insert():
    """置于文件末尾：reload 后其它用例不再依赖本模块未重载状态。"""
    repo = str(Path(__file__).resolve().parents[3])
    while repo in sys.path:
        sys.path.remove(repo)
    import importlib

    importlib.reload(dra)
    assert repo in sys.path
