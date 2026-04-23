from copy import deepcopy
import pytest

from .conftest import dr


def test_load_and_save_json(tmp_path):
    path = tmp_path / "a" / "b.json"
    dr.save_json(path, {"x": 1})
    loaded = dr.load_json(path)
    assert loaded == {"x": 1}


def test_basic_text_helpers():
    assert dr._slug("Hello Tokyo!") == "hello_tokyo"
    assert dr._norm_name({"name": "  Ueno Park "}) == "ueno park"
    assert dr._is_activity_or_restaurant({"icon": "activity"}) is True
    assert dr._contains_any("indoor museum", {"museum", "park"}) is True
    assert dr._ascii_fold("Café") == "Cafe"
    assert dr._is_temple_or_shrine_activity("Senso-ji temple tour") is True
    assert dr._normalize_match_text("Edo-Tokyo Museum!") == "edo tokyo museum"


def test_build_directive_and_context(replan_payload):
    directive = dr._build_llm_replan_directive(replan_payload)
    assert "Dynamic replan instruction" in directive
    ctx = dr._build_structured_replan_context(replan_payload)
    assert ctx["hard_rules"]["replan_scope"]["start_day"] == "Day 1"
    assert "events" in ctx


def test_extract_flight_city_and_date():
    name = "Singapore Changi Airport → Haneda Airport | dep 2026-06-01"
    assert dr._extract_city_from_flight(name) == ("Singapore", "Tokyo")
    assert dr._extract_date_from_flight(name, "dep") == "2026-06-01"
    assert dr._extract_city_from_flight("bad format") == ("Singapore", "Tokyo")
    assert dr._extract_date_from_flight("bad", "dep") == ""


def test_build_rules_and_state(replan_payload):
    rules = dr._build_rules(replan_payload)
    assert rules.start_day_num == 1
    assert "flight_outbound" in rules.must_keep_keys
    assert "edo tokyo museum" in rules.closed_name_aliases
    assert rules.prefer_indoor is True
    assert rules.vegetarian_friendly is True

    state = dr._build_state_from_input(replan_payload)
    assert state["origin"] == "Singapore"
    assert "structured_context" in state["preferences"]
    assert state["hard_constraints"]["requirements"]


def test_prefer_pick_and_convert_candidates():
    item = {"name": "Anime Mall", "description": "indoor shopping"}
    assert dr._prefer_item(item, indoor=True, vegetarian=False) is True
    conv = dr._candidate_to_itinerary_item({"name": "Akihabara"}, "activity", "14:00", 1)
    assert conv["key"].startswith("attraction_replan_1_")

    picked = dr._pick_candidate(
        [{"name": "Temple Walk"}, {"name": "Indoor Museum", "description": "museum"}],
        used_names=set(),
        avoid_keywords={"temple"},
        indoor=True,
        vegetarian=False,
    )
    assert picked["name"] == "Indoor Museum"


def test_disallowed_item_and_return_day_label(replan_payload):
    rules = dr._build_rules(replan_payload)
    blocked, reason = dr._is_disallowed_item(
        {"key": "attraction_01_edo_tokyo_museum", "icon": "activity", "name": "Edo Tokyo Museum"},
        rules,
        visited_names=set(),
    )
    assert blocked is True
    assert reason == "venue_closed"

    label = dr._return_day_label_from_dates({"dates": "2026-06-01 to 2026-06-05"})
    assert label == "Day 5"
    assert dr._return_day_label_from_dates({"dates": "bad"}) == "Day 1"


def test_time_and_scope_helpers():
    assert dr._parse_time_to_min("09:30") == 570
    assert dr._parse_time_to_min("25:10") is None
    assert dr._fmt_min_to_time(574) == "09:34"
    assert dr._fmt_min_to_time(5000) == "23:59"
    assert dr._scope_day_numbers({"user_replan_request": {"replan_scope": {"start_day": "Day 2", "end_day": "Day 0"}}}) == (2, 999)


def test_postprocess_day_level_actions():
    output = {"change_log": [], "replanned_plan": {"plan": []}}
    day = {
        "day": "Day 2",
        "items": [
            {"key": "a1", "icon": "activity", "name": "A", "time": "10:00"},
            {"key": "a2", "icon": "activity", "name": "B", "time": "10:30"},
            {"key": "flight_return", "icon": "flight", "name": "Back", "time": "15:00"},
        ],
    }
    dr._fix_dense_timeline_for_day(output, day)
    dr._ensure_afternoon_coffee(output, day)
    dr._enforce_return_flight_buffer(output, day)
    assert any(x["action"] == "postprocess_adjust_time" for x in output["change_log"])
    assert any(x["action"] in {"postprocess_add_item", "postprocess_remove_item"} for x in output["change_log"])


def test_postprocess_replan_output(replan_payload):
    output = {
        "replanned_plan": {
            "plan": [
                {"day": "Day 1", "items": [{"key": "x1", "icon": "activity", "name": "A", "time": "10:00"}, {"key": "x2", "icon": "activity", "name": "B", "time": "10:20"}]},
                {"day": "Day 3", "items": [{"key": "x3", "icon": "activity", "name": "C", "time": "10:00"}]},
            ]
        }
    }
    fixed = dr.postprocess_replan_output(replan_payload, output)
    assert fixed["postprocess_summary"]["enabled"] is True
    assert "rules" in fixed["postprocess_summary"]


def test_hard_rule_checks_and_verifier_report():
    payload = {
        "user_replan_request": {
            "updated_user_intent": {"must_keep": ["must_keep_1"], "avoid": ["forbidden"]},
            "replan_scope": {"start_day": "Day 1", "end_day": "Day 2"},
        }
    }
    result = {
        "scenario_id": "s1",
        "change_log": [],
        "replanned_plan": {
            "plan": [
                {"day": "Day 1", "items": [{"key": "must_keep_1", "name": "must_keep_1"}, {"key": "flight_return", "name": "return"}]},
            ]
        },
    }
    hard = dr._hard_rule_checks(payload, result)
    assert hard["hard_rule_passed"] is True
    verifier = {"hard_check": hard, "llm_judge": {"final_recommendation": "accept", "reason": "ok"}, "final_verdict": "accept"}
    text = dr.build_verifier_report_text(payload, result, verifier)
    assert "Dynamic Replan Verifier Report" in text
    assert "verdict: accept" in text


def test_llm_judge_missing_key(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judged = dr._llm_judge({}, {}, {"hard_rule_passed": True})
    assert judged["final_recommendation"] == "revise"
    assert "missing_api_key_for_verifier" in judged["risk_flags"]


def test_verify_replan_uses_hard_and_llm(monkeypatch):
    monkeypatch.setattr(dr, "_hard_rule_checks", lambda payload, result: {"hard_rule_passed": True, "violations": []})
    monkeypatch.setattr(dr, "_llm_judge", lambda payload, result, hard: {"final_recommendation": "accept"})
    verifier = dr.verify_replan({}, {})
    assert verifier["final_verdict"] == "accept"


def test_resolve_planner_model_and_root(monkeypatch, tmp_path):
    monkeypatch.setenv("REPLAN_PLANNER_MODEL", "gpt-x")
    assert dr._resolve_planner_model() == "gpt-x"
    monkeypatch.delenv("REPLAN_PLANNER_MODEL")
    monkeypatch.setenv("AGENT3_PLANNER_MODEL", "gpt-y")
    assert dr._resolve_planner_model() == "gpt-y"

    root = tmp_path / "proj"
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "agent_tools.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setenv("REPLAN_PLANNER_ROOT", str(root))
    assert dr._resolve_planner_root() == root.resolve()


def test_replan_wrapper_restores_sys_modules(monkeypatch, tmp_path):
    root = tmp_path / "planner"
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "agent_tools.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(dr, "_resolve_planner_root", lambda: root)
    monkeypatch.setattr(dr, "_resolve_planner_model", lambda: "gpt-test")
    monkeypatch.setattr(dr, "_apply_planner_model_override", lambda model: None)
    monkeypatch.setattr(dr, "_legacy_replan", lambda payload: {"ok": True, "payload": payload})
    out = dr.replan({"x": 1})
    assert out["ok"] is True


def test_llm_generate_alternatives_fallback_and_clean(monkeypatch):
    candidates = [{"name": "AltA", "icon": "activity", "desc": "x"}, {"name": "AltB", "icon": "restaurant", "desc": "y"}]
    monkeypatch.setattr(dr, "DMX_API_KEY", "")
    out = dr._llm_generate_alternatives(
        feedback={},
        plan_payload={},
        consumed_names=set(),
        candidates=candidates,
    )
    assert len(out) == 2

    class _FakeChain:
        def invoke(self, data):
            return {
                "alternatives": [
                    {"name": "UsedOne"},
                    {"name": "FreshOne", "icon": "✨", "desc": "new"},
                ]
            }

    class _FakePrompt:
        def __or__(self, other):
            return self

        def invoke(self, data):
            return _FakeChain().invoke(data)

    monkeypatch.setattr(dr, "DMX_API_KEY", "k")
    monkeypatch.setattr(dr.ChatPromptTemplate, "from_messages", lambda *_args, **_kwargs: _FakePrompt())
    monkeypatch.setattr(dr, "ChatOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(dr, "JsonOutputParser", lambda: object())
    out2 = dr._llm_generate_alternatives(
        feedback={},
        plan_payload={},
        consumed_names={"usedone"},
        candidates=candidates,
    )
    assert len(out2) == 2


def test_resolve_replan_plan_id(monkeypatch):
    class _Query:
        def __init__(self, first_values):
            self.first_values = list(first_values)

        def filter(self, *args, **kwargs):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def first(self):
            return self.first_values.pop(0)

    class _DB:
        def __init__(self):
            self.q = _Query([("debate-plan",), ("latest-plan",)])

        def query(self, *_args, **_kwargs):
            return self.q

        def close(self):
            return None

    monkeypatch.setattr(dr, "SessionLocal", lambda: _DB())
    assert dr._resolve_replan_plan_id() == "debate-plan"

def test_clock_parsing_day_time_and_winner():
    assert dr._parse_clock_minutes("rest of day") == 1439
    assert dr._parse_clock_minutes("2-3 hours") == 180
    assert dr._parse_clock_minutes("09:45") == 585
    assert dr._parse_clock_minutes("bad") is None
    day, minute = dr._extract_day_and_time({"current_day": "Day 3", "current_time": "09:00"})
    assert (day, minute) == (3, 540)
    assert dr._winner_option_from_plan({"debate_verdict": {"winner_option": "B"}, "itineraries": {"A": [], "B": []}}) == "B"
    assert dr._winner_option_from_plan({"itineraries": {"C": []}}) == "C"


def test_collect_consumed_and_candidates():
    days = [
        {"day": "Day 1", "items": [{"icon": "activity", "name": "A", "time": "10:00"}, {"icon": "flight", "name": "F", "time": "09:00"}]},
        {"day": "Day 2", "items": [{"icon": "restaurant", "name": "B", "time": "12:00"}]},
    ]
    consumed = dr._collect_consumed_items(days, until_day=1, until_min=600)
    assert "a" in consumed

    payload = {
        "itineraries": {
            "A": [{"day": "Day 1", "items": [{"icon": "activity", "name": "A", "time": "10:00", "cost": "10"}]}],
            "B": [{"day": "Day 1", "items": [{"icon": "restaurant", "name": "B", "time": "11:00", "cost": "20"}]}],
        }
    }
    candidates = dr._collect_candidate_items(payload, consumed={"a"})
    assert len(candidates) == 1
    assert candidates[0]["name"] == "B"


def test_fallback_ensure_two_and_collect_other_options():
    fallback = dr._fallback_alternatives([{"name": "X", "icon": "activity", "desc": "d"}])
    assert len(fallback) == 2

    ensured = dr._ensure_two_alternatives(
        cleaned=[{"name": "Alpha", "icon": "✨"}],
        candidates=[{"name": "Beta", "icon": "activity", "desc": "d"}],
        consumed_names={"gamma"},
    )
    assert len(ensured) == 2

    itineraries = {
        "A": [{"day": "Day 1", "items": [{"icon": "activity", "name": "Keep", "cost": "1"}]}],
        "B": [{"day": "Day 1", "items": [{"icon": "activity", "name": "Replace1", "cost": "1"}]}],
    }
    pool = dr._collect_replan_candidates_from_other_options(itineraries, "A", {"keep"})
    assert pool[0]["name"] == "Replace1"


def test_apply_key_based_replan():
    days = [
        {"day": "Day 1", "items": [{"key": "a1", "icon": "activity", "name": "OldA"}, {"key": "x1", "icon": "hotel", "name": "H"}]},
        {"day": "Day 2", "items": [{"key": "r1", "icon": "restaurant", "name": "OldR"}]},
    ]
    replanned, change_log, unresolved = dr._apply_key_based_replan(
        days=days,
        replace_item_keys={"a1", "x1", "r1"},
        locked_item_keys={"r1"},
        candidates=[{"name": "NewA", "icon": "activity", "cost": "TBC"}],
    )
    assert replanned[0]["items"][0]["name"] == "NewA"
    assert len(change_log) == 1
    assert unresolved and unresolved[0]["reason"] == "only_activity_or_restaurant_can_be_replaced"


def test_apply_key_based_replan_replace_slots_duplicate_keys():
    """Same catalog key on two rows: only the selected slot must be replaced."""
    days = [
        {
            "day": "Day 1",
            "items": [
                {"key": "rest_dup", "icon": "restaurant", "name": "Sushi Same"},
                {"key": "rest_dup", "icon": "restaurant", "name": "Sushi Same"},
            ],
        }
    ]
    typed = {
        "restaurant": [{"name": "Ramen Other", "cost": "20"}],
        "activity": [],
        "hotel": [],
        "flight_outbound": [],
        "flight_return": [],
    }
    replanned, change_log, unresolved, _generated = dr._apply_key_based_replan(
        days=days,
        replace_item_keys={"rest_dup"},
        locked_item_keys=set(),
        typed_candidates=typed,
        forbidden_names=set(),
        replace_slots={"0:0"},
    )
    assert replanned[0]["items"][0]["name"] == "Ramen Other"
    assert replanned[0]["items"][1]["name"] == "Sushi Same"
    assert len(change_log) == 1
    assert not unresolved


def test_dynamic_replan_agent_key_based_branch():
    state = {
        "plan_id": "p1",
        "destination": "Tokyo",
        "itineraries": {
            "A": [{"day": "Day 1", "items": [{"key": "a1", "icon": "activity", "name": "A"}]}],
            "B": [{"day": "Day 1", "items": [{"key": "b1", "icon": "activity", "name": "B"}]}],
        },
        "replan_request": {
            "selected_option": "A",
            "itinerary_days": [{"day": "Day 1", "items": [{"key": "a1", "icon": "activity", "name": "A"}]}],
            "replace_item_keys": ["a1"],
            "locked_item_keys": [],
            "locked_item_names": [],
            "replace_item_names": ["A"],
            "round": 2,
        },
    }
    out = dr.dynamic_replan_agent(state)
    assert out["plan_id"] == "p1"
    assert out["replanner_output"]["winner_option"] == "A"
    assert out["next_node"] == "orchestrator"


def test_dynamic_replan_agent_no_plan_id(monkeypatch):
    monkeypatch.setattr(dr, "_resolve_replan_plan_id", lambda: None)
    out = dr.dynamic_replan_agent({"user_feedback": {}})
    assert "No plan found" in out["replanner_output"]["error"]


def test_dynamic_replan_agent_plan_not_found(monkeypatch):
    monkeypatch.setattr(dr, "_resolve_replan_plan_id", lambda: "p404")
    monkeypatch.setattr(dr, "load_plan", lambda db, plan_id: None)

    class _DB:
        def close(self):
            return None

    monkeypatch.setattr(dr, "SessionLocal", lambda: _DB())
    out = dr.dynamic_replan_agent({"user_feedback": {}})
    assert "not found" in out["replanner_output"]["error"]


def test_dynamic_replan_agent_feedback_success(monkeypatch):
    plan_payload = {
        "origin": "Singapore",
        "destination": "Tokyo",
        "dates": "2026-06-01 to 2026-06-05",
        "preferences": "culture",
        "debate_verdict": {"winner_option": "A"},
        "itineraries": {
            "A": [{"day": "Day 1", "items": [{"icon": "activity", "name": "Asakusa", "time": "09:00", "cost": "10"}]}],
            "B": [{"day": "Day 1", "items": [{"icon": "activity", "name": "Akiba", "time": "11:00", "cost": "20"}]}],
        },
        "option_meta": {},
    }

    monkeypatch.setattr(dr, "_resolve_replan_plan_id", lambda: "p1")
    monkeypatch.setattr(dr, "load_plan", lambda db, plan_id: deepcopy(plan_payload))
    monkeypatch.setattr(
        dr,
        "_llm_generate_alternatives",
        lambda **kwargs: [
            {"icon": "✨", "name": "Akiba", "desc": "alt1", "price": "TBD", "rating": "⭐ 4.5", "dist": "nearby", "tag": "price"},
            {"icon": "✨", "name": "Skytree", "desc": "alt2", "price": "TBD", "rating": "⭐ 4.5", "dist": "nearby", "tag": "price"},
        ],
    )
    updated = {"called": False}
    monkeypatch.setattr(dr, "update_plan_result", lambda db, pid, payload: updated.__setitem__("called", True))

    class _DB:
        def close(self):
            return None

    monkeypatch.setattr(dr, "SessionLocal", lambda: _DB())
    out = dr.dynamic_replan_agent({"user_feedback": {"current_day": "Day 1", "current_time": "10:00"}})
    assert out["plan_id"] == "p1"
    assert len(out["replanner_output"]["alternatives"]) == 2
    assert updated["called"] is True


def test_more_helper_edges():
    assert dr._extract_city_from_flight("ABC Hub -> XYZ UnknownCity | dep 2026-01-01")[1] == "UnknownCity"
    assert dr._return_day_label_from_dates({"dates": "2026-06-xx to 2026-06-05"}) == "Day 1"
    assert dr._parse_clock_minutes("~1 hour") == 60
    assert dr._parse_clock_minutes("1h") == 60
    d, m = dr._extract_day_and_time({"current_day": "Unknown", "current_time": "bad"})
    assert (d, m) == (1, 0)


def test_disallowed_item_other_reasons(replan_payload):
    rules = dr._build_rules(replan_payload)
    blocked1, reason1 = dr._is_disallowed_item({"key": "k1", "icon": "activity", "name": "Visited Place"}, rules, {"visited place"})
    blocked2, reason2 = dr._is_disallowed_item({"key": "k2", "icon": "activity", "name": "Temple Park"}, rules, set())
    blocked3, reason3 = dr._is_disallowed_item({"key": "k3", "icon": "activity", "name": "Outdoor Park"}, dr.ReplanRules(**{**rules.__dict__, "prefer_indoor": False}), set())
    assert blocked1 and reason1 == "already_visited"
    assert blocked2 and reason2 in {"indoor_preference_violation", "user_avoidance"}
    assert blocked3 and reason3 == "user_avoidance"


def test_postprocess_extra_branches():
    output = {"change_log": []}
    day = {"day": "Day 2", "items": [{"key": "f1", "icon": "flight", "name": "x", "time": "12:00"}, {"key": "flight_return", "icon": "flight", "name": "back", "time": "14:00"}]}
    dr._enforce_return_flight_buffer(output, day)
    assert output["change_log"] == []

    day2 = {"day": "Day 1", "items": [{"key": "cafe", "icon": "restaurant", "name": "Coffee Cafe", "time": "15:00"}]}
    dr._ensure_afternoon_coffee(output, day2)
    assert len(day2["items"]) == 1

    day3 = {"day": "Day 3", "items": [{"key": "x", "icon": "activity", "name": "Untimed"}]}
    dr._fix_dense_timeline_for_day(output, day3)
    assert day3["items"][0]["name"] == "Untimed"


def test_hard_rule_checks_violation_branches():
    payload = {"user_replan_request": {"updated_user_intent": {"must_keep": ["must1"], "avoid": ["badword"]}, "replan_scope": {"start_day": "Day 1", "end_day": "Day 2"}}}
    result = {"replanned_plan": {"plan": [{"day": "Day 1", "items": [{"key": "x", "name": "badword spot"}]}]}}
    hard = dr._hard_rule_checks(payload, result)
    kinds = {v["type"] for v in hard["violations"]}
    assert {"must_keep_missing", "avoid_preference_violation", "return_flight_integrity"} <= kinds


def test_verifier_report_lists_violations_and_flags():
    verifier = {
        "final_verdict": "revise",
        "hard_check": {"hard_rule_passed": False, "scope_checked": {}, "violations": [{"severity": "major", "type": "t1", "message": "m1"}]},
        "llm_judge": {"risk_flags": ["f1"], "final_recommendation": "revise", "reason": "x"},
    }
    text = dr.build_verifier_report_text({}, {"scenario_id": "s", "replanned_plan": {"plan": []}, "change_log": []}, verifier)
    assert "[major] t1: m1" in text
    assert "- f1" in text


def test_llm_judge_success_and_fallback(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    monkeypatch.setenv("REPLAN_VERIFIER_MODEL", "model-1")

    class _OkChain:
        def invoke(self, _data):
            return {"final_recommendation": "accept", "risk_flags": []}

    class _PromptOk:
        def __or__(self, _other):
            return self

        def invoke(self, data):
            return _OkChain().invoke(data)

    monkeypatch.setattr(dr.ChatPromptTemplate, "from_messages", lambda *_args, **_kwargs: _PromptOk())
    monkeypatch.setattr(dr, "ChatOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(dr, "JsonOutputParser", lambda: object())
    ok = dr._llm_judge({}, {}, {})
    assert ok["final_recommendation"] == "accept"

    class _PromptFail:
        def __or__(self, _other):
            return self

        def invoke(self, _data):
            raise RuntimeError("boom")

    monkeypatch.setattr(dr.ChatPromptTemplate, "from_messages", lambda *_args, **_kwargs: _PromptFail())
    fail = dr._llm_judge({}, {}, {})
    assert fail["final_recommendation"] == "revise"
    assert "verifier_fallback_failed:gpt-4.1" in fail["risk_flags"]


def test_resolve_plan_id_fallback_latest(monkeypatch):
    class _Query:
        def __init__(self):
            self.calls = 0

        def filter(self, *args, **kwargs):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def first(self):
            self.calls += 1
            return None if self.calls == 1 else ("latest-only",)

    class _DB:
        def __init__(self):
            self.q = _Query()

        def query(self, *_args, **_kwargs):
            return self.q

        def close(self):
            return None

    monkeypatch.setattr(dr, "SessionLocal", lambda: _DB())
    assert dr._resolve_replan_plan_id() == "latest-only"


def test_apply_planner_model_override(monkeypatch):
    class _M:
        pass

    llm_cfg = _M()
    planner = _M()
    monkeypatch.setattr(
        dr.importlib,
        "import_module",
        lambda name: llm_cfg if name == "agents.llm_config" else planner,
    )
    dr._apply_planner_model_override("gpt-z")
    assert llm_cfg.OPENAI_MODEL == "gpt-z"
    assert planner.OPENAI_MODEL == "gpt-z"


def test_resolve_planner_root_not_found(monkeypatch):
    monkeypatch.delenv("REPLAN_PLANNER_ROOT", raising=False)
    monkeypatch.setattr(dr, "_ROOT", type("P", (), {"parent": dr.Path("Z:/not-exists")})())
    with pytest.raises(FileNotFoundError):
        dr._resolve_planner_root()


def test_replan_wrapper_remove_path_value_error(monkeypatch, tmp_path):
    root = tmp_path / "planner2"
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "agent_tools.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(dr, "_resolve_planner_root", lambda: root)
    monkeypatch.setattr(dr, "_resolve_planner_model", lambda: "gpt-test")
    monkeypatch.setattr(dr, "_apply_planner_model_override", lambda model: None)
    monkeypatch.setattr(dr, "_legacy_replan", lambda payload: {"ok": True})

    class _PathList(list):
        def remove(self, x):
            raise ValueError("cannot remove")

    monkeypatch.setattr(dr.sys, "path", _PathList(dr.sys.path))
    out = dr.replan({})
    assert out["ok"] is True


def test_llm_generate_alternatives_edge_branches(monkeypatch):
    monkeypatch.setattr(dr, "DMX_API_KEY", "k")

    class _Prompt:
        def __init__(self, payload):
            self.payload = payload

        def __or__(self, _other):
            return self

        def invoke(self, _data):
            return self.payload

    monkeypatch.setattr(dr, "ChatOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(dr, "JsonOutputParser", lambda: object())
    monkeypatch.setattr(dr.ChatPromptTemplate, "from_messages", lambda *_args, **_kwargs: _Prompt({"alternatives": "bad"}))
    out1 = dr._llm_generate_alternatives(feedback={}, plan_payload={}, consumed_names=set(), candidates=[])
    assert len(out1) == 2

    monkeypatch.setattr(dr.ChatPromptTemplate, "from_messages", lambda *_args, **_kwargs: _Prompt({"alternatives": [123, {"name": "Done"}]}))
    out2 = dr._llm_generate_alternatives(feedback={}, plan_payload={}, consumed_names={"done"}, candidates=[])
    assert len(out2) == 2


def test_collect_candidates_and_ensure_two_more_edges():
    payload = {"itineraries": {"A": [{"day": "Day 1", "items": [{"icon": "hotel", "name": "H"}, {"icon": "activity", "name": ""}]}]}}
    assert dr._collect_candidate_items(payload, consumed=set()) == []

    ensured = dr._ensure_two_alternatives(cleaned=[], candidates=[], consumed_names=set())
    assert len(ensured) == 2


def test_collect_from_other_options_and_apply_key_no_replacement():
    itineraries = {
        "A": [{"day": "Day 1", "items": [{"icon": "activity", "name": "A"}]}],
        "B": [{"day": "Day 1", "items": [{"icon": "hotel", "name": "Hotel"}, {"icon": "activity", "name": "A"}]}],
    }
    assert dr._collect_replan_candidates_from_other_options(itineraries, "A", {"a"}) == []

    days = [{"day": "Day 1", "items": [{"key": "r1", "icon": "restaurant", "name": "Used"}]}]
    replanned, change_log, unresolved = dr._apply_key_based_replan(
        days=days,
        replace_item_keys={"r1"},
        locked_item_keys=set(),
        candidates=[{"name": "Used", "icon": "restaurant", "cost": "1"}, {"name": "Mismatch", "icon": "activity", "cost": "1"}],
    )
    assert replanned[0]["items"][0]["name"] == "Used"
    assert change_log == []
    assert unresolved and unresolved[0]["reason"] == "no_suitable_replacement_found"


def test_legacy_replan_smoke(monkeypatch, replan_payload):
    class _FakeTools:
        @staticmethod
        def get_tools_for_agent(_name):
            return []

    class _FakePlanner:
        @staticmethod
        def planner_from_research_1(_state, _research):
            return {
                "itineraries": {
                    "C": [
                        {
                            "day": "Day 1",
                            "items": [
                                {"key": "attraction_01_edo_tokyo_museum", "icon": "activity", "name": "Edo-Tokyo Museum", "time": "10:00"},
                                {"key": "restaurant_01_old", "icon": "restaurant", "name": "Old Sushi", "time": "12:00"},
                                {"key": "flight_return", "icon": "flight", "name": "Return", "time": "18:00"},
                            ],
                        },
                        {"day": "Day 2", "items": [{"key": "flight_return", "icon": "flight", "name": "Return2", "time": "18:00"}]},
                    ]
                }
            }

        @staticmethod
        def revise_itinerary_1(_state, _critique, planner_result):
            return planner_result

    class _FakeResearch:
        @staticmethod
        def research_agent_1(_state, _tools):
            return {
                "compact_attractions": [{"name": "Indoor Museum", "description": "indoor"}],
                "compact_restaurants": [{"name": "Veg Cafe", "description": "vegetarian"}],
            }

    monkeypatch.setitem(dr.sys.modules, "agents.agent_tools", _FakeTools)
    monkeypatch.setitem(dr.sys.modules, "agents.specialists.planner_agent_1", _FakePlanner)
    monkeypatch.setitem(dr.sys.modules, "agents.specialists.research_agent_1", _FakeResearch)
    out = dr._legacy_replan(deepcopy(replan_payload))
    assert "replanned_plan" in out
    assert "change_log" in out


def test_enforce_return_flight_buffer_adjust_and_remove():
    output = {"change_log": []}
    # adjust branch: can move previous item earlier (>=08:00)
    day_adjust = {
        "day": "Day 5",
        "items": [
            {"key": "a1", "icon": "activity", "name": "Museum", "time": "14:30"},
            {"key": "flight_return", "icon": "flight", "name": "Return", "time": "16:00"},
        ],
    }
    dr._enforce_return_flight_buffer(output, day_adjust)
    assert day_adjust["items"][0]["time"] == "13:00"
    assert any("return_flight_buffer" in x["reason"] for x in output["change_log"])

    # remove branch: target time falls before 08:00 so remove previous activity
    day_remove = {
        "day": "Day 6",
        "items": [
            {"key": "a2", "icon": "activity", "name": "Early Spot", "time": "07:30"},
            {"key": "flight_return", "icon": "flight", "name": "Return", "time": "09:00"},
        ],
    }
    dr._enforce_return_flight_buffer(output, day_remove)
    keys = [x["key"] for x in day_remove["items"]]
    assert "a2" not in keys
    assert any(x["reason"] == "insufficient_return_flight_buffer" for x in output["change_log"])
