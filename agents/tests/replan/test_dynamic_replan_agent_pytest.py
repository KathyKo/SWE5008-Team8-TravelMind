from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "specialists" / "dynamic_replan_agent.py"

    if "langchain_openai" not in sys.modules:
        lc_openai = types.ModuleType("langchain_openai")

        class _ChatOpenAI:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

            def __or__(self, _other):
                return self

            def invoke(self, _data):
                return {"alternatives": []}

        lc_openai.ChatOpenAI = _ChatOpenAI
        sys.modules["langchain_openai"] = lc_openai

    if "langchain_core.prompts" not in sys.modules:
        lc_prompts = types.ModuleType("langchain_core.prompts")

        class _Pipe:
            def __or__(self, _other):
                return self

            def invoke(self, _data):
                return {}

        class _ChatPromptTemplate:
            @staticmethod
            def from_messages(_messages):
                return _Pipe()

        lc_prompts.ChatPromptTemplate = _ChatPromptTemplate
        sys.modules["langchain_core.prompts"] = lc_prompts

    if "langchain_core.output_parsers" not in sys.modules:
        lc_parsers = types.ModuleType("langchain_core.output_parsers")

        class _JsonOutputParser:
            pass

        lc_parsers.JsonOutputParser = _JsonOutputParser
        sys.modules["langchain_core.output_parsers"] = lc_parsers

    if "agents" not in sys.modules:
        agents_pkg = types.ModuleType("agents")
        agents_pkg.__path__ = [str(module_path.parents[1])]
        sys.modules["agents"] = agents_pkg

    if "agents.specialists" not in sys.modules:
        specialists_pkg = types.ModuleType("agents.specialists")
        specialists_pkg.__path__ = [str(module_path.parent)]
        sys.modules["agents.specialists"] = specialists_pkg

    if "agents.db" not in sys.modules:
        db_pkg = types.ModuleType("agents.db")
        db_pkg.__path__ = [str(module_path.parents[1] / "db")]
        sys.modules["agents.db"] = db_pkg

    if "agents.db.crud" not in sys.modules:
        crud_mod = types.ModuleType("agents.db.crud")
        crud_mod.load_plan = lambda _db, _plan_id: None
        crud_mod.update_plan_result = lambda _db, _plan_id, _revised: None
        sys.modules["agents.db.crud"] = crud_mod

    if "agents.db.database" not in sys.modules:
        db_mod = types.ModuleType("agents.db.database")

        class _DB:
            def close(self):
                pass

        db_mod.SessionLocal = lambda: _DB()
        sys.modules["agents.db.database"] = db_mod

    if "agents.db.models" not in sys.modules:
        models_mod = types.ModuleType("agents.db.models")

        class _Expr:
            def is_not(self, _v):
                return self

            def desc(self):
                return self

        class _Plan:
            plan_id = _Expr()
            debate_verdict = _Expr()
            created_at = _Expr()

        models_mod.Plan = _Plan
        sys.modules["agents.db.models"] = models_mod

    if "agents.llm_config" not in sys.modules:
        llm_mod = types.ModuleType("agents.llm_config")
        llm_mod.DMX_API_KEY = "k"
        llm_mod.DMX_BASE_URL = "https://example.com/v1"
        llm_mod.PLANNER_MODEL = "gpt-4.1"
        sys.modules["agents.llm_config"] = llm_mod

    module_name = "agents.specialists.dynamic_replan_agent"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "agents.specialists"
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


dr = _load_module()


class _FakeDB:
    def __init__(self, answers=None):
        self.answers = answers or [None, None]
        self.idx = 0

    def query(self, _x):
        return self

    def filter(self, _x):
        return self

    def order_by(self, _x):
        return self

    def first(self):
        if self.idx >= len(self.answers):
            return None
        v = self.answers[self.idx]
        self.idx += 1
        return v

    def close(self):
        pass


@pytest.fixture
def sample_payload():
    return {
        "scenario_id": "s1",
        "source": {"kind": "frontend"},
        "original_recommended_plan": {
            "option_key": "A",
            "label": "Recommended",
            "itinerary_id": "IT1",
            "composite_score": 88.8,
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"time": "09:00", "icon": "activity", "key": "old_visited", "name": "Old Museum", "cost": "SGD 10"},
                        {"time": "12:00", "icon": "restaurant", "key": "old_lunch", "name": "Veg Lunch", "cost": "SGD 15"},
                        {"time": "19:00", "icon": "hotel", "key": "hotel_1", "name": "Hotel A", "cost": "SGD 120"},
                    ],
                },
                {
                    "day": "Day 2",
                    "items": [
                        {"time": "10:00", "icon": "activity", "key": "to_close", "name": "Edo Tokyo Museum", "cost": "SGD 12"},
                        {"time": "13:00", "icon": "restaurant", "key": "lunch_2", "name": "Midday Meal", "cost": "SGD 18"},
                    ],
                },
                {
                    "day": "Day 3",
                    "items": [
                        {"time": "11:00", "icon": "activity", "key": "act_3", "name": "City Walk", "cost": "Free"},
                        {"time": "18:00", "icon": "flight", "key": "flight_return", "name": "Tokyo -> Singapore | dep 2026-06-05", "cost": "SGD 400"},
                    ],
                },
            ],
        },
        "user_replan_request": {
            "replan_scope": {
                "start_day": "Day 2",
                "end_day": "Day 3",
                "locked_days": [],
                "allow_replace_flight": False,
                "allow_replace_hotel": False,
            },
            "updated_user_intent": {
                "new_preferences": ["indoor", "anime"],
                "avoid": ["temple", "outdoor"],
                "must_keep": ["flight_return"],
                "meal_preference": "vegetarian",
                "pace_preference": "relaxed",
                "budget_guardrail": {"currency": "SGD"},
            },
            "trigger_events": [
                {"type": "venue_closure", "detail": "Edo Tokyo Museum closed", "affected_item_key": "to_close"},
                {"type": "weather", "detail": "rain expected"},
            ],
            "output_expectation": {"count": 2},
        },
    }


def test_basic_helpers_and_io(tmp_path):
    p = tmp_path / "x.json"
    dr.save_json(p, {"a": 1})
    assert dr.load_json(p)["a"] == 1
    assert dr._slug("Hello World!") == "hello_world"
    assert dr._norm_name({"name": "  X "}) == "x"
    assert dr._is_activity_or_restaurant({"icon": "activity"}) is True
    assert dr._contains_any("hello world", {"wo"}) is True
    assert dr._ascii_fold("café") == "cafe"
    assert dr._is_temple_or_shrine_activity("Senso-ji temple") is True
    assert dr._normalize_match_text("Edo-Tokyo Museum!") == "edo tokyo museum"
    assert dr._extract_city_from_flight("Singapore Changi Airport -> Narita Airport") == ("Singapore", "Tokyo")
    assert dr._extract_city_from_flight("Foo City Airport -> Bar Town Airport") == ("City", "Town")
    assert dr._extract_date_from_flight("dep 2026-06-01", "dep") == "2026-06-01"
    assert dr._day_num("Day 12") == 12
    assert dr._parse_time_to_min("10:30") == 630
    assert dr._parse_time_to_min("bad") is None
    assert dr._parse_time_to_min("25:99") is None
    assert dr._fmt_min_to_time(9999) == "23:59"
    assert dr._norm("  A   b  ") == "a b"


def test_builders_and_candidate_helpers(sample_payload):
    directive = dr._build_llm_replan_directive(sample_payload)
    assert "Dynamic replan instruction" in directive
    ctx = dr._build_structured_replan_context(sample_payload)
    assert "hard_rules" in ctx and "soft_rules" in ctx
    rules = dr._build_rules(sample_payload)
    assert rules.start_day_num == 2
    assert rules.vegetarian_friendly is True
    state = dr._build_state_from_input(sample_payload)
    assert "replan_directive" in state
    assert "structured_context" in state["preferences"]
    assert dr._prefer_item({"name": "Anime indoor museum"}, indoor=True, vegetarian=False) is True
    assert dr._prefer_item({"name": "Plain Park"}, indoor=False, vegetarian=False) is False
    item = dr._candidate_to_itinerary_item({"name": "My Place"}, "activity", "14:00", 1)
    assert item["key"].startswith("attraction_replan_1_")
    pick = dr._pick_candidate(
        [{"name": "Temple Visit", "description": "temple"}, {"name": "Anime Hall", "description": "indoor anime"}],
        used_names=set(),
        avoid_keywords={"outdoor"},
        indoor=True,
        vegetarian=False,
    )
    assert pick and pick["name"] == "Anime Hall"


def test_disallow_return_day_scope_and_postprocess_helpers(sample_payload):
    rules = dr._build_rules(sample_payload)
    blocked, reason = dr._is_disallowed_item({"key": "to_close", "icon": "activity", "name": "Edo Tokyo Museum"}, rules, {"old museum"})
    assert blocked is True and reason in {"venue_closed", "user_avoidance", "indoor_preference_violation"}
    label = dr._return_day_label_from_dates({"dates": "2026-06-01 to 2026-06-05"})
    assert label == "Day 5"
    assert dr._scope_day_numbers(sample_payload) == (2, 3)

    out = {"replanned_plan": {"plan": [{"day": "Day 2", "items": [{"time": "10:00", "icon": "activity", "key": "a", "name": "A"}, {"time": "10:30", "icon": "restaurant", "key": "r", "name": "R"}, {"time": "18:00", "icon": "flight", "key": "flight_return", "name": "F"}]}]}}
    day = out["replanned_plan"]["plan"][0]
    dr._fix_dense_timeline_for_day(out, day)
    dr._ensure_afternoon_coffee(out, day)
    dr._enforce_return_flight_buffer(out, day)
    assert "change_log" in out


def test_legacy_replan_core_flow(monkeypatch, sample_payload):
    fake_tools = types.ModuleType("agents.agent_tools")
    fake_tools.get_tools_for_agent = lambda _name: []
    sys.modules["agents.agent_tools"] = fake_tools

    fake_planner = types.ModuleType("agents.specialists.planner_agent_1")
    fake_planner.planner_from_research_1 = lambda _state, _research: {
        "itineraries": {
            "C": [
                {"day": "Day 1", "items": [{"time": "09:00", "icon": "activity", "key": "keep", "name": "Keep A"}]},
                {"day": "Day 2", "items": [{"time": "10:00", "icon": "activity", "key": "to_close", "name": "Edo Tokyo Museum"}]},
                {"day": "Day 3", "items": [{"time": "18:00", "icon": "flight", "key": "flight_return", "name": "Tokyo -> Singapore | dep 2026-06-05"}]},
            ]
        }
    }
    fake_planner.revise_itinerary_1 = lambda _state, _critique, current: current
    sys.modules["agents.specialists.planner_agent_1"] = fake_planner

    fake_research = types.ModuleType("agents.specialists.research_agent_1")
    fake_research.research_agent_1 = lambda _state, _tools: {
        "compact_attractions": [{"name": "Indoor Anime Center", "description": "indoor anime museum"}],
        "compact_restaurants": [{"name": "Vegan Cafe", "description": "vegetarian"}],
    }
    sys.modules["agents.specialists.research_agent_1"] = fake_research

    out = dr._legacy_replan(sample_payload)
    assert "replanned_plan" in out
    assert out["applied_rules"]["start_day"] == 2
    assert isinstance(out["change_log"], list)

    with pytest.raises(ValueError):
        dr._legacy_replan({"original_recommended_plan": {"plan": []}})


def test_replan_wrapper_and_model_utils(monkeypatch, sample_payload):
    monkeypatch.setenv("REPLAN_PLANNER_MODEL", "m1")
    assert dr._resolve_planner_model() == "m1"

    monkeypatch.setattr(dr, "_resolve_planner_root", lambda: Path("D:/fake-root"))
    monkeypatch.setattr(dr, "_apply_planner_model_override", lambda _m: None)
    monkeypatch.setattr(dr, "_legacy_replan", lambda payload: {"ok": True, "scenario_id": payload.get("scenario_id")})
    out = dr.replan(sample_payload)
    assert out["ok"] is True


def test_apply_planner_model_override(monkeypatch):
    llm_cfg = types.SimpleNamespace()
    planner_mod = types.SimpleNamespace()

    def _fake_import(name):
        if name == "agents.llm_config":
            return llm_cfg
        if name == "agents.specialists.planner_agent_1":
            return planner_mod
        raise RuntimeError("unexpected import")

    monkeypatch.setattr(dr.importlib, "import_module", _fake_import)
    dr._apply_planner_model_override("gpt-x")
    assert getattr(llm_cfg, "OPENAI_MODEL") == "gpt-x"
    assert getattr(planner_mod, "OPENAI_MODEL") == "gpt-x"


def test_resolve_planner_root_paths(monkeypatch, tmp_path):
    ok_root = tmp_path / "ok"
    (ok_root / "agents").mkdir(parents=True)
    (ok_root / "agents" / "agent_tools.py").write_text("x=1", encoding="utf-8")
    monkeypatch.setenv("REPLAN_PLANNER_ROOT", str(ok_root))
    assert dr._resolve_planner_root() == ok_root.resolve()

    monkeypatch.setenv("REPLAN_PLANNER_ROOT", str(tmp_path / "missing"))
    monkeypatch.setattr(dr, "_ROOT", tmp_path / "sandbox" / "inner")
    with pytest.raises(FileNotFoundError):
        dr._resolve_planner_root()


def test_postprocess_verify_and_report(monkeypatch):
    payload = {
        "user_replan_request": {
            "replan_scope": {"start_day": "Day 2", "end_day": "Day 3"},
            "updated_user_intent": {"avoid": ["forbidden"], "must_keep": ["flight_return"]},
        }
    }
    result = {
        "scenario_id": "s1",
        "replanned_plan": {
            "plan": [
                {"day": "Day 2", "items": [{"time": "10:00", "icon": "activity", "key": "k1", "name": "A forbidden"}]},
                {"day": "Day 3", "items": [{"time": "18:00", "icon": "flight", "key": "flight_return", "name": "R"}]},
            ]
        },
        "change_log": [],
    }

    fixed = dr.postprocess_replan_output(payload, result)
    assert fixed["postprocess_summary"]["enabled"] is True
    hard = dr._hard_rule_checks(payload, fixed)
    assert "hard_rule_passed" in hard

    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = dr._llm_judge(payload, fixed, hard)
    assert llm["final_recommendation"] == "revise"

    verifier = dr.verify_replan(payload, fixed)
    text = dr.build_verifier_report_text(payload, fixed, verifier)
    assert "Dynamic Replan Verifier Report" in text

    clean_report = dr.build_verifier_report_text(
        payload,
        {"scenario_id": "s2", "replanned_plan": {"plan": []}, "change_log": []},
        {
            "final_verdict": "accept",
            "hard_check": {"hard_rule_passed": True, "scope_checked": {}, "violations": []},
            "llm_judge": {"risk_flags": [], "final_recommendation": "accept", "reason": "ok"},
        },
    )
    assert "violations: none" in clean_report
    assert "risk_flags: none" in clean_report


def test_llm_judge_success_path(monkeypatch):
    class _Pipe:
        def __or__(self, _other):
            return self

        def invoke(self, _data):
            return {
                "preference_alignment_score": 90,
                "feasibility_score": 88,
                "constraint_compliance_score": 92,
                "quality_confidence_score": 85,
                "risk_flags": [],
                "final_recommendation": "accept",
                "reason": "ok",
            }

    monkeypatch.setenv("JUDGE_API_KEY", "k")
    monkeypatch.setattr(dr, "ChatPromptTemplate", types.SimpleNamespace(from_messages=lambda _m: _Pipe()))
    monkeypatch.setattr(dr, "ChatOpenAI", lambda **_k: _Pipe())
    monkeypatch.setattr(dr, "JsonOutputParser", lambda: _Pipe())
    judged = dr._llm_judge({"user_replan_request": {}}, {"replanned_plan": {"plan": []}}, {"hard_rule_passed": True})
    assert judged["final_recommendation"] == "accept"
    assert judged["_verifier_model_used"] in {"gpt-5-mini-2025-08-07", "gpt-4.1"}


def test_llm_judge_fallback_and_failure(monkeypatch):
    class _FailOnceFactory:
        def __init__(self):
            self.calls = 0

        def __call__(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first model unavailable")

            class _Pipe:
                def __or__(self, _other):
                    return self

                def invoke(self, _data):
                    return {
                        "preference_alignment_score": 70,
                        "feasibility_score": 70,
                        "constraint_compliance_score": 70,
                        "quality_confidence_score": 70,
                        "risk_flags": [],
                        "final_recommendation": "accept",
                        "reason": "fallback ok",
                    }

            return _Pipe()

    class _Prompt:
        def __or__(self, _other):
            return _other

    monkeypatch.setenv("JUDGE_API_KEY", "k")
    monkeypatch.setattr(dr, "ChatPromptTemplate", types.SimpleNamespace(from_messages=lambda _m: _Prompt()))
    monkeypatch.setattr(dr, "JsonOutputParser", lambda: object())
    monkeypatch.setattr(dr, "ChatOpenAI", _FailOnceFactory())
    judged = dr._llm_judge({"user_replan_request": {}}, {"replanned_plan": {"plan": []}}, {"hard_rule_passed": True})
    assert any("model_fallback_used" in x for x in judged.get("risk_flags", []))

    class _AlwaysFail:
        def __call__(self, **_kwargs):
            raise RuntimeError("all down")

    monkeypatch.setattr(dr, "ChatOpenAI", _AlwaysFail())
    failed = dr._llm_judge({"user_replan_request": {}}, {"replanned_plan": {"plan": []}}, {"hard_rule_passed": True})
    assert failed["_verifier_model_used"] == "none"
    assert failed["final_recommendation"] == "revise"


def test_replan_plan_resolution_and_main_agent_paths(monkeypatch):
    monkeypatch.setattr(dr, "SessionLocal", lambda: _FakeDB([("p1",), ("p2",)]))
    assert dr._resolve_replan_plan_id() == "p1"
    monkeypatch.setattr(dr, "SessionLocal", lambda: _FakeDB([None, ("p2",)]))
    assert dr._resolve_replan_plan_id() == "p2"
    monkeypatch.setattr(dr, "SessionLocal", lambda: _FakeDB([None, None]))
    assert dr._resolve_replan_plan_id() is None

    assert dr._parse_clock_minutes("rest of day") == 1439
    assert dr._parse_clock_minutes("2–3 hours") == 180
    assert dr._parse_clock_minutes("~1 hour") == 60
    assert dr._parse_clock_minutes("bad text") is None
    assert dr._extract_day_and_time({"day": "Day 4", "time": "11:20"}) == (4, 680)

    payload = {"debate_verdict": {"winner_option": "B"}, "itineraries": {"A": [], "B": [{"day": "Day 1", "items": []}]}}
    assert dr._winner_option_from_plan(payload) == "B"
    assert dr._winner_option_from_plan({"debate_verdict": {"winner_option": "Z"}, "itineraries": {"A": []}}) == "A"

    days = [{"day": "Day 1", "items": [{"time": "09:00", "icon": "activity", "name": "A"}]}]
    consumed = dr._collect_consumed_items(days, 1, 600)
    assert "a" in consumed

    plan_payload = {"itineraries": {"A": [{"day": "Day 1", "items": [{"time": "12:00", "icon": "activity", "name": "X"}, {"time": "18:00", "icon": "flight", "name": "F"}]}]}}
    cands = dr._collect_candidate_items(plan_payload, {"a"})
    assert len(cands) == 1
    fb = dr._fallback_alternatives([{"name": "One", "icon": "✨", "desc": "d"}])
    assert len(fb) == 2
    ensured = dr._ensure_two_alternatives([{"name": "X", "icon": "✨", "desc": "", "price": "TBD", "rating": "⭐ 4.5", "dist": "nearby", "tag": "price"}], cands, set())
    assert len(ensured) == 2

    monkeypatch.setattr(dr, "DMX_API_KEY", "")
    assert len(dr._llm_generate_alternatives(feedback={}, plan_payload={}, consumed_names=set(), candidates=cands)) == 2

    monkeypatch.setattr(dr, "_resolve_replan_plan_id", lambda: None)
    out = dr.dynamic_replan_agent({})
    assert "No plan found" in out["replanner_output"]["error"]

    monkeypatch.setattr(dr, "_resolve_replan_plan_id", lambda: "p1")
    monkeypatch.setattr(dr, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(dr, "load_plan", lambda _db, _pid: None)
    out2 = dr.dynamic_replan_agent({"user_feedback": {}})
    assert "not found" in out2["replanner_output"]["error"]


def test_dynamic_replan_agent_full_llm_generation(monkeypatch):
    class _Pipe:
        def __or__(self, _other):
            return self

        def invoke(self, _data):
            return {
                "alternatives": [
                    {"icon": "✨", "name": "Used", "desc": "dup", "price": "x", "rating": "⭐4.0", "dist": "1m", "tag": "price"},
                    {"icon": "🍜", "name": "Fresh B", "desc": "ok", "price": "SGD 20", "rating": "⭐4.7", "dist": "200m", "tag": "free"},
                ]
            }

    plan_payload = {
        "origin": "Singapore",
        "destination": "Tokyo",
        "dates": "2026-06-01 to 2026-06-05",
        "preferences": "food",
        "itineraries": {
            "A": [{"day": "Day 1", "items": [{"time": "10:00", "icon": "activity", "name": "Used"}]}],
            "B": [{"day": "Day 2", "items": [{"time": "12:00", "icon": "activity", "name": "Fresh A"}]}],
        },
        "debate_verdict": {"winner_option": "A"},
        "option_meta": {},
        "flight_options_outbound": [{"name": "F1"}],
        "flight_options_return": [{"name": "F2"}],
        "hotel_options": [{"name": "H1"}],
    }
    monkeypatch.setattr(dr, "DMX_API_KEY", "k")
    monkeypatch.setattr(dr, "ChatPromptTemplate", types.SimpleNamespace(from_messages=lambda _m: _Pipe()))
    monkeypatch.setattr(dr, "ChatOpenAI", lambda **_k: _Pipe())
    monkeypatch.setattr(dr, "JsonOutputParser", lambda: _Pipe())
    monkeypatch.setattr(dr, "_resolve_replan_plan_id", lambda: "p3")
    monkeypatch.setattr(dr, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(dr, "load_plan", lambda _db, _pid: plan_payload)
    monkeypatch.setattr(dr, "update_plan_result", lambda _db, _pid, _revised: None)
    out = dr.dynamic_replan_agent({"user_feedback": {"day": "Day 1", "time": "11:00"}})
    alts = out["replanner_output"]["alternatives"]
    assert len(alts) == 2
    assert all(a["name"].lower() != "used" for a in alts)


def test_dynamic_replan_agent_success_and_llm_cleaning(monkeypatch):
    captured = {}
    plan_payload = {
        "origin": "Singapore",
        "destination": "Tokyo",
        "dates": "2026-06-01 to 2026-06-05",
        "preferences": "culture",
        "itineraries": {
            "A": [{"day": "Day 1", "items": [{"time": "10:00", "icon": "activity", "name": "Used"}]}],
            "B": [{"day": "Day 2", "items": [{"time": "14:00", "icon": "activity", "name": "New Place"}]}],
        },
        "debate_verdict": {"winner_option": "A"},
        "option_meta": {},
        "flight_options_outbound": [],
        "flight_options_return": [],
        "hotel_options": [],
    }

    monkeypatch.setattr(dr, "_resolve_replan_plan_id", lambda: "p2")
    monkeypatch.setattr(dr, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(dr, "load_plan", lambda _db, _pid: plan_payload)
    monkeypatch.setattr(dr, "update_plan_result", lambda _db, pid, revised: captured.update({"pid": pid, "revised": revised}))
    monkeypatch.setattr(
        dr,
        "_llm_generate_alternatives",
        lambda **_k: [
            {"icon": "✨", "name": "New Place", "desc": "d1", "price": "TBD", "rating": "⭐ 4.7", "dist": "200m", "tag": "price"},
            {"icon": "🍵", "name": "Tea House", "desc": "d2", "price": "SGD 15", "rating": "⭐ 4.8", "dist": "300m", "tag": "free"},
        ],
    )

    out = dr.dynamic_replan_agent({"user_feedback": {"day": "Day 1", "time": "11:00", "current_location": "Kyoto"}})
    assert out["plan_id"] == "p2"
    assert len(out["replanner_output"]["alternatives"]) == 2
    assert captured["pid"] == "p2"
    assert "replan_last_output" in captured["revised"]["option_meta"]

    class _Pipe:
        def __or__(self, _other):
            return self

        def invoke(self, _data):
            return {"alternatives": [{"name": "Used"}, {"name": "Fresh One"}]}

    monkeypatch.setattr(dr, "ChatPromptTemplate", types.SimpleNamespace(from_messages=lambda _m: _Pipe()))
    monkeypatch.setattr(dr, "ChatOpenAI", lambda **_k: _Pipe())
    monkeypatch.setattr(dr, "JsonOutputParser", lambda: _Pipe())
    monkeypatch.setattr(dr, "DMX_API_KEY", "k")
    cleaned = dr._llm_generate_alternatives(
        feedback={},
        plan_payload={},
        consumed_names={"used"},
        candidates=[{"name": "Fresh One", "icon": "✨", "desc": "d"}],
    )
    assert len(cleaned) == 2


def test_postprocess_branches_for_coffee_and_return_buffer():
    output = {
        "replanned_plan": {
            "plan": [
                {
                    "day": "Day 2",
                    "items": [
                        {"time": "15:30", "icon": "restaurant", "key": "coffee", "name": "Coffee Bar"},
                        {"time": "12:00", "icon": "flight", "key": "flight_x", "name": "X"},
                        {"time": "18:00", "icon": "flight", "key": "flight_return", "name": "Return"},
                    ],
                },
                {
                    "day": "Day 3",
                    "items": [
                        {"time": "09:00", "icon": "activity", "key": "a1", "name": "A1"},
                        {"time": "14:00", "icon": "activity", "key": "a2", "name": "A2"},
                        {"time": "18:00", "icon": "flight", "key": "flight_return", "name": "Return"},
                    ],
                },
            ]
        },
        "change_log": [],
    }
    day2 = output["replanned_plan"]["plan"][0]
    dr._ensure_afternoon_coffee(output, day2)  # early return branch
    before_len = len(day2["items"])
    dr._enforce_return_flight_buffer(output, day2)  # no prev non-flight branch
    assert len(day2["items"]) == before_len

    day3 = output["replanned_plan"]["plan"][1]
    dr._enforce_return_flight_buffer(output, day3)  # already enough buffer branch
    assert any(i["key"] == "a2" for i in day3["items"])

