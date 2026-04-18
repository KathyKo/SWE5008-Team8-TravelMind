from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_debate_agent_module():
    module_path = Path(__file__).resolve().parents[2] / "specialists" / "debate_agent.py"

    if "langchain_openai" not in sys.modules:
        lc_openai = types.ModuleType("langchain_openai")

        class _ChatOpenAI:
            def __init__(self, *args, **kwargs):
                pass

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

    if "agents.specialists.planner_agent" not in sys.modules:
        planner_mod = types.ModuleType("agents.specialists.planner_agent")
        planner_mod.revise_itinerary = lambda _state, _critique, _current: {}
        sys.modules["agents.specialists.planner_agent"] = planner_mod

    if "agents.llm_config" not in sys.modules:
        llm_cfg = types.ModuleType("agents.llm_config")
        llm_cfg.DEBATE_MODEL = "gpt-4.1"
        llm_cfg.JUDGE_MODEL = "gpt-5-mini"
        llm_cfg.DMX_BASE_URL = "https://example.com/v1"
        llm_cfg.DMX_API_KEY = "k"
        sys.modules["agents.llm_config"] = llm_cfg

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
            def is_(self, _v):
                return self

            def desc(self):
                return self

        class _Plan:
            plan_id = _Expr()
            debate_verdict = _Expr()
            created_at = _Expr()

        models_mod.Plan = _Plan
        sys.modules["agents.db.models"] = models_mod

    module_name = "agents.specialists.debate_agent"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "agents.specialists"
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


da = _load_debate_agent_module()


class _FakeDB:
    def close(self):
        pass


@pytest.fixture
def base_plan():
    return {
        "origin": "Singapore",
        "destination": "Tokyo",
        "dates": "2026-06-01 to 2026-06-05",
        "duration": "5 days",
        "budget": "SGD 5000",
        "preferences": "culture, food",
        "itineraries": {"A": [{"day": "Day 1"}], "B": [{"day": "Day 1"}], "C": [{"day": "Day 1"}]},
        "option_meta": {"A": {"label": "A"}, "B": {"label": "B"}, "C": {"label": "C"}},
        "flight_options_outbound": [],
        "flight_options_return": [],
        "hotel_options": [],
        "debate_history": [],
        "debate_verdict": None,
    }


def test_helpers_basic_coverage():
    assert da._extract_duration_days(None) is None
    assert da._extract_duration_days(5) == 5
    assert da._extract_duration_days("7 days") == 7

    scores = da._safe_round_dim_scores(
        {
            "dimension_scores": {
                "A": {"bias_fairness": 101, "logistics": "x", "preference_alignment": -4, "option_diversity": 50},
                "B": {"bias_fairness": 80, "logistics": 90, "preference_alignment": 95, "option_diversity": 85},
            }
        }
    )
    assert scores["A"]["bias_fairness"] == 100.0
    assert scores["A"]["logistics"] == 70.0
    assert scores["A"]["preference_alignment"] == 0.0
    assert "composite" in scores["B"]

    assert da._winner_by_scores({"A": {"composite": 88.0}, "B": {"composite": 91.0}}) == "B"
    assert da._round_from_history([{"sender": "agent4_critic"}, {"sender": "agent3_response"}]) == 2

    merged = da._merge_revised_plan({"itineraries": {"A": []}, "x": 1}, {"itineraries": {"B": []}, "tool_log": ["ok"]})
    assert merged["itineraries"] == {"B": []}
    assert merged["tool_log"] == ["ok"]
    assert merged["x"] == 1


def test_llm_factory_and_score_skip_non_dict(monkeypatch):
    # hit _debate_llm / _judge_llm constructor paths
    _ = da._debate_llm()
    _ = da._judge_llm()

    # hit non-dict option score branch
    scores = da._safe_round_dim_scores({"dimension_scores": {"A": "bad", "B": {"bias_fairness": 88}}})
    assert "A" not in scores
    assert "B" in scores


def test_prompt_builders(monkeypatch, base_plan):
    class _Pipe:
        def __init__(self, payload):
            self.payload = payload

        def __or__(self, _other):
            return self

        def invoke(self, _data):
            return self.payload

    monkeypatch.setattr(
        da.ChatPromptTemplate,
        "from_messages",
        lambda _m: _Pipe(
            {
                "round_decision": "continue",
                "winner_option": None,
                "winner_reason": "",
                "critique_summary": "x",
                "dimension_scores": {"A": {"bias_fairness": 80, "logistics": 82, "preference_alignment": 84, "option_diversity": 86}},
            }
        ),
    )
    monkeypatch.setattr(da, "_debate_llm", lambda **_k: object())
    monkeypatch.setattr(da, "_judge_llm", lambda **_k: object())
    monkeypatch.setattr(da, "JsonOutputParser", lambda: object())

    round_payload = da._build_round_critique_payload(base_plan, [], 1)
    judge_payload = da._build_judge_payload(base_plan, [])
    assert "dimension_scores" in round_payload
    assert "composite" in round_payload["dimension_scores"]["A"]
    assert "dimension_scores" in judge_payload


def test_persist_plan_debate_calls_update(monkeypatch, base_plan):
    captured = {}
    monkeypatch.setattr(da, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(
        da,
        "update_plan_result",
        lambda _db, plan_id, revised: captured.update({"plan_id": plan_id, "revised": revised}),
    )

    da._persist_plan_debate(
        plan_id="p1",
        plan_payload=base_plan,
        debate_history=[{"sender": "agent4_critic"}],
        debate_verdict={"winner_option": "A"},
    )
    assert captured["plan_id"] == "p1"
    assert captured["revised"]["debate_verdict"]["winner_option"] == "A"


def test_resolve_active_plan_id_paths(monkeypatch):
    class _Expr:
        def is_(self, _v):
            return self

        def desc(self):
            return self

    monkeypatch.setattr(da, "Plan", types.SimpleNamespace(plan_id=_Expr(), debate_verdict=_Expr(), created_at=_Expr()))

    class _Q:
        def __init__(self, answers):
            self.answers = answers
            self.idx = 0

        def filter(self, _x):
            return self

        def order_by(self, _x):
            return self

        def first(self):
            v = self.answers[self.idx]
            self.idx += 1
            return v

    class _DB:
        def __init__(self, answers):
            self.q = _Q(answers)

        def query(self, _x):
            return self.q

        def close(self):
            pass

    monkeypatch.setattr(da, "SessionLocal", lambda: _DB([("p1",), ("p2",)]))
    assert da._resolve_active_plan_id() == "p1"

    monkeypatch.setattr(da, "SessionLocal", lambda: _DB([None, ("p2",)]))
    assert da._resolve_active_plan_id() == "p2"

    monkeypatch.setattr(da, "SessionLocal", lambda: _DB([None, None]))
    assert da._resolve_active_plan_id() is None


def test_debate_agent_no_plan_id(monkeypatch):
    monkeypatch.setattr(da, "_resolve_active_plan_id", lambda: None)
    out = da.debate_agent({})
    assert out["is_valid"] is False
    assert out["debate_count"] == 0
    assert "No plan found" in out["debate_output"]["error"]


def test_debate_agent_plan_not_found(monkeypatch):
    monkeypatch.setattr(da, "_resolve_active_plan_id", lambda: "p1")
    monkeypatch.setattr(da, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(da, "load_plan", lambda _db, _pid: None)
    out = da.debate_agent({})
    assert out["plan_id"] == "p1"
    assert out["is_valid"] is False
    assert out["debate_count"] == 0


def test_debate_agent_critique_exception_branch(monkeypatch, base_plan):
    captured = {}
    monkeypatch.setattr(da, "_resolve_active_plan_id", lambda: "p1")
    monkeypatch.setattr(da, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(da, "load_plan", lambda _db, _pid: base_plan)
    monkeypatch.setattr(da, "_build_round_critique_payload", lambda _p, _h, _r: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(da, "revise_itinerary", lambda _s, _c, _p: {"error": "planner failed"})
    monkeypatch.setattr(da, "_persist_plan_debate", lambda **kwargs: captured.update(kwargs))

    out = da.debate_agent({})
    assert out["is_valid"] is False
    assert "Round critique failed" in out["debate_output"]["current_round_summary"]
    assert captured["debate_verdict"] is None


def test_debate_agent_rounds_already_completed(monkeypatch, base_plan):
    history = [{"sender": "agent4_critic"}, {"sender": "agent4_critic"}, {"sender": "agent4_critic"}]
    payload = {**base_plan, "debate_history": history, "debate_verdict": {"winner_option": "B"}}
    monkeypatch.setattr(da, "_resolve_active_plan_id", lambda: "p1")
    monkeypatch.setattr(da, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(da, "load_plan", lambda _db, _pid: payload)
    out = da.debate_agent({})
    assert out["is_valid"] is True
    assert out["debate_count"] == da.MAX_DEBATE_ROUNDS


def test_debate_agent_immediate_win(monkeypatch, base_plan):
    captured = {}

    monkeypatch.setattr(da, "_resolve_active_plan_id", lambda: "p1")
    monkeypatch.setattr(da, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(da, "load_plan", lambda _db, _pid: base_plan)
    monkeypatch.setattr(
        da,
        "_build_round_critique_payload",
        lambda _plan, _h, _r: {
            "round_decision": "decide",
            "winner_option": "A",
            "winner_reason": "A wins",
            "critique_summary": "done",
            "dimension_scores": {"A": {"composite": 92}},
        },
    )
    monkeypatch.setattr(da, "_persist_plan_debate", lambda **kwargs: captured.update(kwargs))
    out = da.debate_agent({})
    assert out["is_valid"] is True
    assert out["debate_count"] == 1
    assert captured["debate_verdict"]["winner_option"] == "A"
    assert captured["debate_verdict"]["via_judge"] is False


def test_debate_agent_continue_with_revision_success(monkeypatch, base_plan):
    captured = {}
    monkeypatch.setattr(da, "_resolve_active_plan_id", lambda: "p1")
    monkeypatch.setattr(da, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(da, "load_plan", lambda _db, _pid: base_plan)
    monkeypatch.setattr(
        da,
        "_build_round_critique_payload",
        lambda _plan, _h, _r: {
            "round_decision": "continue",
            "winner_option": None,
            "winner_reason": "",
            "critique_summary": "please revise",
            "dimension_scores": {"A": {"composite": 80}},
        },
    )
    monkeypatch.setattr(
        da,
        "revise_itinerary",
        lambda _s, _c, _p: {"itineraries": {"A": [{"day": "Day 1 revised"}]}, "planner_chain_of_thought": "revised"},
    )
    monkeypatch.setattr(da, "_persist_plan_debate", lambda **kwargs: captured.update(kwargs))

    out = da.debate_agent({})
    assert out["is_valid"] is False
    assert out["debate_count"] == 1
    assert captured["debate_verdict"] is None
    assert captured["plan_payload"]["itineraries"]["A"][0]["day"] == "Day 1 revised"
    assert any(m.get("sender") == "agent3_response" for m in captured["debate_history"])


def test_debate_agent_continue_with_revision_error(monkeypatch, base_plan):
    captured = {}
    monkeypatch.setattr(da, "_resolve_active_plan_id", lambda: "p1")
    monkeypatch.setattr(da, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(da, "load_plan", lambda _db, _pid: base_plan)
    monkeypatch.setattr(
        da,
        "_build_round_critique_payload",
        lambda _plan, _h, _r: {
            "round_decision": "continue",
            "winner_option": None,
            "winner_reason": "",
            "critique_summary": "revise",
            "dimension_scores": {"A": {"composite": 80}},
        },
    )
    monkeypatch.setattr(da, "revise_itinerary", lambda _s, _c, _p: {"error": "planner failed"})
    monkeypatch.setattr(da, "_persist_plan_debate", lambda **kwargs: captured.update(kwargs))
    out = da.debate_agent({})
    assert out["is_valid"] is False
    assert out["debate_count"] == 1
    assert captured["plan_payload"]["itineraries"]["A"][0]["day"] == "Day 1"


def test_debate_agent_judge_success(monkeypatch, base_plan):
    captured = {}
    payload = {**base_plan, "debate_history": [{"sender": "agent4_critic"}, {"sender": "agent4_critic"}]}
    monkeypatch.setattr(da, "_resolve_active_plan_id", lambda: "p1")
    monkeypatch.setattr(da, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(da, "load_plan", lambda _db, _pid: payload)
    monkeypatch.setattr(
        da,
        "_build_round_critique_payload",
        lambda _plan, _h, _r: {
            "round_decision": "continue",
            "winner_option": None,
            "winner_reason": "",
            "critique_summary": "still no winner",
            "dimension_scores": {"A": {"composite": 88}, "B": {"composite": 91}},
        },
    )
    monkeypatch.setattr(
        da,
        "_build_judge_payload",
        lambda _plan, _h: {"winner_option": "B", "winner_reason": "B better", "dimension_scores": {"B": {"composite": 91}}},
    )
    monkeypatch.setattr(da, "_persist_plan_debate", lambda **kwargs: captured.update(kwargs))

    out = da.debate_agent({})
    assert out["is_valid"] is True
    assert out["debate_count"] == 3
    assert captured["debate_verdict"]["winner_option"] == "B"
    assert captured["debate_verdict"]["via_judge"] is True


def test_debate_agent_judge_fallback(monkeypatch, base_plan):
    captured = {}
    payload = {**base_plan, "debate_history": [{"sender": "agent4_critic"}, {"sender": "agent4_critic"}]}
    monkeypatch.setattr(da, "_resolve_active_plan_id", lambda: "p1")
    monkeypatch.setattr(da, "SessionLocal", lambda: _FakeDB())
    monkeypatch.setattr(da, "load_plan", lambda _db, _pid: payload)
    monkeypatch.setattr(
        da,
        "_build_round_critique_payload",
        lambda _plan, _h, _r: {
            "round_decision": "continue",
            "winner_option": None,
            "winner_reason": "",
            "critique_summary": "judge fallback",
            "dimension_scores": {"A": {"composite": 84}, "B": {"composite": 90}, "C": {"composite": 87}},
        },
    )
    monkeypatch.setattr(da, "_build_judge_payload", lambda _plan, _h: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(da, "_persist_plan_debate", lambda **kwargs: captured.update(kwargs))

    out = da.debate_agent({})
    assert out["is_valid"] is True
    assert captured["debate_verdict"]["winner_option"] == "B"
    assert captured["debate_verdict"]["via_judge"] is True
