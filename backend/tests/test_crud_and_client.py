"""
backend/tests/test_crud_and_client.py
Coverage for:
  - backend/db/crud.py   (save_plan, load_plan, update_plan_result, save_explain,
                          get_user_by_username, create_user, authenticate_user,
                          verify_password edge cases)
  - backend/agent_client.py  (http mode + error paths for all four callers)
  - backend/db/database.py   (get_db generator)
"""
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# crud.py — use MagicMock for the SQLAlchemy session so no real DB is needed
# ---------------------------------------------------------------------------
from backend.db import crud
import backend.agent_client as ac
import requests as _req_lib
from backend.db import database as db_module

_STATE = {
    "origin": "Singapore",
    "destination": "Tokyo",
    "dates": "2026-06-01",
    "duration": "7 days",
    "budget": "SGD 3000",
    "preferences": "culture",
    "hard_constraints": {"max_stops": 1},
    "soft_preferences": {"window_seat": True},
    "search_queries": ["flights SG->TYO"],
}

_RESULT = {
    "itineraries": {
        "A": [{"day": 1, "items": [{"name": "Fushimi", "cost": "Free"}]}],
        "B": [{"day": 1, "items": []}],
        "C": [],
    },
    "option_meta": {"A": {"label": "Option A", "budget": "SGD 3000"}},
    "tool_log": ["log1"],
    "flight_options_outbound": [{"airline": "SQ", "price": "SGD 500"}],
    "flight_options_return": [{"airline": "SQ", "price": "SGD 480"}],
    "hotel_options": [{"name": "Shinjuku Hotel", "price": "SGD 150"}],
    "planner_decision_trace": {"A": ["trace1"]},
    "planner_chain_of_thought": "thought...",
    "debate_verdict": {"winner": "A"},
    "debate_history": [{"round": 1}],
}


# ── save_plan ─────────────────────────────────────────────────────────────────

def test_save_plan_adds_plan_and_children():
    db = MagicMock()
    db.refresh.side_effect = lambda obj: None

    crud.save_plan(db, "plan001", _STATE, _RESULT, via_debate=True)

    # db.add called once per plan + itinerary + flight + hotel rows
    assert db.add.call_count >= 1
    db.commit.assert_called_once()
    db.refresh.assert_called_once()


def test_save_plan_empty_result():
    """save_plan handles result with no itineraries / flights / hotels."""
    db = MagicMock()
    crud.save_plan(db, "plan002", _STATE, {})
    db.commit.assert_called_once()


def test_save_plan_returns_plan_object():
    """save_plan returns the Plan ORM object (after refresh)."""
    db = MagicMock()
    crud.save_plan(db, "plan003", _STATE, _RESULT)
    db.refresh.assert_called_once()


# ── load_plan ─────────────────────────────────────────────────────────────────

def _make_plan_mock():
    plan = MagicMock()
    plan.origin = "Singapore"
    plan.destination = "Tokyo"
    plan.dates = "2026-06-01"
    plan.duration = "7 days"
    plan.budget = "SGD 3000"
    plan.preferences = "culture"
    plan.hard_constraints = {"max_stops": 1}
    plan.soft_preferences = {}
    plan.search_queries = []
    plan.option_meta = {"A": {}}
    plan.tool_log = []
    plan.planner_decision_trace = {}
    plan.chain_of_thought = "thought"
    plan.debate_verdict = None
    plan.debate_history = None

    itin_a = MagicMock()
    itin_a.option = "A"
    itin_a.days = [{"day": 1}]
    plan.itineraries = [itin_a]

    fl_out = MagicMock()
    fl_out.direction = "outbound"
    fl_out.flight_data = {"airline": "SQ"}
    fl_ret = MagicMock()
    fl_ret.direction = "return"
    fl_ret.flight_data = {"airline": "SQ"}
    plan.flights = [fl_out, fl_ret]

    hotel = MagicMock()
    hotel.hotel_data = {"name": "Hotel A"}
    plan.hotels = [hotel]

    return plan


def test_load_plan_returns_full_dict():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = _make_plan_mock()

    result = crud.load_plan(db, "plan001")

    assert result is not None
    assert result["origin"] == "Singapore"
    assert result["destination"] == "Tokyo"
    assert "A" in result["itineraries"]
    assert len(result["flight_options_outbound"]) == 1
    assert len(result["flight_options_return"]) == 1
    assert len(result["hotel_options"]) == 1


def test_load_plan_not_found_returns_none():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    assert crud.load_plan(db, "missing") is None


def test_load_plan_nullable_fields_become_defaults():
    """Nullable DB fields (hard_constraints=None etc.) fall back to {}."""
    db = MagicMock()
    plan = _make_plan_mock()
    plan.hard_constraints = None
    plan.soft_preferences = None
    plan.search_queries = None
    plan.option_meta = None
    plan.tool_log = None
    plan.planner_decision_trace = None
    plan.chain_of_thought = None
    db.query.return_value.filter.return_value.first.return_value = plan

    result = crud.load_plan(db, "plan001")
    assert result["hard_constraints"] == {}
    assert result["soft_preferences"] == {}
    assert result["search_queries"] == []
    assert result["option_meta"] == {}
    assert result["tool_log"] == []
    assert result["planner_decision_trace"] == {}
    assert result["chain_of_thought"] == ""


# ── update_plan_result ────────────────────────────────────────────────────────

def test_update_plan_result_replaces_children():
    db = MagicMock()
    plan = _make_plan_mock()
    db.query.return_value.filter.return_value.first.return_value = plan

    revised = {
        "itineraries": {"A": [{"day": 1, "items": []}]},
        "flight_options_outbound": [{"airline": "JL"}],
        "flight_options_return": [],
        "hotel_options": [{"name": "New Hotel"}],
        "option_meta": {"A": {"label": "Revised"}},
        "tool_log": ["new log"],
        "planner_decision_trace": {},
        "planner_chain_of_thought": "revised thought",
    }
    crud.update_plan_result(db, "plan001", revised)

    db.flush.assert_called_once()
    db.commit.assert_called_once()


def test_update_plan_result_no_op_when_plan_missing():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    crud.update_plan_result(db, "missing", {})
    db.flush.assert_not_called()
    db.commit.assert_not_called()


# ── save_explain ──────────────────────────────────────────────────────────────

def test_save_explain_inserts_new_row():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    crud.save_explain(db, "plan001", {
        "explain_data": {"key": "val"},
        "chain_of_thought": "cot",
        "agent_steps": [{"step": 1}],
    })
    db.add.assert_called_once()
    db.commit.assert_called_once()


def test_save_explain_updates_existing_row():
    db = MagicMock()
    existing = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = existing

    crud.save_explain(db, "plan001", {
        "explain_data": {"k": "v"},
        "chain_of_thought": "cot2",
        "agent_steps": [],
    })
    assert existing.explain_data == {"k": "v"}
    assert existing.chain_of_thought == "cot2"
    db.add.assert_not_called()
    db.commit.assert_called_once()


# ── verify_password edge cases ────────────────────────────────────────────────

def test_verify_password_wrong_algo():
    hashed = crud.hash_password("secret")
    # Replace algo prefix to something invalid
    tampered = hashed.replace("pbkdf2_sha256$", "md5$")
    assert crud.verify_password("secret", tampered) is False


def test_verify_password_malformed_hash():
    assert crud.verify_password("secret", "not$a$valid$hash$extra") is False


def test_verify_password_none_hash():
    assert crud.verify_password("secret", None) is False


# ── get_user_by_username ──────────────────────────────────────────────────────

def test_get_user_by_username_found():
    db = MagicMock()
    user_mock = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = user_mock

    result = crud.get_user_by_username(db, "alice@example.com")
    assert result is user_mock


def test_get_user_by_username_not_found():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    result = crud.get_user_by_username(db, "nobody@example.com")
    assert result is None


# ── create_user ───────────────────────────────────────────────────────────────

def test_create_user_success():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.refresh.side_effect = lambda obj: None

    crud.create_user(db, "newuser@example.com", "password123")
    db.add.assert_called_once()
    db.commit.assert_called_once()


def test_create_user_raises_if_username_exists():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = MagicMock()

    with pytest.raises(ValueError, match="Username already exists"):
        crud.create_user(db, "existing@example.com", "password123")


# ── authenticate_user ─────────────────────────────────────────────────────────

def test_authenticate_user_success():
    db = MagicMock()
    user_mock = MagicMock()
    plain = "mypassword"
    user_mock.password_hash = crud.hash_password(plain)
    db.query.return_value.filter.return_value.first.return_value = user_mock

    result = crud.authenticate_user(db, "alice@example.com", plain)
    assert result is user_mock


def test_authenticate_user_wrong_password():
    db = MagicMock()
    user_mock = MagicMock()
    user_mock.password_hash = crud.hash_password("correct")
    db.query.return_value.filter.return_value.first.return_value = user_mock

    result = crud.authenticate_user(db, "alice@example.com", "wrong")
    assert result is None


def test_authenticate_user_not_found():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    result = crud.authenticate_user(db, "ghost@example.com", "any")
    assert result is None


# ---------------------------------------------------------------------------
# database.py — get_db generator
# ---------------------------------------------------------------------------


def test_get_db_yields_and_closes():
    """get_db yields a session then closes it in the finally block."""
    mock_session = MagicMock()
    original_session_local = db_module.SessionLocal

    db_module.SessionLocal = MagicMock(return_value=mock_session)
    try:
        gen = db_module.get_db()
        session = next(gen)
        assert session is mock_session
        with pytest.raises(StopIteration):
            next(gen)
        mock_session.close.assert_called_once()
    finally:
        db_module.SessionLocal = original_session_local


def test_get_db_closes_on_exception():
    """get_db closes the session even when the consumer raises."""
    mock_session = MagicMock()
    original_session_local = db_module.SessionLocal
    db_module.SessionLocal = MagicMock(return_value=mock_session)
    try:
        gen = db_module.get_db()
        next(gen)
        try:
            gen.throw(RuntimeError("boom"))
        except RuntimeError:
            pass
        mock_session.close.assert_called_once()
    finally:
        db_module.SessionLocal = original_session_local


# ---------------------------------------------------------------------------
# agent_client.py — http mode (mocked requests) + error paths
# ---------------------------------------------------------------------------

def test_call_research_http_success(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "http")

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"flights": []}

    monkeypatch.setattr(_req_lib, "post", lambda url, json, timeout: _Resp())
    monkeypatch.setattr(ac, "requests", _req_lib)
    result = ac.call_research({"origin": "SG"})
    assert "flights" in result


def test_call_research_http_error(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "http")
    monkeypatch.setattr(_req_lib, "post", lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("timeout")))
    monkeypatch.setattr(ac, "requests", _req_lib)
    result = ac.call_research({"origin": "SG"})
    assert "error" in result


def test_call_planner_http_success(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "http")

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"itineraries": {"A": []}}

    monkeypatch.setattr(_req_lib, "post", lambda url, json, timeout: _Resp())
    monkeypatch.setattr(ac, "requests", _req_lib)
    result = ac.call_planner({"origin": "SG"})
    assert "itineraries" in result


def test_call_planner_http_error(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "http")

    def _raise(url, json, timeout):
        raise RuntimeError("network error")

    monkeypatch.setattr(_req_lib, "post", _raise)
    monkeypatch.setattr(ac, "requests", _req_lib)
    result = ac.call_planner({})
    assert "error" in result


def test_call_planner_revise_http_success(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "http")

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"itineraries": {"A": [], "B": [], "C": []}}

    monkeypatch.setattr(_req_lib, "post", lambda url, json, timeout: _Resp())
    monkeypatch.setattr(ac, "requests", _req_lib)
    result = ac.call_planner_revise({}, "too expensive", {})
    assert "itineraries" in result


def test_call_planner_revise_http_error(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "http")

    def _raise(url, json, timeout):
        raise OSError("refused")

    monkeypatch.setattr(_req_lib, "post", _raise)
    monkeypatch.setattr(ac, "requests", _req_lib)
    result = ac.call_planner_revise({}, "critique", {})
    assert "error" in result


def test_call_explainability_http_success(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "http")

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"explain_data": {}}

    monkeypatch.setattr(_req_lib, "post", lambda url, json, timeout: _Resp())
    monkeypatch.setattr(ac, "requests", _req_lib)
    result = ac.call_explainability({"plan_id": "abc"})
    assert "explain_data" in result


def test_call_explainability_http_error(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "http")

    def _raise(url, json, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(_req_lib, "post", _raise)
    monkeypatch.setattr(ac, "requests", _req_lib)
    result = ac.call_explainability({})
    assert "error" in result


# ── local mode: stub agent modules so imports succeed without torch/langchain ──

def _stub_agents(monkeypatch):
    """Insert lightweight stubs for all agent modules imported in local mode."""
    for mod_name in (
        "agents", "agents.specialists", "agents.specialists.research_agent",
        "agents.specialists.planner_agent", "agents.specialists.explainability_agent",
        "agents.agent_tools",
    ):
        if mod_name not in sys.modules:
            monkeypatch.setitem(sys.modules, mod_name, types.ModuleType(mod_name))

    res_mod = sys.modules["agents.specialists.research_agent"]
    res_mod.research_agent = lambda state, tools=None: {"flights": ["stub"]}

    plan_mod = sys.modules["agents.specialists.planner_agent"]
    plan_mod.planner_agent = lambda state: {"itineraries": {"A": []}}
    plan_mod.revise_itinerary = lambda state, critique, current: {"itineraries": {"A": []}}

    exp_mod = sys.modules["agents.specialists.explainability_agent"]
    exp_mod.explainability_agent = lambda state: {"explain_data": {}}

    tools_mod = sys.modules["agents.agent_tools"]
    tools_mod.get_tools_for_agent = lambda name: []


def test_call_research_local_mode(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "local")
    _stub_agents(monkeypatch)
    result = ac.call_research({"origin": "SG"})
    assert "flights" in result


def test_call_planner_local_mode(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "local")
    _stub_agents(monkeypatch)
    result = ac.call_planner({"origin": "SG"})
    assert "itineraries" in result


def test_call_planner_revise_local_mode(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "local")
    _stub_agents(monkeypatch)
    result = ac.call_planner_revise({}, "too expensive", {})
    assert "itineraries" in result


def test_call_explainability_local_mode(monkeypatch):
    monkeypatch.setattr(ac, "AGENT_MODE", "local")
    _stub_agents(monkeypatch)
    result = ac.call_explainability({"plan_id": "abc"})
    assert "explain_data" in result
