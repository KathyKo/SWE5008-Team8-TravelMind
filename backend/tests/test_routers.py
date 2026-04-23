"""
backend/tests/test_routers.py — coverage for research, planner, explainability,
security, fairness, and agent_proxy routers.

All external dependencies (agent_client calls, requests.post) are monkeypatched
so no real network or DB connections are made.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.db.database import get_db
from backend.db import crud

import requests as _requests_lib

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_router(name: str):
    module_path = Path(__file__).resolve().parents[1] / "routers" / f"{name}.py"
    if "backend.routers" not in sys.modules:
        pkg = types.ModuleType("backend.routers")
        pkg.__path__ = [str(module_path.parent)]
        sys.modules["backend.routers"] = pkg
    full_name = f"backend.routers.{name}"
    sys.modules.pop(full_name, None)
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


class _DummySession:
    pass


def _make_client(router, prefix: str = "") -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix=prefix)
    app.dependency_overrides[get_db] = lambda: (yield _DummySession())
    return TestClient(app)


_MINIMAL_STATE = {
    "origin": "Singapore",
    "destination": "Tokyo",
    "dates": "2026-06-01 to 2026-06-07",
    "duration": "7 days",
    "budget": "SGD 3000",
    "preferences": "culture, food",
}

_PLANNER_RESULT = {
    "itineraries": {"A": [{"day": 1, "items": []}], "B": [], "C": []},
    "option_meta": {"A": {"label": "Option A", "budget": "SGD 3000"}},
    "tool_log": [],
    "flight_options_outbound": [{"flight": "SQ601"}],
    "flight_options_return": [{"flight": "SQ602"}],
    "hotel_options": [{"hotel": "Shinjuku Hotel"}],
}

_EXPLAIN_RESULT = {
    "explain_option": "A",
    "summary": {"overall_summary": "Great trip", "day_summaries": {}},
    "item_explanations": {"by_key": {}, "by_occurrence": {}},
    "chain_of_thought": "reasoning...",
    "agent_steps": [],
}


# ── research router ───────────────────────────────────────────────────────────

research = _load_router("research")


@pytest.fixture
def research_client():
    return _make_client(research.router, prefix="/research")


def test_research_health(research_client):
    resp = research_client.get("/research/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_research_run_success(monkeypatch, research_client):
    monkeypatch.setattr(research, "call_research", lambda state: {
        "compact_flights_out": [{"flight": "SQ601"}],
        "compact_flights_ret": [{"flight": "SQ602"}],
        "hotel_opts": [{"hotel": "Park Hotel"}],
        "compact_attractions": [{"name": "Senso-ji"}],
        "compact_restaurants": [{"name": "Ramen Ya"}],
        "tool_log": ["searched flights"],
    })
    resp = research_client.post("/research/run", json=_MINIMAL_STATE)
    assert resp.status_code == 200
    data = resp.json()
    assert data["flight_options_outbound"] == [{"flight": "SQ601"}]
    assert data["hotel_options"] == [{"hotel": "Park Hotel"}]
    assert data["attractions"] == [{"name": "Senso-ji"}]
    assert data["tool_log"] == ["searched flights"]


def test_research_run_agent_error(monkeypatch, research_client):
    monkeypatch.setattr(research, "call_research", lambda state: {"error": "timeout"})
    resp = research_client.post("/research/run", json=_MINIMAL_STATE)
    assert resp.status_code == 500
    assert "Agent2 error" in resp.json()["detail"]


def test_research_run_missing_optional_fields(monkeypatch, research_client):
    monkeypatch.setattr(research, "call_research", lambda state: {"tool_log": []})
    resp = research_client.post("/research/run", json=_MINIMAL_STATE)
    assert resp.status_code == 200
    data = resp.json()
    assert data["flight_options_outbound"] == []
    assert data["restaurants"] == []


# ── planner router ────────────────────────────────────────────────────────────

planner = _load_router("planner")


@pytest.fixture
def planner_client():
    return _make_client(planner.router, prefix="/planner")


def test_planner_health(planner_client):
    resp = planner_client.get("/planner/health")
    assert resp.status_code == 200
    assert resp.json()["agent"] == "Agent3-Planner"


def test_planner_run_success(monkeypatch, planner_client):
    monkeypatch.setattr(planner, "call_planner", lambda state: _PLANNER_RESULT)
    monkeypatch.setattr(crud, "save_plan", lambda db, plan_id, state, result, via_debate: None)
    resp = planner_client.post("/planner/run", json=_MINIMAL_STATE)
    assert resp.status_code == 200
    data = resp.json()
    assert "plan_id" in data
    assert "A" in data["itineraries"]
    assert data["flight_options_outbound"] == [{"flight": "SQ601"}]


def test_planner_run_agent_error(monkeypatch, planner_client):
    monkeypatch.setattr(planner, "call_planner", lambda state: {"error": "LLM failed"})
    monkeypatch.setattr(crud, "save_plan", lambda *a, **kw: None)
    resp = planner_client.post("/planner/run", json=_MINIMAL_STATE)
    assert resp.status_code == 500
    assert "Agent3 error" in resp.json()["detail"]


def test_planner_revise_success(monkeypatch, planner_client):
    cached = {**_MINIMAL_STATE, **_PLANNER_RESULT}
    monkeypatch.setattr(crud, "load_plan", lambda db, plan_id: cached)
    monkeypatch.setattr(planner, "call_planner_revise", lambda state, critique, current: _PLANNER_RESULT)
    monkeypatch.setattr(crud, "update_plan_result", lambda db, plan_id, revised: None)
    resp = planner_client.post("/planner/revise", json={"plan_id": "abc12345", "critique": "too expensive"})
    assert resp.status_code == 200
    assert resp.json()["plan_id"] == "abc12345"


def test_planner_revise_plan_not_found(monkeypatch, planner_client):
    monkeypatch.setattr(crud, "load_plan", lambda db, plan_id: None)
    resp = planner_client.post("/planner/revise", json={"plan_id": "missing", "critique": "..."})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_planner_revise_agent_error(monkeypatch, planner_client):
    cached = {**_MINIMAL_STATE, **_PLANNER_RESULT}
    monkeypatch.setattr(crud, "load_plan", lambda db, plan_id: cached)
    monkeypatch.setattr(planner, "call_planner_revise", lambda *a: {"error": "revision failed"})
    monkeypatch.setattr(crud, "update_plan_result", lambda *a: None)
    resp = planner_client.post("/planner/revise", json={"plan_id": "abc12345", "critique": "..."})
    assert resp.status_code == 500
    assert "revision error" in resp.json()["detail"]


# ── explainability router ─────────────────────────────────────────────────────

explainability = _load_router("explainability")


@pytest.fixture
def explain_client():
    return _make_client(explainability.router, prefix="/explainability")


def test_explainability_health(explain_client):
    resp = explain_client.get("/explainability/health")
    assert resp.status_code == 200
    assert resp.json()["agent"] == "Agent6-Explainability"


def test_explainability_run_success(monkeypatch, explain_client):
    cached = {**_MINIMAL_STATE, **_PLANNER_RESULT}
    monkeypatch.setattr(crud, "load_plan", lambda db, plan_id: cached)
    monkeypatch.setattr(explainability, "call_explainability", lambda state: _EXPLAIN_RESULT)
    monkeypatch.setattr(crud, "save_explain", lambda db, plan_id, result: None)
    resp = explain_client.post("/explainability/run", json={"plan_id": "abc12345", "explain_option": "A"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["plan_id"] == "abc12345"
    assert data["explain_option"] == "A"
    assert data["chain_of_thought"] == "reasoning..."


def test_explainability_plan_not_found(monkeypatch, explain_client):
    monkeypatch.setattr(crud, "load_plan", lambda db, plan_id: None)
    resp = explain_client.post("/explainability/run", json={"plan_id": "missing"})
    assert resp.status_code == 404


def test_explainability_agent_error(monkeypatch, explain_client):
    cached = {**_MINIMAL_STATE, **_PLANNER_RESULT}
    monkeypatch.setattr(crud, "load_plan", lambda db, plan_id: cached)
    monkeypatch.setattr(explainability, "call_explainability", lambda state: {"error": "LLM error"})
    monkeypatch.setattr(crud, "save_explain", lambda *a: None)
    resp = explain_client.post("/explainability/run", json={"plan_id": "abc12345"})
    assert resp.status_code == 500
    assert "Agent6 error" in resp.json()["detail"]


def test_explainability_default_option(monkeypatch, explain_client):
    cached = {**_MINIMAL_STATE, **_PLANNER_RESULT}
    monkeypatch.setattr(crud, "load_plan", lambda db, plan_id: cached)
    monkeypatch.setattr(explainability, "call_explainability", lambda state: {**_EXPLAIN_RESULT, "explain_option": "A"})
    monkeypatch.setattr(crud, "save_explain", lambda *a: None)
    resp = explain_client.post("/explainability/run", json={"plan_id": "abc12345"})
    assert resp.status_code == 200
    assert resp.json()["explain_option"] == "A"


# ── security router ───────────────────────────────────────────────────────────

security_router = _load_router("security")


@pytest.fixture
def security_client():
    app = FastAPI()
    app.include_router(security_router.router, prefix="/travel")
    return TestClient(app)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            err = _requests_lib.exceptions.HTTPError(response=self)
            raise err


def test_security_health(security_client):
    resp = security_client.get("/travel/health")
    assert resp.status_code == 200
    assert resp.json()["router"] == "security-compat"


def test_security_check_success(monkeypatch, security_client):
    monkeypatch.setattr(security_router.requests, "post",
        lambda url, json, timeout: _FakeResponse(200, {"safe": True, "score": 0.1}))
    resp = security_client.post("/travel/security/check", json={"text": "hello", "user_id": "u1"})
    assert resp.status_code == 200
    assert resp.json()["safe"] is True


def test_security_check_output_success(monkeypatch, security_client):
    monkeypatch.setattr(security_router.requests, "post",
        lambda url, json, timeout: _FakeResponse(200, {"safe": True}))
    resp = security_client.post("/travel/security/check-output", json={"text": "result"})
    assert resp.status_code == 200


def test_security_check_http_error(monkeypatch, security_client):
    monkeypatch.setattr(security_router.requests, "post",
        lambda url, json, timeout: _FakeResponse(500, {"detail": "agents down"}))
    resp = security_client.post("/travel/security/check", json={"text": "hello"})
    assert resp.status_code == 502


def test_security_check_connection_error(monkeypatch, security_client):
    def _raise(*a, **kw):
        raise _requests_lib.exceptions.ConnectionError("refused")
    monkeypatch.setattr(security_router.requests, "post", _raise)
    resp = security_client.post("/travel/security/check", json={"text": "hello"})
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["detail"]


def test_security_check_output_connection_error(monkeypatch, security_client):
    def _raise(*a, **kw):
        raise _requests_lib.exceptions.ConnectionError("refused")
    monkeypatch.setattr(security_router.requests, "post", _raise)
    resp = security_client.post("/travel/security/check-output", json={"text": "hello"})
    assert resp.status_code == 502


# ── fairness router ───────────────────────────────────────────────────────────

fairness = _load_router("fairness")


@pytest.fixture
def fairness_client():
    app = FastAPI()
    app.include_router(fairness.router, prefix="/fairness")
    return TestClient(app)


def test_fairness_health(fairness_client):
    resp = fairness_client.get("/fairness/health")
    assert resp.status_code == 200
    assert resp.json()["router"] == "fairness"


def test_fairness_check_success(monkeypatch, fairness_client):
    monkeypatch.setattr(fairness.requests, "post",
        lambda url, json, timeout: _FakeResponse(200, {"fair": True, "score": 0.95}))
    resp = fairness_client.post("/fairness/check-selected-option",
        json={"state": {"itineraries": {}}, "selected_option": "A"})
    assert resp.status_code == 200
    assert resp.json()["fair"] is True


def test_fairness_check_agents_error(monkeypatch, fairness_client):
    monkeypatch.setattr(fairness.requests, "post",
        lambda url, json, timeout: _FakeResponse(422, {"detail": "invalid input"}))
    resp = fairness_client.post("/fairness/check-selected-option",
        json={"state": {}, "selected_option": "B"})
    assert resp.status_code == 422


def test_fairness_check_connection_error(monkeypatch, fairness_client):
    def _raise(*a, **kw):
        raise _requests_lib.exceptions.ConnectionError("down")
    monkeypatch.setattr(fairness.requests, "post", _raise)
    resp = fairness_client.post("/fairness/check-selected-option",
        json={"state": {}, "selected_option": "A"})
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["detail"]


# ── agent_proxy router ────────────────────────────────────────────────────────

agent_proxy = _load_router("agent_proxy")


@pytest.fixture
def proxy_client():
    app = FastAPI()
    app.include_router(agent_proxy.router, prefix="/agent")
    return TestClient(app)


class _FakeStreamResponse:
    status_code = 200
    text = ""

    def iter_content(self, chunk_size=8192):
        yield b'{"node":"planner"}\n'
        yield b'{"node":"done"}\n'

    def close(self):
        pass


def test_proxy_graph_stream_success(monkeypatch, proxy_client):
    monkeypatch.setattr(agent_proxy.requests, "post",
        lambda url, json, stream, timeout: _FakeStreamResponse())
    resp = proxy_client.post("/agent/graph/stream", json={"state": {"destination": "Tokyo"}})
    assert resp.status_code == 200
    assert b"planner" in resp.content


def test_proxy_graph_stream_connection_error(monkeypatch, proxy_client):
    def _raise(*a, **kw):
        raise _requests_lib.exceptions.ConnectionError("down")
    monkeypatch.setattr(agent_proxy.requests, "post", _raise)
    resp = proxy_client.post("/agent/graph/stream", json={"state": {}})
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["detail"]


def test_proxy_graph_stream_non_200(monkeypatch, proxy_client):
    class _Bad:
        status_code = 503
        text = "service unavailable"
        def close(self): pass
    monkeypatch.setattr(agent_proxy.requests, "post", lambda *a, **kw: _Bad())
    resp = proxy_client.post("/agent/graph/stream", json={"state": {}})
    assert resp.status_code == 502


def test_proxy_fairness_check_success(monkeypatch, proxy_client):
    monkeypatch.setattr(agent_proxy.requests, "post",
        lambda url, json, timeout: _FakeResponse(200, {"fair": True}))
    resp = proxy_client.post("/agent/fairness-check",
        json={"state": {}, "selected_option": "A"})
    assert resp.status_code == 200
    assert resp.json()["fair"] is True


def test_proxy_fairness_check_http_error(monkeypatch, proxy_client):
    monkeypatch.setattr(agent_proxy.requests, "post",
        lambda url, json, timeout: _FakeResponse(500, {"detail": "error"}))
    resp = proxy_client.post("/agent/fairness-check", json={"state": {}})
    assert resp.status_code == 502


def test_proxy_fairness_check_connection_error(monkeypatch, proxy_client):
    def _raise(*a, **kw):
        raise _requests_lib.exceptions.ConnectionError("down")
    monkeypatch.setattr(agent_proxy.requests, "post", _raise)
    resp = proxy_client.post("/agent/fairness-check", json={"state": {}})
    assert resp.status_code == 502


def test_proxy_replan_success(monkeypatch, proxy_client):
    monkeypatch.setattr(agent_proxy.requests, "post",
        lambda url, json, timeout: _FakeResponse(200, {"itineraries": {"A": []}}))
    resp = proxy_client.post("/agent/replan", json={"state": {"destination": "Tokyo"}})
    assert resp.status_code == 200
    assert "itineraries" in resp.json()


def test_proxy_replan_http_error(monkeypatch, proxy_client):
    monkeypatch.setattr(agent_proxy.requests, "post",
        lambda url, json, timeout: _FakeResponse(502, {"detail": "agents down"}))
    resp = proxy_client.post("/agent/replan", json={"state": {}})
    assert resp.status_code == 502


def test_proxy_replan_connection_error(monkeypatch, proxy_client):
    def _raise(*a, **kw):
        raise _requests_lib.exceptions.ConnectionError("down")
    monkeypatch.setattr(agent_proxy.requests, "post", _raise)
    resp = proxy_client.post("/agent/replan", json={"state": {}})
    assert resp.status_code == 502
