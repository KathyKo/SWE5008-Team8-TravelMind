from importlib import import_module
from pathlib import Path
import sys
import types

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient


@pytest.fixture
def backend_main_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root))

    fake_research = types.ModuleType("backend.routers.research")
    fake_research.router = APIRouter()

    fake_planner = types.ModuleType("backend.routers.planner")
    fake_planner.router = APIRouter()

    fake_explainability = types.ModuleType("backend.routers.explainability")
    fake_explainability.router = APIRouter()

    fake_security = types.ModuleType("backend.routers.security")
    fake_security.router = APIRouter()

    fake_auth = types.ModuleType("backend.routers.auth")
    fake_auth.router = APIRouter()

    fake_routers = types.ModuleType("backend.routers")
    fake_routers.research = fake_research
    fake_routers.planner = fake_planner
    fake_routers.explainability = fake_explainability
    fake_routers.security = fake_security
    fake_routers.auth = fake_auth
    fake_routers.__path__ = []

    monkeypatch.setitem(sys.modules, "backend.routers", fake_routers)
    monkeypatch.setitem(sys.modules, "backend.routers.research", fake_research)
    monkeypatch.setitem(sys.modules, "backend.routers.planner", fake_planner)
    monkeypatch.setitem(sys.modules, "backend.routers.explainability", fake_explainability)
    monkeypatch.setitem(sys.modules, "backend.routers.security", fake_security)
    monkeypatch.setitem(sys.modules, "backend.routers.auth", fake_auth)

    sys.modules.pop("backend.main", None)
    return import_module("backend.main")


def test_backend_app_metadata(backend_main_module):
    assert backend_main_module.app.title == "TravelMind API"
    assert backend_main_module.app.version == "1.0.0"


def test_backend_health_and_root_endpoints(backend_main_module):
    client = TestClient(backend_main_module.app)

    health_response = client.get("/health")
    root_response = client.get("/")

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "ok",
        "service": "travelmind-backend",
        "agents": ["Agent2", "Agent3", "Agent6"],
    }

    assert root_response.status_code == 200
    assert root_response.json() == {
        "message": "TravelMind API — Kathy scope",
        "agents": {
            "Agent2 Research": "POST /research/run",
            "Agent3 Planner": "POST /planner/run",
            "Agent3 Revise": "POST /planner/revise",
            "Agent6 Explainability": "POST /explainability/run",
        },
        "docs": "/docs",
    }
