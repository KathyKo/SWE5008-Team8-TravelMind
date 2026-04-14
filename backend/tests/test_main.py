from importlib import import_module
from pathlib import Path
import sys
import types

import pytest
from fastapi.testclient import TestClient
from fastapi import APIRouter


@pytest.fixture
def backend_main_module(monkeypatch):
    backend_dir = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(backend_dir))

    fake_travel = types.ModuleType("routers.travel")
    fake_travel.router = APIRouter()

    fake_routers = types.ModuleType("routers")
    fake_routers.travel = fake_travel
    fake_routers.__path__ = []

    monkeypatch.setitem(sys.modules, "routers", fake_routers)
    monkeypatch.setitem(sys.modules, "routers.travel", fake_travel)

    sys.modules.pop("main", None)
    return import_module("main")


def test_backend_app_metadata(backend_main_module):
    assert backend_main_module.app.title == "TravelMind API"
    assert backend_main_module.app.version == "0.1.0"


def test_backend_health_and_root_endpoints(backend_main_module):
    client = TestClient(backend_main_module.app)

    health_response = client.get("/health")
    root_response = client.get("/")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok", "service": "travelmind-backend"}

    assert root_response.status_code == 200
    assert "TravelMind API is running" in root_response.json()["message"]
