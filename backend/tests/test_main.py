import importlib
import importlib.util
import sys
import types
import uuid
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from backend.db import crud
from backend.db.database import get_db


@pytest.fixture
def backend_main_module(monkeypatch):
    fake_pkg = types.ModuleType("backend.routers")
    fake_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "backend.routers", fake_pkg)

    for name in ("research", "planner", "explainability", "security", "auth"):
        module = types.ModuleType(f"backend.routers.{name}")
        module.router = APIRouter()
        monkeypatch.setitem(sys.modules, f"backend.routers.{name}", module)

    sys.modules.pop("backend.main", None)
    return importlib.import_module("backend.main")


def test_backend_app_metadata(backend_main_module):
    app = backend_main_module.app
    assert app.title.startswith("TravelMind API")
    assert app.version == "1.0.0"


def test_backend_health_and_root_endpoints(backend_main_module):
    client = TestClient(backend_main_module.app)

    health_response = client.get("/health")
    root_response = client.get("/")

    assert health_response.status_code == 200
    payload = health_response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "travelmind-backend"
    assert "Agent3" in payload["agents"]

    assert root_response.status_code == 200
    root_payload = root_response.json()
    assert "TravelMind API" in root_payload["message"]
    assert root_payload["docs"] == "/docs"


def _load_auth_module():
    module_path = Path(__file__).resolve().parents[1] / "routers" / "auth.py"
    if "backend.routers" not in sys.modules:
        pkg = types.ModuleType("backend.routers")
        pkg.__path__ = [str(module_path.parent)]
        sys.modules["backend.routers"] = pkg

    spec = importlib.util.spec_from_file_location("backend.routers.auth", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load auth module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["backend.routers.auth"] = module
    spec.loader.exec_module(module)
    return module


auth = _load_auth_module()


class _DummySession:
    pass


class _DummyUser:
    def __init__(self, username: str, user_id: str | None = None):
        self.username = username
        self.id = uuid.UUID(user_id) if user_id else uuid.uuid4()


def _make_auth_client() -> TestClient:
    app = FastAPI()
    app.include_router(auth.router, prefix="/auth")

    def _override_db():
        yield _DummySession()

    app.dependency_overrides[get_db] = _override_db
    return TestClient(app)


def test_register_success(monkeypatch):
    client = _make_auth_client()
    captured = {}

    def fake_create_user(db, username, password):
        captured["username"] = username
        captured["password"] = password
        return _DummyUser(username, "12345678-1234-5678-1234-567812345678")

    monkeypatch.setattr(crud, "create_user", fake_create_user)

    resp = client.post("/auth/register", json={"username": "  alice@example.com ", "password": "123456"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["username"] == "alice@example.com"
    assert payload["message"] == "Register success"
    assert payload["user_id"] == "12345678-1234-5678-1234-567812345678"
    assert captured == {"username": "alice@example.com", "password": "123456"}


def test_register_conflict(monkeypatch):
    client = _make_auth_client()

    def fake_create_user(db, username, password):
        raise ValueError("Username already exists")

    monkeypatch.setattr(crud, "create_user", fake_create_user)
    resp = client.post("/auth/register", json={"username": "alice@example.com", "password": "123456"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Username already exists"


def test_login_success(monkeypatch):
    client = _make_auth_client()
    captured = {}

    def fake_authenticate_user(db, username, password):
        captured["username"] = username
        captured["password"] = password
        return _DummyUser(username, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    monkeypatch.setattr(crud, "authenticate_user", fake_authenticate_user)
    resp = client.post("/auth/login", json={"username": "  bob@example.com ", "password": "123456"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["username"] == "bob@example.com"
    assert payload["message"] == "Login success"
    assert payload["user_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert captured == {"username": "bob@example.com", "password": "123456"}


def test_login_invalid_credentials(monkeypatch):
    client = _make_auth_client()
    monkeypatch.setattr(crud, "authenticate_user", lambda db, username, password: None)
    resp = client.post("/auth/login", json={"username": "bob@example.com", "password": "wrongpass"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password"


def test_auth_validation_and_health():
    client = _make_auth_client()
    short_username = client.post("/auth/login", json={"username": "ab", "password": "123456"})
    short_password = client.post("/auth/register", json={"username": "alice@example.com", "password": "123"})
    health = client.get("/auth/health")

    assert short_username.status_code == 422
    assert short_password.status_code == 422
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "router": "auth"}


def test_crud_password_hash_and_verify():
    plain = "123456"
    hashed = crud.hash_password(plain)

    assert hashed.startswith("pbkdf2_sha256$")
    assert crud.verify_password(plain, hashed) is True
    assert crud.verify_password("bad-password", hashed) is False
    assert crud.verify_password(plain, "invalid-hash-format") is False
