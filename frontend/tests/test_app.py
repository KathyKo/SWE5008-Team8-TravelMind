from importlib import import_module
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def frontend_app_module(monkeypatch):
    frontend_dir = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(frontend_dir))

    class _SessionState(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

    fake_streamlit = types.SimpleNamespace(
        session_state=_SessionState(),
        set_page_config=lambda *args, **kwargs: None,
        markdown=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    sys.modules.pop("app", None)
    return import_module("app")


def test_init_state_populates_expected_defaults(frontend_app_module):
    frontend_app_module.st.session_state.clear()
    frontend_app_module.init_state()

    assert frontend_app_module.st.session_state["logged_in"] is False
    assert frontend_app_module.st.session_state["main_section_key"] == "plan"
    assert frontend_app_module.st.session_state["blocked_count"] == 0
    assert frontend_app_module.st.session_state["login_username_input"] == ""


def test_init_state_preserves_existing_values(frontend_app_module):
    frontend_app_module.st.session_state.clear()
    frontend_app_module.st.session_state["logged_in"] = True
    frontend_app_module.init_state()

    assert frontend_app_module.st.session_state["logged_in"] is True
    assert frontend_app_module.st.session_state["main_section_key"] == "plan"


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_profile_from_username_known_and_unknown(frontend_app_module):
    known = frontend_app_module._profile_from_username("alice@example.com")
    unknown = frontend_app_module._profile_from_username("new_user")

    assert known["name"] == "Alice"
    assert known["prefs"]
    assert unknown["name"] == "new_user"
    assert unknown["prefs"] == []


def test_set_login_state(frontend_app_module):
    frontend_app_module.st.session_state.clear()
    frontend_app_module._set_login_state("bob@example.com", "uid-1")

    assert frontend_app_module.st.session_state["logged_in"] is True
    assert frontend_app_module.st.session_state["user_id"] == "uid-1"
    assert frontend_app_module.st.session_state["user"]["username"] == "bob@example.com"
    assert frontend_app_module.st.session_state["user"]["email"] == "bob@example.com"


def test_login_with_backend_success(monkeypatch, frontend_app_module):
    def fake_post(url, json, timeout):
        assert url.endswith("/auth/login")
        assert json["username"] == "alice@example.com"
        return _FakeResponse(200, {"user_id": "u1", "username": "alice@example.com", "message": "Login success"})

    monkeypatch.setattr(frontend_app_module.requests, "post", fake_post)
    data, err = frontend_app_module._login_with_backend("alice@example.com", "123456")
    assert err is None
    assert data["user_id"] == "u1"


def test_login_with_backend_failure_and_exception(monkeypatch, frontend_app_module):
    def fake_post_unauthorized(url, json, timeout):
        return _FakeResponse(401, {"detail": "Invalid username or password"})

    monkeypatch.setattr(frontend_app_module.requests, "post", fake_post_unauthorized)
    data, err = frontend_app_module._login_with_backend("alice@example.com", "wrong")
    assert data is None
    assert err == "Invalid username or password"

    class _RequestError(frontend_app_module.requests.exceptions.RequestException):
        pass

    def fake_post_error(url, json, timeout):
        raise _RequestError("down")

    monkeypatch.setattr(frontend_app_module.requests, "post", fake_post_error)
    data, err = frontend_app_module._login_with_backend("alice@example.com", "123456")
    assert data is None
    assert err == "Backend unavailable"


def test_register_with_backend_success_and_failure(monkeypatch, frontend_app_module):
    def fake_post_created(url, json, timeout):
        assert url.endswith("/auth/register")
        return _FakeResponse(200, {"user_id": "u2", "username": "new", "message": "Register success"})

    monkeypatch.setattr(frontend_app_module.requests, "post", fake_post_created)
    data, err = frontend_app_module._register_with_backend("new", "123456")
    assert err is None
    assert data["message"] == "Register success"

    def fake_post_conflict(url, json, timeout):
        return _FakeResponse(409, {"detail": "Username already exists"})

    monkeypatch.setattr(frontend_app_module.requests, "post", fake_post_conflict)
    data, err = frontend_app_module._register_with_backend("new", "123456")
    assert data is None
    assert err == "Username already exists"
