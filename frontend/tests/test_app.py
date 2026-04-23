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


def test_register_with_backend_network_error(monkeypatch, frontend_app_module):
    class _RequestError(frontend_app_module.requests.exceptions.RequestException):
        pass

    def fake_post_error(url, json, timeout):
        raise _RequestError("network down")

    monkeypatch.setattr(frontend_app_module.requests, "post", fake_post_error)
    data, err = frontend_app_module._register_with_backend("new_user", "123456")
    assert data is None
    assert err == "Backend unavailable"


def test_fill_demo_credentials_alice(frontend_app_module):
    frontend_app_module.st.session_state.clear()
    frontend_app_module.init_state()
    frontend_app_module._fill_demo_credentials("alice@example.com")

    assert frontend_app_module.st.session_state["login_username_input"] == "alice@example.com"
    assert frontend_app_module.st.session_state["login_password_input"] == "123456"
    assert frontend_app_module.st.session_state["auth_mode"] == "Sign In"


def test_fill_demo_credentials_unknown_user(frontend_app_module):
    frontend_app_module.st.session_state.clear()
    frontend_app_module.init_state()
    frontend_app_module._fill_demo_credentials("unknown@example.com")

    assert frontend_app_module.st.session_state["login_username_input"] == "unknown@example.com"
    assert frontend_app_module.st.session_state["login_password_input"] == "demo123"


def test_set_login_state_unknown_user(frontend_app_module):
    frontend_app_module.st.session_state.clear()
    frontend_app_module._set_login_state("new_user", "uid-99")

    assert frontend_app_module.st.session_state["logged_in"] is True
    assert frontend_app_module.st.session_state["user"]["name"] == "new_user"
    assert frontend_app_module.st.session_state["user"]["prefs"] == []


def test_set_login_state_email_username(frontend_app_module):
    frontend_app_module.st.session_state.clear()
    frontend_app_module._set_login_state("carol@example.com", "uid-carol")

    user = frontend_app_module.st.session_state["user"]
    assert user["username"] == "carol@example.com"
    assert user["email"] == "carol@example.com"


def test_init_state_all_defaults_present(frontend_app_module):
    frontend_app_module.st.session_state.clear()
    frontend_app_module.init_state()

    expected_keys = [
        "logged_in", "user_id", "user", "auth_mode", "selected_option",
        "plan_generated", "plan_itineraries", "plan_id", "security_log",
        "blocked_count", "passed_count", "replan_situation", "replan_done",
        "main_section_key", "visited",
    ]
    for key in expected_keys:
        assert key in frontend_app_module.st.session_state, f"Missing key: {key}"


def test_init_state_does_not_overwrite_multiple_keys(frontend_app_module):
    frontend_app_module.st.session_state.clear()
    frontend_app_module.st.session_state["blocked_count"] = 5
    frontend_app_module.st.session_state["selected_option"] = "C"
    frontend_app_module.init_state()

    assert frontend_app_module.st.session_state["blocked_count"] == 5
    assert frontend_app_module.st.session_state["selected_option"] == "C"


def test_profile_from_username_email_with_at(frontend_app_module):
    profile = frontend_app_module._profile_from_username("hello@domain.com")
    assert profile["name"] == "hello"
    assert profile["prefs"] == []


def test_profile_from_username_no_at_sign(frontend_app_module):
    profile = frontend_app_module._profile_from_username("johndoe")
    assert profile["name"] == "johndoe"
    assert profile["avatar"] == "User"


def test_login_with_backend_non_200_without_detail(monkeypatch, frontend_app_module):
    def fake_post(_url, **_kwargs):
        return _FakeResponse(500, {})

    monkeypatch.setattr(frontend_app_module.requests, "post", fake_post)
    data, err = frontend_app_module._login_with_backend("alice@example.com", "123456")
    assert data is None
    assert err == "Login failed"


# ── Rich st mock for UI function tests ────────────────────────────────────────

class _RichNoop:
    """Context manager and callable that returns itself."""
    def __call__(self, *a, **kw):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def __iter__(self):
        return iter([])
    def __getattr__(self, item):
        return self


_rich_noop = _RichNoop()


class _RichSessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def get(self, key, default=None):
        return self[key] if key in self else default


def _make_rich_st(session_state=None):
    ss = session_state if session_state is not None else _RichSessionState()
    return types.SimpleNamespace(
        session_state=ss,
        set_page_config=lambda *a, **kw: None,
        markdown=lambda *a, **kw: None,
        write=lambda *a, **kw: None,
        caption=lambda *a, **kw: None,
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        success=lambda *a, **kw: None,
        metric=lambda *a, **kw: None,
        radio=lambda label, options, **kw: options[0],
        selectbox=lambda label, options, **kw: options[0] if options else None,
        text_area=lambda *a, **kw: kw.get("value", ""),
        text_input=lambda *a, **kw: kw.get("value", ""),
        number_input=lambda *a, **kw: kw.get("value", 0),
        checkbox=lambda *a, **kw: False,
        button=lambda *a, **kw: False,
        columns=lambda n, **kw: [_rich_noop] * (n if isinstance(n, int) else len(n)),
        container=_rich_noop,
        expander=_rich_noop,
        spinner=_rich_noop,
        empty=lambda: _rich_noop,
        tabs=lambda labels: [_rich_noop] * len(labels),
        divider=lambda: None,
        image=lambda *a, **kw: None,
        json=lambda *a, **kw: None,
        subheader=lambda *a, **kw: None,
        header=lambda *a, **kw: None,
        title=lambda *a, **kw: None,
        progress=lambda *a, **kw: None,
        stop=lambda: None,
        rerun=lambda: None,
        sidebar=_rich_noop,
        form=_rich_noop,
        form_submit_button=lambda *a, **kw: False,
        multiselect=lambda *a, **kw: [],
        code=lambda *a, **kw: None,
        balloons=lambda: None,
        toast=lambda *a, **kw: None,
    )


@pytest.fixture
def app_ui(monkeypatch):
    """Load app.py with a full streamlit mock and initialized state."""
    frontend_dir = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(frontend_dir))

    fake_st = _make_rich_st()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    sys.modules.pop("app", None)
    mod = import_module("app")
    mod.st.session_state.clear()
    mod.init_state()
    return mod


# ── login_screen tests ────────────────────────────────────────────────────────

def test_login_screen_sign_in_branch(app_ui):
    """login_screen runs without error in Sign In mode (button not clicked)."""
    app_ui.login_screen()


def test_login_screen_fills_pending(app_ui):
    """login_screen transfers pending username/password into input keys."""
    app_ui.st.session_state["login_fill_pending_username"] = "testuser"
    app_ui.st.session_state["login_fill_pending_password"] = "testpass"
    app_ui.login_screen()
    assert app_ui.st.session_state.get("login_fill_pending_username") is None
    assert app_ui.st.session_state.get("login_fill_pending_password") is None


def test_login_screen_fills_auth_mode_pending(app_ui):
    """login_screen applies auth_mode_pending to auth_mode_widget."""
    app_ui.st.session_state["auth_mode_pending"] = "Sign In"
    app_ui.login_screen()
    assert app_ui.st.session_state.get("auth_mode_pending") is None


def test_login_screen_register_branch(app_ui, monkeypatch):
    """login_screen runs without error in Register mode."""
    def _radio_register(label, options, **kw):
        return "Register"
    monkeypatch.setattr(app_ui.st, "radio", _radio_register)
    app_ui.login_screen()


def test_login_screen_login_submit_success(app_ui, monkeypatch):
    """login_screen: successful backend login calls _set_login_state."""
    monkeypatch.setattr(app_ui.st, "button", lambda *a, **kw: True)
    called = {}

    def _fake_login(username, password):
        called["login"] = True
        return {"user_id": "u1"}, None

    def _fake_set(username, user_id):
        called["set"] = True

    monkeypatch.setattr(app_ui, "_login_with_backend", _fake_login)
    monkeypatch.setattr(app_ui, "_set_login_state", _fake_set)
    app_ui.login_screen()
    assert called.get("login")
    assert called.get("set")


def test_login_screen_login_backend_unavailable_demo(app_ui, monkeypatch):
    """login_screen: backend unavailable falls back to demo credential check."""
    monkeypatch.setattr(app_ui.st, "button", lambda *a, **kw: True)
    app_ui.st.session_state["login_username_input"] = "alice@example.com"
    app_ui.st.session_state["login_password_input"] = "123456"

    monkeypatch.setattr(app_ui, "_login_with_backend", lambda u, p: (None, "Backend unavailable"))

    called = {}
    monkeypatch.setattr(app_ui, "_set_login_state", lambda u, uid: called.update({"set": True}))
    app_ui.login_screen()
    assert called.get("set")


def test_login_screen_login_error(app_ui, monkeypatch):
    """login_screen: backend error shows st.error."""
    monkeypatch.setattr(app_ui.st, "button", lambda *a, **kw: True)
    errors = []
    monkeypatch.setattr(app_ui.st, "error", lambda msg, **kw: errors.append(msg))
    monkeypatch.setattr(app_ui, "_login_with_backend", lambda u, p: (None, "Invalid credentials."))
    app_ui.login_screen()
    assert errors


def test_login_screen_register_submit(app_ui, monkeypatch):
    """login_screen: register submit sets auth_notice on success."""
    def _radio_register(label, options, **kw):
        return "Register"
    monkeypatch.setattr(app_ui.st, "radio", _radio_register)
    monkeypatch.setattr(app_ui.st, "button", lambda *a, **kw: True)

    def _fake_register(username, password):
        return {"ok": True}, None

    monkeypatch.setattr(app_ui, "_register_with_backend", _fake_register)
    app_ui.login_screen()
    assert app_ui.st.session_state.get("auth_notice") == "Register success. Please sign in."


def test_login_screen_register_backend_unavailable(app_ui, monkeypatch):
    """login_screen: register with backend unavailable shows error."""
    def _radio_register(label, options, **kw):
        return "Register"
    monkeypatch.setattr(app_ui.st, "radio", _radio_register)
    monkeypatch.setattr(app_ui.st, "button", lambda *a, **kw: True)
    errors = []
    monkeypatch.setattr(app_ui.st, "error", lambda msg, **kw: errors.append(msg))
    monkeypatch.setattr(app_ui, "_register_with_backend", lambda u, p: (None, "Backend unavailable"))
    app_ui.login_screen()
    assert errors


# ── topbar tests ──────────────────────────────────────────────────────────────

def test_topbar_renders(app_ui):
    """topbar renders without error when user is set."""
    app_ui.st.session_state["user"] = {"name": "Alice", "avatar": "🧳", "username": "alice@example.com", "email": "alice@example.com", "prefs": []}
    app_ui.topbar()


def test_topbar_sign_out_clears_state(app_ui, monkeypatch):
    """topbar: clicking Sign out clears session state and reinitializes."""
    monkeypatch.setattr(app_ui.st, "button", lambda *a, **kw: True)
    app_ui.st.session_state["user"] = {"name": "Alice", "avatar": "🧳", "username": "alice@example.com", "email": "alice@example.com", "prefs": []}
    app_ui.st.session_state["logged_in"] = True
    app_ui.topbar()
    # After sign out, state is re-initialized (logged_in = False)
    assert app_ui.st.session_state.get("logged_in") is False


# ── main tests ────────────────────────────────────────────────────────────────

def test_main_not_logged_in_shows_login_screen(app_ui, monkeypatch):
    """main: unauthenticated user sees login_screen, not topbar."""
    app_ui.st.session_state["logged_in"] = False
    called = {}
    monkeypatch.setattr(app_ui, "login_screen", lambda: called.update({"login": True}))
    monkeypatch.setattr(app_ui, "topbar", lambda: called.update({"topbar": True}))
    app_ui.main()
    assert called.get("login")
    assert not called.get("topbar")


def test_main_logged_in_calls_topbar(app_ui, monkeypatch):
    """main: authenticated user sees topbar and page render."""
    app_ui.st.session_state["logged_in"] = True
    app_ui.st.session_state["main_section_key"] = "plan"
    app_ui.st.session_state["user"] = {"name": "Alice", "avatar": "🧳", "username": "alice@example.com", "email": "alice@example.com", "prefs": []}
    called = {}
    monkeypatch.setattr(app_ui, "topbar", lambda: called.update({"topbar": True}))

    def fake_render():
        called.update({"render": True})
    plan_mod = types.SimpleNamespace(render=fake_render)
    monkeypatch.setitem(sys.modules, "pages.plan", plan_mod)
    monkeypatch.setitem(sys.modules, "pages.my_trip", types.SimpleNamespace(render=lambda: None))
    monkeypatch.setitem(sys.modules, "pages.replan", types.SimpleNamespace(render=lambda: None))
    monkeypatch.setitem(sys.modules, "pages.security", types.SimpleNamespace(render=lambda: None))

    app_ui.main()
    assert called.get("topbar")


def test_main_pending_nav_applied(app_ui, monkeypatch):
    """main: pending_nav is transferred to main_section_key."""
    app_ui.st.session_state["logged_in"] = True
    app_ui.st.session_state["pending_nav"] = "replan"
    app_ui.st.session_state["user"] = {"name": "Alice", "avatar": "🧳", "username": "alice@example.com", "email": "alice@example.com", "prefs": []}
    monkeypatch.setattr(app_ui, "topbar", lambda: None)
    for name in ("pages.plan", "pages.my_trip", "pages.replan", "pages.security"):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(render=lambda: None))

    app_ui.main()
    assert app_ui.st.session_state.get("pending_nav") is None
