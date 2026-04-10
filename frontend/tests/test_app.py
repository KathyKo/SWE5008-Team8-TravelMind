from importlib import import_module
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def frontend_app_module(monkeypatch):
    frontend_dir = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(frontend_dir))

    fake_streamlit = types.SimpleNamespace(
        session_state={},
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
    assert frontend_app_module.st.session_state["main_section"] == "🛡️ Security"
    assert frontend_app_module.st.session_state["blocked_count"] == 0


def test_init_state_preserves_existing_values(frontend_app_module):
    frontend_app_module.st.session_state.clear()
    frontend_app_module.st.session_state["logged_in"] = True
    frontend_app_module.init_state()

    assert frontend_app_module.st.session_state["logged_in"] is True
    assert frontend_app_module.st.session_state["main_section"] == "🛡️ Security"
