import sys
from unittest.mock import MagicMock
from importlib import import_module

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy transitive dependencies that intent_profile doesn't need
# but get pulled in via agents/__init__.py → input_guard_agent → llm_guard
# and other tool modules. MagicMock auto-creates any sub-attribute on access.
# ---------------------------------------------------------------------------
_STUB_MODULES = [
    "llm_guard",
    "llm_guard.input_scanners",
    "llm_guard.output_scanners",
    "torch",
    "transformers",
]

for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

ip = import_module("agents.specialists.intent_profile")


@pytest.fixture
def intent_state_builder():
    """Build a minimal state dict for intent_profile (no core fields pre-filled)."""

    def _build(
        text: str,
        *,
        origin: str | None = None,
        destination: str | None = None,
        dates: str | None = None,
        budget: str | None = None,
        preferences: str | None = None,
        duration: str | None = None,
        outbound_time_pref: str | None = None,
        return_time_pref: str | None = None,
        user_id: str = "test_user",
    ) -> dict:
        return {
            "messages": [{"role": "user", "content": text}],
            "origin": origin,
            "destination": destination,
            "dates": dates,
            "budget": budget,
            "preferences": preferences,
            "duration": duration,
            "outbound_time_pref": outbound_time_pref,
            "return_time_pref": return_time_pref,
            "user_id": user_id,
        }

    return _build


@pytest.fixture
def full_state_builder():
    """Build a state with all core fields pre-filled so the LLM branch is skipped."""

    def _build(
        text: str = "Plan a trip",
        *,
        origin: str = "Singapore",
        destination: str = "Kyoto, Japan",
        dates: str = "2026-05-10 to 2026-05-15",
        budget: str = "SGD 5000 strict",
        preferences: str = "vegetarian, historical, relaxed",
        duration: str = "5 days",
        outbound_time_pref: str = "morning",
        return_time_pref: str = "afternoon",
        user_id: str = "test_user",
    ) -> dict:
        return {
            "messages": [{"role": "user", "content": text}],
            "origin": origin,
            "destination": destination,
            "dates": dates,
            "budget": budget,
            "preferences": preferences,
            "duration": duration,
            "outbound_time_pref": outbound_time_pref,
            "return_time_pref": return_time_pref,
            "user_id": user_id,
        }

    return _build
