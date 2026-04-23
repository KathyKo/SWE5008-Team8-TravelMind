import sys
from importlib import import_module
from unittest.mock import MagicMock

import pytest

_STUB_MODULES = [
    "llm_guard",
    "llm_guard.input_scanners",
    "llm_guard.output_scanners",
    "torch",
    "transformers",
    "agents.db",
    "agents.db.crud",
    "agents.db.database",
    "agents.db.models",
]

for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

if not getattr(sys.modules["agents.db.database"], "SessionLocal", None):
    sys.modules["agents.db.database"].SessionLocal = MagicMock(return_value=MagicMock())
if not getattr(sys.modules["agents.db.crud"], "update_plan_result", None):
    sys.modules["agents.db.crud"].update_plan_result = MagicMock()

debate = import_module("agents.specialists.debate_agent")


@pytest.fixture
def base_state():
    return {
        "plan_id": "plan-1",
        "origin": "Singapore",
        "destination": "Tokyo",
        "dates": "2026-06-01 to 2026-06-05",
        "duration": "5 days",
        "budget": "SGD 3000",
        "preferences": "food, culture",
        "itineraries": {
            "A": [{"day": "Day 1", "items": [{"key": "a1", "icon": "activity", "name": "Asakusa Walk"}]}],
            "B": [{"day": "Day 1", "items": [{"key": "b1", "icon": "activity", "name": "Shibuya"}]}],
            "C": [{"day": "Day 1", "items": [{"key": "c1", "icon": "activity", "name": "Ueno"}]}],
        },
        "option_meta": {"A": {"style": "budget"}, "B": {"style": "balanced"}, "C": {"style": "comfort"}},
        "flight_options_outbound": [{"name": "SQ632"}],
        "flight_options_return": [{"name": "SQ633"}],
        "hotel_options": [{"name": "Shinjuku Hotel"}],
        "tool_log": [{"tool": "research"}],
        "planner_decision_trace": {"picked": "A"},
        "chain_of_thought": "initial reasoning",
        "debate_history": [],
    }
