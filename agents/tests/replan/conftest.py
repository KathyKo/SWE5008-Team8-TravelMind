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
if not getattr(sys.modules["agents.db.crud"], "load_plan", None):
    sys.modules["agents.db.crud"].load_plan = MagicMock(return_value=None)
if not getattr(sys.modules["agents.db.crud"], "update_plan_result", None):
    sys.modules["agents.db.crud"].update_plan_result = MagicMock()

dr = import_module("agents.specialists.dynamic_replan_agent")


@pytest.fixture
def replan_payload():
    return {
        "scenario_id": "s-1",
        "original_recommended_plan": {
            "option_key": "A",
            "label": "Recommended",
            "itinerary_id": "IT-001",
            "composite_score": 88,
            "plan": [
                {
                    "day": "Day 1",
                    "items": [
                        {"key": "flight_outbound", "icon": "flight", "name": "Singapore Changi Airport → Haneda Airport | dep 2026-06-01", "time": "08:00"},
                        {"key": "hotel_1", "icon": "hotel", "name": "Tokyo Inn", "time": "20:00"},
                        {"key": "a1", "icon": "activity", "name": "Edo-Tokyo Museum", "time": "10:00"},
                    ],
                },
                {
                    "day": "Day 2",
                    "items": [
                        {"key": "r1", "icon": "restaurant", "name": "Sushi Place", "time": "12:00"},
                        {"key": "flight_return", "icon": "flight", "name": "Haneda Airport → Singapore Changi Airport | dep 2026-06-05", "time": "18:00"},
                    ],
                },
            ],
        },
        "user_replan_request": {
            "replan_scope": {
                "start_day": "Day 1",
                "end_day": "Day 2",
                "locked_days": ["Day 2"],
                "allow_replace_flight": False,
                "allow_replace_hotel": False,
            },
            "updated_user_intent": {
                "new_preferences": ["indoor", "anime"],
                "must_keep": ["flight_outbound"],
                "avoid": ["temple", "outdoor park"],
                "meal_preference": "vegetarian",
                "pace_preference": "relaxed",
                "budget_guardrail": {"currency": "SGD"},
            },
            "trigger_events": [
                {
                    "type": "venue_closure",
                    "detail": "Edo Tokyo Museum is closed due to weather",
                    "affected_item_key": "attraction_01_edo_tokyo_museum",
                    "severity": "high",
                    "day": "Day 1",
                }
            ],
            "output_expectation": {"format": "json"},
        },
    }
