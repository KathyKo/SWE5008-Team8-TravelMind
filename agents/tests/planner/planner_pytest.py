"""
Agent3 (Planner) unit tests.

Run from repository root:
    pytest agents/tests/planner/planner_pytest.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from importlib import import_module

import pytest

# Add repo root to path so 'agents' is importable
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stub security deps not installed in this environment
for _mod in [
    "llm_guard", "llm_guard.input_scanners", "llm_guard.output_scanners",
    "tools.security.llm_guard_scanner", "tools.security.sanitiser",
    "tools.security.injection_detector", "tools.security.pii_scanner",
    "tools.security.openai_moderation", "tools.security.hallucination_guard",
    "tools.security.security_logger",
]:
    sys.modules.setdefault(_mod, MagicMock())

pa = import_module("agents.specialists.planner_agent")
ra = import_module("agents.specialists.research_agent")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_state():
    return {
        "origin": "Singapore",
        "destination": "Tokyo",
        "dates": "2026-06-01 to 2026-06-07",
        "duration": "7 days",
        "budget": "SGD 3000",
        "preferences": "culture, food",
    }


@pytest.fixture
def hard_constraints_state():
    return {
        "origin": "Singapore",
        "destination": "Tokyo",
        "hard_constraints": {
            "origin": "Singapore",
            "destination": "Tokyo",
            "start_date": "2026-06-01",
            "end_date": "2026-06-07",
            "budget": {"amount": 3000, "currency": "SGD", "flexibility": "flexible"},
            "requirements": ["vegetarian"],
        },
        "soft_preferences": {
            "travel_style": "balanced",
            "interest_tags": ["culture", "food"],
            "vibe": "relaxed",
            "priority": "cost_effective",
            "pace": "moderate",
        },
    }


@pytest.fixture
def mock_research_result():
    return {
        "flight_options_outbound": [
            {"airline": "SQ", "flight_number": "SQ637", "price_usd": 450,
             "departure_time": "08:00", "arrival_time": "16:00",
             "departure_airport": "SIN", "arrival_airport": "NRT",
             "duration_min": 420, "travel_class": "economy", "display": "SQ637 SIN->NRT $450"}
        ],
        "flight_options_return": [
            {"airline": "SQ", "flight_number": "SQ638", "price_usd": 430,
             "departure_time": "18:00", "arrival_time": "00:00",
             "departure_airport": "NRT", "arrival_airport": "SIN",
             "duration_min": 420, "travel_class": "economy", "display": "SQ638 NRT->SIN $430"}
        ],
        "hotel_options": [
            {"name": "Hotel Gracery Shinjuku", "price_per_night_usd": 120,
             "rating": 4.2, "location": "Shinjuku", "display": "Hotel Gracery $120/night"}
        ],
        "compact_attractions": [
            {"item_key": "senso_ji", "name": "Senso-ji Temple", "type": "attraction",
             "location": "Asakusa", "price_sgd": "Free", "duration_hrs": 2}
        ],
        "compact_restaurants": [
            {"item_key": "ichiran_ramen", "name": "Ichiran Ramen", "type": "restaurant",
             "location": "Shinjuku", "price_sgd": "SGD 20", "duration_hrs": 1}
        ],
        "tool_log": [{"tool": "search_flights", "status": "ok"}],
    }


@pytest.fixture
def mock_itineraries():
    return {
        "A": {"days": [{"day": 1, "date": "2026-06-01", "activities": []}]},
        "B": {"days": [{"day": 1, "date": "2026-06-01", "activities": []}]},
        "C": {"days": [{"day": 1, "date": "2026-06-01", "activities": []}]},
    }


@pytest.fixture
def mock_llm(mock_itineraries):
    instance = MagicMock()
    instance.invoke.return_value = MagicMock(content=json.dumps(mock_itineraries))
    return instance


# ── _normalize_trip_state ─────────────────────────────────────────────────────

class TestNormalizeTripState:

    def test_flat_input_preserved(self, flat_state):
        result = ra._normalize_trip_state(flat_state)
        assert result["origin"] == "Singapore"
        assert result["destination"] == "Tokyo"
        assert result["dates"] == "2026-06-01 to 2026-06-07"
        assert result["duration"] == "7 days"
        assert result["budget"] == "SGD 3000"
        assert result["preferences"] == "culture, food"

    def test_hard_constraints_input(self, hard_constraints_state):
        result = ra._normalize_trip_state(hard_constraints_state)
        assert result["origin"] == "Singapore"
        assert result["destination"] == "Tokyo"
        assert "3000" in result["budget"]
        assert "SGD" in result["budget"]

    def test_budget_as_dict(self, flat_state):
        state = {**flat_state, "budget": {"amount": 2500, "currency": "SGD", "flexibility": "strict"}}
        result = ra._normalize_trip_state(state)
        assert "2500" in result["budget"]
        assert "SGD" in result["budget"]

    def test_duration_computed_from_dates(self):
        state = {
            "origin": "Singapore", "destination": "Tokyo",
            "hard_constraints": {"start_date": "2026-06-01", "end_date": "2026-06-05"},
        }
        assert ra._normalize_trip_state(state)["duration"] == "5 days"

    def test_preferences_built_from_soft_prefs(self, flat_state):
        state = {**flat_state, "preferences": "",
                 "soft_preferences": {"interest_tags": ["culture", "food"],
                                      "travel_style": "relaxed", "vibe": ""}}
        result = ra._normalize_trip_state(state)
        assert "culture" in result["preferences"]
        assert "food" in result["preferences"]


# ── planner_from_research ─────────────────────────────────────────────────────

class TestPlannerFromResearch:

    def test_missing_fields_returns_error(self, mock_research_result):
        result = pa.planner_from_research({"origin": "Singapore"}, mock_research_result)
        assert "error" in result
        assert "Missing required fields" in result["error"]

    def test_output_has_required_keys(self, flat_state, mock_research_result, mock_llm):
        with patch("agents.specialists.planner_agent._llm", return_value=mock_llm), \
             patch("agents.specialists.planner_agent._llm_select_seed",
                   return_value=("seed_A", "culture day", "best match")):
            result = pa.planner_from_research(flat_state, mock_research_result)
        assert "error" not in result
        for key in ("itineraries", "option_meta", "flight_options_outbound",
                    "flight_options_return", "hotel_options"):
            assert key in result

    def test_itineraries_has_abc(self, flat_state, mock_research_result, mock_llm):
        with patch("agents.specialists.planner_agent._llm", return_value=mock_llm), \
             patch("agents.specialists.planner_agent._llm_select_seed",
                   return_value=("seed_A", "culture day", "best match")):
            result = pa.planner_from_research(flat_state, mock_research_result)
        if "error" not in result:
            for opt in ("A", "B", "C"):
                assert opt in result["itineraries"]

    def test_option_meta_has_abc(self, flat_state, mock_research_result, mock_llm):
        with patch("agents.specialists.planner_agent._llm", return_value=mock_llm), \
             patch("agents.specialists.planner_agent._llm_select_seed",
                   return_value=("seed_A", "culture day", "best match")):
            result = pa.planner_from_research(flat_state, mock_research_result)
        if "error" not in result:
            for opt in ("A", "B", "C"):
                assert opt in result["option_meta"]


# ── planner_agent ─────────────────────────────────────────────────────────────

class TestPlannerAgent:

    def test_output_has_required_keys(self, flat_state, mock_research_result, mock_llm):
        with patch("agents.specialists.planner_agent.research_agent", return_value=mock_research_result), \
             patch("agents.specialists.planner_agent._llm", return_value=mock_llm), \
             patch("agents.specialists.planner_agent._llm_select_seed",
                   return_value=("seed_A", "culture day", "best match")):
            result = pa.planner_agent(flat_state)
        assert "error" not in result
        for key in ("itineraries", "option_meta", "flight_options_outbound",
                    "flight_options_return", "hotel_options", "tool_log"):
            assert key in result

    def test_research_error_propagates(self, flat_state):
        with patch("agents.specialists.planner_agent.research_agent",
                   return_value={"error": "API timeout"}):
            result = pa.planner_agent(flat_state)
        assert result.get("error") == "API timeout"

    def test_accepts_hard_constraints_format(self, hard_constraints_state, mock_research_result, mock_llm):
        with patch("agents.specialists.planner_agent.research_agent", return_value=mock_research_result), \
             patch("agents.specialists.planner_agent._llm", return_value=mock_llm), \
             patch("agents.specialists.planner_agent._llm_select_seed",
                   return_value=("seed_A", "culture day", "best match")):
            result = pa.planner_agent(hard_constraints_state)
        assert "error" not in result


# ── revise_itinerary ──────────────────────────────────────────────────────────

class TestReviseItinerary:

    def test_error_when_cache_empty(self, flat_state, mock_itineraries):
        pa._inventory_cache.clear()
        result = pa.revise_itinerary(flat_state, "Add more culture",
                                     {"itineraries": mock_itineraries, "option_meta": {}})
        assert "error" in result
        assert "No cached inventory" in result["error"]

    def test_revises_with_populated_cache(self, flat_state, mock_research_result,
                                          mock_itineraries, mock_llm):
        pa._inventory_cache.update({
            "prompt_inventory": {},
            "flight_out_text": "SQ637 SIN->NRT",
            "flight_ret_text": "SQ638 NRT->SIN",
            "hotel_list_text": "Hotel Gracery",
            "att_list_text": "Senso-ji (Free)",
            "rest_list_text": "Ichiran Ramen",
            "research": {}, "tool_log": [],
            "compact_flights_out": mock_research_result["flight_options_outbound"],
            "compact_flights_ret": mock_research_result["flight_options_return"],
            "compact_hotels": mock_research_result["hotel_options"],
            "compact_attractions": mock_research_result["compact_attractions"],
            "compact_restaurants": mock_research_result["compact_restaurants"],
        })
        with patch("agents.specialists.planner_agent._llm", return_value=mock_llm):
            result = pa.revise_itinerary(flat_state, "Add more cultural activities",
                                         {"itineraries": mock_itineraries, "option_meta": {}})
        assert "error" not in result
        assert "itineraries" in result
