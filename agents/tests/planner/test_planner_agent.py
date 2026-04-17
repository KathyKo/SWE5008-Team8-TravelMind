"""
Agent3 (Planner) unit tests.

Run from repository root:
    pytest agents/tests/planner/test_planner_agent.py -v

Fixtures are defined in conftest.py (same folder).
All tests are mocked — no real LLM or API calls are made.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stub optional security deps not installed in all environments
for _mod in (
    "llm_guard",
    "llm_guard.input_scanners",
    "llm_guard.output_scanners",
    "tools.security.llm_guard_scanner",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from agents.specialists.research_agent import _normalize_trip_state  # noqa: E402
from agents.specialists.planner_agent import (  # noqa: E402
    planner_agent,
    planner_from_research,
    revise_itinerary,
    _inventory_cache,
)


# ── 1. _normalize_trip_state ──────────────────────────────────────────────────

class TestNormalizeTripState:

    def test_flat_input_preserved(self, flat_state):
        result = _normalize_trip_state(flat_state)
        assert result["origin"] == "Singapore"
        assert result["destination"] == "Tokyo"
        assert result["dates"] == "2026-06-01 to 2026-06-07"
        assert result["duration"] == "7 days"
        assert result["budget"] == "SGD 3000"
        assert result["preferences"] == "culture, food"

    def test_hard_constraints_input(self, hard_constraints_state):
        result = _normalize_trip_state(hard_constraints_state)
        assert result["origin"] == "Singapore"
        assert result["destination"] == "Tokyo"
        assert result["dates"] == "2026-06-01 to 2026-06-07"
        assert result["duration"] == "7 days"
        assert "3000" in result["budget"]
        assert "SGD" in result["budget"]

    def test_budget_dict_input(self, flat_state):
        state = {**flat_state, "budget": {"amount": 2500, "currency": "SGD", "flexibility": "strict"}}
        result = _normalize_trip_state(state)
        assert "2500" in result["budget"]
        assert "SGD" in result["budget"]

    def test_preferences_built_from_soft_prefs(self, flat_state):
        state = {**flat_state, "preferences": ""}
        state["soft_preferences"] = {
            "interest_tags": ["culture", "food"],
            "travel_style": "relaxed",
            "vibe": "",
        }
        result = _normalize_trip_state(state)
        assert "culture" in result["preferences"]
        assert "food" in result["preferences"]

    def test_duration_computed_from_dates(self):
        state = {
            "origin": "Singapore",
            "destination": "Tokyo",
            "hard_constraints": {
                "start_date": "2026-06-01",
                "end_date": "2026-06-05",
            },
        }
        result = _normalize_trip_state(state)
        assert result["duration"] == "5 days"


# ── 2. planner_from_research ──────────────────────────────────────────────────

class TestPlannerFromResearch:

    def test_output_schema(self, flat_state, mock_research_result, patch_planner_llm):
        result = planner_from_research(flat_state, mock_research_result)
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "itineraries" in result
        assert "option_meta" in result
        assert "flight_options_outbound" in result
        assert "flight_options_return" in result
        assert "hotel_options" in result

    def test_itineraries_has_abc_options(self, flat_state, mock_research_result, patch_planner_llm):
        result = planner_from_research(flat_state, mock_research_result)
        if "error" not in result:
            itins = result.get("itineraries", {})
            for option in ("A", "B", "C"):
                assert option in itins, f"Missing itinerary option {option}"

    def test_missing_required_fields_returns_error(self, mock_research_result):
        incomplete_state = {"origin": "Singapore"}
        result = planner_from_research(incomplete_state, mock_research_result)
        assert "error" in result
        assert "Missing required fields" in result["error"]

    def test_option_meta_has_abc(self, flat_state, mock_research_result, patch_planner_llm):
        result = planner_from_research(flat_state, mock_research_result)
        if "error" not in result:
            meta = result.get("option_meta", {})
            for option in ("A", "B", "C"):
                assert option in meta, f"Missing option_meta for {option}"


# ── 3. planner_agent ─────────────────────────────────────────────────────────

class TestPlannerAgent:

    def test_output_schema(self, flat_state, patch_research_agent, patch_planner_llm):
        result = planner_agent(flat_state)
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "itineraries" in result
        assert "option_meta" in result
        assert "flight_options_outbound" in result
        assert "flight_options_return" in result
        assert "hotel_options" in result
        assert "tool_log" in result

    def test_propagates_research_error(self, flat_state):
        from unittest.mock import patch
        with patch("agents.specialists.planner_agent.research_agent",
                   return_value={"error": "API timeout"}):
            result = planner_agent(flat_state)
        assert "error" in result
        assert result["error"] == "API timeout"

    def test_accepts_hard_constraints_state(
        self, hard_constraints_state, patch_research_agent, patch_planner_llm
    ):
        result = planner_agent(hard_constraints_state)
        assert "error" not in result, f"Unexpected error: {result.get('error')}"


# ── 4. revise_itinerary ───────────────────────────────────────────────────────

class TestReviseItinerary:

    def test_returns_error_when_cache_empty(self, flat_state, mock_planner_result):
        _inventory_cache.clear()
        result = revise_itinerary(flat_state, "Add more cultural activities", mock_planner_result)
        assert "error" in result
        assert "No cached inventory" in result["error"]

    def test_output_schema_with_populated_cache(
        self, flat_state, mock_planner_result, populated_inventory, patch_planner_llm
    ):
        result = revise_itinerary(
            flat_state,
            "Option A lacks cultural activities on Day 3",
            mock_planner_result,
        )
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "itineraries" in result
        assert "option_meta" in result
