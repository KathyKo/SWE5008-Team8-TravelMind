"""
Agent6 (Explainability) unit tests.

Run from repository root:
    pytest agents/tests/explainability/test_explainability_agent.py -v

Fixtures are defined in conftest.py (same folder).
All tests are mocked — no real LLM calls are made.
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

from agents.specialists.explainability_agent import (  # noqa: E402
    explainability_agent,
    _collect_itineraries,
    _selected_option,
    _first_present,
    _build_place_lookup,
)


# ── 1. Helper functions ───────────────────────────────────────────────────────

class TestHelpers:

    def test_first_present_returns_first_non_empty(self):
        state = {"explain_option": "B", "selected_option": "A"}
        assert _first_present(state, "explain_option") == "B"

    def test_first_present_skips_empty_string(self):
        state = {"explain_option": "", "selected_option": "A"}
        assert _first_present(state, "explain_option", "selected_option") == "A"

    def test_first_present_returns_default_when_missing(self):
        state = {}
        assert _first_present(state, "explain_option", default="X") == "X"

    def test_collect_itineraries_finds_itineraries_key(self):
        state = {"itineraries": {"A": [{"day": 1}], "B": [], "C": []}}
        result = _collect_itineraries(state)
        assert "A" in result

    def test_collect_itineraries_prefers_final_itineraries(self):
        state = {
            "final_itineraries": {"A": [{"day": 1}]},
            "itineraries": {"A": [{"day": 99}]},
        }
        result = _collect_itineraries(state)
        assert result["A"][0]["day"] == 1  # final_itineraries wins

    def test_collect_itineraries_returns_empty_when_missing(self):
        result = _collect_itineraries({})
        assert result == {}

    def test_selected_option_uses_explain_option(self):
        state = {"explain_option": "B"}
        itineraries = {"A": [], "B": [], "C": []}
        assert _selected_option(state, itineraries) == "B"

    def test_selected_option_defaults_to_a(self):
        state = {}
        itineraries = {"A": [], "B": [], "C": []}
        assert _selected_option(state, itineraries) == "A"

    def test_selected_option_falls_back_to_first_key(self):
        state = {}
        itineraries = {"C": []}
        assert _selected_option(state, itineraries) == "C"

    def test_build_place_lookup_indexes_attractions(self):
        # _build_place_lookup reads from state["research"]["maps_attractions"]
        # or state["inventory"]["attractions"]
        state = {
            "inventory": {
                "attractions": [{"name": "Senso-ji Temple", "type": "attraction"}]
            }
        }
        lookup = _build_place_lookup(state)
        # Key is normalized: "Senso-ji Temple" -> "senso ji temple"
        assert any("senso" in k for k in lookup), \
            f"Expected senso-ji in lookup keys, got: {list(lookup.keys())}"


# ── 2. explainability_agent — empty itinerary path ────────────────────────────

class TestExplainabilityAgentEmpty:

    def test_empty_itineraries_returns_empty_summary(self, state_no_itineraries):
        result = explainability_agent(state_no_itineraries)
        assert "error" not in result
        assert result["summary"]["overall_summary"] == ""
        assert result["item_explanations"] == {"by_key": {}, "by_occurrence": {}}

    def test_empty_itineraries_still_returns_required_keys(self, state_no_itineraries):
        result = explainability_agent(state_no_itineraries)
        for key in ("explain_option", "summary", "item_explanations",
                    "chain_of_thought", "agent_steps"):
            assert key in result, f"Missing key: {key}"

    def test_empty_itineraries_explain_option_defaults_to_a(self, state_no_itineraries):
        state = {**state_no_itineraries, "explain_option": ""}
        result = explainability_agent(state)
        assert result["explain_option"] == "A"


# ── 3. explainability_agent — full path ───────────────────────────────────────

class TestExplainabilityAgentFull:

    def test_output_schema(self, base_state, patch_explain_llm):
        result = explainability_agent(base_state)
        assert "error" not in result
        for key in ("explain_option", "summary", "item_explanations",
                    "chain_of_thought", "agent_steps", "evidence"):
            assert key in result, f"Missing key: {key}"

    def test_explain_option_matches_request(self, base_state, patch_explain_llm):
        result = explainability_agent(base_state)
        assert result["explain_option"] == "A"

    def test_summary_has_required_fields(self, base_state, patch_explain_llm):
        result = explainability_agent(base_state)
        summary = result["summary"]
        assert "overall_summary" in summary
        assert "day_summaries" in summary

    def test_item_explanations_has_by_key_and_by_occurrence(self, base_state, patch_explain_llm):
        result = explainability_agent(base_state)
        item_exp = result["item_explanations"]
        assert "by_key" in item_exp
        assert "by_occurrence" in item_exp

    def test_senso_ji_appears_in_item_explanations(self, base_state, patch_explain_llm):
        result = explainability_agent(base_state)
        by_key = result["item_explanations"]["by_key"]
        assert any("senso" in k.lower() for k in by_key), \
            f"Expected senso_ji in item_explanations, got keys: {list(by_key.keys())}"

    def test_chain_of_thought_includes_planner_cot(self, base_state, patch_explain_llm):
        result = explainability_agent(base_state)
        assert "Agent3" in result["chain_of_thought"] or \
               result["planner_chain_of_thought"] in result["chain_of_thought"]

    def test_selects_option_b_when_requested(self, state_with_option_b, patch_explain_llm):
        result = explainability_agent(state_with_option_b)
        assert result["explain_option"] == "B"

    def test_agent_steps_is_list(self, base_state, patch_explain_llm):
        result = explainability_agent(base_state)
        assert isinstance(result["agent_steps"], list)
