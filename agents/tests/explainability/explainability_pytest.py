"""
Agent6 (Explainability) unit tests.

Run from repository root:
    pytest agents/tests/explainability/explainability_pytest.py -v
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

ea = import_module("agents.specialists.explainability_agent")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    return {
        "origin": "Singapore",
        "destination": "Tokyo",
        "dates": "2026-06-01 to 2026-06-07",
        "duration": "7 days",
        "budget": "SGD 3000",
        "preferences": "culture, food",
        "explain_option": "A",
        "itineraries": {
            "A": [{"day": 1, "date": "2026-06-01", "items": [
                {"key": "senso_ji", "name": "Senso-ji Temple",
                 "type": "attraction", "location": "Asakusa",
                 "duration_hrs": 2, "price_sgd": "Free"}
            ]}],
            "B": [{"day": 1, "date": "2026-06-01", "items": []}],
            "C": [{"day": 1, "date": "2026-06-01", "items": []}],
        },
        "option_meta": {
            "A": {"label": "Budget", "style": "budget", "budget": "SGD 2000"},
            "B": {"label": "Balanced", "style": "balanced", "budget": "SGD 2500"},
            "C": {"label": "Comfort", "style": "comfort", "budget": "SGD 3000"},
        },
        "planner_decision_trace": {
            "A": [{"day": 1, "name": "Senso-ji Temple",
                   "type": "attraction", "reason": "Top cultural landmark"}]
        },
        "chain_of_thought": "Agent3 selected Senso-ji for Day 1 due to cultural relevance.",
        "tool_log": [{"tool": "search_flights", "status": "ok"}],
        "hotel_options": [{"name": "Hotel Gracery Shinjuku", "location": "Shinjuku", "rating": 4.2}],
    }


@pytest.fixture
def state_no_itineraries():
    return {
        "origin": "Singapore", "destination": "Tokyo",
        "explain_option": "A", "itineraries": {}, "tool_log": [],
    }


@pytest.fixture
def state_with_option_b(base_state):
    return {**base_state, "explain_option": "B"}


@pytest.fixture
def patch_explain_llm():
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MagicMock(content=json.dumps({
        "overall_summary": "A culturally rich 7-day Tokyo itinerary.",
        "day_summaries": {"Day 1": "Start with Senso-ji Temple in Asakusa."},
    }))
    with patch("agents.specialists.explainability_agent._llm", return_value=mock_instance):
        yield


# ── Helper functions ──────────────────────────────────────────────────────────

class TestHelpers:

    def test_first_present_returns_first_non_empty(self):
        state = {"explain_option": "B", "selected_option": "A"}
        assert ea._first_present(state, "explain_option") == "B"

    def test_first_present_skips_empty_string(self):
        state = {"explain_option": "", "selected_option": "A"}
        assert ea._first_present(state, "explain_option", "selected_option") == "A"

    def test_first_present_returns_default_when_missing(self):
        assert ea._first_present({}, "explain_option", default="X") == "X"

    def test_collect_itineraries_finds_itineraries_key(self):
        state = {"itineraries": {"A": [{"day": 1}], "B": [], "C": []}}
        assert "A" in ea._collect_itineraries(state)

    def test_collect_itineraries_prefers_final_itineraries(self):
        state = {
            "final_itineraries": {"A": [{"day": 1}]},
            "itineraries": {"A": [{"day": 99}]},
        }
        assert ea._collect_itineraries(state)["A"][0]["day"] == 1

    def test_collect_itineraries_empty_when_missing(self):
        assert ea._collect_itineraries({}) == {}

    def test_selected_option_uses_explain_option(self):
        assert ea._selected_option({"explain_option": "B"}, {"A": [], "B": [], "C": []}) == "B"

    def test_selected_option_defaults_to_a(self):
        assert ea._selected_option({}, {"A": [], "B": [], "C": []}) == "A"

    def test_selected_option_falls_back_to_first_key(self):
        assert ea._selected_option({}, {"C": []}) == "C"

    def test_build_place_lookup_indexes_by_normalized_name(self):
        state = {"inventory": {"attractions": [{"name": "Senso-ji Temple", "type": "attraction"}]}}
        lookup = ea._build_place_lookup(state)
        assert any("senso" in k for k in lookup)


# ── Empty itinerary path ──────────────────────────────────────────────────────

class TestEmptyItineraries:

    def test_returns_empty_summary(self, state_no_itineraries):
        result = ea.explainability_agent(state_no_itineraries)
        assert result["summary"]["overall_summary"] == ""
        assert result["item_explanations"] == {"by_key": {}, "by_occurrence": {}}

    def test_returns_required_keys(self, state_no_itineraries):
        result = ea.explainability_agent(state_no_itineraries)
        for key in ("explain_option", "summary", "item_explanations",
                    "chain_of_thought", "agent_steps"):
            assert key in result

    def test_explain_option_defaults_to_a(self):
        result = ea.explainability_agent({"itineraries": {}, "explain_option": "", "tool_log": []})
        assert result["explain_option"] == "A"


# ── Full explainability path ──────────────────────────────────────────────────

class TestExplainabilityAgentFull:

    def test_output_has_required_keys(self, base_state, patch_explain_llm):
        result = ea.explainability_agent(base_state)
        for key in ("explain_option", "summary", "item_explanations",
                    "chain_of_thought", "agent_steps", "evidence"):
            assert key in result

    def test_explain_option_matches_request(self, base_state, patch_explain_llm):
        assert ea.explainability_agent(base_state)["explain_option"] == "A"

    def test_summary_has_required_fields(self, base_state, patch_explain_llm):
        summary = ea.explainability_agent(base_state)["summary"]
        assert "overall_summary" in summary
        assert "day_summaries" in summary

    def test_item_explanations_structure(self, base_state, patch_explain_llm):
        item_exp = ea.explainability_agent(base_state)["item_explanations"]
        assert "by_key" in item_exp
        assert "by_occurrence" in item_exp

    def test_senso_ji_in_item_explanations(self, base_state, patch_explain_llm):
        by_key = ea.explainability_agent(base_state)["item_explanations"]["by_key"]
        assert any("senso" in k.lower() for k in by_key)

    def test_chain_of_thought_includes_planner_cot(self, base_state, patch_explain_llm):
        result = ea.explainability_agent(base_state)
        assert result["planner_chain_of_thought"] in result["chain_of_thought"]

    def test_selects_option_b_when_requested(self, state_with_option_b, patch_explain_llm):
        assert ea.explainability_agent(state_with_option_b)["explain_option"] == "B"

    def test_agent_steps_is_list(self, base_state, patch_explain_llm):
        assert isinstance(ea.explainability_agent(base_state)["agent_steps"], list)
