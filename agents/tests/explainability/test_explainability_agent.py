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
    _metadata_score,
    _rating_text,
    _normalized_name,
    _numeric,
    _integer,
    _occurrence_id,
    _item_explanation,
    _item_display,
    _get_decision_trace,
    _strip_numeric_artifacts,
    _build_agent_steps,
    _summary_language,
    _collect_pref_tags,
    _register_place,
    _trace_name_in_items,
    _sanitize_trace_entry,
    _fallback_summary,
    _summary_text,
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


# ── 4. Scoring helpers ────────────────────────────────────────────────────────

class TestMetadataScore:

    def test_empty_metadata_scores_zero(self):
        assert _metadata_score({}) == 0

    def test_single_field_scores_one(self):
        assert _metadata_score({"rating": 4.2}) == 1

    def test_all_fields_scores_seven(self):
        meta = {
            "rating": 4.2,
            "reviews": 1000,
            "hours": "9AM-6PM",
            "weekday_descriptions": ["Mon: 9AM-6PM"],
            "lat": 35.71,
            "lng": 139.79,
            "address": "1-2-3 Asakusa",
        }
        assert _metadata_score(meta) == 7

    def test_none_values_not_counted(self):
        assert _metadata_score({"rating": None, "reviews": "", "lat": []}) == 0

    def test_partial_fields_counted_correctly(self):
        assert _metadata_score({"rating": 4.0, "lat": 35.71, "lng": 139.79}) == 3


# ── 5. Rating text helper ─────────────────────────────────────────────────────

class TestRatingText:

    def test_rating_with_reviews(self):
        result = _rating_text({"rating": 4.2, "reviews": 1500})
        assert "4.2" in result
        assert "1,500" in result
        assert "reviews" in result

    def test_rating_without_reviews(self):
        result = _rating_text({"rating": 3.8})
        assert "3.8" in result
        assert "/ 5" in result
        assert "reviews" not in result

    def test_no_rating_returns_empty(self):
        assert _rating_text({}) == ""
        assert _rating_text({"rating": None}) == ""
        assert _rating_text({"rating": ""}) == ""

    def test_string_rating_parsed(self):
        result = _rating_text({"rating": "4.5"})
        assert "4.5" in result


# ── 6. Low-level normalisation helpers ───────────────────────────────────────

class TestNormalizedName:

    def test_lowercases_input(self):
        assert _normalized_name("Senso-Ji Temple") == _normalized_name("senso-ji temple")

    def test_strips_punctuation(self):
        result = _normalized_name("Café & Bar!")
        assert "&" not in result
        assert "!" not in result

    def test_collapses_whitespace(self):
        assert "  " not in _normalized_name("too  many   spaces")

    def test_empty_string(self):
        assert _normalized_name("") == ""


class TestNumericAndInteger:

    def test_numeric_float_string(self):
        assert _numeric("4.2") == pytest.approx(4.2)

    def test_numeric_comma_number(self):
        assert _numeric("1,234.5") == pytest.approx(1234.5)

    def test_numeric_none_returns_none(self):
        assert _numeric(None) is None

    def test_numeric_non_numeric_string(self):
        assert _numeric("hello") is None

    def test_integer_extracts_first_digits(self):
        assert _integer("1,500 reviews") == 1500

    def test_integer_empty_returns_none(self):
        assert _integer("") is None

    def test_integer_no_digits_returns_none(self):
        assert _integer("no numbers here") is None


# ── 7. Occurrence ID ─────────────────────────────────────────────────────────

class TestOccurrenceId:

    def test_format_is_correct(self):
        oid = _occurrence_id("A", 0, 0, {"key": "senso_ji"})
        assert oid == "A__d01__i01__senso_ji"

    def test_day_and_item_are_one_indexed(self):
        oid = _occurrence_id("B", 2, 4, {"key": "meiji"})
        assert "d03" in oid
        assert "i05" in oid

    def test_missing_key_uses_item_fallback(self):
        oid = _occurrence_id("A", 0, 0, {})
        assert "item" in oid


# ── 8. Item explanation ───────────────────────────────────────────────────────

class TestItemExplanation:

    def _place_lookup(self):
        return {
            "senso ji temple": {
                "name": "Senso-ji Temple", "kind": "activity",
                "type": "attraction", "category": "",
                "rating": "4.5", "reviews": "2000",
                "address": "Asakusa", "lat": 35.71, "lng": 139.79,
                "description": "", "search_query": "",
            }
        }

    def test_attraction_has_rating(self):
        item = {"key": "senso_ji", "name": "Senso-ji Temple",
                "icon": "attraction", "time": "09:00"}
        result = _item_explanation("A", "Day 1", 0, 0, item, self._place_lookup(), {})
        assert "4.5" in result["rating"]

    def test_flight_has_no_rating(self):
        item = {"key": "flight_outbound", "name": "SQ637",
                "icon": "flight", "time": "08:00"}
        result = _item_explanation("A", "Day 1", 0, 0, item, {}, {})
        assert result["rating"] == ""

    def test_occurrence_id_in_result(self):
        item = {"key": "senso_ji", "name": "Senso-ji Temple",
                "icon": "attraction", "time": "09:00"}
        result = _item_explanation("A", "Day 1", 0, 0, item, self._place_lookup(), {})
        assert "occurrence_id" in result
        assert result["occurrence_id"].startswith("A__")

    def test_item_display_subset(self):
        item = {"key": "senso_ji", "name": "Senso-ji Temple",
                "icon": "attraction", "time": "09:00"}
        explanation = _item_explanation("A", "Day 1", 0, 0, item, self._place_lookup(), {})
        display = _item_display(explanation)
        for key in ("item_key", "occurrence_id", "day", "time", "icon", "name", "type", "rating"):
            assert key in display


# ── 9. Decision trace reader ──────────────────────────────────────────────────

class TestGetDecisionTrace:

    def test_returns_list_for_known_option(self):
        state = {
            "planner_decision_trace": {
                "A": [{"day": 1, "name": "Senso-ji Temple", "reason": "Top landmark"}]
            }
        }
        result = _get_decision_trace(state, "A")
        assert isinstance(result, list)
        assert result[0]["name"] == "Senso-ji Temple"

    def test_returns_empty_for_missing_option(self):
        state = {"planner_decision_trace": {"A": []}}
        assert _get_decision_trace(state, "B") == []

    def test_parses_json_string_trace(self):
        import json as _json
        trace = _json.dumps({"A": [{"day": 1, "name": "X", "reason": "Y"}]})
        state = {"planner_decision_trace": trace}
        result = _get_decision_trace(state, "A")
        assert result[0]["name"] == "X"

    def test_missing_trace_returns_empty(self):
        assert _get_decision_trace({}, "A") == []


# ── 10. Strip numeric artifacts ───────────────────────────────────────────────

class TestStripNumericArtifacts:

    def test_removes_rating_pattern(self):
        assert "4.2 / 5" not in _strip_numeric_artifacts("Great place 4.2 / 5.")

    def test_removes_review_count(self):
        assert "1,500 reviews" not in _strip_numeric_artifacts("See (1,500 reviews) here.")

    def test_removes_km_distances(self):
        assert "km" not in _strip_numeric_artifacts("Located 2.5 km away.")

    def test_leaves_plain_text_unchanged(self):
        text = "A beautiful cultural landmark."
        assert _strip_numeric_artifacts(text) == text

    def test_empty_string_returns_empty(self):
        assert _strip_numeric_artifacts("") == ""


# ── 11. Build agent steps ─────────────────────────────────────────────────────

class TestBuildAgentSteps:

    def test_returns_three_steps(self):
        steps = _build_agent_steps([], "Tokyo")
        assert len(steps) == 3

    def test_steps_have_icon_name_detail(self):
        for step in _build_agent_steps([], "Tokyo"):
            assert "icon" in step
            assert "name" in step
            assert "detail" in step

    def test_destination_appears_in_research_step(self):
        steps = _build_agent_steps([], "Tokyo", "A")
        assert "Tokyo" in steps[0]["detail"]

    def test_option_key_appears_in_planner_step(self):
        steps = _build_agent_steps([], "Tokyo", "B")
        assert "B" in steps[1]["detail"]

    def test_tool_log_count_reflected(self):
        tool_log = ["[step1]", "[step2]", "no bracket"]
        steps = _build_agent_steps(tool_log, "Tokyo")
        assert "2" in steps[0]["detail"]


# ── 12. Summary language ──────────────────────────────────────────────────────

class TestSummaryLanguage:

    def test_defaults_to_english_when_empty(self):
        assert _summary_language({}) == "English"

    def test_returns_non_english_value(self):
        assert _summary_language({"summary_language": "Japanese"}) == "Japanese"

    def test_english_prefix_passes_through(self):
        result = _summary_language({"summary_language": "English"})
        assert result == "English"


# ── 13. Collect pref tags ─────────────────────────────────────────────────────

class TestCollectPrefTags:

    def test_extracts_comma_separated_preferences(self):
        tags = _collect_pref_tags({"preferences": "culture, food, nature"})
        assert "culture" in tags
        assert "food" in tags

    def test_skips_non_dict_scope(self):
        # state.get("state") returns a non-dict — the inner loop must skip it
        state = {"preferences": "culture", "state": "not-a-dict"}
        tags = _collect_pref_tags(state)
        assert "culture" in tags  # outer scope still works

    def test_deduplicates_tags(self):
        tags = _collect_pref_tags({"preferences": "food, Food, FOOD"})
        assert tags.count("food") <= 1

    def test_extracts_from_soft_preferences(self):
        state = {"soft_preferences": {"interest_tags": ["temples", "markets"]}}
        tags = _collect_pref_tags(state)
        assert "temples" in tags


# ── 14. Register place guards ─────────────────────────────────────────────────

class TestRegisterPlace:

    def test_non_dict_item_is_ignored(self):
        lookup = {}
        _register_place(lookup, "not-a-dict", "activity")  # type: ignore[arg-type]
        assert lookup == {}

    def test_item_without_name_is_ignored(self):
        lookup = {}
        _register_place(lookup, {"type": "attraction"}, "activity")
        assert lookup == {}

    def test_item_with_name_is_registered(self):
        lookup = {}
        _register_place(lookup, {"name": "Senso-ji Temple"}, "activity")
        assert any("senso" in k for k in lookup)

    def test_higher_score_item_replaces_lower(self):
        lookup = {}
        _register_place(lookup, {"name": "X"}, "activity")
        _register_place(lookup, {"name": "X", "rating": 4.5, "lat": 35.0, "lng": 139.0}, "activity")
        key = _normalized_name("X")
        assert lookup[key]["rating"] == 4.5


# ── 15. Build place lookup with research data ─────────────────────────────────

class TestBuildPlaceLookupResearch:

    def test_uses_research_maps_attractions(self):
        state = {
            "research": {
                "maps_attractions": [{"name": "Senso-ji Temple", "type": "attraction"}]
            }
        }
        lookup = _build_place_lookup(state)
        assert any("senso" in k for k in lookup)

    def test_uses_research_maps_restaurants(self):
        state = {
            "research": {
                "maps_restaurants": [{"name": "Ichiran Ramen", "type": "restaurant"}]
            }
        }
        lookup = _build_place_lookup(state)
        assert any("ichiran" in k for k in lookup)

    def test_uses_inventory_restaurants(self):
        state = {
            "inventory": {
                "restaurants": [{"name": "Sushi Dai", "type": "restaurant"}]
            }
        }
        lookup = _build_place_lookup(state)
        assert any("sushi" in k for k in lookup)


# ── 16. Item explanation — hotel checkout ────────────────────────────────────

class TestItemExplanationHotelCheckout:

    def test_checkout_strips_prefix_for_lookup(self):
        hotel_lookup = {
            _normalized_name("Hotel Gracery Shinjuku"): {
                "name": "Hotel Gracery Shinjuku", "rating": "4.2"
            }
        }
        item = {"key": "hotel_stay", "name": "Checkout from Hotel Gracery Shinjuku",
                "icon": "hotel", "time": "11:00"}
        result = _item_explanation("A", "Day 7", 6, 0, item, {}, hotel_lookup)
        assert result["icon"] == "hotel"

    def test_plain_hotel_name_works(self):
        hotel_lookup = {
            _normalized_name("Park Hyatt Tokyo"): {"name": "Park Hyatt Tokyo", "rating": "4.8"}
        }
        item = {"key": "hotel_stay", "name": "Park Hyatt Tokyo",
                "icon": "hotel", "time": "14:00"}
        result = _item_explanation("A", "Day 1", 0, 0, item, {}, hotel_lookup)
        assert result["icon"] == "hotel"


# ── 17. Decision trace error paths ───────────────────────────────────────────

class TestGetDecisionTraceErrors:

    def test_invalid_json_string_returns_empty(self):
        state = {"planner_decision_trace": "{not valid json}"}
        assert _get_decision_trace(state, "A") == []

    def test_list_trace_returns_empty(self):
        # trace is a list (not a dict) — should return []
        state = {"planner_decision_trace": [{"day": 1}]}
        assert _get_decision_trace(state, "A") == []


# ── 18. Trace name in items ───────────────────────────────────────────────────

class TestTraceNameInItems:

    def test_empty_name_returns_false(self):
        assert _trace_name_in_items("", [{"name": "Senso-ji Temple"}]) is False

    def test_matching_name_returns_true(self):
        items = [{"name": "Senso-ji Temple", "icon": "activity"}]
        assert _trace_name_in_items("Senso-ji Temple", items) is True

    def test_icon_filter_skips_wrong_type(self):
        items = [{"name": "Senso-ji Temple", "icon": "restaurant"}]
        assert _trace_name_in_items("Senso-ji Temple", items, icon="activity") is False

    def test_no_match_returns_false(self):
        items = [{"name": "Meiji Shrine", "icon": "activity"}]
        assert _trace_name_in_items("Senso-ji Temple", items) is False

    def test_no_items_returns_false(self):
        assert _trace_name_in_items("Senso-ji Temple", []) is False


# ── 19. Sanitize trace entry ──────────────────────────────────────────────────

class TestSanitizeTraceEntry:

    def test_non_dict_returns_empty_dict(self):
        assert _sanitize_trace_entry("not-a-dict", []) == {}  # type: ignore[arg-type]

    def test_valid_entry_returns_expected_keys(self):
        entry = {
            "day_type": "sightseeing",
            "theme": "cultural heritage",
            "seed_name": "Senso-ji Temple",
            "seed_reason": "Top cultural site.",
            "activities": [{"name": "Senso-ji Temple"}],
            "lunch": None,
            "dinner": None,
        }
        items = [{"name": "Senso-ji Temple", "icon": "activity"}]
        result = _sanitize_trace_entry(entry, items)
        for key in ("day_type", "theme", "seed_name", "seed_reason", "activities"):
            assert key in result

    def test_activity_not_in_items_is_removed(self):
        entry = {
            "activities": [{"name": "Phantom Place"}],
            "lunch": None, "dinner": None,
            "seed_name": "", "seed_reason": "",
        }
        items = [{"name": "Senso-ji Temple", "icon": "activity"}]
        result = _sanitize_trace_entry(entry, items)
        assert result["activities"] == []


# ── 20. Fallback summary ──────────────────────────────────────────────────────

class TestFallbackSummary:

    def _payload(self, day_type="sightseeing"):
        return {
            "option_label": "Budget",
            "user_preferences": ["culture", "food"],
            "days": [
                {
                    "day": "Day 1", "day_type": day_type, "theme": "cultural heritage",
                    "activities": [{"name": "Senso-ji Temple"}],
                    "seed_reason": "Strong cultural anchor.",
                }
            ],
        }

    def test_returns_overall_summary(self):
        result = _fallback_summary(self._payload(), "English", ["Day 1"])
        assert result["overall_summary"] != ""

    def test_returns_day_summaries(self):
        result = _fallback_summary(self._payload(), "English", ["Day 1"])
        assert "Day 1" in result["day_summaries"]

    def test_arrival_day_type(self):
        result = _fallback_summary(self._payload(day_type="arrival"), "English", ["Day 1"])
        assert "arrival" in result["day_summaries"]["Day 1"].lower() or \
               result["day_summaries"]["Day 1"] != ""

    def test_departure_day_type(self):
        result = _fallback_summary(self._payload(day_type="departure"), "English", ["Day 1"])
        assert result["day_summaries"]["Day 1"] != ""

    def test_generation_mode_is_fallback(self):
        result = _fallback_summary(self._payload(), "English", ["Day 1"])
        assert result["generation_mode"] == "fallback_trace"


# ── 21. Summary text ──────────────────────────────────────────────────────────

class TestSummaryText:

    def test_includes_overall_and_day(self):
        summary = {
            "overall_summary": "A wonderful trip.",
            "day_summaries": {"Day 1": "Visit Senso-ji Temple."},
        }
        text = _summary_text(summary)
        assert "wonderful trip" in text
        assert "Day 1" in text

    def test_empty_day_summary_excluded(self):
        summary = {
            "overall_summary": "Great.",
            "day_summaries": {"Day 1": "", "Day 2": "Active day."},
        }
        text = _summary_text(summary)
        assert "Day 1" not in text
        assert "Day 2" in text

    def test_empty_summary_returns_empty_string(self):
        assert _summary_text({}) == ""


# ── 22. LLM exception fallback path ──────────────────────────────────────────

class TestExplainabilityAgentLLMFallback:

    def test_llm_exception_does_not_crash(self, base_state):
        from unittest.mock import patch, MagicMock
        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = RuntimeError("LLM unavailable")
        with patch("agents.specialists.explainability_agent._llm", return_value=mock_instance):
            result = explainability_agent(base_state)
        # Should return a result (fallback), not raise
        assert "summary" in result
        assert result["summary"].get("overall_summary") is not None
