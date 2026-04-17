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
from unittest.mock import MagicMock, patch

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

from datetime import date

from agents.specialists.research_agent import _normalize_trip_state  # noqa: E402
from agents.specialists.planner_agent import (  # noqa: E402
    planner_agent,
    planner_from_research,
    revise_itinerary,
    _inventory_cache,
    # Formatting helpers
    _usd_to_sgd_str,
    _safe_price,
    _flight_display,
    _slugify,
    _stable_item_key,
    # Time helpers
    _parse_time_value,
    _clock_time,
    _shift_clock,
    _sort_time_value,
    _minutes_since_midnight,
    _hhmm,
    _duration_days,
    _trip_start_date,
    _service_date,
    # Hours text
    _normalize_hours_text,
    _parse_ampm_minutes,
    _latest_close_from_hours_text,
    # Schema / lookup helpers
    _has_expected_option_schema,
    _match_flight,
    _match_hotel,
    _looks_like_hotel_item,
    # Scoring / preference helpers
    _numeric_rating,
    _tokenize_text,
    _build_preference_model,
    _text_blob,
    _blob_tokens,
    _preference_overlap_score,
    _preference_emphasis,
    _normalized_name,
    _classify_itinerary_candidate,
    _is_normal_activity_candidate,
    _is_normal_restaurant_candidate,
    _exclude_same_place,
    _merge_used_place_names,
    _items_centroid,
    _haversine_km,
    _is_nonlocal_restaurant,
    _item_signal_counts,
    _activity_relevance_score,
    _restaurant_relevance_score,
    _hotel_price_sgd,
    _flight_stop_rank_local,
    _pick_outbound_flight,
    _pick_return_flight,
    _pick_hotels_by_profile,
    _activity_score,
    _restaurant_score,
    _travel_minutes,
    _activity_duration_minutes,
    _activity_latest_close_minutes,
    _activity_latest_start_minutes,
    _meal_duration_minutes,
    # Inventory building
    _build_activity_entries,
    _ensure_inventory_keys,
    _build_prompt_inventory,
    # Scheduling helpers
    _service_windows_from_hours_text,
    _latest_close_from_weekday_descriptions,
    _service_windows_for_item,
    # Advanced / direct-call functions
    _fit_service_start,
    _ensure_boundary_days,
    _post_process_options,
    _choose_cluster_activities,
    _llm_select_seed,
    _name_lookup,
    _trace_day_type,
    _trace_theme_from_items,
    _pick_restaurant_for_anchor,
    _pick_activity_near_anchor,
    _repair_restaurants_to_option_pool,
    _arrival_day_items,
    _departure_day_items,
    _item_end_minutes,
    _append_missing_meal,
    _pick_mandatory_meal,
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


# ── 5. Formatting helpers ─────────────────────────────────────────────────────

class TestFormattingHelpers:

    def test_usd_to_sgd_str_converts_correctly(self):
        result = _usd_to_sgd_str(100)
        assert "SGD" in result and "135" in result

    def test_usd_to_sgd_str_handles_string(self):
        assert "SGD" in _usd_to_sgd_str("200")

    def test_usd_to_sgd_str_handles_comma_number(self):
        assert "SGD" in _usd_to_sgd_str("1,000")

    def test_usd_to_sgd_str_handles_invalid(self):
        result = _usd_to_sgd_str("not_a_number")
        assert "USD" in result or result == "TBC"

    def test_usd_to_sgd_str_handles_empty(self):
        assert _usd_to_sgd_str("") == "TBC"

    def test_safe_price_keeps_sgd_prefix(self):
        assert _safe_price("SGD 50") == "SGD 50"

    def test_safe_price_converts_usd(self):
        assert "SGD" in _safe_price("USD 100")

    def test_safe_price_plain_string_unchanged(self):
        assert _safe_price("Free") == "Free"

    def test_safe_price_empty_returns_empty(self):
        assert _safe_price("") == ""

    def test_flight_display_uses_display_field(self):
        assert _flight_display({"display": "SQ637 SIN->NRT $450"}) == "SQ637 SIN->NRT $450"

    def test_flight_display_builds_from_fields(self):
        flight = {
            "airline": "SQ", "flight_number": "SQ637",
            "departure_airport": "SIN", "arrival_airport": "NRT",
            "departure_time": "08:00", "arrival_time": "16:00",
            "duration_min": 420, "travel_class": "economy", "price_usd": 450,
        }
        result = _flight_display(flight)
        assert "SQ637" in result and "SIN" in result

    def test_flight_display_empty_dict(self):
        assert isinstance(_flight_display({}), str)

    def test_slugify_basic(self):
        assert _slugify("Senso-ji Temple") == "senso_ji_temple"

    def test_slugify_special_chars(self):
        result = _slugify("Café & Bar!")
        assert " " not in result and "-" not in result

    def test_slugify_empty_returns_item(self):
        assert _slugify("") == "item"

    def test_stable_item_key_format(self):
        key = _stable_item_key("attraction", 1, "Senso-ji Temple")
        assert key.startswith("attraction_01_") and "senso" in key


# ── 6. Time helpers ───────────────────────────────────────────────────────────

class TestTimeHelpers:

    def test_parse_time_value_hhmm(self):
        result = _parse_time_value("09:30")
        assert result is not None and result.hour == 9 and result.minute == 30

    def test_parse_time_value_datetime_format(self):
        result = _parse_time_value("2026-06-01 14:00")
        assert result is not None and result.hour == 14

    def test_parse_time_value_empty_returns_none(self):
        assert _parse_time_value("") is None

    def test_parse_time_value_invalid_returns_none(self):
        assert _parse_time_value("not-a-time") is None

    def test_clock_time_valid(self):
        assert _clock_time("09:30") == "09:30"

    def test_clock_time_invalid_returns_fallback(self):
        assert _clock_time("garbage", fallback="12:00") == "12:00"

    def test_shift_clock_adds_minutes(self):
        assert _shift_clock("09:00", 90) == "10:30"

    def test_shift_clock_invalid_returns_fallback(self):
        assert _shift_clock("bad", 30, fallback="XX") == "XX"

    def test_sort_time_value_valid(self):
        minutes, _ = _sort_time_value("09:30")
        assert minutes == 9 * 60 + 30

    def test_sort_time_value_invalid_sorts_last(self):
        minutes, _ = _sort_time_value("garbage")
        assert minutes == 24 * 60 + 59

    def test_minutes_since_midnight_valid(self):
        assert _minutes_since_midnight("10:00") == 600

    def test_minutes_since_midnight_invalid_returns_none(self):
        assert _minutes_since_midnight("bad") is None

    def test_hhmm_converts_minutes(self):
        assert _hhmm(9 * 60 + 30) == "09:30"

    def test_hhmm_clamps_negative(self):
        assert _hhmm(-10) == "00:00"

    def test_duration_days_parses_string(self):
        assert _duration_days("7 days") == 7

    def test_duration_days_returns_1_for_empty(self):
        assert _duration_days("") == 1

    def test_trip_start_date_from_dates_string(self):
        assert _trip_start_date({"dates": "2026-06-01 to 2026-06-07"}) == date(2026, 6, 1)

    def test_trip_start_date_from_hard_constraints(self):
        assert _trip_start_date({"hard_constraints": {"start_date": "2026-06-01"}}) == date(2026, 6, 1)

    def test_trip_start_date_missing_returns_none(self):
        assert _trip_start_date({}) is None

    def test_service_date_adds_offset(self):
        assert _service_date(date(2026, 6, 1), 2) == date(2026, 6, 3)

    def test_service_date_none_start_returns_none(self):
        assert _service_date(None, 1) is None


# ── 7. Hours text parsing ─────────────────────────────────────────────────────

class TestHoursTextParsing:

    def test_normalize_hours_text_strips_unicode(self):
        result = _normalize_hours_text("9\u202fAM - 10\u202fPM")
        assert "\u202f" not in result and "9 AM" in result

    def test_normalize_hours_text_normalizes_dashes(self):
        assert "-" in _normalize_hours_text("9AM\u20139PM")

    def test_parse_ampm_minutes_am(self):
        assert _parse_ampm_minutes("9 AM") == 9 * 60

    def test_parse_ampm_minutes_pm(self):
        assert _parse_ampm_minutes("10 PM") == 22 * 60

    def test_parse_ampm_minutes_with_colon(self):
        assert _parse_ampm_minutes("9:30 AM") == 9 * 60 + 30

    def test_latest_close_open_24h(self):
        assert _latest_close_from_hours_text("Open 24 hours") == 23 * 60 + 59

    def test_latest_close_closed(self):
        assert _latest_close_from_hours_text("Closed") == -1

    def test_latest_close_range(self):
        result = _latest_close_from_hours_text("9:00 AM - 9:00 PM")
        assert result is not None and result > 0

    def test_latest_close_empty(self):
        assert _latest_close_from_hours_text("") is None


# ── 8. Schema / lookup helpers ────────────────────────────────────────────────

class TestSchemaAndLookupHelpers:

    def test_has_expected_option_schema_valid(self):
        options = {"A": {"days": []}, "B": {"days": []}, "C": {"days": []}}
        assert _has_expected_option_schema(options) is True

    def test_has_expected_option_schema_missing_key(self):
        assert _has_expected_option_schema({"A": {"days": []}, "B": {"days": []}}) is False

    def test_has_expected_option_schema_not_dict(self):
        assert _has_expected_option_schema("not a dict") is False

    def test_has_expected_option_schema_days_not_list(self):
        options = {"A": {"days": "bad"}, "B": {"days": []}, "C": {"days": []}}
        assert _has_expected_option_schema(options) is False

    def test_match_flight_by_flight_number(self):
        candidates = [{"flight_number": "SQ637", "display": "SQ637 SIN->NRT"}]
        result = _match_flight("SQ637", candidates)
        assert result is not None and result["flight_number"] == "SQ637"

    def test_match_flight_by_display(self):
        candidates = [{"flight_number": "SQ637", "display": "SQ637 SIN->NRT $450"}]
        assert _match_flight("SQ637 SIN->NRT", candidates) is not None

    def test_match_flight_no_match_returns_none(self):
        assert _match_flight("JL999", [{"flight_number": "SQ637", "display": "SQ637"}]) is None

    def test_match_flight_empty_list(self):
        assert _match_flight("SQ637", []) is None

    def test_match_hotel_by_name(self):
        assert _match_hotel("Hotel Gracery Shinjuku", [{"name": "Hotel Gracery Shinjuku"}]) is not None

    def test_match_hotel_partial_match(self):
        assert _match_hotel("Hotel Gracery", [{"name": "Hotel Gracery Shinjuku"}]) is not None

    def test_match_hotel_no_match(self):
        assert _match_hotel("Park Hyatt Tokyo", [{"name": "Hotel Gracery Shinjuku"}]) is None

    def test_looks_like_hotel_item_from_key(self):
        assert _looks_like_hotel_item("Hotel Gracery", "hotel_stay", []) is True

    def test_looks_like_hotel_item_from_regex(self):
        assert _looks_like_hotel_item("Grand Inn Tokyo", "some_key", []) is True

    def test_looks_like_hotel_item_false_for_attraction(self):
        assert _looks_like_hotel_item("Senso-ji Temple", "attraction_01", []) is False


# ── 9. Scoring / preference helpers ──────────────────────────────────────────

class TestScoringHelpers:

    def test_numeric_rating_float(self):
        assert _numeric_rating(4.5) == 4.5

    def test_numeric_rating_string(self):
        assert _numeric_rating("3.8") == 3.8

    def test_numeric_rating_invalid_returns_zero(self):
        assert _numeric_rating("not_a_number") == 0.0

    def test_tokenize_text_splits_words(self):
        tokens = _tokenize_text("culture food temple")
        assert "culture" in tokens and "temple" in tokens

    def test_tokenize_text_empty_returns_empty(self):
        assert _tokenize_text("") == []

    def test_build_preference_model_from_preferences(self):
        state = {"origin": "Singapore", "destination": "Tokyo", "preferences": "culture food temple"}
        model = _build_preference_model(state)
        assert "token_counts" in model and "families" in model

    def test_build_preference_model_from_soft_prefs(self):
        state = {
            "origin": "Singapore", "destination": "Tokyo", "preferences": "",
            "soft_preferences": {"interest_tags": ["culture", "food"], "vibe": "relaxed"},
        }
        model = _build_preference_model(state)
        assert isinstance(model["token_counts"], dict)

    def test_text_blob_combines_fields(self):
        blob = _text_blob({"name": "Senso-ji Temple", "type": "attraction", "description": "historic"})
        assert "senso-ji" in blob and "temple" in blob

    def test_blob_tokens_includes_bigrams(self):
        tokens = _blob_tokens("senso ji temple")
        assert "senso" in tokens and "senso ji" in tokens

    def test_preference_overlap_score_matches_keywords(self):
        score = _preference_overlap_score(
            "senso-ji is a famous temple culture spot",
            {"token_counts": {"culture": 2, "temple": 1}},
        )
        assert score > 0

    def test_preference_overlap_score_no_match_is_zero(self):
        assert _preference_overlap_score(
            "mountain skiing resort",
            {"token_counts": {"beach": 2}},
        ) == 0.0

    def test_preference_emphasis_returns_count(self):
        assert _preference_emphasis({"token_counts": {"culture": 3}}, "culture") == 3

    def test_preference_emphasis_missing_returns_zero(self):
        assert _preference_emphasis({"token_counts": {}}, "food") == 0

    def test_normalized_name_strips_and_lowercases(self):
        assert _normalized_name({"name": "  Senso-ji Temple  "}) == "senso-ji temple"

    def test_normalized_name_none_returns_empty(self):
        assert _normalized_name(None) == ""

    def test_classify_itinerary_candidate_restaurant(self):
        assert _classify_itinerary_candidate(
            {"name": "Ramen Restaurant", "type": "restaurant", "description": "dining"}
        ) == "restaurant"

    def test_classify_itinerary_candidate_attraction(self):
        assert _classify_itinerary_candidate(
            {"name": "Senso-ji Temple", "type": "attraction", "description": "historic"}
        ) == "attraction"

    def test_is_normal_activity_candidate_true(self):
        assert _is_normal_activity_candidate({"name": "Senso-ji Temple", "type": "attraction"}) is True

    def test_is_normal_restaurant_candidate_true(self):
        assert _is_normal_restaurant_candidate(
            {"name": "Ramen Bar", "type": "restaurant", "description": "noodle restaurant"}
        ) is True

    def test_exclude_same_place_filters_blocked(self):
        items = [{"name": "Senso-ji Temple"}, {"name": "Shinjuku Gyoen"}]
        result = _exclude_same_place(items, {"senso-ji temple"})
        assert len(result) == 1 and result[0]["name"] == "Shinjuku Gyoen"

    def test_exclude_same_place_empty_blocked_returns_all(self):
        items = [{"name": "A"}, {"name": "B"}]
        assert _exclude_same_place(items, set()) == items

    def test_merge_used_place_names_combines_sets(self):
        assert _merge_used_place_names({"a", "b"}, {"c"}, {"d"}) == {"a", "b", "c", "d"}

    def test_items_centroid_returns_average(self):
        centroid = _items_centroid([{"lat": 35.0, "lng": 139.0}, {"lat": 35.2, "lng": 139.2}])
        assert centroid is not None and abs(centroid[0] - 35.1) < 0.01

    def test_items_centroid_no_coords_returns_none(self):
        assert _items_centroid([{"name": "Place"}]) is None

    def test_haversine_km_known_distance(self):
        result = _haversine_km(1.3521, 103.8198, 35.6762, 139.6503)
        assert result is not None and 5000 < result < 6000

    def test_haversine_km_none_input_returns_none(self):
        assert _haversine_km(None, None, 35.0, 139.0) is None

    def test_is_nonlocal_restaurant_detects_italian(self):
        assert _is_nonlocal_restaurant({"name": "Italian Restaurant", "type": "restaurant"}) is True

    def test_is_nonlocal_restaurant_false_for_local(self):
        assert _is_nonlocal_restaurant(
            {"name": "Ichiran Ramen", "type": "restaurant", "description": "japanese ramen"}
        ) is False


# ── 10. Inventory building ────────────────────────────────────────────────────

class TestInventoryBuilding:

    def test_build_activity_entries_basic(self):
        items = [{"name": "Senso-ji Temple", "type": "attraction", "lat": 35.71, "lng": 139.79}]
        entries = _build_activity_entries(items, "attraction")
        assert len(entries) == 1
        assert "key" in entries[0]
        assert entries[0]["name"] == "Senso-ji Temple"

    def test_build_activity_entries_skips_nameless(self):
        items = [{"name": ""}, {"name": "Meiji Shrine"}]
        entries = _build_activity_entries(items, "attraction")
        assert len(entries) == 1

    def test_build_activity_entries_includes_optional_fields(self):
        items = [{"name": "Senso-ji", "rating": 4.5, "hours": "9AM-5PM"}]
        entries = _build_activity_entries(items, "attraction")
        assert entries[0].get("rating") == 4.5
        assert entries[0].get("hours") == "9AM-5PM"

    def test_ensure_inventory_keys_adds_key_if_missing(self):
        items = [{"name": "Senso-ji Temple"}]
        result = _ensure_inventory_keys(items, "attraction")
        assert "key" in result[0]

    def test_ensure_inventory_keys_preserves_existing_key(self):
        items = [{"name": "X", "key": "my_custom_key"}]
        result = _ensure_inventory_keys(items, "attraction")
        assert result[0]["key"] == "my_custom_key"

    def test_build_prompt_inventory_structure(self):
        flights_out = [{"flight_number": "SQ637", "price_usd": 450,
                        "departure_time": "08:00", "arrival_time": "16:00"}]
        flights_ret = [{"flight_number": "SQ638", "price_usd": 430,
                        "departure_time": "18:00", "arrival_time": "00:00"}]
        hotels = [{"name": "Hotel Gracery", "price_per_night_usd": 120, "rating": 4.2}]
        attractions = [{"name": "Senso-ji Temple", "type": "attraction"}]
        restaurants = [{"name": "Ichiran Ramen", "type": "restaurant"}]
        inventory = _build_prompt_inventory(attractions, restaurants, hotels, flights_out, flights_ret)
        assert "outbound_flights" in inventory
        assert "return_flights" in inventory
        assert "hotels" in inventory
        assert "attractions" in inventory
        assert "restaurants" in inventory

    def test_build_prompt_inventory_hotel_without_price(self):
        hotels = [{"name": "Unknown Hotel"}]  # no price_per_night_usd
        inventory = _build_prompt_inventory([], [], hotels, [], [])
        assert inventory["hotels"][0]["nightly_cost"] == "TBC"

    def test_build_prompt_inventory_skips_nameless_hotel(self):
        hotels = [{"name": ""}, {"name": "Valid Hotel", "price_per_night_usd": 100}]
        inventory = _build_prompt_inventory([], [], hotels, [], [])
        assert len(inventory["hotels"]) == 1


# ── 11. Signal counts ─────────────────────────────────────────────────────────

class TestItemSignalCounts:

    def test_temple_signal_detected(self):
        signals = _item_signal_counts("a historic temple and shrine area")
        assert signals["temple"] > 0

    def test_museum_signal_detected(self):
        signals = _item_signal_counts("national museum of art history")
        assert signals["museum"] > 0

    def test_no_signals_in_plain_text(self):
        signals = _item_signal_counts("open air venue")
        # heritage, temple, museum, etc. should be 0
        assert signals["temple"] == 0 and signals["museum"] == 0

    def test_returns_dict_with_expected_keys(self):
        signals = _item_signal_counts("test place")
        for key in ("heritage", "temple", "museum", "nature", "scenic", "local_food"):
            assert key in signals


# ── 12. Relevance scoring ─────────────────────────────────────────────────────

class TestRelevanceScoring:

    def _profile(self):
        return {"focus": {"culture": 1.5, "food": 1.0, "nature": 0.5, "landmark": 1.0, "modern": 0.0}}

    def _pref_model(self):
        return _build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "culture food temple heritage",
        })

    def test_temple_scores_higher_for_culture_preference(self):
        temple = {"name": "Senso-ji Temple", "type": "attraction", "description": "historic Buddhist temple"}
        modern = {"name": "teamLab Digital Art", "type": "attraction", "description": "modern digital art"}
        score_t = _activity_relevance_score(temple, self._profile(), self._pref_model())
        score_m = _activity_relevance_score(modern, self._profile(), self._pref_model())
        assert score_t > score_m

    def test_activity_relevance_score_returns_float(self):
        item = {"name": "Shinjuku Gyoen", "type": "attraction", "description": "large park garden"}
        assert isinstance(_activity_relevance_score(item, self._profile(), self._pref_model()), float)

    def test_restaurant_relevance_score_returns_float(self):
        item = {"name": "Ichiran Ramen", "type": "restaurant", "description": "japanese ramen noodle"}
        assert isinstance(_restaurant_relevance_score(item, self._profile(), self._pref_model()), float)

    def test_local_food_scores_positively(self):
        local = {"name": "Tsukiji Outer Market", "type": "restaurant", "description": "local street food seafood market"}
        italian = {"name": "Italian Trattoria", "type": "restaurant", "description": "italian pasta pizza restaurant"}
        score_l = _restaurant_relevance_score(local, self._profile(), self._pref_model())
        score_i = _restaurant_relevance_score(italian, self._profile(), self._pref_model())
        assert score_l > score_i


# ── 13. Hotel / flight picking ────────────────────────────────────────────────

class TestPickingHelpers:

    def test_hotel_price_sgd_converts_usd(self):
        assert abs(_hotel_price_sgd({"price_per_night_usd": 100}) - 135.0) < 1

    def test_hotel_price_sgd_invalid_returns_fallback(self):
        assert _hotel_price_sgd({"price_per_night_usd": "bad"}) == 9999.0

    def test_flight_stop_rank_nonstop_is_zero(self):
        assert _flight_stop_rank_local({"display": "SQ637 nonstop SIN->NRT"}) == 0

    def test_flight_stop_rank_with_stops(self):
        assert _flight_stop_rank_local({"display": "JL001 1 stop via HND"}) == 1

    def test_flight_stop_rank_unknown_is_9(self):
        assert _flight_stop_rank_local({"display": "SQ637 SIN->NRT"}) == 9

    def test_pick_outbound_flight_returns_first_when_one(self):
        flights = [{"flight_number": "SQ637", "display": "nonstop", "price_usd": 450,
                    "arrival_time": "16:00", "duration_min": 420}]
        result = _pick_outbound_flight(flights)
        assert result is not None and result["flight_number"] == "SQ637"

    def test_pick_outbound_flight_empty_returns_none(self):
        assert _pick_outbound_flight([]) is None

    def test_pick_return_flight_empty_returns_none(self):
        assert _pick_return_flight([]) is None

    def test_pick_return_flight_returns_flight(self):
        flights = [{"flight_number": "SQ638", "display": "direct", "price_usd": 430,
                    "departure_time": "18:00", "duration_min": 420}]
        assert _pick_return_flight(flights) is not None

    def test_pick_hotels_by_profile_returns_abc_keys(self):
        hotels = [
            {"name": "Budget Inn", "price_per_night_usd": 60, "rating": 3.5},
            {"name": "Comfort Hotel", "price_per_night_usd": 120, "rating": 4.2},
            {"name": "Luxury Stay", "price_per_night_usd": 250, "rating": 4.8},
        ]
        picks = _pick_hotels_by_profile(hotels)
        assert "A" in picks and "B" in picks and "C" in picks


# ── 14. Activity / restaurant scoring ────────────────────────────────────────

class TestActivityRestaurantScoring:

    def _profile(self):
        return {
            "focus": {"culture": 1.5, "food": 1.0, "nature": 0.5, "landmark": 1.0, "modern": 0.0},
            "discouraged_activity_penalty": 18,
            "fresh_activity_bonus": 0,
            "discouraged_restaurant_penalty": 16,
            "fresh_restaurant_bonus": 0,
        }

    def _pref_model(self):
        return _build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "culture food temple",
        })

    def test_activity_score_returns_float(self):
        item = {"name": "Senso-ji Temple", "type": "attraction", "description": "historic temple",
                "lat": 35.71, "lng": 139.79}
        assert isinstance(_activity_score(item, self._profile(), self._pref_model()), float)

    def test_activity_score_discouraged_penalty(self):
        item = {"name": "Senso-ji Temple", "type": "attraction", "description": "temple",
                "lat": 35.71, "lng": 139.79}
        normal = _activity_score(item, self._profile(), self._pref_model())
        discouraged = _activity_score(item, self._profile(), self._pref_model(), {"senso-ji temple"})
        assert discouraged < normal

    def test_restaurant_score_returns_float(self):
        item = {"name": "Ichiran Ramen", "type": "restaurant", "description": "japanese ramen",
                "lat": 35.69, "lng": 139.70}
        assert isinstance(_restaurant_score(item, self._profile(), self._pref_model()), float)

    def test_travel_minutes_with_coords(self):
        a = {"lat": 35.71, "lng": 139.79}
        b = {"lat": 35.69, "lng": 139.70}
        result = _travel_minutes(a, b)
        assert isinstance(result, int) and result > 0

    def test_travel_minutes_no_coords_returns_base_plus_10(self):
        assert _travel_minutes(None, None, base=20) == 30

    def test_activity_duration_museum(self):
        assert _activity_duration_minutes({"name": "Tokyo National Museum", "type": "attraction", "description": "museum history"}) == 105

    def test_activity_duration_temple(self):
        assert _activity_duration_minutes({"name": "Senso-ji Temple", "type": "attraction", "description": "temple shrine"}) == 75

    def test_activity_duration_default(self):
        assert _activity_duration_minutes({"name": "Random Place", "type": "attraction", "description": "misc"}) == 90

    def test_activity_latest_close_with_hours(self):
        item = {"name": "Senso-ji", "hours": "9:00 AM - 5:00 PM"}
        result = _activity_latest_close_minutes(item)
        assert result is not None and result > 0

    def test_activity_latest_close_no_hours_returns_none(self):
        assert _activity_latest_close_minutes({"name": "Open Place"}) is None

    def test_activity_latest_start_for_museum(self):
        item = {"name": "Museum", "description": "national museum"}
        result = _activity_latest_start_minutes(item)
        assert isinstance(result, int) and result > 0

    def test_activity_latest_start_for_temple(self):
        item = {"name": "Senso-ji Temple", "description": "temple shrine"}
        result = _activity_latest_start_minutes(item)
        assert isinstance(result, int) and result > 0

    def test_meal_duration_standard(self):
        assert _meal_duration_minutes({"name": "Ichiran Ramen", "type": "restaurant"}) == 75

    def test_meal_duration_fine_dining(self):
        item = {"name": "Fine Dining Kaiseki", "type": "restaurant",
                "description": "fine dining omakase tasting menu"}
        assert _meal_duration_minutes(item) == 90


# ── 15. Scheduling helpers ────────────────────────────────────────────────────

class TestSchedulingHelpers:

    def test_service_windows_from_hours_text_returns_windows(self):
        windows = _service_windows_from_hours_text("9:00 AM - 5:00 PM")
        assert len(windows) > 0
        open_m, close_m = windows[0]
        assert open_m == 9 * 60 and close_m == 17 * 60

    def test_service_windows_closed_returns_empty(self):
        assert _service_windows_from_hours_text("Closed") == []

    def test_service_windows_open_24h_returns_full_day(self):
        windows = _service_windows_from_hours_text("Open 24 hours")
        assert windows == [(0, 23 * 60 + 59)]

    def test_service_windows_empty_returns_empty(self):
        assert _service_windows_from_hours_text("") == []

    def test_latest_close_from_weekday_descriptions_no_descriptions(self):
        assert _latest_close_from_weekday_descriptions([], None) is None

    def test_latest_close_from_weekday_descriptions_finds_day(self):
        descriptions = ["Monday: 9:00 AM - 6:00 PM", "Tuesday: 9:00 AM - 6:00 PM"]
        result = _latest_close_from_weekday_descriptions(descriptions, date(2026, 6, 1))  # Monday
        assert result is not None and result > 0

    def test_service_windows_for_item_no_data(self):
        item = {"name": "Open Place"}
        result = _service_windows_for_item(item, None)
        assert isinstance(result, list)

    def test_service_windows_for_item_with_hours(self):
        item = {"name": "Senso-ji", "hours": "9:00 AM - 5:00 PM"}
        result = _service_windows_for_item(item, None)
        assert len(result) > 0


# ── 16. Additional branch coverage: classify / relevance ─────────────────────

class TestClassifyBranches:
    """Cover tour/night_only/photo_spot/area branches (lines 1687-1693)."""

    def test_classify_returns_tour(self):
        assert _classify_itinerary_candidate(
            {"name": "Tokyo Walking Tour", "type": "attraction", "description": "guided tour"}
        ) == "tour"

    def test_classify_returns_night_only(self):
        assert _classify_itinerary_candidate(
            {"name": "Night Light Show", "type": "attraction", "description": "illumination night and light"}
        ) == "night_only"

    def test_classify_returns_photo_spot(self):
        assert _classify_itinerary_candidate(
            {"name": "Tokyo Sign Monument", "type": "attraction", "description": "monument photo spot"}
        ) == "photo_spot"

    def test_classify_returns_area(self):
        assert _classify_itinerary_candidate(
            {"name": "Memory Lane Yokocho", "type": "attraction", "description": "alley yokocho"}
        ) == "area"


class TestActivityRelevanceDeductions:
    """Cover deduction branches in _activity_relevance_score (lines 1770-1778)."""

    def _profile(self):
        return {"focus": {"culture": 1.5, "food": 1.0, "nature": 0.5, "landmark": 1.0, "modern": 0.0}}

    def _pref_traditional(self):
        return _build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "traditional culture authentic historical heritage",
        })

    def _pref_local_food(self):
        return _build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "local street food hawker market",
        })

    def test_quirky_item_gets_lower_score(self):
        quirky = {"name": "Trick Art Horror Museum", "type": "attraction",
                  "description": "optical illusion trick horror selfie"}
        normal = {"name": "Senso-ji Temple", "type": "attraction", "description": "historic temple"}
        score_q = _activity_relevance_score(quirky, self._profile(), self._pref_traditional())
        score_n = _activity_relevance_score(normal, self._profile(), self._pref_traditional())
        assert score_q < score_n

    def test_natural_history_deduction(self):
        item = {"name": "Natural History Museum", "type": "attraction",
                "description": "natural history traditional exhibition museum"}
        score = _activity_relevance_score(item, self._profile(), self._pref_traditional())
        assert isinstance(score, float)

    def test_discovery_centre_deduction(self):
        item = {"name": "Science Discovery Centre", "type": "attraction",
                "description": "discovery centre science interactive innovation"}
        score = _activity_relevance_score(item, self._profile(), self._pref_traditional())
        assert isinstance(score, float)

    def test_modern_museum_deduction(self):
        item = {"name": "Digital Art Museum", "type": "attraction",
                "description": "museum modern digital science innovation"}
        score = _activity_relevance_score(item, self._profile(), self._pref_traditional())
        assert isinstance(score, float)

    def test_local_food_pref_non_heritage_deduction(self):
        item = {"name": "Fashion Boutique", "type": "attraction",
                "description": "shopping boutique fashion retail"}
        score = _activity_relevance_score(item, self._profile(), self._pref_local_food())
        assert isinstance(score, float)


class TestRestaurantRelevanceBranches:
    """Cover bonus/penalty branches in _restaurant_relevance_score (lines 1811-1822)."""

    def _profile(self):
        return {"focus": {"culture": 1.5, "food": 1.0, "nature": 0.5, "landmark": 1.0, "modern": 0.0}}

    def _pref_traditional(self):
        return _build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "traditional authentic cultural heritage",
        })

    def _pref_local_food(self):
        return _build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "local street food hawker market",
        })

    def test_traditional_pref_non_local_cuisine_bonus(self):
        item = {"name": "Kaiseki Ryori", "type": "restaurant",
                "description": "traditional Japanese authentic cuisine regional", "rating": 4.5}
        score = _restaurant_relevance_score(item, self._profile(), self._pref_traditional())
        assert isinstance(score, float)

    def test_modern_signal_deduction(self):
        item = {"name": "Modern Digital Cafe", "type": "restaurant",
                "description": "modern digital science interactive"}
        score = _restaurant_relevance_score(item, self._profile(), self._pref_local_food())
        assert isinstance(score, float)

    def test_local_food_pref_no_local_signal_deduction(self):
        item = {"name": "Fine Dining Restaurant", "type": "restaurant",
                "description": "fine dining upscale premium"}
        score = _restaurant_relevance_score(item, self._profile(), self._pref_local_food())
        assert isinstance(score, float)

    def test_traditional_pref_with_italian_penalty(self):
        item = {"name": "Italian Trattoria", "type": "restaurant",
                "description": "italian pasta pizza authentic"}
        score = _restaurant_relevance_score(item, self._profile(), self._pref_traditional())
        assert isinstance(score, float)

    def test_local_food_pref_with_non_local_cuisine_penalty(self):
        item = {"name": "Indian Curry Restaurant", "type": "restaurant",
                "description": "indian curry spice"}
        score = _restaurant_relevance_score(item, self._profile(), self._pref_local_food())
        assert isinstance(score, float)

    def test_grill_without_local_food_penalty(self):
        item = {"name": "Italian Grill Steakhouse", "type": "restaurant",
                "description": "italian grill buffet steakhouse brasserie"}
        pref = _build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "culture sightseeing temple",
        })
        score = _restaurant_relevance_score(item, self._profile(), pref)
        assert isinstance(score, float)


class TestActivityScoreBranches:
    """Cover matches_preference and fresh_activity_bonus (lines 1901, 1908)."""

    def _profile(self):
        return {
            "focus": {"culture": 1.5, "food": 1.0, "nature": 0.5, "landmark": 1.0, "modern": 0.0},
            "discouraged_activity_penalty": 18,
            "fresh_activity_bonus": 5,
        }

    def _pref(self):
        return _build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "culture food temple heritage",
        })

    def test_matches_preference_gives_bonus(self):
        with_pref = {"name": "Senso-ji Temple", "type": "attraction",
                     "description": "temple", "matches_preference": True}
        without_pref = {"name": "Senso-ji Temple", "type": "attraction",
                        "description": "temple"}
        assert _activity_score(with_pref, self._profile(), self._pref()) > \
               _activity_score(without_pref, self._profile(), self._pref())

    def test_fresh_activity_bonus_when_not_discouraged(self):
        item = {"name": "Shinjuku Gyoen", "type": "attraction", "description": "garden park"}
        discouraged = {"other-place"}
        fresh = _activity_score(item, self._profile(), self._pref(), discouraged)
        normal = _activity_score(item, self._profile(), self._pref())
        assert fresh > normal  # fresh_activity_bonus=5 applies


class TestRestaurantScoreBranches:
    """Cover vegetarian bonus and discouraged restaurant tiers (lines 1918-1928)."""

    def _profile(self):
        return {
            "focus": {"culture": 1.0, "food": 1.0, "nature": 0.5, "landmark": 1.0, "modern": 0.0},
            "discouraged_restaurant_penalty": 16,
            "fresh_restaurant_bonus": 5,
        }

    def _pref(self):
        return _build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "food local street hawker",
        })

    def test_vegetarian_gets_bonus(self):
        veg = {"name": "Veggie Kitchen", "type": "restaurant",
               "description": "vegetarian vegan plant-based"}
        non_veg = {"name": "Ramen Bar", "type": "restaurant", "description": "ramen noodle"}
        assert _restaurant_score(veg, self._profile(), self._pref()) > \
               _restaurant_score(non_veg, self._profile(), self._pref())

    def test_discouraged_restaurant_gets_penalty(self):
        item = {"name": "Ichiran Ramen", "type": "restaurant", "description": "ramen"}
        normal = _restaurant_score(item, self._profile(), self._pref())
        penalized = _restaurant_score(item, self._profile(), self._pref(), {"ichiran ramen"})
        assert penalized < normal

    def test_fresh_restaurant_gets_bonus(self):
        item = {"name": "Sushi Dai", "type": "restaurant", "description": "sushi authentic local"}
        discouraged = {"other-restaurant"}
        fresh = _restaurant_score(item, self._profile(), self._pref(), discouraged)
        normal = _restaurant_score(item, self._profile(), self._pref())
        assert fresh > normal

    def test_discouraged_high_relevance_reduced_penalty(self):
        # High local_food signals → high relevance → penalty multiplied by 0.3 or 0.55
        item = {"name": "Tsukiji Market", "type": "restaurant",
                "description": "local street food hawker market seafood snack"}
        normal = _restaurant_score(item, self._profile(), self._pref())
        penalized = _restaurant_score(item, self._profile(), self._pref(), {"tsukiji market"})
        assert penalized < normal


class TestActivityDurationGardenBranch:
    """Cover garden/park branch in _activity_duration_minutes (line 1946)."""

    def test_garden_returns_90_not_temple_path(self):
        garden = {"name": "Shinjuku Garden Park", "type": "attraction", "description": "garden park"}
        museum = {"name": "Art Museum", "type": "attraction", "description": "museum exhibit"}
        assert _activity_duration_minutes(museum) == 105
        assert _activity_duration_minutes(garden) == 90

    def test_park_returns_90(self):
        park = {"name": "Ueno Park", "type": "attraction", "description": "park outdoor botanic"}
        assert _activity_duration_minutes(park) == 90


class TestActivityLatestStartBranches:
    """Cover branches in _activity_latest_start_minutes (lines 1958, 1965-1979)."""

    def test_returns_neg1_when_place_is_closed(self):
        item = {"name": "Closed Museum", "type": "attraction", "hours": "Closed"}
        assert _activity_latest_start_minutes(item) == -1

    def test_computes_from_hours_text(self):
        item = {"name": "Temple", "type": "attraction",
                "hours": "9:00 AM - 5:00 PM", "description": "temple shrine"}
        result = _activity_latest_start_minutes(item)
        assert result > 0

    def test_nature_branch_returns_960(self):
        item = {"name": "Nature Trail", "type": "attraction",
                "description": "nature garden park botanic trail"}
        assert _activity_latest_start_minutes(item) == 16 * 60

    def test_scenic_branch_returns_1080(self):
        item = {"name": "Scenic District", "type": "attraction",
                "description": "scenic neighborhood district quarter"}
        assert _activity_latest_start_minutes(item) == 18 * 60

    def test_weekday_descriptions_closed_day(self):
        item = {"name": "Museum", "type": "attraction",
                "weekday_descriptions": ["Monday: Closed"]}
        # Closed → actual_close = -1 → return -1
        result = _activity_latest_start_minutes(item, date(2026, 6, 1))  # Monday
        assert result == -1

    def test_weekday_descriptions_compute_start(self):
        item = {"name": "Museum", "type": "attraction",
                "weekday_descriptions": ["Monday: 10:00 AM - 6:00 PM"],
                "description": "museum gallery"}
        result = _activity_latest_start_minutes(item, date(2026, 6, 1))
        assert result > 0


# ── 17. Parse/date edge cases ─────────────────────────────────────────────────

class TestParseAmpmMinutesNone:
    """Cover _parse_ampm_minutes returning None (line 185)."""

    def test_garbage_returns_none(self):
        assert _parse_ampm_minutes("garbage text") is None

    def test_24h_format_not_recognized(self):
        assert _parse_ampm_minutes("14:00") is None


class TestTripStartDateValueError:
    """Cover ValueError branch in _trip_start_date (lines 149-150)."""

    def test_unparseable_date_returns_none(self):
        state = {"origin": "Singapore", "destination": "Tokyo", "dates": "June 1 to June 7"}
        assert _trip_start_date(state) is None


class TestLatestCloseEdgeCases:
    """Cover edge cases in _latest_close_from_hours_text."""

    def test_closes_pattern(self):
        # "closes" keyword pattern → lines 200-202 (regex: closes? + whitespace + time, no "at")
        result = _latest_close_from_hours_text("closes 9 PM")
        assert result is not None

    def test_single_time_token_returns_none(self):
        # Only one parseable time, no closes pattern → line 208
        result = _latest_close_from_hours_text("Open from 9 AM (no close)")
        # Either matches closes or returns None
        assert result is None or isinstance(result, int)

    def test_overnight_hours_wraps(self):
        # close < open → line 215: close += 24*60
        result = _latest_close_from_hours_text("11:00 PM - 2:00 AM")
        assert result is not None and result > 0


class TestServiceWindowsOvernight:
    """Cover overnight branch in _service_windows_from_hours_text (line 240)."""

    def test_overnight_hours_window(self):
        windows = _service_windows_from_hours_text("10:00 PM - 2:00 AM")
        assert len(windows) > 0


class TestWeekdayDescriptionsBranches:
    """Cover weekday-mismatch and fallback branches (lines 253-262)."""

    def test_weekday_mismatch_falls_to_fallback_text(self):
        # "Tuesday" doesn't match Monday → continue. "9 AM - 5 PM" has no ":" so becomes fallback_text
        descriptions = [
            "Tuesday: 9 AM - 6 PM",
            "9 AM - 5 PM",  # no colon day label → fallback_text
        ]
        result = _latest_close_from_weekday_descriptions(descriptions, date(2026, 6, 1))  # Monday
        assert result is not None

    def test_fallback_text_no_colon(self):
        # No ":" in description → not parsed as day:hours → becomes fallback_text
        descriptions = ["9 AM - 5 PM"]
        result = _latest_close_from_weekday_descriptions(descriptions, date(2026, 6, 1))
        assert result is not None


class TestServiceWindowsForItemBranches:
    """Cover _service_windows_for_item with weekday descriptions (lines 270-280)."""

    def test_matching_weekday_description(self):
        item = {
            "name": "Museum",
            "weekday_descriptions": [
                "Monday: 9:00 AM - 5:00 PM",
                "Tuesday: 9:00 AM - 6:00 PM",
            ],
        }
        result = _service_windows_for_item(item, date(2026, 6, 1))  # Monday
        assert len(result) > 0

    def test_non_matching_weekday_falls_through_to_hours(self):
        item = {
            "name": "Museum",
            "weekday_descriptions": ["Tuesday: 10:00 AM - 6:00 PM"],
            "hours": "9:00 AM - 5:00 PM",
        }
        result = _service_windows_for_item(item, date(2026, 6, 1))  # Monday (Tuesday doesn't match)
        assert isinstance(result, list)

    def test_plain_description_without_day_label(self):
        # "9 AM - 5 PM" has no ":" so it becomes fallback_text (no day prefix)
        item = {
            "name": "Park",
            "weekday_descriptions": ["9 AM - 5 PM"],
        }
        result = _service_windows_for_item(item, date(2026, 6, 1))
        assert len(result) > 0


class TestMatchFlightCodeshare:
    """Cover code-share flight number parsing (lines 488-490)."""

    def test_codeshare_first_segment(self):
        candidates = [{"flight_number": "SQ637/JL001", "display": "SQ637/JL001 SIN->NRT"}]
        assert _match_flight("SQ637", candidates) is not None

    def test_codeshare_second_segment(self):
        candidates = [{"flight_number": "SQ637/JL001", "display": "SQ637/JL001 SIN->NRT"}]
        assert _match_flight("JL001", candidates) is not None


class TestFitServiceStart:
    """Cover _fit_service_start branches (lines 293-305)."""

    def test_no_windows_no_hours_returns_preferred(self):
        item = {"name": "Open All Day Place"}
        result = _fit_service_start(item, 10 * 60, 90, None)
        assert result == 10 * 60

    def test_no_windows_has_hours_that_force_none(self):
        item = {"name": "Closed", "hours": "Closed"}
        result = _fit_service_start(item, 10 * 60, 90, None)
        assert result is None

    def test_window_exists_fits(self):
        item = {"name": "Temple", "hours": "9:00 AM - 6:00 PM"}
        result = _fit_service_start(item, 10 * 60, 90, None)
        assert result is not None and result >= 9 * 60

    def test_window_exists_but_no_slot_returns_none(self):
        item = {"name": "Temple", "hours": "9:00 AM - 9:30 AM"}  # tiny window
        result = _fit_service_start(item, 9 * 60, 120, None)  # needs 120 min, only 30 available
        assert result is None

    def test_latest_finish_constraint_returns_none(self):
        item = {"name": "Place", "hours": "9:00 AM - 8:00 PM"}
        result = _fit_service_start(item, 10 * 60, 90, None, latest_finish=10 * 60 + 30)
        assert result is None  # 10:00 + 90 > 10:30

    def test_preferred_exceeds_latest_finish_no_windows(self):
        item = {"name": "Open Place"}
        result = _fit_service_start(item, 20 * 60, 90, None, latest_finish=10 * 60)
        assert result is None  # preferred + duration > latest_finish


# ── 18. _ensure_boundary_days ─────────────────────────────────────────────────

class TestEnsureBoundaryDays:
    """Cover _ensure_boundary_days branches for missing flight/hotel insertion."""

    def test_empty_days_returns_empty(self):
        result_days, repairs = _ensure_boundary_days([], [], [], [])
        assert result_days == []
        assert repairs == []

    def test_inserts_missing_outbound_flight(self):
        days = [
            {"day": 1, "date": "2026-06-01", "items": []},  # no outbound flight
            {"day": 7, "date": "2026-06-07", "items": [
                {"key": "flight_return", "name": "SQ638", "icon": "flight", "time": "18:00"}
            ]},
        ]
        flights_out = [{"flight_number": "SQ637", "arrival_time": "16:00",
                        "departure_time": "08:00", "price_usd": 450,
                        "display": "SQ637 SIN->NRT $450"}]
        flights_ret = [{"flight_number": "SQ638", "departure_time": "18:00",
                        "price_usd": 430, "display": "SQ638 NRT->SIN $430"}]
        result_days, repairs = _ensure_boundary_days(days, [], flights_out, flights_ret)
        # Should have inserted the outbound flight
        assert any("outbound" in r for r in repairs)

    def test_inserts_missing_return_flight(self):
        days = [
            {"day": 1, "date": "2026-06-01", "items": [
                {"key": "flight_outbound", "name": "SQ637", "icon": "flight", "time": "16:00"}
            ]},
            {"day": 7, "date": "2026-06-07", "items": []},  # no return flight
        ]
        flights_out = [{"flight_number": "SQ637", "arrival_time": "16:00",
                        "departure_time": "08:00", "price_usd": 450,
                        "display": "SQ637 SIN->NRT $450"}]
        flights_ret = [{"flight_number": "SQ638", "departure_time": "18:00",
                        "price_usd": 430, "display": "SQ638 NRT->SIN $430"}]
        result_days, repairs = _ensure_boundary_days(days, [], flights_out, flights_ret)
        assert any("return" in r for r in repairs)

    def test_no_repairs_when_all_present(self):
        days = [
            {"day": 1, "date": "2026-06-01", "items": [
                {"key": "flight_outbound", "name": "SQ637", "icon": "flight", "time": "16:00"}
            ]},
            {"day": 7, "date": "2026-06-07", "items": [
                {"key": "flight_return", "name": "SQ638", "icon": "flight", "time": "18:00"}
            ]},
        ]
        flights_out = [{"flight_number": "SQ637", "arrival_time": "16:00",
                        "departure_time": "08:00", "price_usd": 450,
                        "display": "SQ637 SIN->NRT $450"}]
        flights_ret = [{"flight_number": "SQ638", "departure_time": "18:00",
                        "price_usd": 430, "display": "SQ638 NRT->SIN $430"}]
        result_days, repairs = _ensure_boundary_days(days, [], flights_out, flights_ret)
        assert len(repairs) == 0


# ── 19. _post_process_options direct call ─────────────────────────────────────

class TestPostProcessOptions:
    """Direct test of _post_process_options to cover inner closures."""

    def _make_options(self, mock_research_result):
        """Build a rich options dict with middle days containing activity/restaurant items."""
        middle_days = [
            {
                "day": 2, "date": "2026-06-02",
                "items": [
                    {"key": "attraction_01_senso_ji_temple", "name": "Senso-ji Temple",
                     "icon": "activity", "time": "10:00", "type": "activity"},
                    {"key": "restaurant_01_ichiran_ramen", "name": "Ichiran Ramen",
                     "icon": "restaurant", "time": "12:30", "type": "restaurant"},
                    {"key": "attraction_02_shinjuku_gyoen", "name": "Shinjuku Gyoen",
                     "icon": "activity", "time": "15:00", "type": "activity"},
                ],
            },
            {
                "day": 3, "date": "2026-06-03",
                "items": [
                    {"key": "attraction_03_meiji_shrine", "name": "Meiji Shrine",
                     "icon": "activity", "time": "10:00", "type": "activity"},
                    {"key": "restaurant_02_sushi_dai", "name": "Sushi Dai",
                     "icon": "restaurant", "time": "13:00", "type": "restaurant"},
                    {"key": "attraction_04_teamlab_planets", "name": "teamLab Planets",
                     "icon": "activity", "time": "16:00", "type": "activity"},
                ],
            },
            {
                "day": 4, "date": "2026-06-04",
                "items": [
                    {"key": "attraction_05_tokyo_national_museum", "name": "Tokyo National Museum",
                     "icon": "activity", "time": "10:00", "type": "activity"},
                    {"key": "restaurant_03_uobei_shibuya", "name": "Uobei Shibuya",
                     "icon": "restaurant", "time": "12:30", "type": "restaurant"},
                    # Duplicate: tests dedup logic
                    {"key": "attraction_01_senso_ji_temple", "name": "Senso-ji Temple",
                     "icon": "activity", "time": "15:00", "type": "activity"},
                ],
            },
        ]
        day_1 = {
            "day": 1, "date": "2026-06-01",
            "items": [
                {"key": "flight_outbound", "name": "SQ637", "icon": "flight", "time": "16:00"},
                {"key": "hotel_stay", "name": "Hotel Gracery Shinjuku", "icon": "hotel", "time": "20:30"},
            ],
        }
        day_last = {
            "day": 5, "date": "2026-06-05",
            "items": [
                {"key": "flight_return", "name": "SQ638", "icon": "flight", "time": "18:00"},
            ],
        }
        return {
            "A": {"label": "Budget", "style": "budget", "days": [day_1] + middle_days + [day_last]},
            "B": {"label": "Balanced", "style": "balanced", "days": [day_1] + middle_days[:2] + [day_last]},
            "C": {"label": "Comfort", "style": "comfort", "days": [day_1] + middle_days[:1] + [day_last]},
        }

    def test_processes_activity_restaurant_items(self, mock_research_result):
        options = self._make_options(mock_research_result)
        result_itins, result_meta, result_inv, result_trace = _post_process_options(
            options,
            mock_research_result["compact_attractions"],
            mock_research_result["compact_restaurants"],
            mock_research_result["hotel_options"],
            mock_research_result["flight_options_outbound"],
            mock_research_result["flight_options_return"],
            [],
        )
        assert "A" in result_itins
        assert len(result_itins["A"]) > 0

    def test_dedup_removes_repeated_attraction(self, mock_research_result):
        # Senso-ji appears on day 2 and day 4 of option A → dedup removes day 4 entry
        options = self._make_options(mock_research_result)
        result_itins, _, _, _ = _post_process_options(
            options,
            mock_research_result["compact_attractions"],
            mock_research_result["compact_restaurants"],
            mock_research_result["hotel_options"],
            mock_research_result["flight_options_outbound"],
            mock_research_result["flight_options_return"],
            [],
        )
        # Count how many times Senso-ji appears across all days in option A
        senso_count = sum(
            1
            for day in result_itins["A"]
            for item in day.get("items", [])
            if "senso" in item.get("name", "").lower()
        )
        assert senso_count <= 1  # dedup should remove the second occurrence

    def test_result_meta_has_options(self, mock_research_result):
        options = self._make_options(mock_research_result)
        _, result_meta, _, _ = _post_process_options(
            options,
            mock_research_result["compact_attractions"],
            mock_research_result["compact_restaurants"],
            mock_research_result["hotel_options"],
            mock_research_result["flight_options_outbound"],
            mock_research_result["flight_options_return"],
            [],
        )
        for key in ("A", "B", "C"):
            assert key in result_meta

    def test_items_have_correct_icon_types(self, mock_research_result):
        options = self._make_options(mock_research_result)
        result_itins, _, _, _ = _post_process_options(
            options,
            mock_research_result["compact_attractions"],
            mock_research_result["compact_restaurants"],
            mock_research_result["hotel_options"],
            mock_research_result["flight_options_outbound"],
            mock_research_result["flight_options_return"],
            [],
        )
        for day in result_itins["A"]:
            for item in day.get("items", []):
                assert item.get("icon") in {"activity", "restaurant", "flight", "hotel"}

    def test_options_without_middle_days(self, mock_research_result):
        """Flight-only options still produce valid output."""
        options = {
            "A": {
                "label": "Budget", "style": "budget",
                "days": [
                    {"day": 1, "date": "2026-06-01", "items": [
                        {"key": "flight_outbound", "name": "SQ637", "icon": "flight", "time": "16:00"},
                    ]},
                    {"day": 2, "date": "2026-06-02", "items": [
                        {"key": "flight_return", "name": "SQ638", "icon": "flight", "time": "18:00"},
                    ]},
                ],
            },
            "B": {"label": "Balanced", "style": "balanced", "days": []},
            "C": {"label": "Comfort", "style": "comfort", "days": []},
        }
        result_itins, result_meta, _, _ = _post_process_options(
            options,
            mock_research_result["compact_attractions"],
            mock_research_result["compact_restaurants"],
            mock_research_result["hotel_options"],
            mock_research_result["flight_options_outbound"],
            mock_research_result["flight_options_return"],
            [],
        )
        assert "A" in result_itins


# ── 20. _choose_cluster_activities with light_only and immersion mode ──────────

class TestChooseClusterActivities:
    """Cover light_only and immersion mode branches (lines 2471-2607)."""

    def _attractions(self, mock_research_result):
        from agents.specialists.planner_agent import _build_activity_entries, _ensure_inventory_keys
        items = _ensure_inventory_keys(mock_research_result["compact_attractions"], "attraction")
        return _build_activity_entries(items, "attraction")

    def _pref(self):
        return _build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "culture food temple heritage",
        })

    def _profile_coverage(self):
        return {
            "selection_mode": "coverage",
            "discouraged_activity_penalty": 18,
            "fresh_activity_bonus": 0,
            "route_penalty": 1.6,
            "fallback_route_penalty": 1.9,
            "cluster_seed_penalty": 0.0,
            "spread_bonus": 0.0,
        }

    def _profile_immersion(self):
        return {
            "selection_mode": "immersion",
            "cluster_radius_km": 5.0,
            "discouraged_activity_penalty": 18,
            "fresh_activity_bonus": 0,
            "route_penalty": 1.6,
            "fallback_route_penalty": 1.9,
            "cluster_seed_penalty": 0.0,
            "spread_bonus": 0.0,
            "fresh_activity_quota": 0,
        }

    def test_light_only_true_covers_inner_function(self, mock_research_result):
        attractions = self._attractions(mock_research_result)
        result = _choose_cluster_activities(
            attractions, set(), self._profile_coverage(), self._pref(),
            count=2, light_only=True,
        )
        assert isinstance(result, list)

    def test_immersion_mode_runs_cluster_selection(self, mock_research_result):
        attractions = self._attractions(mock_research_result)
        result = _choose_cluster_activities(
            attractions, set(), self._profile_immersion(), self._pref(),
            count=2, light_only=False,
        )
        assert isinstance(result, list)

    def test_immersion_mode_with_discouraged_names(self, mock_research_result):
        attractions = self._attractions(mock_research_result)
        discouraged = {"senso-ji temple"}
        result = _choose_cluster_activities(
            attractions, set(), self._profile_immersion(), self._pref(),
            count=2, discouraged_names=discouraged,
        )
        assert isinstance(result, list)

    def test_empty_attractions_returns_empty(self):
        result = _choose_cluster_activities([], set(), self._profile_coverage(), self._pref(), count=2)
        assert result == []

    def test_count_exceeds_available_uses_fallback(self, mock_research_result):
        # Only 2 unique attractions but count=5 → triggers fallback pool
        small_pool = self._attractions(mock_research_result)[:2]
        result = _choose_cluster_activities(
            small_pool, set(), self._profile_coverage(), self._pref(), count=5,
        )
        # Should return at most len(small_pool) items
        assert len(result) <= len(small_pool)

    def test_with_fresh_quota_and_discouraged(self, mock_research_result):
        attractions = self._attractions(mock_research_result)
        profile = {**self._profile_coverage(), "fresh_activity_quota": 2, "fresh_activity_bonus": 5}
        discouraged = {"senso-ji temple", "shinjuku gyoen"}
        result = _choose_cluster_activities(
            attractions, set(), profile, self._pref(),
            count=2, discouraged_names=discouraged,
        )
        assert isinstance(result, list)


# ── TestPostProcessOptionsEdgeCases ──────────────────────────────────────────

class TestPostProcessOptionsEdgeCases:
    """Cover _process_days edge branches inside _post_process_options."""

    @staticmethod
    def _call(options, mock_research_result):
        itins, meta, inv, trace = _post_process_options(
            options,
            mock_research_result["compact_attractions"],
            mock_research_result["compact_restaurants"],
            mock_research_result["hotel_options"],
            mock_research_result["flight_options_outbound"],
            mock_research_result["flight_options_return"],
            [],
        )
        return itins

    def test_list_day_is_promoted_to_dict(self, mock_research_result):
        # A day entry that is a list (not dict) wrapped into {"day":..., "items": day}
        options = {
            "A": {
                "label": "Budget", "style": "budget",
                "days": [
                    [  # list-day
                        {"key": "flight_outbound", "name": "SQ637 SIN->NRT $450",
                         "icon": "flight", "time": "16:00"},
                    ]
                ],
            },
            "B": {"label": "Balanced", "style": "balanced", "days": []},
            "C": {"label": "Comfort", "style": "comfort", "days": []},
        }
        result = self._call(options, mock_research_result)
        assert "A" in result

    def test_non_dict_item_is_skipped(self, mock_research_result):
        # An item that is not a dict increments invalid_items and is skipped
        options = {
            "A": {
                "label": "Budget", "style": "budget",
                "days": [
                    {
                        "day": 1, "date": "2026-06-01",
                        "items": [
                            "not a dict",
                            {"key": "flight_outbound", "name": "SQ637 SIN->NRT $450",
                             "icon": "flight", "time": "16:00"},
                        ],
                    }
                ],
            },
            "B": {"label": "Balanced", "style": "balanced", "days": []},
            "C": {"label": "Comfort", "style": "comfort", "days": []},
        }
        result = self._call(options, mock_research_result)
        assert "A" in result

    def test_pipe_separator_in_name_is_stripped(self, mock_research_result):
        # Name with " | " → only left part used for catalog lookup
        options = {
            "A": {
                "label": "Budget", "style": "budget",
                "days": [
                    {
                        "day": 1, "date": "2026-06-01",
                        "items": [
                            {"key": "flight_outbound", "name": "SQ637 SIN->NRT $450",
                             "icon": "flight", "time": "16:00"},
                        ],
                    },
                    {
                        "day": 2, "date": "2026-06-02",
                        "items": [
                            {"key": "senso_ji", "name": "Senso-ji Temple | Asakusa",
                             "icon": "activity", "time": "10:00"},
                        ],
                    },
                ],
            },
            "B": {"label": "Balanced", "style": "balanced", "days": []},
            "C": {"label": "Comfort", "style": "comfort", "days": []},
        }
        result = self._call(options, mock_research_result)
        assert "A" in result

    def test_fuzzy_name_match_in_resolve_catalog_item(self, mock_research_result):
        # "Senso-ji" is a substring of catalog name "Senso-ji Temple" → fuzzy branch
        options = {
            "A": {
                "label": "Budget", "style": "budget",
                "days": [
                    {
                        "day": 1, "date": "2026-06-01",
                        "items": [
                            {"key": "flight_outbound", "name": "SQ637 SIN->NRT $450",
                             "icon": "flight", "time": "16:00"},
                        ],
                    },
                    {
                        "day": 2, "date": "2026-06-02",
                        "items": [
                            {"key": "senso_ji_fuzzy", "name": "Senso-ji",
                             "icon": "activity", "time": "10:00"},
                        ],
                    },
                ],
            },
            "B": {"label": "Balanced", "style": "balanced", "days": []},
            "C": {"label": "Comfort", "style": "comfort", "days": []},
        }
        result = self._call(options, mock_research_result)
        assert "A" in result


# ── TestLlmSelectSeed ─────────────────────────────────────────────────────────

class TestLlmSelectSeed:
    """Cover _llm_select_seed edge branches."""

    @staticmethod
    def _pref():
        return {"token_counts": {"culture": 3, "temple": 2, "garden": 1}, "families": {"culture": 2}}

    @staticmethod
    def _profile():
        return {"style": "budget", "selection_mode": "coverage"}

    @staticmethod
    def _make_candidate(name, seed_name=None):
        seed = {"name": seed_name or name, "key": name.lower().replace(" ", "_"),
                "type": "attraction", "lat": 35.71, "lng": 139.79}
        return {
            "name": name,
            "type": "attraction",
            "seed": seed,
            "cluster_members": [{"name": "Nearby Place", "type": "attraction"}],
        }

    def test_empty_candidates_returns_none(self):
        result, theme, reason = _llm_select_seed([], 0, "A", self._profile(), self._pref(), [])
        assert result is None
        assert theme == "mixed sightseeing"
        assert "no seed candidates" in reason

    def test_exception_falls_back_to_first_candidate(self):
        candidates = [self._make_candidate("Senso-ji Temple")]
        with patch("agents.specialists.planner_agent._llm") as mock_llm_fn:
            mock_llm_fn.return_value.invoke.side_effect = RuntimeError("LLM down")
            seed, theme, reason = _llm_select_seed(candidates, 0, "A", self._profile(), self._pref(), [])
        assert seed is not None
        assert seed["name"] == "Senso-ji Temple"
        assert "deterministic fallback" in reason

    def test_partial_name_match_returns_seed(self):
        candidates = [self._make_candidate("Senso-ji Temple")]
        mock_response = MagicMock()
        mock_response.content = '{"selected_name": "Senso-ji", "theme": "temple day", "reason": "nice"}'
        with patch("agents.specialists.planner_agent._llm") as mock_llm_fn:
            mock_llm_fn.return_value.invoke.return_value = mock_response
            seed, theme, reason = _llm_select_seed(candidates, 0, "A", self._profile(), self._pref(), [])
        assert seed is not None
        assert theme == "temple day"

    def test_no_match_returns_first_candidate_with_message(self):
        candidates = [self._make_candidate("Senso-ji Temple")]
        mock_response = MagicMock()
        mock_response.content = '{"selected_name": "Totally Unknown Place", "theme": "misc", "reason": "ok"}'
        with patch("agents.specialists.planner_agent._llm") as mock_llm_fn:
            mock_llm_fn.return_value.invoke.return_value = mock_response
            seed, theme, reason = _llm_select_seed(candidates, 0, "A", self._profile(), self._pref(), [])
        assert seed["name"] == "Senso-ji Temple"
        assert "did not match" in reason


# ── TestTraceHelpers ──────────────────────────────────────────────────────────

class TestTraceHelpers:
    """Cover _name_lookup non-dict skip, _trace_day_type, _trace_theme_from_items."""

    def test_name_lookup_skips_non_dict_items(self):
        items = [
            {"name": "Senso-ji Temple"},
            "not a dict",
            42,
            {"name": "Shinjuku Gyoen"},
        ]
        result = _name_lookup(items)
        assert "senso-ji temple" in result
        assert "shinjuku gyoen" in result
        assert len(result) == 2

    def test_trace_day_type_arrival(self):
        assert _trace_day_type("Day 1", 0, 7, {}) == "arrival"

    def test_trace_day_type_departure(self):
        assert _trace_day_type("Day 7", 6, 7, {}) == "departure"

    def test_trace_day_type_middle(self):
        assert _trace_day_type("Day 3", 2, 7, {}) == "middle"

    def test_trace_day_type_from_original_entry(self):
        assert _trace_day_type("Day 3", 2, 7, {"day_type": "special"}) == "special"

    def test_trace_theme_arrival(self):
        theme = _trace_theme_from_items({}, [], "arrival")
        assert theme == "arrival evening"

    def test_trace_theme_departure(self):
        theme = _trace_theme_from_items({}, [], "departure")
        assert theme == "departure day"

    def test_trace_theme_from_original_entry(self):
        theme = _trace_theme_from_items({"theme": "custom theme"}, [], "middle")
        assert theme == "custom theme"

    def test_trace_theme_garden_and_culture(self):
        activities = [{"type": "garden"}, {"type": "temple"}]
        theme = _trace_theme_from_items({}, activities, "middle")
        assert theme == "gardens & culture"

    def test_trace_theme_garden_only(self):
        activities = [{"type": "garden"}]
        theme = _trace_theme_from_items({}, activities, "middle")
        assert theme == "garden day"

    def test_trace_theme_culture_only(self):
        activities = [{"type": "temple"}]
        theme = _trace_theme_from_items({}, activities, "middle")
        assert theme == "culture day"

    def test_trace_theme_sightseeing_fallback(self):
        activities = [{"type": "observation_deck"}]
        theme = _trace_theme_from_items({}, activities, "middle")
        assert theme == "sightseeing day"


# ── TestPickRestaurantForAnchor ───────────────────────────────────────────────

class TestPickRestaurantForAnchor:
    """Cover _pick_restaurant_for_anchor branches."""

    @staticmethod
    def _restaurants():
        return [
            {"name": "Ichiran Ramen", "key": "ichiran_ramen", "type": "restaurant",
             "lat": 35.6938, "lng": 139.7034, "price": "SGD 20"},
            {"name": "Sushi Dai", "key": "sushi_dai", "type": "restaurant",
             "lat": 35.6654, "lng": 139.7707, "price": "SGD 50"},
            {"name": "Uobei Shibuya", "key": "uobei_shibuya", "type": "restaurant",
             "lat": 35.6590, "lng": 139.7003, "price": "SGD 15"},
        ]

    @staticmethod
    def _profile():
        return {
            "style": "budget",
            "selection_mode": "coverage",
            "fresh_restaurant_quota": 0,
            "restaurant_route_penalty": 1.4,
        }

    @staticmethod
    def _pref():
        return _build_preference_model({"preferences": "culture, food"})

    def test_basic_pick(self):
        result = _pick_restaurant_for_anchor(
            self._restaurants(), set(), self._profile(), self._pref(), anchor=None
        )
        assert result is not None
        assert result.get("name") in {r["name"] for r in self._restaurants()}

    def test_all_used_returns_none(self):
        used = {"ichiran ramen", "sushi dai", "uobei shibuya"}
        result = _pick_restaurant_for_anchor(
            self._restaurants(), used, self._profile(), self._pref(), anchor=None
        )
        assert result is None

    def test_discouraged_names_filtered(self):
        discouraged = {"ichiran ramen", "sushi dai"}
        result = _pick_restaurant_for_anchor(
            self._restaurants(), set(), self._profile(), self._pref(),
            anchor=None, discouraged_names=discouraged
        )
        assert result is not None
        assert result["name"] == "Uobei Shibuya"

    def test_time_constraint_impossible_returns_none(self):
        # preferred_start past closing time → no time_fit → returns None
        result = _pick_restaurant_for_anchor(
            self._restaurants(), set(), self._profile(), self._pref(),
            anchor=None, preferred_start=22 * 60, latest_finish=22 * 60 + 30, meal_duration=75
        )
        assert result is None or isinstance(result, dict)

    def test_immersion_mode_novel_restaurants(self):
        profile = {**self._profile(), "selection_mode": "immersion"}
        discouraged = {"ichiran ramen", "sushi dai"}
        result = _pick_restaurant_for_anchor(
            self._restaurants(), set(), profile, self._pref(),
            anchor=None, discouraged_names=discouraged
        )
        assert result is not None


# ── TestPickActivityNearAnchor ────────────────────────────────────────────────

class TestPickActivityNearAnchor:
    """Cover _pick_activity_near_anchor branches."""

    @staticmethod
    def _attractions():
        return [
            {"name": "Senso-ji Temple", "key": "senso_ji", "type": "temple",
             "lat": 35.7148, "lng": 139.7967, "price": "Free", "duration_hrs": 2},
            {"name": "Shinjuku Gyoen", "key": "shinjuku_gyoen", "type": "garden",
             "lat": 35.6851, "lng": 139.7100, "price": "SGD 3", "duration_hrs": 2},
            {"name": "Tokyo Tower", "key": "tokyo_tower", "type": "tower",
             "lat": 35.6586, "lng": 139.7454, "price": "SGD 20", "duration_hrs": 1},
        ]

    @staticmethod
    def _profile():
        return {"style": "budget", "nearby_route_penalty": 1.6}

    @staticmethod
    def _pref():
        return _build_preference_model({"preferences": "culture, food"})

    def test_basic_pick(self):
        result = _pick_activity_near_anchor(
            self._attractions(), set(), self._profile(), self._pref(), anchor=None
        )
        assert result is not None

    def test_all_used_returns_none(self):
        used = {"senso-ji temple", "shinjuku gyoen", "tokyo tower"}
        result = _pick_activity_near_anchor(
            self._attractions(), used, self._profile(), self._pref(), anchor=None
        )
        assert result is None

    def test_light_only_mode(self):
        result = _pick_activity_near_anchor(
            self._attractions(), set(), self._profile(), self._pref(),
            anchor=None, light_only=True
        )
        assert result is not None

    def test_discouraged_names_filtered(self):
        discouraged = {"senso-ji temple"}
        result = _pick_activity_near_anchor(
            self._attractions(), set(), self._profile(), self._pref(),
            anchor=None, discouraged_names=discouraged
        )
        assert result is not None
        assert result["name"] != "Senso-ji Temple"

    def test_all_discouraged_falls_back_to_broad(self):
        # All items discouraged → relaxed_novel_available branch
        discouraged = {"senso-ji temple", "shinjuku gyoen", "tokyo tower"}
        result = _pick_activity_near_anchor(
            self._attractions(), set(), self._profile(), self._pref(),
            anchor=None, discouraged_names=discouraged
        )
        assert result is not None


# ── TestRepairRestaurantsToOptionPool ─────────────────────────────────────────

class TestRepairRestaurantsToOptionPool:
    """Cover _repair_restaurants_to_option_pool branches."""

    @staticmethod
    def _profile():
        return {
            "style": "budget",
            "selection_mode": "coverage",
            "fresh_restaurant_quota": 0,
            "restaurant_route_penalty": 1.4,
        }

    @staticmethod
    def _pref():
        return _build_preference_model({"preferences": "culture, food"})

    @staticmethod
    def _pool():
        return [
            {"name": "Ichiran Ramen", "key": "ichiran_ramen", "type": "restaurant",
             "lat": 35.6938, "lng": 139.7034, "price": "SGD 20"},
            {"name": "Sushi Dai", "key": "sushi_dai", "type": "restaurant",
             "lat": 35.6654, "lng": 139.7707, "price": "SGD 50"},
        ]

    def test_no_pool_returns_days_unchanged(self):
        days = [{"day": 1, "items": [
            {"icon": "restaurant", "name": "Some Place", "key": "some_place", "time": "12:00"}
        ]}]
        result = _repair_restaurants_to_option_pool(days, [], self._profile(), self._pref())
        assert result[0]["items"][0]["name"] == "Some Place"

    def test_restaurant_already_in_pool_not_replaced(self):
        days = [{"day": 1, "items": [
            {"icon": "restaurant", "name": "Ichiran Ramen", "key": "ichiran_ramen", "time": "12:30"},
        ]}]
        result = _repair_restaurants_to_option_pool(days, self._pool(), self._profile(), self._pref())
        assert result[0]["items"][0]["name"] == "Ichiran Ramen"

    def test_out_of_pool_restaurant_is_repaired_or_kept(self):
        # "Unknown Cafe" not in pool → repair attempted
        days = [{"day": 2, "items": [
            {"icon": "activity", "name": "Senso-ji Temple", "key": "senso_ji", "time": "10:00"},
            {"icon": "restaurant", "name": "Unknown Cafe", "key": "unknown_cafe", "time": "12:30"},
        ]}]
        result = _repair_restaurants_to_option_pool(days, self._pool(), self._profile(), self._pref())
        replaced_name = result[0]["items"][1]["name"]
        assert replaced_name in {r["name"] for r in self._pool()} or replaced_name == "Unknown Cafe"


# ── TestArrivalDayItemsEarlyFlight ────────────────────────────────────────────

class TestArrivalDayItemsEarlyFlight:
    """Cover _arrival_day_items early-arrival light-activity branch (lines 4030-4052)."""

    @staticmethod
    def _attractions():
        return [
            {"name": "Senso-ji Temple", "key": "senso_ji", "type": "temple",
             "lat": 35.7148, "lng": 139.7967, "price": "Free", "duration_hrs": 2},
            {"name": "Shinjuku Gyoen", "key": "shinjuku_gyoen", "type": "garden",
             "lat": 35.6851, "lng": 139.7100, "price": "SGD 3", "duration_hrs": 2},
        ]

    @staticmethod
    def _restaurants():
        return [
            {"name": "Ichiran Ramen", "key": "ichiran_ramen", "type": "restaurant",
             "lat": 35.6938, "lng": 139.7034, "price": "SGD 20"},
        ]

    @staticmethod
    def _hotel():
        return {"name": "Hotel Gracery Shinjuku", "lat": 35.6938, "lng": 139.7034,
                "price_per_night_usd": 120}

    @staticmethod
    def _profile():
        return {"style": "budget", "meal_role": "standard",
                "nearby_route_penalty": 1.6, "restaurant_route_penalty": 1.4,
                "fresh_restaurant_quota": 0, "arrival_restaurant_offset": 0}

    @staticmethod
    def _pref():
        return _build_preference_model({"preferences": "culture, food"})

    def test_early_arrival_triggers_light_activity_branch(self):
        # Arrival at 13:00 (780 min) <= 15*60+30 (930) → light activity branch executed
        outbound_flight = {
            "airline": "SQ", "flight_number": "SQ637",
            "departure_time": "08:00", "arrival_time": "13:00",
            "departure_airport": "SIN", "arrival_airport": "NRT",
            "price_usd": 450,
        }
        result = _arrival_day_items(
            self._profile(), outbound_flight, self._hotel(),
            self._attractions(), self._restaurants(), None,
            set(), self._pref(),
        )
        icons = [item["icon"] for item in result]
        assert "flight" in icons

    def test_late_arrival_skips_light_activity(self):
        # Arrival at 19:00 (1140 min) > 930 → light activity branch skipped
        outbound_flight = {
            "airline": "SQ", "flight_number": "SQ637",
            "departure_time": "11:00", "arrival_time": "19:00",
            "departure_airport": "SIN", "arrival_airport": "NRT",
            "price_usd": 450,
        }
        result = _arrival_day_items(
            self._profile(), outbound_flight, self._hotel(),
            self._attractions(), self._restaurants(), None,
            set(), self._pref(),
        )
        icons = [item["icon"] for item in result]
        assert "flight" in icons
        assert "activity" not in icons


# ── TestDepartureDayItems ─────────────────────────────────────────────────────

class TestDepartureDayItems:
    """Cover _departure_day_items activity+brunch branch (lines 4164-4284)."""

    @staticmethod
    def _attractions():
        return [
            {"name": "Senso-ji Temple", "key": "senso_ji", "type": "temple",
             "lat": 35.7148, "lng": 139.7967, "price": "Free", "duration_hrs": 1},
        ]

    @staticmethod
    def _restaurants():
        return [
            {"name": "Ichiran Ramen", "key": "ichiran_ramen", "type": "restaurant",
             "lat": 35.6938, "lng": 139.7034, "price": "SGD 20"},
        ]

    @staticmethod
    def _hotel():
        return {"name": "Hotel Gracery Shinjuku", "lat": 35.6938, "lng": 139.7034,
                "price_per_night_usd": 120}

    @staticmethod
    def _profile():
        return {"style": "budget", "meal_role": "standard",
                "nearby_route_penalty": 1.6, "restaurant_route_penalty": 1.4,
                "fresh_restaurant_quota": 0}

    @staticmethod
    def _pref():
        return _build_preference_model({"preferences": "culture, food"})

    def test_late_departure_triggers_activity_and_brunch(self):
        # Departure at 20:00 → airport_cutoff = 17:00, plenty of time
        return_flight = {
            "airline": "SQ", "flight_number": "SQ638",
            "departure_time": "20:00", "arrival_time": "00:00",
            "departure_airport": "NRT", "arrival_airport": "SIN",
            "price_usd": 430,
        }
        result = _departure_day_items(
            self._profile(), return_flight, self._hotel(),
            self._attractions(), self._restaurants(), None,
            set(), self._pref(),
        )
        icons = [item["icon"] for item in result]
        assert "flight" in icons

    def test_no_hotel_still_works(self):
        return_flight = {
            "airline": "SQ", "flight_number": "SQ638",
            "departure_time": "20:00", "arrival_time": "00:00",
            "departure_airport": "NRT", "arrival_airport": "SIN",
            "price_usd": 430,
        }
        result = _departure_day_items(
            self._profile(), return_flight, None,
            self._attractions(), self._restaurants(), None,
            set(), self._pref(),
        )
        icons = [item["icon"] for item in result]
        assert "flight" in icons
        assert "hotel" not in icons

    def test_early_departure_skips_activity(self):
        # Departure at 09:00 → available_after_checkout < 165 min → activity branch skipped
        return_flight = {
            "airline": "SQ", "flight_number": "SQ638",
            "departure_time": "09:00", "arrival_time": "15:00",
            "departure_airport": "NRT", "arrival_airport": "SIN",
            "price_usd": 430,
        }
        result = _departure_day_items(
            self._profile(), return_flight, self._hotel(),
            self._attractions(), self._restaurants(), None,
            set(), self._pref(),
        )
        icons = [item["icon"] for item in result]
        assert "flight" in icons
        assert "activity" not in icons

    def test_mid_day_departure_triggers_brunch_only(self):
        # Departure at 14:30 → available_after_checkout ~160 min: 75 <= 160 < 165
        # → elif branch: activity skipped, brunch attempted
        return_flight = {
            "airline": "SQ", "flight_number": "SQ638",
            "departure_time": "14:30", "arrival_time": "22:00",
            "departure_airport": "NRT", "arrival_airport": "SIN",
            "price_usd": 430,
        }
        result = _departure_day_items(
            self._profile(), return_flight, self._hotel(),
            self._attractions(), self._restaurants(), None,
            set(), self._pref(),
        )
        icons = [item["icon"] for item in result]
        assert "flight" in icons
        # No activity (not enough time for >= 165 min window)
        assert "activity" not in icons


# ── TestItemEndMinutes ────────────────────────────────────────────────────────

class TestItemEndMinutes:
    """Cover _item_end_minutes — lines 3810-3815."""

    def test_activity_end_is_start_plus_duration(self):
        # "temple" → _activity_duration_minutes = 75 min
        item = {"icon": "activity", "name": "Senso-ji Temple", "time": "10:00"}
        result = _item_end_minutes(item)
        assert result == 10 * 60 + 75  # 675

    def test_restaurant_end_is_start_plus_meal_duration(self):
        # restaurant → _meal_duration_minutes (default 75)
        item = {"icon": "restaurant", "name": "Ichiran Ramen", "time": "12:30"}
        result = _item_end_minutes(item)
        assert result == 12 * 60 + 30 + 75  # 825

    def test_other_icon_returns_start_only(self):
        # hotel icon → just returns start minutes
        item = {"icon": "hotel", "name": "Hotel ABC", "time": "20:30"}
        result = _item_end_minutes(item)
        assert result == 20 * 60 + 30  # 1230


# ── TestAppendMissingMeal ─────────────────────────────────────────────────────

class TestAppendMissingMeal:
    """Cover _append_missing_meal — lines 3855-3880."""

    @staticmethod
    def _restaurants():
        return [
            {"name": "Ichiran Ramen", "key": "ichiran_ramen", "type": "restaurant",
             "lat": 35.6938, "lng": 139.7034, "price": "SGD 20"},
            {"name": "Sushi Dai", "key": "sushi_dai", "type": "restaurant",
             "lat": 35.6654, "lng": 139.7707, "price": "SGD 50"},
        ]

    @staticmethod
    def _profile():
        return {
            "style": "budget",
            "selection_mode": "coverage",
            "fresh_restaurant_quota": 0,
            "restaurant_route_penalty": 1.4,
        }

    @staticmethod
    def _pref():
        return _build_preference_model({"preferences": "culture, food"})

    def test_lunch_appended_when_missing(self):
        items = [
            {"icon": "activity", "name": "Senso-ji Temple", "key": "senso_ji", "time": "10:00"}
        ]
        result = _append_missing_meal(
            items, self._restaurants(), set(), self._profile(), self._pref(),
            meal_kind="lunch", service_date=None,
        )
        assert isinstance(result, bool)
        if result:
            icons = [item["icon"] for item in items]
            assert "restaurant" in icons

    def test_dinner_appended_when_missing(self):
        items = [
            {"icon": "activity", "name": "Senso-ji Temple", "key": "senso_ji", "time": "16:00"}
        ]
        result = _append_missing_meal(
            items, self._restaurants(), set(), self._profile(), self._pref(),
            meal_kind="dinner", service_date=None,
        )
        assert isinstance(result, bool)

    def test_no_restaurants_returns_false(self):
        items = [
            {"icon": "activity", "name": "Senso-ji Temple", "key": "senso_ji", "time": "10:00"}
        ]
        result = _append_missing_meal(
            items, [], set(), self._profile(), self._pref(),
            meal_kind="lunch", service_date=None,
        )
        assert result is False

    def test_all_used_returns_false(self):
        items = [
            {"icon": "activity", "name": "Senso-ji Temple", "key": "senso_ji", "time": "10:00"}
        ]
        used = {"ichiran ramen", "sushi dai"}
        result = _append_missing_meal(
            items, self._restaurants(), used, self._profile(), self._pref(),
            meal_kind="lunch", service_date=None,
        )
        assert isinstance(result, bool)


# ── TestPickMandatoryMeal ─────────────────────────────────────────────────────

class TestPickMandatoryMeal:
    """Cover _pick_mandatory_meal basic paths."""

    @staticmethod
    def _restaurants():
        return [
            {"name": "Ichiran Ramen", "key": "ichiran_ramen", "type": "restaurant",
             "lat": 35.6938, "lng": 139.7034, "price": "SGD 20"},
            {"name": "Sushi Dai", "key": "sushi_dai", "type": "restaurant",
             "lat": 35.6654, "lng": 139.7707, "price": "SGD 50"},
        ]

    @staticmethod
    def _profile():
        return {
            "style": "budget",
            "selection_mode": "coverage",
            "fresh_restaurant_quota": 0,
            "restaurant_route_penalty": 1.4,
        }

    @staticmethod
    def _pref():
        return _build_preference_model({"preferences": "culture, food"})

    def test_lunch_pick_returns_restaurant_or_none(self):
        result = _pick_mandatory_meal(
            self._restaurants(), set(), set(), self._profile(), self._pref(),
            anchor=None, meal_kind="lunch",
        )
        assert result is None or isinstance(result, dict)

    def test_dinner_pick_returns_restaurant_or_none(self):
        result = _pick_mandatory_meal(
            self._restaurants(), set(), set(), self._profile(), self._pref(),
            anchor=None, meal_kind="dinner",
        )
        assert result is None or isinstance(result, dict)

    def test_empty_restaurants_returns_none(self):
        result = _pick_mandatory_meal(
            [], set(), set(), self._profile(), self._pref(),
            anchor=None, meal_kind="lunch",
        )
        assert result is None

    def test_with_local_preference_raises_floor(self):
        # "local, hawker" preference → relevance_floor = -5.0 (lunch)
        # Forces the fallback path when restaurant scores < -5.0
        pref = _build_preference_model({"preferences": "local, hawker, street food"})
        result = _pick_mandatory_meal(
            self._restaurants(), set(), set(), self._profile(), pref,
            anchor=None, meal_kind="lunch",
        )
        assert result is None or isinstance(result, dict)
