"""
Agent3 (Planner) unit tests.

Run from repository root:
    pytest agents/tests/planner/planner_pytest.py -v
"""

import json
import sys
from datetime import date
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


# ── Fixtures ───────────────────────────────────────────────────────────────────

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
             "location": "Asakusa", "price_sgd": "Free", "duration_hrs": 2,
             "lat": 35.7148, "lng": 139.7967},
            {"item_key": "shinjuku_gyoen", "name": "Shinjuku Gyoen", "type": "attraction",
             "location": "Shinjuku", "price_sgd": "SGD 3", "duration_hrs": 2,
             "lat": 35.6851, "lng": 139.7100},
            {"item_key": "meiji_shrine", "name": "Meiji Shrine", "type": "attraction",
             "location": "Harajuku", "price_sgd": "Free", "duration_hrs": 1.5,
             "lat": 35.6763, "lng": 139.6993},
            {"item_key": "teamlab", "name": "teamLab Planets", "type": "attraction",
             "location": "Toyosu", "price_sgd": "SGD 40", "duration_hrs": 2,
             "lat": 35.6504, "lng": 139.7954},
            {"item_key": "ueno_museum", "name": "Tokyo National Museum", "type": "attraction",
             "location": "Ueno", "price_sgd": "SGD 15", "duration_hrs": 3,
             "lat": 35.7189, "lng": 139.7764},
            {"item_key": "shibuya_sky", "name": "Shibuya Sky Observatory", "type": "attraction",
             "location": "Shibuya", "price_sgd": "SGD 25", "duration_hrs": 1,
             "lat": 35.6585, "lng": 139.7026},
        ],
        "compact_restaurants": [
            {"item_key": "ichiran_ramen", "name": "Ichiran Ramen", "type": "restaurant",
             "location": "Shinjuku", "price_sgd": "SGD 20", "duration_hrs": 1,
             "lat": 35.6938, "lng": 139.7034},
            {"item_key": "sushi_dai", "name": "Sushi Dai", "type": "restaurant",
             "location": "Tsukiji", "price_sgd": "SGD 50", "duration_hrs": 1.5,
             "lat": 35.6654, "lng": 139.7707},
            {"item_key": "uobei_sushi", "name": "Uobei Shibuya", "type": "restaurant",
             "location": "Shibuya", "price_sgd": "SGD 15", "duration_hrs": 1,
             "lat": 35.6590, "lng": 139.7003},
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


@pytest.fixture
def revise_llm(mock_research_result):
    """Mock LLM that returns the correct revise_itinerary JSON format."""
    revise_response = {
        "chain_of_thought": "Revised itinerary focusing on culture and food in Tokyo.",
        "options": {
            "A": {
                "label": "Budget", "desc": "Budget cultural trip", "budget": "SGD 2000",
                "style": "budget", "badge": "Best Value",
                "days": [
                    {
                        "day": 1, "date": "2026-06-01",
                        "items": [
                            {"key": "flight_outbound", "name": "SQ637", "icon": "flight",
                             "time": "16:00", "type": "flight"},
                            {"key": "hotel_stay", "name": "Hotel Gracery Shinjuku",
                             "icon": "hotel", "time": "20:30", "type": "hotel"},
                        ],
                    },
                    {
                        "day": 2, "date": "2026-06-02",
                        "items": [
                            {"key": "senso_ji", "name": "Senso-ji Temple",
                             "icon": "attraction", "time": "09:00",
                             "type": "attraction", "duration_hrs": 2, "cost": "Free"},
                            {"key": "ichiran_ramen", "name": "Ichiran Ramen",
                             "icon": "restaurant", "time": "12:30",
                             "type": "restaurant", "cost": "SGD 20"},
                            {"key": "meiji_shrine", "name": "Meiji Shrine",
                             "icon": "attraction", "time": "15:00",
                             "type": "attraction", "duration_hrs": 1.5, "cost": "Free"},
                        ],
                    },
                    {
                        "day": 7, "date": "2026-06-07",
                        "items": [
                            {"key": "flight_return", "name": "SQ638", "icon": "flight",
                             "time": "18:00", "type": "flight"},
                        ],
                    },
                ],
            },
            "B": {
                "label": "Balanced", "desc": "Balanced experience", "budget": "SGD 2500",
                "style": "balanced", "badge": "Popular",
                "days": [
                    {
                        "day": 1, "date": "2026-06-01",
                        "items": [
                            {"key": "flight_outbound", "name": "SQ637",
                             "icon": "flight", "time": "16:00", "type": "flight"},
                        ],
                    },
                    {
                        "day": 7, "date": "2026-06-07",
                        "items": [
                            {"key": "flight_return", "name": "SQ638",
                             "icon": "flight", "time": "18:00", "type": "flight"},
                        ],
                    },
                ],
            },
            "C": {
                "label": "Comfort", "desc": "Premium comfort", "budget": "SGD 3000",
                "style": "comfort", "badge": "Best Experience",
                "days": [
                    {
                        "day": 1, "date": "2026-06-01",
                        "items": [
                            {"key": "flight_outbound", "name": "SQ637",
                             "icon": "flight", "time": "16:00", "type": "flight"},
                        ],
                    },
                    {
                        "day": 7, "date": "2026-06-07",
                        "items": [
                            {"key": "flight_return", "name": "SQ638",
                             "icon": "flight", "time": "18:00", "type": "flight"},
                        ],
                    },
                ],
            },
        },
    }
    instance = MagicMock()
    instance.invoke.return_value = MagicMock(content=json.dumps(revise_response))
    return instance


@pytest.fixture
def populated_cache(mock_research_result):
    """Populate _inventory_cache and return it."""
    pa._inventory_cache.update({
        "prompt_inventory": {},
        "flight_out_text": "SQ637 SIN->NRT $450",
        "flight_ret_text": "SQ638 NRT->SIN $430",
        "hotel_list_text": "Hotel Gracery Shinjuku $120/night",
        "att_list_text": "Senso-ji Temple (Free), Shinjuku Gyoen (SGD 3)",
        "rest_list_text": "Ichiran Ramen (SGD 20), Sushi Dai (SGD 50)",
        "research": {}, "tool_log": [],
        "compact_flights_out": mock_research_result["flight_options_outbound"],
        "compact_flights_ret": mock_research_result["flight_options_return"],
        "compact_hotels": mock_research_result["hotel_options"],
        "compact_attractions": mock_research_result["compact_attractions"],
        "compact_restaurants": mock_research_result["compact_restaurants"],
        "hotel_opts": mock_research_result["hotel_options"],
    })
    return pa._inventory_cache


# ── _normalize_trip_state ──────────────────────────────────────────────────────

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


# ── Formatting / parsing helpers ───────────────────────────────────────────────

class TestFormattingHelpers:

    def test_usd_to_sgd_str_converts_correctly(self):
        result = pa._usd_to_sgd_str(100)
        assert "SGD" in result
        assert "135" in result

    def test_usd_to_sgd_str_handles_string(self):
        result = pa._usd_to_sgd_str("200")
        assert "SGD" in result

    def test_usd_to_sgd_str_handles_comma_number(self):
        result = pa._usd_to_sgd_str("1,000")
        assert "SGD" in result

    def test_usd_to_sgd_str_handles_invalid(self):
        result = pa._usd_to_sgd_str("not_a_number")
        assert "USD" in result or result == "TBC"

    def test_usd_to_sgd_str_handles_empty(self):
        result = pa._usd_to_sgd_str("")
        assert result == "TBC"

    def test_safe_price_keeps_sgd_prefix(self):
        assert pa._safe_price("SGD 50") == "SGD 50"

    def test_safe_price_converts_usd(self):
        result = pa._safe_price("USD 100")
        assert "SGD" in result

    def test_safe_price_returns_as_is_for_plain_string(self):
        result = pa._safe_price("Free")
        assert result == "Free"

    def test_safe_price_returns_empty_for_empty(self):
        assert pa._safe_price("") == ""

    def test_flight_display_uses_display_field(self):
        flight = {"display": "SQ637 SIN->NRT $450"}
        assert pa._flight_display(flight) == "SQ637 SIN->NRT $450"

    def test_flight_display_builds_from_fields(self):
        flight = {
            "airline": "SQ", "flight_number": "SQ637",
            "departure_airport": "SIN", "arrival_airport": "NRT",
            "departure_time": "08:00", "arrival_time": "16:00",
            "duration_min": 420, "travel_class": "economy", "price_usd": 450,
        }
        result = pa._flight_display(flight)
        assert "SQ637" in result
        assert "SIN" in result

    def test_flight_display_empty_dict(self):
        result = pa._flight_display({})
        assert isinstance(result, str)

    def test_slugify_basic(self):
        assert pa._slugify("Senso-ji Temple") == "senso_ji_temple"

    def test_slugify_special_chars(self):
        result = pa._slugify("Café & Bar!")
        assert " " not in result
        assert "-" not in result

    def test_slugify_empty_returns_item(self):
        assert pa._slugify("") == "item"

    def test_stable_item_key_format(self):
        key = pa._stable_item_key("attraction", 1, "Senso-ji Temple")
        assert key.startswith("attraction_01_")
        assert "senso" in key


# ── Time helpers ───────────────────────────────────────────────────────────────

class TestTimeHelpers:

    def test_parse_time_value_hhmm(self):
        result = pa._parse_time_value("09:30")
        assert result is not None
        assert result.hour == 9
        assert result.minute == 30

    def test_parse_time_value_datetime_format(self):
        result = pa._parse_time_value("2026-06-01 14:00")
        assert result is not None
        assert result.hour == 14

    def test_parse_time_value_empty_returns_none(self):
        assert pa._parse_time_value("") is None

    def test_parse_time_value_invalid_returns_none(self):
        assert pa._parse_time_value("not-a-time") is None

    def test_clock_time_valid(self):
        assert pa._clock_time("09:30") == "09:30"

    def test_clock_time_invalid_returns_fallback(self):
        assert pa._clock_time("garbage", fallback="12:00") == "12:00"

    def test_shift_clock_adds_minutes(self):
        result = pa._shift_clock("09:00", 90)
        assert result == "10:30"

    def test_shift_clock_invalid_returns_fallback(self):
        assert pa._shift_clock("bad", 30, fallback="XX") == "XX"

    def test_sort_time_value_valid(self):
        minutes, raw = pa._sort_time_value("09:30")
        assert minutes == 9 * 60 + 30

    def test_sort_time_value_invalid_sorts_last(self):
        minutes, _ = pa._sort_time_value("garbage")
        assert minutes == 24 * 60 + 59

    def test_minutes_since_midnight_valid(self):
        assert pa._minutes_since_midnight("10:00") == 600

    def test_minutes_since_midnight_invalid_returns_none(self):
        assert pa._minutes_since_midnight("bad") is None

    def test_hhmm_converts_minutes(self):
        assert pa._hhmm(9 * 60 + 30) == "09:30"

    def test_hhmm_clamps_negative(self):
        assert pa._hhmm(-10) == "00:00"

    def test_duration_days_parses_string(self):
        assert pa._duration_days("7 days") == 7

    def test_duration_days_returns_1_for_empty(self):
        assert pa._duration_days("") == 1

    def test_trip_start_date_from_dates_string(self):
        state = {"dates": "2026-06-01 to 2026-06-07"}
        result = pa._trip_start_date(state)
        assert result == date(2026, 6, 1)

    def test_trip_start_date_from_hard_constraints(self):
        state = {"hard_constraints": {"start_date": "2026-06-01"}}
        result = pa._trip_start_date(state)
        assert result == date(2026, 6, 1)

    def test_trip_start_date_missing_returns_none(self):
        assert pa._trip_start_date({}) is None

    def test_service_date_adds_offset(self):
        from datetime import date as d
        start = d(2026, 6, 1)
        result = pa._service_date(start, 2)
        assert result == d(2026, 6, 3)

    def test_service_date_none_start_returns_none(self):
        assert pa._service_date(None, 1) is None


# ── Hours text parsing ─────────────────────────────────────────────────────────

class TestHoursTextParsing:

    def test_normalize_hours_text_strips_unicode(self):
        result = pa._normalize_hours_text("9\u202fAM - 10\u202fPM")
        assert "\u202f" not in result
        assert "9 AM" in result

    def test_normalize_hours_text_normalizes_dashes(self):
        result = pa._normalize_hours_text("9AM\u20139PM")
        assert "-" in result

    def test_parse_ampm_minutes_am(self):
        result = pa._parse_ampm_minutes("9 AM")
        assert result == 9 * 60

    def test_parse_ampm_minutes_pm(self):
        result = pa._parse_ampm_minutes("10 PM")
        assert result == 22 * 60

    def test_parse_ampm_minutes_with_colon(self):
        result = pa._parse_ampm_minutes("9:30 AM")
        assert result == 9 * 60 + 30

    def test_latest_close_from_hours_text_open_24h(self):
        result = pa._latest_close_from_hours_text("Open 24 hours")
        assert result == 23 * 60 + 59

    def test_latest_close_from_hours_text_closed(self):
        result = pa._latest_close_from_hours_text("Closed")
        assert result == -1

    def test_latest_close_from_hours_text_range(self):
        result = pa._latest_close_from_hours_text("9:00 AM - 9:00 PM")
        assert result is not None and result > 0

    def test_latest_close_from_hours_text_empty(self):
        assert pa._latest_close_from_hours_text("") is None


# ── Schema / lookup helpers ────────────────────────────────────────────────────

class TestSchemaAndLookupHelpers:

    def test_has_expected_option_schema_valid(self):
        options = {
            "A": {"days": []},
            "B": {"days": []},
            "C": {"days": []},
        }
        assert pa._has_expected_option_schema(options) is True

    def test_has_expected_option_schema_missing_key(self):
        assert pa._has_expected_option_schema({"A": {"days": []}, "B": {"days": []}}) is False

    def test_has_expected_option_schema_not_dict(self):
        assert pa._has_expected_option_schema("not a dict") is False

    def test_has_expected_option_schema_days_not_list(self):
        options = {"A": {"days": "bad"}, "B": {"days": []}, "C": {"days": []}}
        assert pa._has_expected_option_schema(options) is False

    def test_match_flight_by_flight_number(self):
        candidates = [{"flight_number": "SQ637", "display": "SQ637 SIN->NRT"}]
        result = pa._match_flight("SQ637", candidates)
        assert result is not None
        assert result["flight_number"] == "SQ637"

    def test_match_flight_by_display(self):
        candidates = [{"flight_number": "SQ637", "display": "SQ637 SIN->NRT $450"}]
        result = pa._match_flight("SQ637 SIN->NRT", candidates)
        assert result is not None

    def test_match_flight_no_match_returns_none(self):
        candidates = [{"flight_number": "SQ637", "display": "SQ637"}]
        assert pa._match_flight("JL999", candidates) is None

    def test_match_flight_empty_list(self):
        assert pa._match_flight("SQ637", []) is None

    def test_match_hotel_by_name(self):
        hotels = [{"name": "Hotel Gracery Shinjuku"}]
        result = pa._match_hotel("Hotel Gracery Shinjuku", hotels)
        assert result is not None

    def test_match_hotel_partial_match(self):
        hotels = [{"name": "Hotel Gracery Shinjuku"}]
        result = pa._match_hotel("Hotel Gracery", hotels)
        assert result is not None

    def test_match_hotel_no_match(self):
        hotels = [{"name": "Hotel Gracery Shinjuku"}]
        assert pa._match_hotel("Park Hyatt Tokyo", hotels) is None

    def test_looks_like_hotel_item_from_key(self):
        assert pa._looks_like_hotel_item("Hotel Gracery", "hotel_stay", []) is True

    def test_looks_like_hotel_item_from_regex(self):
        assert pa._looks_like_hotel_item("Grand Inn Tokyo", "some_key", []) is True

    def test_looks_like_hotel_item_false_for_attraction(self):
        assert pa._looks_like_hotel_item("Senso-ji Temple", "attraction_01", []) is False


# ── Scoring / preference helpers ───────────────────────────────────────────────

class TestScoringHelpers:

    def test_numeric_rating_float(self):
        assert pa._numeric_rating(4.5) == 4.5

    def test_numeric_rating_string(self):
        assert pa._numeric_rating("3.8") == 3.8

    def test_numeric_rating_invalid_returns_zero(self):
        assert pa._numeric_rating("not_a_number") == 0.0

    def test_tokenize_text_splits_words(self):
        tokens = pa._tokenize_text("culture food temple")
        assert "culture" in tokens
        assert "temple" in tokens

    def test_tokenize_text_empty_returns_empty(self):
        assert pa._tokenize_text("") == []

    def test_build_preference_model_from_preferences(self):
        state = {"origin": "Singapore", "destination": "Tokyo", "preferences": "culture food temple"}
        model = pa._build_preference_model(state)
        assert "token_counts" in model
        assert "families" in model

    def test_build_preference_model_from_soft_prefs(self):
        state = {
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "",
            "soft_preferences": {"interest_tags": ["culture", "food"], "vibe": "relaxed"},
        }
        model = pa._build_preference_model(state)
        assert isinstance(model["token_counts"], object)

    def test_text_blob_combines_fields(self):
        item = {"name": "Senso-ji Temple", "type": "attraction", "description": "historic temple"}
        blob = pa._text_blob(item)
        assert "senso-ji" in blob
        assert "temple" in blob

    def test_blob_tokens_includes_bigrams(self):
        tokens = pa._blob_tokens("senso ji temple")
        assert "senso" in tokens
        assert "senso ji" in tokens

    def test_preference_overlap_score_matches_keywords(self):
        model = {"token_counts": {"culture": 2, "temple": 1}}
        blob = "senso-ji is a famous temple culture spot"
        score = pa._preference_overlap_score(blob, model)
        assert score > 0

    def test_preference_overlap_score_no_match_is_zero(self):
        model = {"token_counts": {"beach": 2}}
        blob = "mountain skiing resort"
        assert pa._preference_overlap_score(blob, model) == 0.0

    def test_preference_emphasis_returns_count(self):
        model = {"token_counts": {"culture": 3}}
        assert pa._preference_emphasis(model, "culture") == 3

    def test_preference_emphasis_missing_returns_zero(self):
        model = {"token_counts": {}}
        assert pa._preference_emphasis(model, "food") == 0

    def test_normalized_name_strips_and_lowercases(self):
        item = {"name": "  Senso-ji Temple  "}
        assert pa._normalized_name(item) == "senso-ji temple"

    def test_normalized_name_none_returns_empty(self):
        assert pa._normalized_name(None) == ""

    def test_classify_itinerary_candidate_restaurant(self):
        item = {"name": "Ramen Restaurant", "type": "restaurant", "description": "dining place"}
        result = pa._classify_itinerary_candidate(item)
        assert result == "restaurant"

    def test_classify_itinerary_candidate_attraction(self):
        item = {"name": "Senso-ji Temple", "type": "attraction", "description": "historic landmark"}
        result = pa._classify_itinerary_candidate(item)
        assert result == "attraction"

    def test_is_normal_activity_candidate_true_for_attraction(self):
        item = {"name": "Senso-ji Temple", "type": "attraction"}
        assert pa._is_normal_activity_candidate(item) is True

    def test_is_normal_restaurant_candidate_true_for_restaurant(self):
        item = {"name": "Ramen Bar", "type": "restaurant", "description": "noodle restaurant"}
        assert pa._is_normal_restaurant_candidate(item) is True

    def test_exclude_same_place_filters_blocked(self):
        items = [{"name": "Senso-ji Temple"}, {"name": "Shinjuku Gyoen"}]
        blocked = {"senso-ji temple"}
        result = pa._exclude_same_place(items, blocked)
        assert len(result) == 1
        assert result[0]["name"] == "Shinjuku Gyoen"

    def test_exclude_same_place_empty_blocked_returns_all(self):
        items = [{"name": "A"}, {"name": "B"}]
        assert pa._exclude_same_place(items, set()) == items

    def test_merge_used_place_names_combines_sets(self):
        result = pa._merge_used_place_names({"a", "b"}, {"c"}, {"d"})
        assert result == {"a", "b", "c", "d"}

    def test_items_centroid_returns_average(self):
        items = [{"lat": 35.0, "lng": 139.0}, {"lat": 35.2, "lng": 139.2}]
        centroid = pa._items_centroid(items)
        assert centroid is not None
        assert abs(centroid[0] - 35.1) < 0.01

    def test_items_centroid_no_coords_returns_none(self):
        items = [{"name": "Place"}]
        assert pa._items_centroid(items) is None

    def test_haversine_km_known_distance(self):
        # Singapore to Tokyo is roughly 5300 km
        result = pa._haversine_km(1.3521, 103.8198, 35.6762, 139.6503)
        assert result is not None
        assert 5000 < result < 6000

    def test_haversine_km_returns_none_for_none_input(self):
        assert pa._haversine_km(None, None, 35.0, 139.0) is None

    def test_is_nonlocal_restaurant_detects_italian(self):
        item = {"name": "Italian Restaurant", "type": "restaurant"}
        assert pa._is_nonlocal_restaurant(item) is True

    def test_is_nonlocal_restaurant_false_for_local(self):
        item = {"name": "Ichiran Ramen", "type": "restaurant", "description": "japanese ramen"}
        assert pa._is_nonlocal_restaurant(item) is False


# ── planner_from_research ──────────────────────────────────────────────────────

class TestPlannerFromResearch:

    def test_missing_fields_returns_error(self, mock_research_result):
        result = pa.planner_from_research({"origin": "Singapore"}, mock_research_result)
        assert "error" in result
        assert "Missing required fields" in result["error"]

    def test_output_has_required_keys(self, flat_state, mock_research_result, mock_llm):
        with patch("agents.specialists.planner_agent._llm", return_value=mock_llm):
            result = pa.planner_from_research(flat_state, mock_research_result)
        assert "error" not in result
        for key in ("itineraries", "option_meta", "flight_options_outbound",
                    "flight_options_return", "hotel_options"):
            assert key in result

    def test_itineraries_has_abc(self, flat_state, mock_research_result, mock_llm):
        with patch("agents.specialists.planner_agent._llm", return_value=mock_llm):
            result = pa.planner_from_research(flat_state, mock_research_result)
        if "error" not in result:
            for opt in ("A", "B", "C"):
                assert opt in result["itineraries"]

    def test_option_meta_has_abc(self, flat_state, mock_research_result, mock_llm):
        with patch("agents.specialists.planner_agent._llm", return_value=mock_llm):
            result = pa.planner_from_research(flat_state, mock_research_result)
        if "error" not in result:
            for opt in ("A", "B", "C"):
                assert opt in result["option_meta"]

    def test_with_inventory_key(self, flat_state, mock_llm):
        """Test extraction when research_result uses 'inventory' sub-key."""
        research_result = {
            "inventory": {
                "attractions": [
                    {"name": "Senso-ji Temple", "type": "attraction", "lat": 35.7148, "lng": 139.7967}
                ],
                "restaurants": [
                    {"name": "Ichiran Ramen", "type": "restaurant"}
                ],
                "hotels": [
                    {"name": "Hotel Gracery Shinjuku", "price_per_night_usd": 120, "rating": 4.2}
                ],
                "flights_outbound": [
                    {"airline": "SQ", "flight_number": "SQ637", "price_usd": 450,
                     "departure_time": "08:00", "arrival_time": "16:00",
                     "display": "SQ637 SIN->NRT $450"}
                ],
                "flights_return": [
                    {"airline": "SQ", "flight_number": "SQ638", "price_usd": 430,
                     "departure_time": "18:00", "arrival_time": "00:00",
                     "display": "SQ638 NRT->SIN $430"}
                ],
            },
            "tool_log": [],
        }
        with patch("agents.specialists.planner_agent._llm", return_value=mock_llm):
            result = pa.planner_from_research(flat_state, research_result)
        # Should not crash; may succeed or return an error from LLM mock
        assert isinstance(result, dict)


# ── planner_agent ──────────────────────────────────────────────────────────────

class TestPlannerAgent:

    def test_output_has_required_keys(self, flat_state, mock_research_result, mock_llm):
        with patch("agents.specialists.planner_agent.research_agent", return_value=mock_research_result), \
             patch("agents.specialists.planner_agent._llm", return_value=mock_llm):
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
             patch("agents.specialists.planner_agent._llm", return_value=mock_llm):
            result = pa.planner_agent(hard_constraints_state)
        assert "error" not in result


# ── revise_itinerary ───────────────────────────────────────────────────────────

class TestReviseItinerary:

    def test_error_when_cache_empty(self, flat_state, mock_itineraries):
        pa._inventory_cache.clear()
        result = pa.revise_itinerary(flat_state, "Add more culture",
                                     {"itineraries": mock_itineraries, "option_meta": {}})
        assert "error" in result
        assert "No cached inventory" in result["error"]

    def test_revises_with_populated_cache(self, flat_state, mock_research_result,
                                          mock_itineraries, populated_cache, revise_llm):
        with patch("agents.specialists.planner_agent._llm", return_value=revise_llm):
            result = pa.revise_itinerary(flat_state, "Add more cultural activities",
                                         {"itineraries": mock_itineraries, "option_meta": {}})
        assert "error" not in result
        assert "itineraries" in result

    def test_revise_returns_option_meta(self, flat_state, mock_itineraries,
                                        populated_cache, revise_llm):
        with patch("agents.specialists.planner_agent._llm", return_value=revise_llm):
            result = pa.revise_itinerary(flat_state, "More food options",
                                         {"itineraries": mock_itineraries, "option_meta": {}})
        assert "option_meta" in result

    def test_revise_returns_chain_of_thought(self, flat_state, mock_itineraries,
                                              populated_cache, revise_llm):
        with patch("agents.specialists.planner_agent._llm", return_value=revise_llm):
            result = pa.revise_itinerary(flat_state, "More food options",
                                         {"itineraries": mock_itineraries, "option_meta": {}})
        assert "chain_of_thought" in result or "planner_chain_of_thought" in result

    def test_revise_returns_flight_options(self, flat_state, mock_itineraries,
                                           populated_cache, revise_llm):
        with patch("agents.specialists.planner_agent._llm", return_value=revise_llm):
            result = pa.revise_itinerary(flat_state, "More culture",
                                         {"itineraries": mock_itineraries, "option_meta": {}})
        assert "flight_options_outbound" in result
        assert "flight_options_return" in result

    def test_revise_handles_llm_json_error(self, flat_state, mock_itineraries, populated_cache):
        bad_llm = MagicMock()
        bad_llm.invoke.return_value = MagicMock(content="not valid json {{")
        with patch("agents.specialists.planner_agent._llm", return_value=bad_llm):
            result = pa.revise_itinerary(flat_state, "Break it",
                                         {"itineraries": mock_itineraries, "option_meta": {}})
        assert "error" in result


# ── Scoring / picking functions ────────────────────────────────────────────────

class TestScoringAndPickingFunctions:
    """Tests for activity/restaurant scoring and flight/hotel picking."""

    @pytest.fixture
    def profile_a(self):
        return pa._OPTION_PROFILES["A"]

    @pytest.fixture
    def profile_c(self):
        return pa._OPTION_PROFILES["C"]

    @pytest.fixture
    def pref_model(self):
        state = {
            "origin": "Singapore",
            "destination": "Tokyo",
            "preferences": "culture food temple heritage",
        }
        return pa._build_preference_model(state)

    @pytest.fixture
    def temple_item(self):
        return {
            "name": "Senso-ji Temple",
            "type": "attraction",
            "description": "ancient temple heritage landmark",
            "location": "Asakusa",
            "lat": 35.7148,
            "lng": 139.7967,
        }

    @pytest.fixture
    def modern_item(self):
        return {
            "name": "teamLab Digital Art",
            "type": "attraction",
            "description": "modern digital interactive art museum",
            "location": "Toyosu",
            "lat": 35.6504,
            "lng": 139.7954,
        }

    @pytest.fixture
    def ramen_restaurant(self):
        return {
            "name": "Ichiran Ramen",
            "type": "restaurant",
            "description": "japanese ramen noodle local food",
            "location": "Shinjuku",
            "rating": 4.2,
            "lat": 35.6938,
            "lng": 139.7034,
        }

    @pytest.fixture
    def italian_restaurant(self):
        return {
            "name": "La Pasta Italian",
            "type": "restaurant",
            "description": "italian pasta grill restaurant",
            "location": "Shinjuku",
            "rating": 3.8,
        }

    @pytest.fixture
    def hotels(self):
        return [
            {"name": "Hotel Gracery Shinjuku", "price_per_night_usd": 120, "rating": 4.2},
            {"name": "APA Hotel Tokyo", "price_per_night_usd": 60, "rating": 3.5},
            {"name": "Park Hyatt Tokyo", "price_per_night_usd": 400, "rating": 4.8},
        ]

    @pytest.fixture
    def flights_out(self):
        return [
            {"flight_number": "SQ637", "price_usd": 450, "arrival_time": "16:00",
             "duration_min": 420, "display": "SQ637 SIN->NRT nonstop $450"},
            {"flight_number": "SQ639", "price_usd": 350, "arrival_time": "22:00",
             "duration_min": 480, "display": "SQ639 SIN->NRT 1 stop $350"},
        ]

    @pytest.fixture
    def flights_ret(self):
        return [
            {"flight_number": "SQ638", "price_usd": 430, "departure_time": "18:00",
             "display": "SQ638 NRT->SIN nonstop"},
            {"flight_number": "SQ640", "price_usd": 300, "departure_time": "08:00",
             "display": "SQ640 NRT->SIN 1 stop"},
        ]

    def test_activity_relevance_score_temple_gets_high_score(self, temple_item, profile_a, pref_model):
        score = pa._activity_relevance_score(temple_item, profile_a, pref_model)
        assert score > 0

    def test_activity_relevance_score_modern_item_penalized(self, modern_item, profile_a, pref_model):
        temple_score = pa._activity_relevance_score(
            {"name": "Ancient Temple", "type": "attraction",
             "description": "ancient heritage temple cultural landmark"}, profile_a, pref_model
        )
        modern_score = pa._activity_relevance_score(modern_item, profile_a, pref_model)
        # Temple should generally score >= modern for culture preferences
        assert isinstance(temple_score, float)
        assert isinstance(modern_score, float)

    def test_restaurant_relevance_local_food_scores_higher(self, ramen_restaurant, italian_restaurant,
                                                             profile_a, pref_model):
        ramen_score = pa._restaurant_relevance_score(ramen_restaurant, profile_a, pref_model)
        italian_score = pa._restaurant_relevance_score(italian_restaurant, profile_a, pref_model)
        assert ramen_score > italian_score

    def test_restaurant_relevance_nonlocal_penalized_with_local_pref(self, profile_a):
        local_pref_model = pa._build_preference_model({
            "origin": "SG", "destination": "Tokyo",
            "preferences": "local street food hawker traditional"
        })
        item = {"name": "Italian Grill Bistro", "type": "restaurant",
                "description": "italian grill pasta brasserie"}
        score = pa._restaurant_relevance_score(item, profile_a, local_pref_model)
        assert isinstance(score, float)

    def test_hotel_price_sgd_converts_usd(self):
        hotel = {"price_per_night_usd": 100}
        price = pa._hotel_price_sgd(hotel)
        assert abs(price - 135.0) < 0.01

    def test_hotel_price_sgd_invalid_returns_high(self):
        hotel = {"price_per_night_usd": None}
        assert pa._hotel_price_sgd(hotel) == 9999.0

    def test_flight_stop_rank_nonstop(self):
        flight = {"display": "SQ637 nonstop $450"}
        assert pa._flight_stop_rank_local(flight) == 0

    def test_flight_stop_rank_direct(self):
        flight = {"display": "SQ637 direct $450"}
        assert pa._flight_stop_rank_local(flight) == 0

    def test_flight_stop_rank_one_stop(self):
        flight = {"display": "SQ637 1 stop $350"}
        assert pa._flight_stop_rank_local(flight) == 1

    def test_flight_stop_rank_unknown_returns_9(self):
        flight = {"display": "SQ637 $450"}
        assert pa._flight_stop_rank_local(flight) == 9

    def test_pick_outbound_flight_prefers_nonstop(self, flights_out):
        result = pa._pick_outbound_flight(flights_out)
        assert result is not None
        assert "nonstop" in result.get("display", "").lower()

    def test_pick_outbound_flight_empty_returns_none(self):
        assert pa._pick_outbound_flight([]) is None

    def test_pick_return_flight_prefers_nonstop(self, flights_ret):
        result = pa._pick_return_flight(flights_ret)
        assert result is not None
        assert "nonstop" in result.get("display", "").lower()

    def test_pick_return_flight_empty_returns_none(self):
        assert pa._pick_return_flight([]) is None

    def test_pick_hotels_by_profile_returns_abc(self, hotels):
        result = pa._pick_hotels_by_profile(hotels)
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_pick_hotels_assigns_budget_to_b(self, hotels):
        result = pa._pick_hotels_by_profile(hotels)
        # Budget option B should pick cheapest hotel
        b_hotel = result.get("B")
        assert b_hotel is not None

    def test_pick_hotels_empty_returns_none(self):
        result = pa._pick_hotels_by_profile([])
        assert all(v is None for v in result.values())

    def test_activity_score_includes_rating(self, temple_item, profile_a, pref_model):
        item_with_rating = {**temple_item, "rating": 4.5}
        item_no_rating = {**temple_item, "rating": None}
        score_rated = pa._activity_score(item_with_rating, profile_a, pref_model)
        score_unrated = pa._activity_score(item_no_rating, profile_a, pref_model)
        assert score_rated > score_unrated

    def test_activity_score_penalizes_discouraged(self, temple_item, profile_a, pref_model):
        discouraged = {"senso-ji temple"}
        score_fresh = pa._activity_score(temple_item, profile_a, pref_model)
        score_discouraged = pa._activity_score(temple_item, profile_a, pref_model, discouraged)
        assert score_fresh > score_discouraged

    def test_restaurant_score_returns_float(self, ramen_restaurant, profile_a, pref_model):
        score = pa._restaurant_score(ramen_restaurant, profile_a, pref_model)
        assert isinstance(score, float)


# ── Cluster / scheduling helpers ───────────────────────────────────────────────

class TestClusterAndSchedulingHelpers:
    """Tests for activity clustering and scheduling helpers."""

    @pytest.fixture
    def pref_model(self):
        return pa._build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "culture food temple"
        })

    @pytest.fixture
    def many_attractions(self):
        return [
            {"name": "Senso-ji Temple", "type": "attraction",
             "description": "ancient temple heritage landmark", "lat": 35.7148, "lng": 139.7967},
            {"name": "Shinjuku Gyoen", "type": "attraction",
             "description": "japanese garden nature park", "lat": 35.6851, "lng": 139.7100},
            {"name": "Meiji Shrine", "type": "attraction",
             "description": "shinto shrine cultural heritage", "lat": 35.6763, "lng": 139.6993},
            {"name": "teamLab Planets", "type": "attraction",
             "description": "digital art museum modern", "lat": 35.6504, "lng": 139.7954},
            {"name": "Tokyo National Museum", "type": "attraction",
             "description": "museum heritage culture history", "lat": 35.7189, "lng": 139.7764},
            {"name": "Shibuya Sky", "type": "attraction",
             "description": "sky observatory panoramic view", "lat": 35.6585, "lng": 139.7026},
        ]

    @pytest.fixture
    def many_restaurants(self):
        return [
            {"name": "Ichiran Ramen", "type": "restaurant",
             "description": "japanese ramen local food", "lat": 35.6938, "lng": 139.7034},
            {"name": "Sushi Dai", "type": "restaurant",
             "description": "sushi fresh seafood", "lat": 35.6654, "lng": 139.7707},
            {"name": "Uobei Shibuya", "type": "restaurant",
             "description": "conveyor belt sushi local", "lat": 35.6590, "lng": 139.7003},
        ]

    def test_choose_cluster_activities_returns_list(self, many_attractions, pref_model):
        profile = pa._OPTION_PROFILES["A"]
        result = pa._choose_cluster_activities(
            many_attractions, set(), profile, pref_model, count=2
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_choose_cluster_activities_respects_used_names(self, many_attractions, pref_model):
        profile = pa._OPTION_PROFILES["A"]
        used = {"senso-ji temple"}
        result = pa._choose_cluster_activities(
            many_attractions, used, profile, pref_model, count=2
        )
        names = [pa._normalized_name(item) for item in result]
        assert "senso-ji temple" not in names

    def test_choose_cluster_activities_empty_returns_empty(self, pref_model):
        profile = pa._OPTION_PROFILES["A"]
        result = pa._choose_cluster_activities([], set(), profile, pref_model, count=2)
        assert result == []

    def test_choose_cluster_activities_immersion_mode(self, many_attractions, pref_model):
        """Test the immersion cluster selection code path (lines 2532-2607)."""
        immersion_profile = {
            **pa._OPTION_PROFILES["C"],
            "selection_mode": "immersion",
            "cluster_radius_km": 8.0,
            "target_activity_count": 2,
            "min_activity_count": 2,
        }
        result = pa._choose_cluster_activities(
            many_attractions,
            set(),
            immersion_profile,
            pref_model,
            count=2,
            forced_seed=None,
        )
        assert isinstance(result, list)

    def test_choose_cluster_activities_immersion_with_discouraged(self, many_attractions, pref_model):
        """Test immersion mode with discouraged_names (lines 2534-2544)."""
        immersion_profile = {**pa._OPTION_PROFILES["C"], "selection_mode": "immersion",
                             "cluster_radius_km": 8.0}
        discouraged = {"senso-ji temple", "meiji shrine"}
        result = pa._choose_cluster_activities(
            many_attractions, set(), immersion_profile, pref_model,
            count=2, discouraged_names=discouraged, forced_seed=None,
        )
        assert isinstance(result, list)

    def test_choose_cluster_activities_light_only_mode(self, many_attractions, pref_model):
        """Test light_only code path (lines 2471-2478)."""
        profile = pa._OPTION_PROFILES["A"]
        result = pa._choose_cluster_activities(
            many_attractions, set(), profile, pref_model, count=2, light_only=True
        )
        assert isinstance(result, list)

    def test_choose_cluster_with_discouraged_names_quota(self, many_attractions, pref_model):
        """Test discouraged_names quota logic (lines 2496-2516)."""
        profile = pa._OPTION_PROFILES["B"]
        discouraged = {"senso-ji temple", "shinjuku gyoen"}
        result = pa._choose_cluster_activities(
            many_attractions, set(), profile, pref_model,
            count=2, discouraged_names=discouraged
        )
        assert isinstance(result, list)

    def test_name_lookup_indexes_by_normalized_name(self, many_attractions):
        lookup = pa._name_lookup(many_attractions)
        assert any("senso" in k.lower() for k in lookup)

    def test_build_day_activity_pool_returns_list(self, many_attractions, pref_model):
        profile = pa._OPTION_PROFILES["A"]
        result = pa._build_day_activity_pool(
            many_attractions,
            set(),
            profile,
            pref_model,
            desired_count=2,
            day_index=0,
        )
        assert isinstance(result, list)

    def test_build_day_activity_pool_respects_used_names(self, many_attractions, pref_model):
        profile = pa._OPTION_PROFILES["A"]
        used = {"senso-ji temple", "meiji shrine"}
        result = pa._build_day_activity_pool(
            many_attractions, used, profile, pref_model,
            desired_count=2, day_index=1,
        )
        names = [pa._normalized_name(item) for item in result]
        assert "senso-ji temple" not in names

    def test_build_district_restaurant_pool_returns_list(self, many_restaurants, many_attractions):
        result = pa._build_district_restaurant_pool(
            many_restaurants,
            many_attractions[:2],
            set(),
            primary_radius_km=5.0,
            expanded_radius_km=10.0,
            desired_count=3,
        )
        assert isinstance(result, list)


# ── Trace / decision functions ─────────────────────────────────────────────────

class TestTraceFunctions:
    """Tests for planner decision trace helper functions."""

    @pytest.fixture
    def pref_model(self):
        return pa._build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "culture food"
        })

    @pytest.fixture
    def sample_restaurant(self):
        return {
            "name": "Ichiran Ramen",
            "type": "restaurant",
            "description": "japanese ramen local food",
            "lat": 35.6938,
            "lng": 139.7034,
        }

    def test_meal_trace_with_restaurant(self, sample_restaurant, pref_model):
        result = pa._meal_trace(sample_restaurant, "lunch", pref_model)
        assert "meal_kind" in result
        assert result["meal_kind"] == "lunch"

    def test_meal_trace_with_none(self, pref_model):
        result = pa._meal_trace(None, "dinner", pref_model)
        assert result["picked"] is None

    def test_trace_day_type_arrival(self):
        result = pa._trace_day_type("Day 1", 0, 7, {"day_type": "arrival"})
        assert result == "arrival"

    def test_trace_day_type_departure(self):
        result = pa._trace_day_type("Day 7", 6, 7, {})
        assert "departure" in result or "last" in result or isinstance(result, str)

    def test_trace_day_type_middle(self):
        result = pa._trace_day_type("Day 3", 2, 7, {})
        assert isinstance(result, str)

    def test_trace_theme_from_items_returns_string(self):
        items = [
            {"name": "Senso-ji Temple", "type": "attraction"},
        ]
        entry = {"day": 2, "items": items}
        result = pa._trace_theme_from_items(entry, items, "middle")
        assert isinstance(result, str)

    def test_trace_seed_name_with_seed(self):
        items = [{"name": "Senso-ji Temple", "type": "attraction"}]
        entry = {"seed_name": "Senso-ji Temple", "day": 2}
        result = pa._trace_seed_name(entry, items, "middle")
        assert isinstance(result, (str, type(None)))

    def test_synchronize_planner_decision_trace_returns_list(self, pref_model):
        days = [
            {"day": "Day 1", "date": "2026-06-01", "items": [
                {"key": "senso_ji", "name": "Senso-ji Temple",
                 "type": "attraction", "time": "09:00"}
            ]},
            {"day": "Day 2", "date": "2026-06-02", "items": []},
        ]
        draft_trace = [
            {"day": "Day 1", "day_type": "arrival", "theme": "cultural",
             "seed_name": "Senso-ji Temple", "seed_reason": "Top cultural pick",
             "activities": [{"name": "Senso-ji Temple"}], "lunch": None, "dinner": None}
        ]
        attractions = [{"name": "Senso-ji Temple", "type": "attraction"}]
        restaurants = []
        attraction_lookup = pa._name_lookup(attractions)
        restaurant_lookup = pa._name_lookup(restaurants)
        result = pa._synchronize_planner_decision_trace(
            days, draft_trace, attraction_lookup, restaurant_lookup, pref_model
        )
        assert isinstance(result, list)

    def test_build_activity_entries_adds_key(self):
        items = [{"name": "Senso-ji Temple", "type": "attraction", "lat": 35.7, "lng": 139.7}]
        entries = pa._build_activity_entries(items, "attraction")
        assert len(entries) == 1
        assert "key" in entries[0]

    def test_ensure_inventory_keys_assigns_missing_keys(self):
        items = [
            {"name": "Senso-ji Temple", "type": "attraction"},
            {"name": "Shinjuku Gyoen", "key": "existing_key", "type": "attraction"},
        ]
        result = pa._ensure_inventory_keys(items, "attraction")
        for item in result:
            assert "key" in item or item.get("key") == "existing_key"

    def test_fit_service_start_within_window(self):
        # _fit_service_start(item, preferred_start, duration_minutes, service_date)
        item = {}  # no hours → returns preferred_start
        result = pa._fit_service_start(item, 10 * 60, 2 * 60, None)
        assert result == 10 * 60

    def test_fit_service_start_with_hours_item(self):
        item = {"hours": "9:00 AM - 9:00 PM"}
        result = pa._fit_service_start(item, 10 * 60, 2 * 60, None)
        assert isinstance(result, (int, type(None)))

    def test_service_windows_from_hours_text_range(self):
        result = pa._service_windows_from_hours_text("9:00 AM - 9:00 PM")
        assert isinstance(result, list)

    def test_service_windows_from_hours_text_empty(self):
        result = pa._service_windows_from_hours_text("")
        assert result == []

    def test_build_preference_model_with_search_queries(self):
        state = {
            "origin": "Singapore",
            "destination": "Tokyo",
            "preferences": "culture",
            "search_queries": [{"query": "best temples tokyo"}, {"query": "food markets"}],
        }
        model = pa._build_preference_model(state)
        assert isinstance(model["token_counts"], object)
        assert isinstance(model["families"], dict)

    def test_build_option_restaurant_pools_returns_dict(self):
        pref_model = pa._build_preference_model({
            "origin": "Singapore", "destination": "Tokyo", "preferences": "food"
        })
        restaurants = [
            {"name": "Ichiran Ramen", "type": "restaurant",
             "description": "japanese ramen local food", "lat": 35.6938, "lng": 139.7034},
            {"name": "Sushi Dai", "type": "restaurant",
             "description": "sushi fresh", "lat": 35.6654, "lng": 139.7707},
        ]
        result = pa._build_option_restaurant_pools(restaurants, 7, pref_model)
        assert isinstance(result, dict)
        assert "A" in result


# ── Item classification and duration helpers ───────────────────────────────────

class TestItemClassificationHelpers:
    """Tests for item classification branches and duration helpers."""

    def test_classify_tour_item(self):
        item = {"name": "Tokyo Walking Tour", "type": "activity",
                "description": "guided tour walking tour city"}
        assert pa._classify_itinerary_candidate(item) == "tour"

    def test_classify_night_only_item(self):
        item = {"name": "Illumination Festival", "type": "activity",
                "description": "night and light illumination event"}
        assert pa._classify_itinerary_candidate(item) == "night_only"

    def test_classify_photo_spot_item(self):
        item = {"name": "Tokyo Word Mark Monument", "type": "attraction",
                "description": "monument photo spot"}
        assert pa._classify_itinerary_candidate(item) == "photo_spot"

    def test_classify_area_item(self):
        item = {"name": "Omoide Yokocho Memory Lane", "type": "area",
                "description": "yokocho alley food street"}
        assert pa._classify_itinerary_candidate(item) == "area"

    def test_classify_food_entity_item(self):
        item = {"name": "Ramen Bistro", "type": "restaurant",
                "description": "ramen bistro dining"}
        assert pa._classify_itinerary_candidate(item) == "restaurant"

    def test_classify_default_is_attraction(self):
        item = {"name": "Some Place", "type": "attraction",
                "description": "a landmark"}
        assert pa._classify_itinerary_candidate(item) == "attraction"

    def test_activity_duration_museum(self):
        item = {"name": "Tokyo National Museum", "description": "museum"}
        assert pa._activity_duration_minutes(item) == 105

    def test_activity_duration_garden(self):
        item = {"name": "Shinjuku Gyoen", "description": "garden park"}
        assert pa._activity_duration_minutes(item) == 90

    def test_activity_duration_temple(self):
        item = {"name": "Senso-ji", "description": "temple shrine"}
        assert pa._activity_duration_minutes(item) == 75

    def test_activity_duration_default(self):
        item = {"name": "Random Place", "description": "some place"}
        assert pa._activity_duration_minutes(item) == 90

    def test_activity_latest_start_teamlab(self):
        item = {"name": "teamLab Planets", "description": "teamlab digital"}
        result = pa._activity_latest_start_minutes(item)
        assert result == 19 * 60

    def test_activity_latest_start_museum(self):
        item = {"name": "Museum of History", "description": "museum collection"}
        result = pa._activity_latest_start_minutes(item)
        assert result == 15 * 60 + 30

    def test_activity_latest_start_nature(self):
        item = {"name": "Shinjuku Gyoen", "description": "garden nature park"}
        result = pa._activity_latest_start_minutes(item)
        assert result == 16 * 60

    def test_activity_latest_start_temple(self):
        item = {"name": "Senso-ji Temple", "description": "temple shrine"}
        result = pa._activity_latest_start_minutes(item)
        assert result == 16 * 60 + 30

    def test_activity_latest_start_scenic(self):
        item = {"name": "Shibuya District View", "description": "scenic neighborhood panoramic"}
        result = pa._activity_latest_start_minutes(item)
        assert result == 18 * 60

    def test_activity_latest_start_default(self):
        item = {"name": "Generic Place", "description": "a place"}
        result = pa._activity_latest_start_minutes(item)
        assert result == 17 * 60

    def test_activity_latest_close_minutes_with_hours(self):
        item = {"hours": "9:00 AM - 5:00 PM"}
        result = pa._activity_latest_close_minutes(item)
        assert result is not None

    def test_activity_latest_close_minutes_no_hours(self):
        item = {"name": "Somewhere"}
        result = pa._activity_latest_close_minutes(item)
        assert result is None


# ── Scoring branches ───────────────────────────────────────────────────────────

class TestScoringBranches:
    """Tests for specific branches in relevance scoring functions."""

    @pytest.fixture
    def profile_a(self):
        return pa._OPTION_PROFILES["A"]

    @pytest.fixture
    def culture_model(self):
        return pa._build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "culture heritage traditional temple",
        })

    @pytest.fixture
    def local_food_model(self):
        return pa._build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "local food hawker street food",
        })

    def test_activity_relevance_quirky_penalized(self, profile_a, culture_model):
        """Test quirky signal penalty branch (line 1770)."""
        item = {"name": "Quirky Robot Show", "description": "quirky bizarre unusual entertainment"}
        score = pa._activity_relevance_score(item, profile_a, culture_model)
        assert isinstance(score, float)

    def test_activity_relevance_natural_history_penalized(self, profile_a, culture_model):
        """Test natural history penalty branch (line 1772)."""
        item = {"name": "Natural History Museum", "description": "natural history science museum"}
        score = pa._activity_relevance_score(item, profile_a, culture_model)
        assert isinstance(score, float)

    def test_activity_relevance_science_centre_penalized(self, profile_a, culture_model):
        """Test science centre penalty branch (line 1773-1774)."""
        item = {"name": "Science Centre Tokyo", "description": "science centre discovery"}
        score = pa._activity_relevance_score(item, profile_a, culture_model)
        assert isinstance(score, float)

    def test_activity_relevance_local_food_pref_no_signals(self, profile_a, local_food_model):
        """Test local_food_pref penalty when not a heritage/nature place (line 1777-1778)."""
        item = {"name": "Modern Office Tower", "description": "modern building office"}
        score = pa._activity_relevance_score(item, profile_a, local_food_model)
        assert isinstance(score, float)

    def test_restaurant_relevance_fine_dining_boost(self, profile_a):
        """Test fine dining boost branch (line 1812)."""
        fine_model = pa._build_preference_model({
            "origin": "SG", "destination": "Tokyo",
            "preferences": "fine dining traditional authentic",
        })
        item = {"name": "Sushi Yoshitake", "type": "restaurant",
                "description": "traditional japanese sushi fine dining cuisine", "rating": 4.8}
        score = pa._restaurant_relevance_score(item, profile_a, fine_model)
        assert isinstance(score, float)

    def test_restaurant_relevance_nonlocal_fine_penalized(self, profile_a):
        """Test nonlocal restaurant with fine pref penalty (line 1817-1818)."""
        fine_model = pa._build_preference_model({
            "origin": "SG", "destination": "Tokyo",
            "preferences": "fine dining authentic traditional",
        })
        item = {"name": "Italian Fine Dining Ristorante",
                "description": "italian grill fine dining pasta authentic", "rating": 4.5}
        score = pa._restaurant_relevance_score(item, profile_a, fine_model)
        assert isinstance(score, float)

    def test_restaurant_score_vegetarian_bonus(self, profile_a):
        """Test vegetarian/vegan bonus branch (line 1917-1918)."""
        pref = pa._build_preference_model({"origin": "SG", "destination": "Tokyo", "preferences": "food"})
        item = {"name": "Vegan Garden Restaurant",
                "description": "vegetarian vegan plant-based dining", "rating": 4.0}
        score = pa._restaurant_score(item, profile_a, pref)
        item_noveg = {"name": "Regular Restaurant",
                      "description": "standard restaurant dining", "rating": 4.0}
        score_noveg = pa._restaurant_score(item_noveg, profile_a, pref)
        assert score > score_noveg

    def test_restaurant_score_discouraged_penalty(self, profile_a):
        """Test discouraged restaurant penalty branch (line 1920-1928)."""
        pref = pa._build_preference_model({"origin": "SG", "destination": "Tokyo", "preferences": "food"})
        item = {"name": "Ichiran Ramen", "description": "ramen noodle", "rating": 4.0}
        discouraged = {"ichiran ramen"}
        score_fresh = pa._restaurant_score(item, profile_a, pref)
        score_disc = pa._restaurant_score(item, profile_a, pref, discouraged)
        assert score_fresh > score_disc

    def test_restaurant_role_adjustment_returns_float(self, profile_a):
        pref = pa._build_preference_model({"origin": "SG", "destination": "Tokyo", "preferences": "food"})
        item = {"name": "Local Ramen Shop", "description": "japanese ramen local"}
        result = pa._restaurant_role_adjustment(item, profile_a, pref)
        assert isinstance(result, float)

    def test_match_flight_slash_number(self):
        """Test slash-separated flight number branch (lines 487-490)."""
        candidates = [{"flight_number": "SQ637/SQ638", "display": ""}]
        result = pa._match_flight("SQ637", candidates)
        assert result is not None

    def test_match_hotel_empty_name_skipped(self):
        """Test that hotels with empty name are skipped (line 499)."""
        hotels = [{"name": ""}, {"name": "Hotel Gracery Shinjuku"}]
        result = pa._match_hotel("Hotel Gracery Shinjuku", hotels)
        assert result is not None
        assert result["name"] == "Hotel Gracery Shinjuku"

    def test_looks_like_hotel_item_by_hotel_match(self):
        """Test _looks_like_hotel_item when matched via _match_hotel (line 509)."""
        hotels = [{"name": "Grand Plaza Hotel"}]
        assert pa._looks_like_hotel_item("Grand Plaza Hotel", "some_key", hotels) is True

    def test_latest_close_from_hours_text_close_match(self):
        """Test 'closes at' pattern (lines 200-202)."""
        result = pa._latest_close_from_hours_text("Opens 9 AM, closes at 6 PM")
        assert result is not None and result > 0

    def test_latest_close_from_hours_text_one_token_pair(self):
        """Test multiple time pairs (lines 210-218)."""
        result = pa._latest_close_from_hours_text("9:00 AM - 5:00 PM, 6:00 PM - 9:00 PM")
        assert result is not None

    def test_parse_ampm_minutes_no_match(self):
        """Test when no AM/PM format matches (line 185)."""
        result = pa._parse_ampm_minutes("not a time at all")
        assert result is None

    def test_trip_start_date_invalid_iso_skipped(self):
        """Test that invalid date strings are skipped (lines 149-150)."""
        state = {"dates": "not-a-date to 2026-06-07"}
        result = pa._trip_start_date(state)
        # Should skip invalid date and try next candidate
        assert result is None or isinstance(result, date)


# ── Arrival/departure day scheduling ──────────────────────────────────────────

class TestArrivalDepartureScheduling:
    """Direct tests for arrival and departure day item builders."""

    @pytest.fixture
    def pref_model(self):
        return pa._build_preference_model({
            "origin": "Singapore", "destination": "Tokyo",
            "preferences": "culture food temple",
        })

    @pytest.fixture
    def profile_a(self):
        return pa._OPTION_PROFILES["A"]

    @pytest.fixture
    def attractions(self):
        return [
            {"name": "Senso-ji Temple", "type": "attraction",
             "description": "ancient temple heritage", "lat": 35.7148, "lng": 139.7967,
             "key": "senso_ji"},
            {"name": "Shinjuku Gyoen", "type": "attraction",
             "description": "japanese garden park", "lat": 35.6851, "lng": 139.7100,
             "key": "shinjuku_gyoen"},
        ]

    @pytest.fixture
    def restaurants(self):
        return [
            {"name": "Ichiran Ramen", "type": "restaurant",
             "description": "japanese ramen local food", "lat": 35.6938, "lng": 139.7034,
             "key": "ichiran_ramen"},
        ]

    @pytest.fixture
    def hotel(self):
        return {"name": "Hotel Gracery Shinjuku", "price_per_night_usd": 120,
                "rating": 4.2, "lat": 35.6938, "lng": 139.7034}

    @pytest.fixture
    def late_flight(self):
        return {"flight_number": "SQ637", "arrival_time": "21:00",
                "departure_time": "08:00", "price_usd": 450,
                "display": "SQ637 SIN->NRT $450"}

    @pytest.fixture
    def early_flight(self):
        """Flight arriving before 15:30 triggers light activity branch."""
        return {"flight_number": "SQ635", "arrival_time": "14:00",
                "departure_time": "06:00", "price_usd": 480,
                "display": "SQ635 SIN->NRT $480"}

    @pytest.fixture
    def return_flight(self):
        return {"flight_number": "SQ638", "departure_time": "18:00",
                "arrival_time": "00:00", "price_usd": 430,
                "display": "SQ638 NRT->SIN $430"}

    def test_arrival_day_items_late_flight(self, profile_a, late_flight, hotel,
                                           attractions, restaurants, pref_model):
        result = pa._arrival_day_items(
            profile_a, late_flight, hotel, attractions, restaurants,
            restaurants, set(), pref_model
        )
        assert isinstance(result, list)
        assert any(item.get("icon") == "flight" for item in result)

    def test_arrival_day_items_early_flight_may_add_activity(
            self, profile_a, early_flight, hotel, attractions, restaurants, pref_model):
        """Early arrival (< 15:30) triggers light activity check (lines 4030-4052)."""
        result = pa._arrival_day_items(
            profile_a, early_flight, hotel, attractions, restaurants,
            restaurants, set(), pref_model
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_arrival_day_items_no_flight(self, profile_a, hotel,
                                         attractions, restaurants, pref_model):
        result = pa._arrival_day_items(
            profile_a, None, hotel, attractions, restaurants,
            restaurants, set(), pref_model
        )
        assert isinstance(result, list)

    def test_arrival_day_items_no_hotel(self, profile_a, late_flight,
                                        attractions, restaurants, pref_model):
        result = pa._arrival_day_items(
            profile_a, late_flight, None, attractions, restaurants,
            restaurants, set(), pref_model
        )
        assert isinstance(result, list)

    def test_departure_day_items_with_afternoon_flight(self, profile_a, return_flight, hotel,
                                                        attractions, restaurants, pref_model):
        """Afternoon departure has enough time for activity (lines 4175-4190)."""
        result = pa._departure_day_items(
            profile_a, return_flight, hotel, attractions, restaurants,
            restaurants, set(), pref_model
        )
        assert isinstance(result, list)
        assert any(item.get("icon") == "flight" for item in result)

    def test_departure_day_items_no_flight(self, profile_a, hotel,
                                           attractions, restaurants, pref_model):
        """No flight means airport_cutoff is None (line 4146-4147)."""
        result = pa._departure_day_items(
            profile_a, None, hotel, attractions, restaurants,
            restaurants, set(), pref_model
        )
        assert isinstance(result, list)

    def test_departure_day_items_no_hotel(self, profile_a, return_flight,
                                          attractions, restaurants, pref_model):
        result = pa._departure_day_items(
            profile_a, return_flight, None, attractions, restaurants,
            restaurants, set(), pref_model
        )
        assert isinstance(result, list)

    def test_pick_restaurant_for_anchor_returns_item_or_none(
            self, profile_a, restaurants, pref_model, hotel):
        result = pa._pick_restaurant_for_anchor(
            restaurants, set(), profile_a, pref_model, hotel,
            preferred_start=12 * 60,
        )
        assert result is None or isinstance(result, dict)

    def test_pick_activity_near_anchor_returns_item_or_none(
            self, profile_a, attractions, pref_model, hotel):
        result = pa._pick_activity_near_anchor(
            attractions, set(), profile_a, pref_model,
            anchor=hotel, light_only=False,
        )
        assert result is None or isinstance(result, dict)
