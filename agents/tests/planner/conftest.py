"""
Shared fixtures for Agent3 (Planner) tests.
"""

from __future__ import annotations

import json
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

from agents.specialists.planner_agent import _inventory_cache  # noqa: E402


# ── State fixtures ────────────────────────────────────────────────────────────

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


# ── Research result fixture ───────────────────────────────────────────────────

@pytest.fixture
def mock_research_result():
    return {
        "flight_options_outbound": [
            {
                "airline": "SQ", "flight_number": "SQ637", "price_usd": 450,
                "departure_time": "08:00", "arrival_time": "16:00",
                "departure_airport": "SIN", "arrival_airport": "NRT",
                "duration_min": 420, "travel_class": "economy",
                "display": "SQ637 SIN->NRT $450",
            }
        ],
        "flight_options_return": [
            {
                "airline": "SQ", "flight_number": "SQ638", "price_usd": 430,
                "departure_time": "18:00", "arrival_time": "00:00",
                "departure_airport": "NRT", "arrival_airport": "SIN",
                "duration_min": 420, "travel_class": "economy",
                "display": "SQ638 NRT->SIN $430",
            }
        ],
        "hotel_options": [
            {
                "name": "Hotel Gracery Shinjuku", "price_per_night_usd": 120,
                "rating": 4.2, "location": "Shinjuku",
                "display": "Hotel Gracery $120/night",
            }
        ],
        "compact_attractions": [
            {
                "item_key": "senso_ji", "name": "Senso-ji Temple", "type": "attraction",
                "location": "Asakusa", "price_sgd": "Free", "duration_hrs": 2,
                "lat": 35.7148, "lng": 139.7967,
            },
            {
                "item_key": "shinjuku_gyoen", "name": "Shinjuku Gyoen", "type": "attraction",
                "location": "Shinjuku", "price_sgd": "SGD 3", "duration_hrs": 2,
                "lat": 35.6851, "lng": 139.7100,
            },
            {
                "item_key": "meiji_shrine", "name": "Meiji Shrine", "type": "attraction",
                "location": "Harajuku", "price_sgd": "Free", "duration_hrs": 1.5,
                "lat": 35.6763, "lng": 139.6993,
            },
            {
                "item_key": "teamlab", "name": "teamLab Planets", "type": "attraction",
                "location": "Toyosu", "price_sgd": "SGD 40", "duration_hrs": 2,
                "lat": 35.6504, "lng": 139.7954,
            },
            {
                "item_key": "ueno_museum", "name": "Tokyo National Museum", "type": "attraction",
                "location": "Ueno", "price_sgd": "SGD 15", "duration_hrs": 3,
                "lat": 35.7189, "lng": 139.7764,
            },
            {
                "item_key": "shibuya_sky", "name": "Shibuya Sky Observatory", "type": "attraction",
                "location": "Shibuya", "price_sgd": "SGD 25", "duration_hrs": 1,
                "lat": 35.6585, "lng": 139.7026,
            },
        ],
        "compact_restaurants": [
            {
                "item_key": "ichiran_ramen", "name": "Ichiran Ramen", "type": "restaurant",
                "location": "Shinjuku", "price_sgd": "SGD 20", "duration_hrs": 1,
                "lat": 35.6938, "lng": 139.7034,
            },
            {
                "item_key": "sushi_dai", "name": "Sushi Dai", "type": "restaurant",
                "location": "Tsukiji", "price_sgd": "SGD 50", "duration_hrs": 1.5,
                "lat": 35.6654, "lng": 139.7707,
            },
            {
                "item_key": "uobei_sushi", "name": "Uobei Shibuya", "type": "restaurant",
                "location": "Shibuya", "price_sgd": "SGD 15", "duration_hrs": 1,
                "lat": 35.6590, "lng": 139.7003,
            },
        ],
        "tool_log": [{"tool": "search_flights", "status": "ok"}],
        "att_list_text": "Senso-ji Temple (Free), Shinjuku Gyoen (SGD 3)",
        "rest_list_text": "Ichiran Ramen (SGD 20), Sushi Dai (SGD 50)",
        "hotel_list_text": "Hotel Gracery Shinjuku $120/night",
        "flight_out_text": "SQ637 SIN->NRT $450",
        "flight_ret_text": "SQ638 NRT->SIN $430",
    }


# ── Mock planner result ───────────────────────────────────────────────────────

@pytest.fixture
def mock_planner_result():
    return {
        "itineraries": {
            "A": [
                {
                    "day": 1, "date": "2026-06-01",
                    "items": [
                        {"key": "flight_outbound", "name": "SQ637", "type": "flight", "time": "16:00"},
                    ],
                }
            ],
            "B": [
                {
                    "day": 1, "date": "2026-06-01",
                    "items": [
                        {"key": "flight_outbound", "name": "SQ637", "type": "flight", "time": "16:00"},
                    ],
                }
            ],
            "C": [
                {
                    "day": 1, "date": "2026-06-01",
                    "items": [
                        {"key": "flight_outbound", "name": "SQ637", "type": "flight", "time": "16:00"},
                    ],
                }
            ],
        },
        "option_meta": {
            "A": {"label": "Budget", "style": "budget", "budget": "SGD 2000"},
            "B": {"label": "Balanced", "style": "balanced", "budget": "SGD 2500"},
            "C": {"label": "Comfort", "style": "comfort", "budget": "SGD 3000"},
        },
        "flight_options_outbound": [],
        "flight_options_return": [],
        "hotel_options": [],
        "tool_log": [],
    }


# ── Populated inventory cache ─────────────────────────────────────────────────

@pytest.fixture
def populated_inventory(mock_research_result):
    """Fill _inventory_cache so revise_itinerary can run without calling planner_agent."""
    _inventory_cache.clear()
    _inventory_cache.update({
        "prompt_inventory": {},
        "flight_out_text": "SQ637 SIN->NRT $450",
        "flight_ret_text": "SQ638 NRT->SIN $430",
        "hotel_list_text": "Hotel Gracery Shinjuku $120/night",
        "att_list_text": "Senso-ji Temple (Free), Shinjuku Gyoen (SGD 3)",
        "rest_list_text": "Ichiran Ramen (SGD 20), Sushi Dai (SGD 50)",
        "ret_cutoff_note": "",
        "research": {},
        "tool_log": [],
        "compact_flights_out": mock_research_result["flight_options_outbound"],
        "compact_flights_ret": mock_research_result["flight_options_return"],
        "compact_hotels": mock_research_result["hotel_options"],
        "compact_attractions": mock_research_result["compact_attractions"],
        "compact_restaurants": mock_research_result["compact_restaurants"],
        "hotel_opts": mock_research_result["hotel_options"],
    })
    yield _inventory_cache
    _inventory_cache.clear()


# ── LLM patch fixtures ────────────────────────────────────────────────────────

# JSON response that satisfies both _llm_select_seed (selected_name / theme / reason)
# and revise_itinerary (chain_of_thought / options A-B-C with days).
_MIDDLE_DAYS_A = [
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
            {"key": "attraction_06_shibuya_sky_observatory", "name": "Shibuya Sky Observatory",
             "icon": "activity", "time": "16:00", "type": "activity"},
        ],
    },
    {
        "day": 5, "date": "2026-06-05",
        "items": [
            # duplicate attraction → tests dedup logic in _process_days
            {"key": "attraction_01_senso_ji_temple", "name": "Senso-ji Temple",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_01_ichiran_ramen", "name": "Ichiran Ramen",
             "icon": "restaurant", "time": "12:30", "type": "restaurant"},
        ],
    },
    {
        "day": 6, "date": "2026-06-06",
        "items": [
            {"key": "attraction_02_shinjuku_gyoen", "name": "Shinjuku Gyoen",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_02_sushi_dai", "name": "Sushi Dai",
             "icon": "restaurant", "time": "13:00", "type": "restaurant"},
        ],
    },
]

_MIDDLE_DAYS_B = [
    {
        "day": 2, "date": "2026-06-02",
        "items": [
            {"key": "attraction_02_shinjuku_gyoen", "name": "Shinjuku Gyoen",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_03_uobei_shibuya", "name": "Uobei Shibuya",
             "icon": "restaurant", "time": "12:30", "type": "restaurant"},
            {"key": "attraction_03_meiji_shrine", "name": "Meiji Shrine",
             "icon": "activity", "time": "15:00", "type": "activity"},
        ],
    },
    {
        "day": 3, "date": "2026-06-03",
        "items": [
            {"key": "attraction_04_teamlab_planets", "name": "teamLab Planets",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_01_ichiran_ramen", "name": "Ichiran Ramen",
             "icon": "restaurant", "time": "12:30", "type": "restaurant"},
        ],
    },
    {
        "day": 4, "date": "2026-06-04",
        "items": [
            {"key": "attraction_05_tokyo_national_museum", "name": "Tokyo National Museum",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_02_sushi_dai", "name": "Sushi Dai",
             "icon": "restaurant", "time": "13:00", "type": "restaurant"},
        ],
    },
    {
        "day": 5, "date": "2026-06-05",
        "items": [
            {"key": "attraction_06_shibuya_sky_observatory", "name": "Shibuya Sky Observatory",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_03_uobei_shibuya", "name": "Uobei Shibuya",
             "icon": "restaurant", "time": "12:30", "type": "restaurant"},
        ],
    },
    {
        "day": 6, "date": "2026-06-06",
        "items": [
            {"key": "attraction_01_senso_ji_temple", "name": "Senso-ji Temple",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_01_ichiran_ramen", "name": "Ichiran Ramen",
             "icon": "restaurant", "time": "12:30", "type": "restaurant"},
        ],
    },
]

_MIDDLE_DAYS_C = [
    {
        "day": 2, "date": "2026-06-02",
        "items": [
            {"key": "attraction_03_meiji_shrine", "name": "Meiji Shrine",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_02_sushi_dai", "name": "Sushi Dai",
             "icon": "restaurant", "time": "12:30", "type": "restaurant"},
            {"key": "attraction_05_tokyo_national_museum", "name": "Tokyo National Museum",
             "icon": "activity", "time": "15:00", "type": "activity"},
        ],
    },
    {
        "day": 3, "date": "2026-06-03",
        "items": [
            {"key": "attraction_06_shibuya_sky_observatory", "name": "Shibuya Sky Observatory",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_03_uobei_shibuya", "name": "Uobei Shibuya",
             "icon": "restaurant", "time": "13:00", "type": "restaurant"},
            {"key": "attraction_04_teamlab_planets", "name": "teamLab Planets",
             "icon": "activity", "time": "16:00", "type": "activity"},
        ],
    },
    {
        "day": 4, "date": "2026-06-04",
        "items": [
            {"key": "attraction_01_senso_ji_temple", "name": "Senso-ji Temple",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_01_ichiran_ramen", "name": "Ichiran Ramen",
             "icon": "restaurant", "time": "12:30", "type": "restaurant"},
        ],
    },
    {
        "day": 5, "date": "2026-06-05",
        "items": [
            {"key": "attraction_02_shinjuku_gyoen", "name": "Shinjuku Gyoen",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_02_sushi_dai", "name": "Sushi Dai",
             "icon": "restaurant", "time": "13:00", "type": "restaurant"},
        ],
    },
    {
        "day": 6, "date": "2026-06-06",
        "items": [
            {"key": "attraction_04_teamlab_planets", "name": "teamLab Planets",
             "icon": "activity", "time": "10:00", "type": "activity"},
            {"key": "restaurant_03_uobei_shibuya", "name": "Uobei Shibuya",
             "icon": "restaurant", "time": "12:30", "type": "restaurant"},
        ],
    },
]

_REVISE_RESPONSE = {
    "selected_name": "Senso-ji Temple",
    "theme": "cultural heritage",
    "reason": "Strong cultural focus anchors the day.",
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
                *_MIDDLE_DAYS_A,
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
                        {"key": "flight_outbound", "name": "SQ637", "icon": "flight",
                         "time": "16:00", "type": "flight"},
                        {"key": "hotel_stay", "name": "Hotel Gracery Shinjuku",
                         "icon": "hotel", "time": "20:30", "type": "hotel"},
                    ],
                },
                *_MIDDLE_DAYS_B,
                {
                    "day": 7, "date": "2026-06-07",
                    "items": [
                        {"key": "flight_return", "name": "SQ638", "icon": "flight",
                         "time": "18:00", "type": "flight"},
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
                        {"key": "flight_outbound", "name": "SQ637", "icon": "flight",
                         "time": "16:00", "type": "flight"},
                        {"key": "hotel_stay", "name": "Hotel Gracery Shinjuku",
                         "icon": "hotel", "time": "20:30", "type": "hotel"},
                    ],
                },
                *_MIDDLE_DAYS_C,
                {
                    "day": 7, "date": "2026-06-07",
                    "items": [
                        {"key": "flight_return", "name": "SQ638", "icon": "flight",
                         "time": "18:00", "type": "flight"},
                    ],
                },
            ],
        },
    },
}


@pytest.fixture
def patch_planner_llm():
    """Patch _llm in planner_agent so no real LLM calls are made."""
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MagicMock(content=json.dumps(_REVISE_RESPONSE))
    with patch("agents.specialists.planner_agent._llm", return_value=mock_instance):
        yield mock_instance


@pytest.fixture
def patch_research_agent(mock_research_result):
    """Patch research_agent in planner_agent so no real API calls are made."""
    with patch(
        "agents.specialists.planner_agent.research_agent",
        return_value=mock_research_result,
    ):
        yield
