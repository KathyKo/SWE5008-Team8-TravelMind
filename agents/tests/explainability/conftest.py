"""
Shared fixtures for Agent6 (Explainability) tests.
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


# ── State fixtures ────────────────────────────────────────────────────────────

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
            "A": [
                {
                    "day": 1, "date": "2026-06-01",
                    "items": [
                        {
                            "key": "senso_ji", "name": "Senso-ji Temple",
                            "type": "attraction", "location": "Asakusa",
                            "duration_hrs": 2, "price_sgd": "Free",
                        }
                    ],
                }
            ],
            "B": [{"day": 1, "date": "2026-06-01", "items": []}],
            "C": [{"day": 1, "date": "2026-06-01", "items": []}],
        },
        "option_meta": {
            "A": {"label": "Budget", "style": "budget", "budget": "SGD 2000"},
            "B": {"label": "Balanced", "style": "balanced", "budget": "SGD 2500"},
            "C": {"label": "Comfort", "style": "comfort", "budget": "SGD 3000"},
        },
        "planner_decision_trace": {
            "A": [
                {
                    "day": 1, "name": "Senso-ji Temple",
                    "type": "attraction", "reason": "Top cultural landmark",
                }
            ]
        },
        "chain_of_thought": "Agent3 selected Senso-ji for Day 1 due to cultural relevance.",
        "tool_log": [{"tool": "search_flights", "status": "ok"}],
        "hotel_options": [
            {"name": "Hotel Gracery Shinjuku", "location": "Shinjuku", "rating": 4.2}
        ],
    }


@pytest.fixture
def state_no_itineraries():
    return {
        "origin": "Singapore",
        "destination": "Tokyo",
        "explain_option": "A",
        "itineraries": {},
        "tool_log": [],
    }


@pytest.fixture
def state_with_option_b(base_state):
    return {**base_state, "explain_option": "B"}


# ── LLM patch fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def patch_explain_llm():
    """Patch _llm in explainability_agent so no real LLM calls are made."""
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MagicMock(
        content=json.dumps({
            "overall_summary": "A culturally rich 7-day Tokyo itinerary.",
            "day_summaries": {"Day 1": "Start with Senso-ji Temple in Asakusa."},
        })
    )
    with patch("agents.specialists.explainability_agent._llm", return_value=mock_instance):
        yield mock_instance
