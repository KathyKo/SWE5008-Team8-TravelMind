"""
Agent 4 — Debate & critique (minimal stub).

The HTTP service imports this module at startup. A full LLM-based debate
implementation can replace this later; for now we approve when itineraries
exist so the LangGraph orchestrator can reach explain / output_guard.
"""

from __future__ import annotations

from typing import Any


def _itineraries_dict(state: dict[str, Any]) -> dict[str, Any]:
    for key in ("final_itineraries", "validated_itineraries", "itineraries"):
        raw = state.get(key)
        if isinstance(raw, dict) and raw:
            return raw
    return {}


def debate_agent(state: dict[str, Any], tools: dict[str, Any] | None = None) -> dict[str, Any]:
    itins = _itineraries_dict(state)
    if not itins:
        return {
            "is_valid": False,
            "composite_score": 0.0,
            "critique": "No itinerary options to review.",
            "debate_output": {"verdict": "rejected", "reason": "empty_itineraries"},
        }

    return {
        "is_valid": True,
        "composite_score": 82.0,
        "critique": None,
        "debate_output": {
            "verdict": "approved",
            "reason": "stub_debate_agent: itineraries present",
            "options_reviewed": list(itins.keys()),
        },
    }
