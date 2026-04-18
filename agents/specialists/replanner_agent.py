"""
Agent 7 — Replanner (minimal stub).

Triggered when `user_feedback` is set. Returns a state patch; the
`replanner_node` wrapper in nodes.py clears feedback and downstream fields.
Replace with a full replan pipeline (e.g. re-call planner) when ready.
"""

from __future__ import annotations

from typing import Any


def replanner_agent(state: dict[str, Any], tools: dict[str, Any] | None = None) -> dict[str, Any]:
    feedback = (state.get("user_feedback") or "").strip()
    itins = state.get("final_itineraries") or state.get("itineraries")

    return {
        "replanner_output": {
            "status": "stub",
            "feedback_preview": feedback[:500] if feedback else "",
        },
        "itineraries": itins,
        "final_itineraries": itins,
    }
