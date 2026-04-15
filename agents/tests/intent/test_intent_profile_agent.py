"""
Agent1 (Intent Profile) test cases.

Run from repository root:
  python -m agents.test_case.test_intent_profile_agent
  python agents/test_case/test_intent_profile_agent.py

Pytest (CI):
  pytest agents/test_case/test_intent_profile_agent.py -v
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repository root: agents/test_case/<this_file> -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.specialists.intent_profile import intent_profile  # noqa: E402

ALLOWED_FLEXIBILITY = {"strict", "flexible", "no_limit"}
ALLOWED_TRAVEL_STYLE = {"relaxed", "balanced", "intense"}
ALLOWED_PRIORITY = {"cost_effective", "time_saving", "comfort"}
ALLOWED_QUERY_TYPES = {"api_flight", "api_hotel", "rag_attraction", "rag_local_info"}
ALLOWED_TIME_PREFS = {"midnight", "early morning", "morning", "afternoon", "evening", "night"}

DEFAULT_CASES: list[dict] = [
    {
        "user_id": "u-alice-001",
        "message": (
            "I want to visit Kyoto for 5 days from 2026-05-10 to 2026-05-15, budget SGD 5000 strict, "
            "two travelers, vegetarian, no red-eye flights, relaxed traditional cultural trip."
        ),
        "origin": "Singapore",
        "destination": "Kyoto, Japan",
        "dates": "2026-05-10 to 2026-05-15",
        "budget": "SGD 5000 strict",
        "preferences": "vegetarian, historical, zen_gardens, fine dining, relaxed, traditional",
        "duration": "5 days",
        "outbound_time_pref": "morning",
        "return_time_pref": "afternoon",
    },
    {
        "user_id": "u-bob-002",
        "message": (
            "Plan a balanced 4-day Tokyo trip for one person. Budget is flexible around USD 3000, "
            "prefer modern and food."
        ),
        "origin": "Singapore",
        "destination": "Tokyo, Japan",
        "dates": "2026-06-01 to 2026-06-04",
        "budget": "USD 3000 flexible",
        "preferences": "modern, food, balanced",
        "duration": "4 days",
        "outbound_time_pref": "early morning",
        "return_time_pref": "evening",
    },
]


def validate_output(payload: dict) -> dict:
    issues = []
    required_top = {
        "session_id",
        "outbound_time_pref",
        "return_time_pref",
        "hard_constraints",
        "soft_preferences",
        "search_queries",
    }
    missing_top = [k for k in required_top if k not in payload]
    if missing_top:
        issues.append(f"missing top-level keys: {missing_top}")
    if payload.get("outbound_time_pref") not in ALLOWED_TIME_PREFS:
        issues.append("outbound_time_pref invalid")
    if payload.get("return_time_pref") not in ALLOWED_TIME_PREFS:
        issues.append("return_time_pref invalid")

    hc = payload.get("hard_constraints", {})
    for key in ["origin", "destination", "start_date", "end_date", "budget", "travelers", "requirements"]:
        if key not in hc:
            issues.append(f"hard_constraints missing '{key}'")

    budget = hc.get("budget", {})
    if not isinstance(budget, dict):
        issues.append("hard_constraints.budget must be object")
    else:
        for key in ["amount", "currency", "flexibility"]:
            if key not in budget:
                issues.append(f"budget missing '{key}'")
        if budget.get("flexibility") not in ALLOWED_FLEXIBILITY:
            issues.append("budget.flexibility must be strict/flexible/no_limit")

    sp = payload.get("soft_preferences", {})
    for key in ["travel_style", "interest_tags", "pace", "vibe", "priority"]:
        if key not in sp:
            issues.append(f"soft_preferences missing '{key}'")
    if sp.get("travel_style") not in ALLOWED_TRAVEL_STYLE:
        issues.append("soft_preferences.travel_style must be relaxed/balanced/intense")
    if sp.get("priority") not in ALLOWED_PRIORITY:
        issues.append("soft_preferences.priority must be cost_effective/time_saving/comfort")

    sq = payload.get("search_queries", [])
    if not isinstance(sq, list):
        issues.append("search_queries must be list")
    else:
        for i, item in enumerate(sq):
            if not isinstance(item, dict):
                issues.append(f"search_queries[{i}] must be object")
                continue
            if item.get("type") not in ALLOWED_QUERY_TYPES:
                issues.append(f"search_queries[{i}].type invalid")
            if not item.get("query"):
                issues.append(f"search_queries[{i}].query empty")

    return {"is_valid": len(issues) == 0, "issues": issues}


def run_one_case(case: dict) -> dict:
    state = {
        "messages": [{"role": "user", "content": case["message"]}],
        "origin": case.get("origin"),
        "destination": case.get("destination"),
        "dates": case.get("dates"),
        "budget": case.get("budget"),
        "preferences": case.get("preferences"),
        "duration": case.get("duration"),
        "outbound_time_pref": case.get("outbound_time_pref"),
        "return_time_pref": case.get("return_time_pref"),
        "user_id": case.get("user_id", "local_test_user"),
    }
    patch = intent_profile(state)
    payload = patch.get("intent_profile_output", {})
    check = validate_output(payload)
    return {"input": case, "output": payload, "validation": check}


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / "outputs" / "intent_profile_tests" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [run_one_case(c) for c in DEFAULT_CASES]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output_dir": str(out_dir),
        "total_cases": len(results),
        "valid_cases": sum(1 for r in results if r["validation"]["is_valid"]),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for i, r in enumerate(results, start=1):
        (out_dir / f"case_{i:02d}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved detailed files to: {out_dir}")


# --- Pytest / CI --------------------------------------------------------------

def test_intent_profile_schema_all_default_cases() -> None:
    """Assert Agent1 output passes schema validation for built-in cases."""
    for case in DEFAULT_CASES:
        result = run_one_case(case)
        assert result["validation"]["is_valid"], (
            f"case {case.get('user_id')}: {result['validation']['issues']}"
        )


if __name__ == "__main__":
    main()
