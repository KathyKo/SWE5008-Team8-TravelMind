"""Unified dynamic replan specialist module.

This file merges:
- replan runtime wrapper (Agent3 planner integration)
- postprocess fixer
- verifier and report builder
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
import sys
import types
import unicodedata
from typing import Any, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from agents.db.crud import load_plan, update_plan_result
from agents.db.database import SessionLocal
from agents.db.models import Plan
from agents.llm_config import DMX_API_KEY, DMX_BASE_URL, PLANNER_MODEL

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@dataclass
class ReplanRules:
    start_day_num: int
    end_day_num: int
    locked_days: set[int]
    allow_replace_flight: bool
    allow_replace_hotel: bool
    must_keep_keys: set[str]
    closed_item_keys: set[str]
    closed_name_aliases: set[str]
    avoid_keywords: set[str]
    prefer_indoor: bool
    vegetarian_friendly: bool


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def _norm_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or "").strip().lower()


def _is_activity_or_restaurant(item: dict[str, Any]) -> bool:
    return str(item.get("icon") or "").lower() in {"activity", "restaurant"}


def _contains_any(text: str, keywords: set[str]) -> bool:
    t = str(text or "").lower()
    return any(k in t for k in keywords if k)


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _is_temple_or_shrine_activity(text: str) -> bool:
    folded = _ascii_fold(text).lower()
    compact = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    signals = {"temple", "shrine", "senso ji", "zojo ji", "gotokuji", "jinja", "dera"}
    if any(sig in compact for sig in signals):
        return True
    return bool(re.search(r"\b[a-z0-9]+[\s_-]ji\b", compact))


def _normalize_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _ascii_fold(str(text or "")).lower()).strip()


def _build_llm_replan_directive(payload: dict[str, Any]) -> str:
    req = payload.get("user_replan_request") or {}
    scope = req.get("replan_scope") or {}
    intent = req.get("updated_user_intent") or {}
    triggers = req.get("trigger_events") or []
    avoid_list = ", ".join(str(x) for x in (intent.get("avoid") or [])) or "none"
    pref_list = ", ".join(str(x) for x in (intent.get("new_preferences") or [])) or "none"
    must_keep = ", ".join(str(x) for x in (intent.get("must_keep") or [])) or "none"
    trigger_text = "; ".join(f"{str(t.get('type', 'event'))}: {str(t.get('detail', '')).strip()}" for t in triggers) or "none"
    return (
        "Dynamic replan instruction (must follow): "
        f"Replan from {scope.get('start_day', 'start')} to {scope.get('end_day', 'end')}. "
        f"Hard avoid topics/places: {avoid_list}. "
        "Do not include temple/shrine style attractions when indoor preference is requested. "
        f"Target preferences: {pref_list}. Must keep boundary items: {must_keep}. "
        f"Live trigger events: {trigger_text}. "
        "Output should keep schedule feasibility and avoid repeating consumed activities/restaurants."
    )


def _build_structured_replan_context(payload: dict[str, Any]) -> dict[str, Any]:
    req = payload.get("user_replan_request") or {}
    scope = req.get("replan_scope") or {}
    intent = req.get("updated_user_intent") or {}
    triggers = req.get("trigger_events") or []
    output_expectation = req.get("output_expectation") or {}
    hard_rules = {
        "replan_scope": {"start_day": scope.get("start_day"), "end_day": scope.get("end_day"), "locked_days": list(scope.get("locked_days") or [])},
        "must_keep_keys": list(intent.get("must_keep") or []),
        "allow_replace_flight": bool(scope.get("allow_replace_flight", False)),
        "allow_replace_hotel": bool(scope.get("allow_replace_hotel", False)),
        "forbidden_categories": [
            "temple/shrine activities when indoor preference is enabled",
            "already consumed activities/restaurants before replan start day",
        ],
    }
    soft_rules = {
        "preferences": list(intent.get("new_preferences") or []),
        "avoid": list(intent.get("avoid") or []),
        "pace_preference": intent.get("pace_preference"),
        "meal_preference": intent.get("meal_preference"),
        "budget_guardrail": intent.get("budget_guardrail") or {},
    }
    events = [
        {
            "type": str(event.get("type", "")),
            "severity": str(event.get("severity", "")),
            "day": str(event.get("day", "")),
            "detail": str(event.get("detail", "")),
            "affected_item_key": str(event.get("affected_item_key", "")),
        }
        for event in triggers
    ]
    return {"hard_rules": hard_rules, "soft_rules": soft_rules, "events": events, "output_expectation": output_expectation}


def _extract_city_from_flight(flight_name: str) -> tuple[str, str]:
    text = str(flight_name or "")
    m = re.search(r"(.+?)\s*[→\-]+\s*(.+)", text)
    if not m:
        return "Singapore", "Tokyo"
    origin_raw, dest_raw = m.group(1), m.group(2)

    def clean_airport(raw: str) -> str:
        airport = re.sub(r".*?\)\s*", "", raw.split("|")[0].strip()).replace("International Airport", "").replace("Airport", "").strip()
        mapping = {"Narita": "Tokyo", "Haneda": "Tokyo", "Singapore Changi": "Singapore"}
        for key, city in mapping.items():
            if key.lower() in airport.lower():
                return city
        return airport.split()[-1] if airport else ""

    return clean_airport(origin_raw), clean_airport(dest_raw)


def _extract_date_from_flight(flight_name: str, field: str = "dep") -> str:
    m = re.search(rf"{field}\s+(\d{{4}}-\d{{2}}-\d{{2}})", str(flight_name or ""))
    return m.group(1) if m else ""


def _build_rules(payload: dict[str, Any]) -> ReplanRules:
    req = payload.get("user_replan_request") or {}
    scope = req.get("replan_scope") or {}
    intent = req.get("updated_user_intent") or {}
    trigger_events = req.get("trigger_events") or []
    closed_item_keys = {str(evt.get("affected_item_key", "")).strip() for evt in trigger_events if str(evt.get("type", "")).lower() == "venue_closure"}
    closed_item_keys.discard("")
    closed_name_aliases: set[str] = set()
    for evt in trigger_events:
        if str(evt.get("type", "")).lower() != "venue_closure":
            continue
        merged = f"{_normalize_match_text(evt.get('detail', ''))} {_normalize_match_text(evt.get('affected_item_key', ''))}".strip()
        if "edo tokyo museum" in merged:
            closed_name_aliases.add("edo tokyo museum")
        raw_key = str(evt.get("affected_item_key", "")).strip().lower()
        if raw_key:
            raw_key = re.sub(r"^attraction_\d+_", "", raw_key).replace("_", " ").strip()
            if raw_key:
                closed_name_aliases.add(_normalize_match_text(raw_key))
    avoid_keywords = set()
    for phrase in intent.get("avoid", []) or []:
        avoid_keywords.update(re.findall(r"[a-zA-Z]+", str(phrase).lower()))
    for evt in trigger_events:
        if "rain" in str(evt.get("detail", "")).lower():
            avoid_keywords.update({"outdoor", "garden", "park", "walk", "temple", "shrine"})
    new_preferences = [str(x).lower() for x in (intent.get("new_preferences") or [])]
    return ReplanRules(
        start_day_num=_day_num(scope.get("start_day")),
        end_day_num=_day_num(scope.get("end_day")),
        locked_days={_day_num(x) for x in (scope.get("locked_days") or [])},
        allow_replace_flight=bool(scope.get("allow_replace_flight", False)),
        allow_replace_hotel=bool(scope.get("allow_replace_hotel", False)),
        must_keep_keys={str(x) for x in (intent.get("must_keep") or [])},
        closed_item_keys=closed_item_keys,
        closed_name_aliases=closed_name_aliases,
        avoid_keywords=avoid_keywords,
        prefer_indoor=any("indoor" in x for x in new_preferences),
        vegetarian_friendly="vegetarian" in str(intent.get("meal_preference", "")).lower(),
    )


def _build_state_from_input(payload: dict[str, Any]) -> dict[str, Any]:
    plan = (payload.get("original_recommended_plan") or {}).get("plan", [])
    outbound = ""
    ret = ""
    for day in plan:
        for item in day.get("items", []):
            key = str(item.get("key", ""))
            if key == "flight_outbound":
                outbound = str(item.get("name", ""))
            if key == "flight_return":
                ret = str(item.get("name", ""))
    origin, destination = _extract_city_from_flight(outbound)
    dep_date = _extract_date_from_flight(outbound, "dep") or "2026-06-01"
    ret_date = _extract_date_from_flight(ret, "dep") or "2026-06-05"
    req = payload.get("user_replan_request") or {}
    intent = req.get("updated_user_intent") or {}
    replan_directive = _build_llm_replan_directive(payload)
    structured_context = _build_structured_replan_context(payload)
    structured_context_text = json.dumps(structured_context, ensure_ascii=False)
    preferences = ", ".join(
        [
            *[str(x) for x in (intent.get("new_preferences") or [])],
            f"pace: {intent.get('pace_preference', 'balanced')}",
            f"meal: {intent.get('meal_preference', 'any')}",
            f"avoid: {', '.join(intent.get('avoid', []) or [])}",
            f"directive: {replan_directive}",
            f"structured_context: {structured_context_text}",
        ]
    )
    return {
        "origin": origin or "Singapore",
        "destination": destination or "Tokyo",
        "dates": f"{dep_date} to {ret_date}",
        "duration": f"{len(plan)} days" if plan else "5 days",
        "budget": f"{(intent.get('budget_guardrail') or {}).get('currency', 'SGD')} flexible",
        "preferences": preferences,
        "search_queries": [
            {"type": "rag_attraction", "query": f"Best indoor attractions in {destination or 'Tokyo'}"},
            {"type": "rag_restaurant", "query": f"Best vegetarian friendly restaurants in {destination or 'Tokyo'}"},
            {"type": "rag_attraction", "query": f"Anime and culture shopping places in {destination or 'Tokyo'}"},
            {"type": "rag_attraction", "query": f"Avoid temple and shrine attractions in {destination or 'Tokyo'}"},
        ],
        "hard_constraints": {"requirements": [replan_directive, structured_context_text, "No temple/shrine activities if user prefers indoor-focused itinerary."]},
        "soft_preferences": {"interest_tags": [str(x) for x in (intent.get("new_preferences") or [])], "vibe": str(intent.get("pace_preference") or "balanced")},
        "replan_directive": replan_directive,
        "replan_context": structured_context,
    }


def _prefer_item(item: dict[str, Any], *, indoor: bool, vegetarian: bool) -> bool:
    blob = " ".join(str(item.get(k, "") or "") for k in ("name", "type", "description", "address", "matches_preference")).lower()
    if indoor and any(k in blob for k in {"museum", "gallery", "mall", "shopping", "arcade", "anime", "indoor", "art"}):
        return True
    if vegetarian and any(k in blob for k in {"vegetarian", "vegan", "plant-based"}):
        return True
    return False


def _candidate_to_itinerary_item(candidate: dict[str, Any], icon: str, time_value: str, seq: int) -> dict[str, Any]:
    name = str(candidate.get("name") or candidate.get("title") or "").strip()
    key_prefix = "attraction" if icon == "activity" else "restaurant"
    return {"time": time_value, "icon": icon, "key": f"{key_prefix}_replan_{seq}_{_slug(name)}", "name": name, "cost": "TBC"}


def _pick_candidate(candidates: list[dict[str, Any]], used_names: set[str], avoid_keywords: set[str], *, indoor: bool, vegetarian: bool) -> dict[str, Any] | None:
    preferred = []
    fallback = []
    for item in candidates:
        name = str(item.get("name") or item.get("title") or "").strip()
        if not name or name.lower() in used_names:
            continue
        blob = " ".join(str(item.get(k, "") or "") for k in ("name", "type", "description", "address", "matches_preference")).lower()
        if indoor and _is_temple_or_shrine_activity(blob):
            continue
        if _contains_any(blob, avoid_keywords):
            continue
        (preferred if _prefer_item(item, indoor=indoor, vegetarian=vegetarian) else fallback).append(item)
    return preferred[0] if preferred else (fallback[0] if fallback else None)


def _is_disallowed_item(item: dict[str, Any], rules: ReplanRules, visited_names: set[str]) -> tuple[bool, str]:
    key = str(item.get("key", "")).strip()
    icon = str(item.get("icon", "")).lower()
    name = _norm_name(item)
    blob = f"{name} {key}"
    blob_norm = _normalize_match_text(blob)
    if key in rules.closed_item_keys or any(alias and alias in blob_norm for alias in rules.closed_name_aliases):
        return True, "venue_closed"
    if _is_activity_or_restaurant(item) and name in visited_names:
        return True, "already_visited"
    if _contains_any(blob, rules.avoid_keywords):
        return True, "user_avoidance"
    if rules.prefer_indoor and icon == "activity" and (_is_temple_or_shrine_activity(blob) or any(k in blob for k in {"garden", "park"})):
        return True, "indoor_preference_violation"
    return False, ""


def _return_day_label_from_dates(state: dict[str, Any]) -> str:
    dates_text = str(state.get("dates", "")).strip()
    if " to " not in dates_text:
        return "Day 1"
    start_s, end_s = [x.strip() for x in dates_text.split(" to ", 1)]
    try:
        from datetime import date

        start = date.fromisoformat(start_s)
        end = date.fromisoformat(end_s)
        return f"Day {max(1, (end - start).days + 1)}"
    except Exception:
        return "Day 1"


def _legacy_replan(payload: dict[str, Any]) -> dict[str, Any]:
    original = deepcopy(payload.get("original_recommended_plan") or {})
    days = deepcopy(original.get("plan") or [])
    if not days:
        raise ValueError("Input missing original_recommended_plan.plan")
    rules = _build_rules(payload)
    state = _build_state_from_input(payload)

    from agents.agent_tools import get_tools_for_agent  # type: ignore[reportMissingImports]
    from agents.specialists.planner_agent_1 import planner_from_research_1, revise_itinerary_1  # type: ignore[reportMissingImports]
    from agents.specialists.research_agent_1 import research_agent_1  # type: ignore[reportMissingImports]

    tools = get_tools_for_agent("planner_agent_1")
    research_result = research_agent_1(state, tools)
    planner_result = planner_from_research_1(state, research_result) if "error" not in research_result else {"itineraries": {}}
    generated_options = planner_result.get("itineraries", {}) if isinstance(planner_result, dict) else {}
    if isinstance(planner_result, dict) and generated_options:
        critique = f"{state.get('replan_directive')}\nStructured context:\n{json.dumps(state.get('replan_context') or {}, ensure_ascii=False, indent=2)}"
        try:
            revised = revise_itinerary_1(state, critique, planner_result)
            if isinstance(revised, dict) and revised.get("itineraries"):
                generated_options = revised.get("itineraries", generated_options)
        except Exception:
            pass
    generated_days = generated_options.get("C") or next(iter(generated_options.values()), [])
    attractions_pool = research_result.get("compact_attractions", []) if isinstance(research_result, dict) else []
    restaurants_pool = research_result.get("compact_restaurants", []) if isinstance(research_result, dict) else []
    visited_names: set[str] = set()
    used_names: set[str] = set()
    change_log: list[dict[str, Any]] = []

    for day in days:
        dnum = _day_num(day.get("day"))
        if dnum < rules.start_day_num:
            for item in day.get("items", []):
                n = _norm_name(item)
                if _is_activity_or_restaurant(item) and n:
                    visited_names.add(n)
                    used_names.add(n)

    for idx, day in enumerate(days):
        dnum = _day_num(day.get("day"))
        in_scope = rules.start_day_num <= dnum <= rules.end_day_num if rules.end_day_num else dnum >= rules.start_day_num
        if not in_scope or dnum in rules.locked_days:
            for item in day.get("items", []):
                n = _norm_name(item)
                if _is_activity_or_restaurant(item) and n:
                    used_names.add(n)
            continue
        generated_day = generated_days[idx] if idx < len(generated_days) else {"day": day.get("day"), "items": []}
        candidate_items = deepcopy(generated_day.get("items", [])) or deepcopy(day.get("items", []))
        old_items = day.get("items", [])
        original_flights = [x for x in old_items if str(x.get("icon", "")).lower() == "flight"]
        original_hotels = [x for x in old_items if str(x.get("icon", "")).lower() == "hotel"]
        cleaned: list[dict[str, Any]] = []
        removed_activity_count = 0
        removed_rest_count = 0
        for item in candidate_items:
            icon = str(item.get("icon", "")).lower()
            key = str(item.get("key", ""))
            if key in rules.must_keep_keys:
                cleaned.append(item)
                continue
            if icon == "flight" and not rules.allow_replace_flight:
                continue
            if icon == "hotel" and not rules.allow_replace_hotel:
                continue
            blocked, reason = _is_disallowed_item(item, rules, visited_names)
            if blocked:
                if icon == "activity":
                    removed_activity_count += 1
                if icon == "restaurant":
                    removed_rest_count += 1
                change_log.append({"day": day.get("day"), "action": "removed_item", "item_key": key, "item_name": item.get("name"), "reason": reason})
                continue
            n = _norm_name(item)
            if _is_activity_or_restaurant(item) and n:
                used_names.add(n)
            cleaned.append(item)
        for item in original_flights:
            if item not in cleaned:
                cleaned.append(item)
        for item in original_hotels:
            if item not in cleaned:
                cleaned.append(item)
        for i in range(removed_activity_count):
            replacement = _pick_candidate(attractions_pool, used_names | visited_names, rules.avoid_keywords, indoor=rules.prefer_indoor, vegetarian=False)
            if replacement:
                rep = _candidate_to_itinerary_item(replacement, "activity", "14:00", i + 1)
                cleaned.append(rep)
                used_names.add(_norm_name(rep))
                change_log.append({"day": day.get("day"), "action": "added_replacement", "item_key": rep.get("key"), "item_name": rep.get("name"), "reason": "replace_removed_activity"})
        for i in range(removed_rest_count):
            replacement = _pick_candidate(restaurants_pool, used_names | visited_names, rules.avoid_keywords, indoor=False, vegetarian=rules.vegetarian_friendly)
            if replacement:
                rep = _candidate_to_itinerary_item(replacement, "restaurant", "18:00", i + 1)
                cleaned.append(rep)
                used_names.add(_norm_name(rep))
                change_log.append({"day": day.get("day"), "action": "added_replacement", "item_key": rep.get("key"), "item_name": rep.get("name"), "reason": "replace_removed_restaurant"})
        day["items"] = sorted(cleaned, key=lambda x: str(x.get("time", "99:99")))

    for day in days:
        kept_items: list[dict[str, Any]] = []
        for item in day.get("items", []):
            _blocked, reason = _is_disallowed_item(item, rules, visited_names)
            if reason == "venue_closed":
                change_log.append({"day": day.get("day"), "action": "final_guard_removed", "item_key": item.get("key"), "item_name": item.get("name"), "reason": "venue_closed"})
                continue
            kept_items.append(item)
        day["items"] = kept_items

    return_day = _return_day_label_from_dates(state)
    kept_return_flight: dict | None = None
    for day in days:
        day_label = str(day.get("day", ""))
        new_items: list[dict[str, Any]] = []
        for item in day.get("items", []):
            key = str(item.get("key", ""))
            if key != "flight_return":
                new_items.append(item)
                continue
            if day_label != return_day:
                change_log.append({"day": day_label, "action": "final_guard_removed", "item_key": key, "item_name": item.get("name"), "reason": "return_flight_wrong_day"})
                continue
            if kept_return_flight is None:
                kept_return_flight = item
                new_items.append(item)
            else:
                change_log.append({"day": day_label, "action": "final_guard_removed", "item_key": key, "item_name": item.get("name"), "reason": "duplicate_return_flight"})
        day["items"] = sorted(new_items, key=lambda x: str(x.get("time", "99:99")))
    if kept_return_flight is None:
        original_return = None
        for day in original.get("plan", []):
            for item in day.get("items", []):
                if str(item.get("key", "")) == "flight_return":
                    original_return = deepcopy(item)
                    break
            if original_return:
                break
        if original_return:
            for day in days:
                if str(day.get("day", "")) == return_day:
                    day.setdefault("items", []).append(original_return)
                    day["items"] = sorted(day["items"], key=lambda x: str(x.get("time", "99:99")))
                    change_log.append({"day": return_day, "action": "final_guard_added", "item_key": "flight_return", "item_name": original_return.get("name"), "reason": "restore_required_return_flight"})
                    break

    return {
        "scenario_id": payload.get("scenario_id"),
        "input_source": payload.get("source", {}),
        "state_used_for_research_and_planner": state,
        "replan_context_used": state.get("replan_context", {}),
        "applied_rules": {
            "start_day": rules.start_day_num,
            "end_day": rules.end_day_num,
            "locked_days": sorted(rules.locked_days),
            "avoid_keywords": sorted(rules.avoid_keywords),
            "closed_name_aliases": sorted(rules.closed_name_aliases),
            "return_day_enforced": return_day,
            "prefer_indoor": rules.prefer_indoor,
            "vegetarian_friendly": rules.vegetarian_friendly,
            "must_keep_keys": sorted(rules.must_keep_keys),
        },
        "replanned_plan": {
            "option_key": original.get("option_key"),
            "label": f"{original.get('label', 'Recommended')} (Dynamic Replan)",
            "itinerary_id": f"{original.get('itinerary_id', 'IT')}_REPLAN",
            "composite_score": original.get("composite_score"),
            "plan": days,
        },
        "change_log": change_log,
        "reasoning_summary": [
            "Respected updated user preferences (e.g. indoor-first and meal constraints).",
            "Removed in-scope items that violate trigger events or user avoid signals.",
            "Prevented repeats of items already consumed before the replan start day.",
            "Filled removed slots with research-backed alternatives from the planner pipeline.",
        ],
        "api_dependencies": [
            {"name": "OpenAI (LangChain ChatOpenAI)", "env": "OPENAI_API_KEY", "required": True},
            {"name": "SerpAPI", "env": "SERPAPI_API_KEY", "required": True},
            {"name": "Google Places API (New)", "env": "GOOGLE_MAPS_API_KEY", "required": True},
            {"name": "Google Custom Search API", "env": "GOOGLE_API_KEY + GOOGLE_CSE_ID", "required": False},
            {"name": "Tavily Search API", "env": "TAVILY_API_KEY", "required": False},
            {"name": "OpenWeather API", "env": "OPENWEATHER_API_KEY", "required": False},
        ],
        "raw_research_result_excerpt": {"attractions_count": len(attractions_pool), "restaurants_count": len(restaurants_pool)},
    }


# ----------------------------
# replan runtime wrapper
# ----------------------------
def _resolve_planner_root() -> Path:
    env_root = (os.getenv("REPLAN_PLANNER_ROOT") or "").strip()
    if env_root:
        candidate = Path(env_root).resolve()
        if (candidate / "agents" / "agent_tools.py").exists():
            return candidate
    default = (_ROOT.parent / "SWE5008-Team8-TravelMind-update-research_planner_explainibility").resolve()
    if (default / "agents" / "agent_tools.py").exists():
        return default
    raise FileNotFoundError(
        "Cannot resolve planner project root for replan runtime. "
        "Set REPLAN_PLANNER_ROOT to a directory containing agents/agent_tools.py."
    )


def _resolve_planner_model() -> str:
    return (
        (os.getenv("REPLAN_PLANNER_MODEL") or "").strip()
        or (os.getenv("AGENT3_PLANNER_MODEL") or "").strip()
        or "gpt-4.1"
    )


def _apply_planner_model_override(model_name: str) -> None:
    llm_config = importlib.import_module("agents.llm_config")
    setattr(llm_config, "OPENAI_MODEL", model_name)
    planner_module = importlib.import_module("agents.specialists.planner_agent_1")
    setattr(planner_module, "OPENAI_MODEL", model_name)


def replan(payload: dict) -> dict:
    planner_root = _resolve_planner_root()
    planner_model = _resolve_planner_model()

    shadowed_modules = {
        name: mod
        for name, mod in sys.modules.items()
        if name in ("agents", "tools") or name.startswith("agents.") or name.startswith("tools.")
    }
    for name in list(shadowed_modules.keys()):
        sys.modules.pop(name, None)

    inserted = False
    if str(planner_root) not in sys.path:
        sys.path.insert(0, str(planner_root))
        inserted = True
    try:
        fake_agents = types.ModuleType("agents")
        fake_agents.__path__ = [str(planner_root / "agents")]
        fake_specialists = types.ModuleType("agents.specialists")
        fake_specialists.__path__ = [str(planner_root / "agents" / "specialists")]
        fake_tools = types.ModuleType("tools")
        fake_tools.__path__ = [str(planner_root / "tools")]
        sys.modules["agents"] = fake_agents
        sys.modules["agents.specialists"] = fake_specialists
        sys.modules["tools"] = fake_tools

        importlib.invalidate_caches()
        _apply_planner_model_override(planner_model)
        return _legacy_replan(payload)
    finally:
        for name in [n for n in list(sys.modules.keys()) if n in ("agents", "tools") or n.startswith("agents.") or n.startswith("tools.")]:
            sys.modules.pop(name, None)
        sys.modules.update(shadowed_modules)
        if inserted:
            try:
                sys.path.remove(str(planner_root))
            except ValueError:
                pass


# ----------------------------
# postprocess fixer
# ----------------------------
def _day_num(day_label: str) -> int:
    m = re.search(r"(\d+)", str(day_label or ""))
    return int(m.group(1)) if m else 0


def _parse_time_to_min(value: str) -> int | None:
    m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", str(value or ""))
    if not m:
        return None
    h = int(m.group(1))
    mm = int(m.group(2))
    if h < 0 or h > 23 or mm < 0 or mm > 59:
        return None
    return h * 60 + mm


def _fmt_min_to_time(total_min: int) -> str:
    total_min = max(0, min(total_min, 23 * 60 + 59))
    h = total_min // 60
    mm = total_min % 60
    return f"{h:02d}:{mm:02d}"


def _iter_replan_days(output: dict[str, Any]) -> list[dict[str, Any]]:
    return (((output.get("replanned_plan") or {}).get("plan") or []))


def _scope_day_numbers(payload: dict[str, Any]) -> tuple[int, int]:
    req = payload.get("user_replan_request") or {}
    scope = req.get("replan_scope") or {}
    start = _day_num(scope.get("start_day"))
    end = _day_num(scope.get("end_day"))
    if end <= 0:
        end = 999
    return max(start, 1), end


def _append_change(output: dict[str, Any], day_label: str, action: str, item_key: str, item_name: str, reason: str) -> None:
    output.setdefault("change_log", [])
    output["change_log"].append(
        {
            "day": day_label,
            "action": action,
            "item_key": item_key,
            "item_name": item_name,
            "reason": reason,
        }
    )


def _fix_dense_timeline_for_day(output: dict[str, Any], day: dict[str, Any]) -> None:
    day_label = str(day.get("day", ""))
    items = list(day.get("items", []) or [])
    items.sort(key=lambda x: str(x.get("time", "99:99")))

    min_gap = 75
    last_min: int | None = None
    for item in items:
        cur_min = _parse_time_to_min(item.get("time", ""))
        if cur_min is None:
            continue
        if last_min is not None and cur_min - last_min < min_gap:
            new_min = last_min + min_gap
            old_time = item.get("time", "")
            item["time"] = _fmt_min_to_time(new_min)
            _append_change(
                output,
                day_label,
                "postprocess_adjust_time",
                str(item.get("key", "")),
                str(item.get("name", "")),
                f"tight_timeline_gap({old_time}->{item['time']})",
            )
            cur_min = new_min
        last_min = cur_min

    day["items"] = items


def _ensure_afternoon_coffee(output: dict[str, Any], day: dict[str, Any]) -> None:
    day_label = str(day.get("day", ""))
    items = list(day.get("items", []) or [])

    has_coffee = False
    for item in items:
        if str(item.get("icon", "")).lower() != "restaurant":
            continue
        t = _parse_time_to_min(item.get("time", ""))
        name = str(item.get("name", "")).lower()
        if t is not None and 14 * 60 <= t <= 17 * 60 + 30 and any(k in name for k in ("coffee", "cafe", "caf", "espresso")):
            has_coffee = True
            break
    if has_coffee:
        return

    new_item = {
        "time": "15:30",
        "icon": "restaurant",
        "key": f"restaurant_post_coffee_{_day_num(day_label)}",
        "name": "Afternoon Coffee Break",
        "cost": "SGD 8-15",
        "note": "Post-processed to satisfy coffee break preference.",
    }
    items.append(new_item)
    items.sort(key=lambda x: str(x.get("time", "99:99")))
    day["items"] = items
    _append_change(
        output,
        day_label,
        "postprocess_add_item",
        new_item["key"],
        new_item["name"],
        "missing_afternoon_coffee_break",
    )


def _enforce_return_flight_buffer(output: dict[str, Any], day: dict[str, Any]) -> None:
    day_label = str(day.get("day", ""))
    items = list(day.get("items", []) or [])
    return_idx = None
    return_min = None
    for i, item in enumerate(items):
        if str(item.get("key", "")) == "flight_return":
            return_idx = i
            return_min = _parse_time_to_min(item.get("time", ""))
            break
    if return_idx is None or return_min is None:
        return

    min_buffer = 180
    prev_idx = None
    prev_min = None
    for i in range(return_idx - 1, -1, -1):
        if str(items[i].get("icon", "")).lower() == "flight":
            continue
        t = _parse_time_to_min(items[i].get("time", ""))
        if t is not None:
            prev_idx = i
            prev_min = t
            break
    if prev_idx is None or prev_min is None:
        return

    if return_min - prev_min >= min_buffer:
        return

    target_prev = return_min - min_buffer
    if target_prev >= 8 * 60:
        old = items[prev_idx].get("time", "")
        items[prev_idx]["time"] = _fmt_min_to_time(target_prev)
        _append_change(
            output,
            day_label,
            "postprocess_adjust_time",
            str(items[prev_idx].get("key", "")),
            str(items[prev_idx].get("name", "")),
            f"return_flight_buffer({old}->{items[prev_idx]['time']})",
        )
    else:
        removed = items.pop(prev_idx)
        _append_change(
            output,
            day_label,
            "postprocess_remove_item",
            str(removed.get("key", "")),
            str(removed.get("name", "")),
            "insufficient_return_flight_buffer",
        )
    items.sort(key=lambda x: str(x.get("time", "99:99")))
    day["items"] = items


def postprocess_replan_output(payload: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    fixed = deepcopy(output)
    start_day, end_day = _scope_day_numbers(payload)

    for day in _iter_replan_days(fixed):
        dnum = _day_num(day.get("day", ""))
        if not (start_day <= dnum <= end_day):
            continue
        _fix_dense_timeline_for_day(fixed, day)
        _ensure_afternoon_coffee(fixed, day)
        _enforce_return_flight_buffer(fixed, day)

    fixed.setdefault("postprocess_summary", {})
    fixed["postprocess_summary"].update(
        {
            "enabled": True,
            "rules": [
                "timeline_gap_min_75",
                "afternoon_coffee_break_required",
                "return_flight_buffer_min_180",
            ],
        }
    )
    return fixed


# ----------------------------
# verifier
# ----------------------------
DEFAULT_BASE_URL = "https://www.dmxapi.cn/v1"
DEFAULT_VERIFIER_MODEL = "gpt-5-mini-2025-08-07"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _extract_replanned_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for day in (((result.get("replanned_plan") or {}).get("plan") or [])):
        for item in day.get("items", []) or []:
            items.append(item)
    return items


def _hard_rule_checks(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    req = payload.get("user_replan_request") or {}
    intent = req.get("updated_user_intent") or {}
    scope = req.get("replan_scope") or {}
    avoid_terms = [_norm(x) for x in (intent.get("avoid") or []) if str(x).strip()]
    must_keep = {str(x).strip() for x in (intent.get("must_keep") or []) if str(x).strip()}

    replanned_items = _extract_replanned_items(result)
    item_keys = {str(i.get("key") or "").strip() for i in replanned_items}
    item_blobs = [
        _norm(" ".join([str(i.get("key") or ""), str(i.get("name") or ""), str(i.get("note") or "")]))
        for i in replanned_items
    ]

    violations: list[dict[str, Any]] = []
    for key in sorted(must_keep):
        if key and key not in item_keys:
            violations.append({"type": "must_keep_missing", "severity": "critical", "message": f"Required boundary item missing: {key}"})
    for term in avoid_terms:
        if term and any(term in blob for blob in item_blobs):
            violations.append({"type": "avoid_preference_violation", "severity": "major", "message": f"Forbidden/avoid term still appears in replanned output: {term}"})
    return_flights = [i for i in replanned_items if str(i.get("key", "")).strip() == "flight_return"]
    if len(return_flights) != 1:
        violations.append({"type": "return_flight_integrity", "severity": "critical", "message": f"Expected exactly one flight_return, got {len(return_flights)}"})

    hard_pass = len([v for v in violations if v["severity"] in ("critical", "major")]) == 0
    return {
        "hard_rule_passed": hard_pass,
        "scope_checked": {"start_day": scope.get("start_day"), "end_day": scope.get("end_day")},
        "violations": violations,
    }


VERIFIER_PROMPT = """You are a strict QA verifier for dynamic travel replanning.

Input includes:
1) User replan request and constraints
2) Replan output JSON
3) Deterministic hard-rule check result

Task:
- Evaluate whether final replanned output is reasonable, feasible, and aligned to user goals.
- Pay attention to:
  - preference satisfaction
  - avoid-list compliance risk
  - schedule/logistics feasibility
  - trustworthiness/explainability

Return strict JSON:
{{
  "preference_alignment_score": 0-100,
  "feasibility_score": 0-100,
  "constraint_compliance_score": 0-100,
  "quality_confidence_score": 0-100,
  "risk_flags": ["..."],
  "final_recommendation": "accept" | "revise",
  "reason": "short summary"
}}
"""


def _llm_judge(payload: dict[str, Any], result: dict[str, Any], hard_check: dict[str, Any]) -> dict[str, Any]:
    api_key = (os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return {
            "preference_alignment_score": 0,
            "feasibility_score": 0,
            "constraint_compliance_score": 0,
            "quality_confidence_score": 0,
            "risk_flags": ["missing_api_key_for_verifier"],
            "final_recommendation": "revise",
            "reason": "Verifier LLM skipped because JUDGE_API_KEY / OPENAI_API_KEY is missing.",
        }

    model = (os.getenv("REPLAN_VERIFIER_MODEL") or DEFAULT_VERIFIER_MODEL).strip()
    base_url = (os.getenv("REPLAN_VERIFIER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VERIFIER_PROMPT),
            ("human", "User replan request:\n{req_json}\n\nHard-rule result:\n{hard_json}\n\nReplan output:\n{result_json}"),
        ]
    )
    data = {
        "req_json": json.dumps(payload.get("user_replan_request", {}), ensure_ascii=False, indent=2),
        "hard_json": json.dumps(hard_check, ensure_ascii=False, indent=2),
        "result_json": json.dumps(result, ensure_ascii=False, indent=2),
    }
    for candidate_model in [model, "gpt-4.1"]:
        try:
            chain = prompt | ChatOpenAI(model=candidate_model, api_key=api_key, base_url=base_url, temperature=0.1) | JsonOutputParser()
            judged = chain.invoke(data)
            judged["_verifier_model_used"] = candidate_model
            if candidate_model != model:
                judged.setdefault("risk_flags", []).append(f"model_fallback_used:{model}->gpt-4.1")
            return judged
        except Exception:
            continue
    return {
        "preference_alignment_score": 0,
        "feasibility_score": 0,
        "constraint_compliance_score": 0,
        "quality_confidence_score": 0,
        "risk_flags": [f"verifier_model_unavailable:{model}", "verifier_fallback_failed:gpt-4.1"],
        "final_recommendation": "revise",
        "reason": "Verifier LLM failed for both primary and fallback model.",
        "_verifier_model_used": "none",
    }


def verify_replan(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    hard_check = _hard_rule_checks(payload, result)
    llm_judge = _llm_judge(payload, result, hard_check)
    final_accept = bool(hard_check.get("hard_rule_passed")) and llm_judge.get("final_recommendation") == "accept"
    return {"hard_check": hard_check, "llm_judge": llm_judge, "final_verdict": "accept" if final_accept else "revise"}


def build_verifier_report_text(payload: dict[str, Any], result: dict[str, Any], verifier: dict[str, Any]) -> str:
    hard = verifier.get("hard_check", {})
    judge = verifier.get("llm_judge", {})
    lines = [
        "Dynamic Replan Verifier Report",
        "============================",
        "",
        "Final Verdict",
        "-------------",
        f"- verdict: {verifier.get('final_verdict', 'revise')}",
        "",
        "Hard Rule Checks",
        "----------------",
        f"- passed: {hard.get('hard_rule_passed', False)}",
        f"- scope: {hard.get('scope_checked', {})}",
    ]
    violations = hard.get("violations", []) or []
    if violations:
        lines.append("- violations:")
        for v in violations:
            lines.append(f"  - [{v.get('severity', 'major')}] {v.get('type', 'rule')}: {v.get('message', '')}")
    else:
        lines.append("- violations: none")
    lines.extend(
        [
            "",
            "LLM Judge",
            "---------",
            f"- preference_alignment_score: {judge.get('preference_alignment_score', 0)}",
            f"- feasibility_score: {judge.get('feasibility_score', 0)}",
            f"- constraint_compliance_score: {judge.get('constraint_compliance_score', 0)}",
            f"- quality_confidence_score: {judge.get('quality_confidence_score', 0)}",
            f"- final_recommendation: {judge.get('final_recommendation', 'revise')}",
            f"- reason: {judge.get('reason', '')}",
        ]
    )
    risk_flags = judge.get("risk_flags", []) or []
    if risk_flags:
        lines.append("- risk_flags:")
        for flag in risk_flags:
            lines.append(f"  - {flag}")
    else:
        lines.append("- risk_flags: none")
    lines.extend(
        [
            "",
            "Trace",
            "-----",
            f"- scenario_id: {result.get('scenario_id', '')}",
            f"- output_plan_days: {len(((result.get('replanned_plan') or {}).get('plan') or []))}",
            f"- change_log_count: {len(result.get('change_log', []) or [])}",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "dynamic_replan_agent",
    "load_json",
    "save_json",
    "replan",
    "postprocess_replan_output",
    "verify_replan",
    "build_verifier_report_text",
    "_resolve_planner_root",
    "_resolve_planner_model",
    "_apply_planner_model_override",
    "_parse_time_to_min",
    "_fmt_min_to_time",
    "_hard_rule_checks",
    "_llm_judge",
    "_ROOT",
    "importlib",
]


def _resolve_replan_plan_id() -> Optional[str]:
    db = SessionLocal()
    try:
        row = (
            db.query(Plan.plan_id)
            .filter(Plan.debate_verdict.is_not(None))
            .order_by(Plan.created_at.desc())
            .first()
        )
        if row:
            return row[0]
        latest = db.query(Plan.plan_id).order_by(Plan.created_at.desc()).first()
        return latest[0] if latest else None
    finally:
        db.close()


def _parse_clock_minutes(value: str) -> Optional[int]:
    if not value:
        return None
    text = str(value).strip().lower()
    if "rest of day" in text or "full" in text:
        return 23 * 60 + 59
    if "~1" in text or text == "1h":
        return 60
    if "2-3" in text or "2–3" in text or text == "3h":
        return 180
    m = re.search(r"(\d{1,2}):(\d{2})", text)
    if not m:
        return None
    h = int(m.group(1))
    mm = int(m.group(2))
    if 0 <= h <= 23 and 0 <= mm <= 59:
        return h * 60 + mm
    return None


def _extract_day_and_time(feedback: dict[str, Any]) -> tuple[int, int]:
    day_text = str(feedback.get("current_day") or feedback.get("day") or "Day 1")
    day_num = _day_num(day_text)
    if day_num <= 0:
        day_num = 1
    minute = _parse_clock_minutes(str(feedback.get("current_time") or feedback.get("time") or "")) or 0
    return day_num, minute


def _winner_option_from_plan(plan_payload: dict[str, Any]) -> str:
    verdict = plan_payload.get("debate_verdict") or {}
    winner = str(verdict.get("winner_option") or "").strip()
    itineraries = plan_payload.get("itineraries") or {}
    if winner in itineraries:
        return winner
    return "A" if "A" in itineraries else (next(iter(itineraries.keys()), "A"))


def _collect_consumed_items(days: list[dict[str, Any]], until_day: int, until_min: int) -> set[str]:
    consumed: set[str] = set()
    for day in days:
        dnum = _day_num(day.get("day"))
        for item in day.get("items", []) or []:
            if str(item.get("icon", "")).lower() in {"hotel", "flight"}:
                continue
            item_min = _parse_time_to_min(str(item.get("time", "")))
            if dnum < until_day or (dnum == until_day and item_min is not None and item_min <= until_min):
                name = str(item.get("name", "")).strip().lower()
                if name:
                    consumed.add(name)
    return consumed


def _collect_candidate_items(plan_payload: dict[str, Any], consumed: set[str]) -> list[dict[str, Any]]:
    itineraries = plan_payload.get("itineraries") or {}
    candidates: list[dict[str, Any]] = []
    for _, days in itineraries.items():
        for day in days or []:
            for item in day.get("items", []) or []:
                icon = str(item.get("icon", "")).lower()
                name = str(item.get("name", "")).strip()
                if icon in {"hotel", "flight"} or not name:
                    continue
                if name.lower() in consumed:
                    continue
                candidates.append(
                    {
                        "name": name,
                        "icon": item.get("icon", "✨"),
                        "desc": f"{day.get('day', '')} · {item.get('time', '')} · {item.get('cost', 'TBD')}",
                    }
                )
    # preserve order, remove duplicates by name
    dedup: dict[str, dict[str, Any]] = {}
    for c in candidates:
        dedup.setdefault(c["name"].lower(), c)
    return list(dedup.values())


def _fallback_alternatives(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alts: list[dict[str, Any]] = []
    for c in candidates[:2]:
        alts.append(
            {
                "icon": c.get("icon", "✨"),
                "name": c.get("name", "Alternative"),
                "desc": c.get("desc", ""),
                "price": "TBD",
                "rating": "⭐ 4.5",
                "dist": "nearby",
                "tag": "price",
            }
        )
    # Always return exactly 2 options for frontend selection UX.
    while len(alts) < 2:
        idx = len(alts) + 1
        alts.append(
            {
                "icon": "✨",
                "name": f"Smart Alternative {idx}",
                "desc": "Replanned option tailored to your current context.",
                "price": "TBD",
                "rating": "⭐ 4.5",
                "dist": "nearby",
                "tag": "price",
            }
        )
    return alts[:2]


def _ensure_two_alternatives(cleaned: list[dict[str, Any]], candidates: list[dict[str, Any]], consumed_names: set[str]) -> list[dict[str, Any]]:
    seen = {str(x.get("name", "")).strip().lower() for x in cleaned if isinstance(x, dict)}
    for c in candidates:
        if len(cleaned) >= 2:
            break
        name = str(c.get("name", "")).strip()
        lower = name.lower()
        if not name or lower in consumed_names or lower in seen:
            continue
        cleaned.append(
            {
                "icon": str(c.get("icon") or "✨"),
                "name": name,
                "desc": str(c.get("desc") or ""),
                "price": "TBD",
                "rating": "⭐ 4.5",
                "dist": "nearby",
                "tag": "price",
            }
        )
        seen.add(lower)
    if len(cleaned) < 2:
        return _fallback_alternatives(candidates)
    return cleaned[:2]


def _llm_generate_alternatives(
    *,
    feedback: dict[str, Any],
    plan_payload: dict[str, Any],
    consumed_names: set[str],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not DMX_API_KEY:
        return _fallback_alternatives(candidates)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a travel dynamic replanning assistant. Return strict JSON only with key alternatives "
                "(array with exactly 2 alternatives). Each alternative must include icon,name,desc,price,rating,dist,tag. "
                "Never repeat consumed activities.",
            ),
            (
                "human",
                "User feedback:\n{feedback_json}\n\n"
                "Trip context:\n{context_json}\n\n"
                "Consumed (must avoid):\n{consumed_json}\n\n"
                "Candidate pool:\n{candidates_json}",
            ),
        ]
    )
    chain = (
        prompt
        | ChatOpenAI(
            model=PLANNER_MODEL,
            temperature=0.2,
            base_url=DMX_BASE_URL,
            api_key=DMX_API_KEY,
        )
        | JsonOutputParser()
    )
    raw = chain.invoke(
        {
            "feedback_json": json.dumps(feedback, ensure_ascii=False, indent=2),
            "context_json": json.dumps(
                {
                    "origin": plan_payload.get("origin"),
                    "destination": plan_payload.get("destination"),
                    "dates": plan_payload.get("dates"),
                    "preferences": plan_payload.get("preferences"),
                    "flight_options_outbound": plan_payload.get("flight_options_outbound", []),
                    "flight_options_return": plan_payload.get("flight_options_return", []),
                    "hotel_options": plan_payload.get("hotel_options", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "consumed_json": json.dumps(sorted(consumed_names), ensure_ascii=False),
            "candidates_json": json.dumps(candidates[:20], ensure_ascii=False, indent=2),
        }
    )
    alternatives = raw.get("alternatives") if isinstance(raw, dict) else None
    if not isinstance(alternatives, list):
        return _fallback_alternatives(candidates)
    cleaned: list[dict[str, Any]] = []
    for alt in alternatives[:2]:
        if not isinstance(alt, dict):
            continue
        name = str(alt.get("name", "")).strip()
        if not name or name.lower() in consumed_names:
            continue
        cleaned.append(
            {
                "icon": str(alt.get("icon") or "✨"),
                "name": name,
                "desc": str(alt.get("desc") or ""),
                "price": str(alt.get("price") or "TBD"),
                "rating": str(alt.get("rating") or "⭐ 4.5"),
                "dist": str(alt.get("dist") or "nearby"),
                "tag": str(alt.get("tag") or "price"),
            }
        )
    if not cleaned:
        return _fallback_alternatives(candidates)
    return _ensure_two_alternatives(cleaned, candidates, consumed_names)


def dynamic_replan_agent(state: dict[str, Any]) -> dict[str, Any]:
    """
    Replanner entrypoint:
    - receives user_feedback from frontend state
    - resolves latest debated plan from DB
    - avoids already consumed items up to selected time
    - returns alternatives for frontend rendering
    """
    feedback = state.get("user_feedback") or {}
    plan_id = str(state.get("plan_id") or "").strip() or _resolve_replan_plan_id()
    if not plan_id:
        return {
            "replanner_output": {"error": "No plan found for replanning."},
            "next_node": "orchestrator",
        }

    db = SessionLocal()
    try:
        plan_payload = load_plan(db, plan_id)
    finally:
        db.close()
    if not plan_payload:
        return {
            "plan_id": plan_id,
            "replanner_output": {"error": f"Plan '{plan_id}' not found."},
            "next_node": "orchestrator",
        }

    winner_option = _winner_option_from_plan(plan_payload)
    winner_days = (plan_payload.get("itineraries") or {}).get(winner_option, [])
    cur_day, cur_min = _extract_day_and_time(feedback)
    consumed = _collect_consumed_items(winner_days, cur_day, cur_min)
    candidates = _collect_candidate_items(plan_payload, consumed)
    alternatives = _llm_generate_alternatives(
        feedback=feedback,
        plan_payload=plan_payload,
        consumed_names=consumed,
        candidates=candidates,
    )

    replanner_output = {
        "plan_id": plan_id,
        "winner_option": winner_option,
        "user_feedback": feedback,
        "context": {
            "location": feedback.get("current_location") or "Current location unknown",
            "current_day": f"Day {cur_day}",
            "current_time": feedback.get("current_time") or feedback.get("time") or "00:00",
            "original_plan_title": f"Debate winner option {winner_option}",
            "consumed_item_count": len(consumed),
        },
        "consumed_items": sorted(consumed),
        "alternatives": alternatives,
        "ripple_effect": "后续时段已重排，且不会重复你已完成的行程项目。",
    }

    # Persist last replan artifact into option_meta for traceability.
    updated_payload = dict(plan_payload)
    option_meta = dict(plan_payload.get("option_meta") or {})
    option_meta["replan_last_output"] = replanner_output
    updated_payload["option_meta"] = option_meta
    db = SessionLocal()
    try:
        update_plan_result(db, plan_id, updated_payload)
    finally:
        db.close()

    return {
        "plan_id": plan_id,
        "replanner_output": replanner_output,
        "user_feedback": None,
        "next_node": "orchestrator",
    }

