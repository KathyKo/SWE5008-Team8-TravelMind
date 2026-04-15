"""
Agent 6: Trace-based explainability for the selected itinerary option.

This module reads the structured decision trace produced by the planner
(planner_decision_trace), enriches it with place metadata, then uses a
single LLM call to produce semantic natural-language summaries.

Primary outputs (unchanged keys for frontend compatibility):
  - summary.overall_summary / summary.day_summaries
  - item_explanations.by_key / item_explanations.by_occurrence
  - evidence (debug / trace)
"""

import json
import re
import unicodedata
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..llm_config import OPENAI_MODEL


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)


# ===================================================================
# State helpers (kept from original)
# ===================================================================

def _first_present(state: dict, *keys: str, default: Any = "") -> Any:
    for scope in (state, state.get("state", {})):
        if not isinstance(scope, dict):
            continue
        for key in keys:
            value = scope.get(key)
            if value not in (None, "", [], {}):
                return value
    return default


def _collect_itineraries(state: dict) -> dict:
    for key in ("final_itineraries", "validated_itineraries", "itineraries"):
        value = state.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _selected_option(state: dict, itineraries: dict) -> str:
    candidates = [
        _first_present(state, "explain_option"),
        _first_present(state, "selected_option"),
        _first_present(state, "confirmed_option"),
        _first_present(state, "my_trip_option"),
        _first_present(state, "selected_itinerary_option"),
        _first_present(state, "option_key"),
        _first_present(state, "option"),
    ]
    for candidate in candidates:
        option_key = str(candidate or "").strip().upper()
        if option_key in itineraries:
            return option_key
    if "A" in itineraries:
        return "A"
    return next(iter(itineraries), "")


def _summary_language(state: dict) -> str:
    raw = str(
        _first_present(state, "summary_language", "language", "ui_language", "locale", default="")
    ).strip()
    if not raw:
        return "English"
    return raw if not raw.casefold().startswith("en") else raw


def _collect_pref_tags(state: dict) -> list[str]:
    tags: list[str] = []
    for scope in (state, state.get("state", {})):
        if not isinstance(scope, dict):
            continue
        raw = scope.get("preferences")
        if raw:
            tags.extend(part.strip() for part in str(raw).split(",") if part.strip())
        user_profile = scope.get("user_profile") or {}
        if isinstance(user_profile, dict):
            tags.extend(str(p).strip() for p in user_profile.get("prefs", []) or [] if str(p).strip())
        soft = scope.get("soft_preferences") or {}
        if isinstance(soft, dict):
            tags.extend(str(t).strip() for t in soft.get("interest_tags", []) or [] if str(t).strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        normalized = tag.casefold()
        if normalized and normalized not in seen:
            ordered.append(tag)
            seen.add(normalized)
    return ordered


# ===================================================================
# Place / hotel / flight metadata lookup (simplified from original)
# ===================================================================

def _normalized_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _numeric(value: Any) -> float | None:
    try:
        text = str(value).replace(",", "").strip()
        return float(text) if text else None
    except Exception:
        return None


def _integer(value: Any) -> int | None:
    match = re.search(r"\d+", str(value or "").replace(",", ""))
    return int(match.group()) if match else None


def _metadata_score(metadata: dict) -> int:
    score = 0
    for key in ("rating", "reviews", "hours", "weekday_descriptions", "lat", "lng", "address"):
        if metadata.get(key) not in (None, "", [], {}):
            score += 1
    return score


def _register_place(lookup: dict, item: dict, kind: str) -> None:
    if not isinstance(item, dict):
        return
    name = item.get("name")
    if not name:
        return
    normalized = {
        "name": name,
        "kind": kind,
        "type": item.get("type") or item.get("category") or "",
        "category": item.get("category") or item.get("type") or "",
        "rating": item.get("rating") or "",
        "reviews": item.get("reviews") or "",
        "address": item.get("address") or "",
        "lat": item.get("lat"),
        "lng": item.get("lng"),
        "description": item.get("description") or "",
        "search_query": item.get("search_query") or "",
    }
    key = _normalized_name(name)
    current = lookup.get(key)
    if current is None or _metadata_score(normalized) > _metadata_score(current):
        lookup[key] = normalized


def _build_place_lookup(state: dict) -> dict:
    lookup: dict[str, dict] = {}
    research = state.get("research", {}) or {}
    inventory = state.get("inventory", {}) or {}
    for key in ("maps_attractions", "ta_attractions"):
        for item in research.get(key, []) or []:
            _register_place(lookup, item, "activity")
    for key in ("maps_restaurants", "ta_restaurants"):
        for item in research.get(key, []) or []:
            _register_place(lookup, item, "restaurant")
    for item in inventory.get("attractions", []) or []:
        _register_place(lookup, item, "activity")
    for item in inventory.get("restaurants", []) or []:
        _register_place(lookup, item, "restaurant")
    return lookup


def _build_hotel_lookup(state: dict) -> dict:
    lookup: dict[str, dict] = {}
    for item in state.get("hotel_options", []) or []:
        if isinstance(item, dict) and item.get("name"):
            lookup[_normalized_name(item["name"])] = item
    return lookup


def _find_metadata(name: str, place_lookup: dict) -> dict:
    return place_lookup.get(_normalized_name(name), {})


# ===================================================================
# Item-level explanation (simplified — no signal/match system)
# ===================================================================

def _rating_text(metadata: dict) -> str:
    rating = _numeric(metadata.get("rating"))
    reviews = _integer(metadata.get("reviews"))
    if rating is None:
        return ""
    if reviews:
        return f"{rating:.1f} / 5 ({reviews:,} reviews)"
    return f"{rating:.1f} / 5"


def _occurrence_id(option_key: str, day_index: int, item_index: int, item: dict) -> str:
    item_key = str(item.get("key") or "item")
    return f"{option_key}__d{day_index + 1:02d}__i{item_index + 1:02d}__{item_key}"


def _item_explanation(
    option_key: str,
    day_label: str,
    day_index: int,
    item_index: int,
    item: dict,
    place_lookup: dict,
    hotel_lookup: dict,
) -> dict:
    """Build a lightweight explanation dict for one itinerary item."""
    icon = str(item.get("icon") or "")
    name = str(item.get("name") or "")
    time_text = str(item.get("time") or "")

    if icon == "hotel":
        lookup_name = name[len("Checkout from "):].strip() if name.lower().startswith("checkout from ") else name
        metadata = hotel_lookup.get(_normalized_name(lookup_name), {})
    elif icon == "flight":
        metadata = {}
    else:
        metadata = _find_metadata(name, place_lookup)

    rating = _rating_text(metadata) if icon != "flight" else ""
    item_type = str(metadata.get("type") or metadata.get("category") or "").strip()

    return {
        "item_key": item.get("key", ""),
        "occurrence_id": _occurrence_id(option_key, day_index, item_index, item),
        "option": option_key,
        "day": day_label,
        "name": name,
        "time": time_text,
        "icon": icon,
        "type": item_type,
        "rating": rating,
    }


def _item_display(explanation: dict) -> dict:
    """Frontend-facing subset."""
    return {
        "item_key": explanation.get("item_key", ""),
        "occurrence_id": explanation.get("occurrence_id", ""),
        "day": explanation.get("day", ""),
        "time": explanation.get("time", ""),
        "icon": explanation.get("icon", ""),
        "name": explanation.get("name", ""),
        "type": explanation.get("type", ""),
        "rating": explanation.get("rating", ""),
    }


# ===================================================================
# Decision trace reader
# ===================================================================

def _get_decision_trace(state: dict, option_key: str) -> list[dict]:
    """Extract the planner decision trace for one option."""
    trace = state.get("planner_decision_trace", {})
    if isinstance(trace, str):
        try:
            trace = json.loads(trace)
        except Exception:
            trace = {}
    if isinstance(trace, dict):
        return trace.get(option_key, [])
    return []


def _trace_name_in_items(name: str, items: list[dict], icon: str | None = None) -> bool:
    normalized_name = _normalized_name(name)
    if not normalized_name:
        return False
    for item in items:
        if icon and str(item.get("icon") or "") != icon:
            continue
        if _normalized_name(item.get("name", "")) == normalized_name:
            return True
    return False


def _sanitize_trace_entry(trace_entry: dict, day_items: list[dict]) -> dict:
    """Drop any trace content that is not present in the final itinerary items."""
    if not isinstance(trace_entry, dict):
        return {}

    activities = [
        activity
        for activity in trace_entry.get("activities", []) or []
        if _trace_name_in_items(activity.get("name", ""), day_items, icon="activity")
    ]
    original_activities = trace_entry.get("activities", []) or []

    original_lunch = trace_entry.get("lunch")
    lunch = original_lunch
    if not (isinstance(lunch, dict) and _trace_name_in_items(lunch.get("name", ""), day_items, icon="restaurant")):
        lunch = None

    original_dinner = trace_entry.get("dinner")
    dinner = original_dinner
    if not (isinstance(dinner, dict) and _trace_name_in_items(dinner.get("name", ""), day_items, icon="restaurant")):
        dinner = None

    original_seed_name = str(trace_entry.get("seed_name") or "").strip()
    seed_name = original_seed_name or None
    if seed_name and not _trace_name_in_items(seed_name, day_items, icon="activity"):
        seed_name = None

    allowed_names = {
        _normalized_name(item.get("name", ""))
        for item in day_items
        if item.get("name")
    }
    trace_names = [
        str(activity.get("name") or "").strip()
        for activity in (trace_entry.get("activities", []) or [])
    ]
    if isinstance(trace_entry.get("lunch"), dict):
        trace_names.append(str(trace_entry["lunch"].get("name") or "").strip())
    if isinstance(trace_entry.get("dinner"), dict):
        trace_names.append(str(trace_entry["dinner"].get("name") or "").strip())
    if original_seed_name:
        trace_names.append(original_seed_name)

    removed_names = [
        name for name in trace_names
        if name and _normalized_name(name) not in allowed_names
    ]
    seed_reason = str(trace_entry.get("seed_reason") or "").strip()
    normalized_reason = _normalized_name(seed_reason)
    trace_mismatch = (
        len(activities) != len(original_activities)
        or (isinstance(original_lunch, dict) and lunch is None)
        or (isinstance(original_dinner, dict) and dinner is None)
        or (original_seed_name and seed_name is None)
    )
    if trace_mismatch or any(_normalized_name(name) in normalized_reason for name in removed_names if _normalized_name(name)):
        seed_reason = ""

    return {
        "day_type": trace_entry.get("day_type", ""),
        "theme": trace_entry.get("theme", ""),
        "seed_name": seed_name,
        "seed_reason": seed_reason,
        "activities": activities,
        "lunch": lunch,
        "dinner": dinner,
    }


def _decision_trace_for_itinerary(itinerary: list[dict], decision_trace: list[dict]) -> list[dict]:
    trace_by_day = {
        str(entry.get("day") or "").strip(): entry
        for entry in decision_trace or []
        if isinstance(entry, dict)
    }
    sanitized: list[dict] = []
    for day_index, day in enumerate(itinerary):
        day_label = str(day.get("day") or f"Day {day_index + 1}")
        filtered = _sanitize_trace_entry(trace_by_day.get(day_label, {}), day.get("items", []) or [])
        sanitized.append({"day": day_label, **filtered})
    return sanitized


# ===================================================================
# Summary generation — trace-based
# ===================================================================

def _build_summary_payload(
    option_key: str,
    itinerary: list[dict],
    decision_trace: list[dict],
    option_meta: dict,
    pref_tags: list[str],
) -> dict:
    """Build the payload that the LLM receives for summary generation."""
    days_payload: list[dict] = []

    for day_index, day in enumerate(itinerary):
        day_label = str(day.get("day") or f"Day {day_index + 1}")
        items_display = [
            {
                "name": item.get("name", ""),
                "icon": item.get("icon", ""),
                "time": item.get("time", ""),
            }
            for item in day.get("items", [])
        ]

        # Find matching trace entry
        trace_entry = {}
        for t in decision_trace:
            if t.get("day") == day_label:
                trace_entry = t
                break
        trace_entry = _sanitize_trace_entry(trace_entry, day.get("items", []))

        days_payload.append({
            "day": day_label,
            "day_type": trace_entry.get("day_type", ""),
            "theme": trace_entry.get("theme", ""),
            "seed_name": trace_entry.get("seed_name"),
            "seed_reason": trace_entry.get("seed_reason", ""),
            "activities": trace_entry.get("activities", []),
            "lunch": trace_entry.get("lunch"),
            "dinner": trace_entry.get("dinner"),
            "items_in_order": items_display,
        })

    meta = option_meta.get(option_key, {}) if isinstance(option_meta, dict) else {}
    return {
        "selected_option": option_key,
        "option_label": meta.get("label", option_key),
        "option_style": meta.get("style", ""),
        "user_preferences": pref_tags[:8],
        "days": days_payload,
    }


def _generate_summary(
    state: dict,
    option_key: str,
    itinerary: list[dict],
    decision_trace: list[dict],
) -> dict:
    """Call LLM to produce overall + per-day summaries from the decision trace."""
    language = _summary_language(state)
    option_meta = state.get("option_meta", {}) or {}
    pref_tags = _collect_pref_tags(state)

    payload = _build_summary_payload(
        option_key, itinerary, decision_trace, option_meta, pref_tags,
    )
    day_order = [d["day"] for d in payload["days"]]

    system_prompt = (
        "You are a travel itinerary explainability writer. "
        "You receive structured decision data from a planner and write "
        "natural-language summaries that explain the planning logic.\n\n"
        f"Write in {language}.\n\n"
        "For each day summary, explain the day using only the final stops in "
        "items_in_order plus any aligned trace fields.\n"
        "Cover these points when the data supports them:\n"
        "1. the day's theme,\n"
        "2. why these final stops belong on the same day,\n"
        "3. why the final order makes sense,\n"
        "4. what role the meals play, if lunch/dinner data is meaningful,\n"
        "5. any tradeoff, only if a tradeoff is clearly supported.\n\n"
        "For arrival and departure days, explain the structural constraint "
        "(late arrival / early departure) and what was fit around it.\n\n"
        "RULES:\n"
        "- Do NOT include numeric ratings, scores, review counts, or distances (km, miles).\n"
        "- NEVER use the word 'anchor' in any form (anchor, anchored, anchoring). "
        "Say what the place IS, not that it 'anchors' the day.\n"
        "- Do NOT use these words or phrases: bridge, connector, "
        "timing anchor, meal anchor, lunch bridge, dinner close, "
        "sightseeing block, sightseeing window, sightseeing continuation, "
        "arrival anchor, departure anchor, meal-based theme shift, "
        "ground stop, ground segment, handoff, one-stop, fuel, functional.\n"
        "- Do NOT describe meals as 'practical' or 'functional'. "
        "Describe what cuisine they are and how that fits the day theme.\n"
        "- Do NOT use internal labels like lunch_bridge, dinner_close, "
        "  sightseeing_continuation, arrival_anchor.\n"
        "- Do NOT just list preferences the user stated. Explain the planning logic.\n"
        "- Use only stops that appear in items_in_order. If theme or seed_reason is broader "
        "  than the final stops, trust items_in_order and ignore the extra trace detail.\n"
        "- Do NOT invent temples, museums, gardens, cuisines, or tradeoffs that are not "
        "  grounded in the provided data.\n"
        "- If matched_preferences is empty or cuisine is generic, do not force a cuisine-theme link.\n"
        "- If there is not enough evidence for a tradeoff, omit tradeoff entirely.\n"
        "- Describe quality with words only: 'well reviewed', 'highly rated'.\n"
        "- Keep overall_summary to 2-4 sentences.\n"
        "- Keep each day_summary to 2-4 sentences, or fewer when the day is structurally simple.\n"
        "- Return ONLY valid JSON: {\"overall_summary\": \"...\", "
        "\"day_summaries\": {\"Day 1\": \"...\", ...}}\n"
    )

    human_prompt = (
        "Here is the structured decision data for the selected itinerary. "
        "Write the summaries.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        response = _llm().invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.content)
        if not isinstance(parsed, dict):
            parsed = {}

        overall = ""
        for key in ("overall_summary", "overall", "summary"):
            val = parsed.get(key)
            if val and str(val).strip():
                overall = str(val).strip()
                break

        raw_days = parsed.get("day_summaries", parsed.get("days", {}))
        if isinstance(raw_days, dict):
            day_summaries = {
                day_label: str(raw_days.get(day_label, "")).strip()
                for day_label in day_order
            }
        elif isinstance(raw_days, list):
            day_summaries = {}
            for i, entry in enumerate(raw_days):
                if isinstance(entry, dict):
                    label = str(
                        entry.get("day") or entry.get("day_label") or
                        (day_order[i] if i < len(day_order) else "")
                    ).strip()
                    text = str(
                        entry.get("summary") or entry.get("text") or
                        entry.get("content") or ""
                    ).strip()
                    if label:
                        day_summaries[label] = text
                elif i < len(day_order):
                    day_summaries[day_order[i]] = str(entry or "").strip()
        else:
            day_summaries = {}

        # Sanitize stray numbers
        overall = _strip_numeric_artifacts(overall)
        day_summaries = {k: _strip_numeric_artifacts(v) for k, v in day_summaries.items()}

        if not overall and not any(day_summaries.values()):
            return _fallback_summary(payload, language, day_order)

        return {
            "overall_summary": overall,
            "day_summaries": day_summaries,
            "generation_mode": "llm_trace",
            "language": language,
            "source": payload,
        }
    except Exception as exc:
        fallback = _fallback_summary(payload, language, day_order)
        fallback["generation_mode"] = "llm_error_fallback"
        fallback["error"] = str(exc)
        return fallback


def _strip_numeric_artifacts(text: str) -> str:
    """Remove stray rating numbers and review counts the LLM might leak."""
    cleaned = str(text or "")
    cleaned = re.sub(r"\b\d+\.\d+\s*/\s*5\b", "", cleaned)
    cleaned = re.sub(r"\(\s*[\d,]+\s+reviews?\s*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b[\d,]+\s+reviews?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b\d+(\.\d+)?\s*km\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _fallback_summary(
    payload: dict,
    language: str,
    day_order: list[str],
) -> dict:
    """Deterministic fallback when LLM fails — uses trace themes directly."""
    days = payload.get("days", [])
    option_label = payload.get("option_label", "This option")
    prefs = payload.get("user_preferences", [])
    pref_text = ", ".join(prefs[:4]) if prefs else "the main trip interests"

    overall = (
        f"{option_label} is organized so each day has a distinct theme "
        f"built around {pref_text}."
    )

    day_summaries: dict[str, str] = {}
    for day_data in days:
        label = day_data.get("day", "Day")
        theme = day_data.get("theme", "")
        activities = day_data.get("activities", [])
        names = [a.get("name", "") for a in activities if a.get("name")]
        seed_reason = day_data.get("seed_reason", "")

        if day_data.get("day_type") == "arrival":
            day_summaries[label] = (
                f"{label} is a light arrival evening with one dinner stop before the hotel."
            )
        elif day_data.get("day_type") == "departure":
            day_summaries[label] = (
                f"{label} is structured around checkout, one last visit, "
                f"and the return flight."
            )
        elif names:
            names_text = ", ".join(names[:3])
            day_summaries[label] = (
                f"{label} focuses on {theme or 'mixed sightseeing'}, "
                f"visiting {names_text}. "
                f"{seed_reason}" if seed_reason else ""
            ).strip()
        else:
            day_summaries[label] = f"{label} is organized as {theme or 'a sightseeing day'}."

    return {
        "overall_summary": overall,
        "day_summaries": day_summaries,
        "generation_mode": "fallback_trace",
        "language": language,
        "source": payload,
    }


def _summary_text(summary: dict) -> str:
    parts: list[str] = []
    overall = str(summary.get("overall_summary") or "").strip()
    if overall:
        parts.append(overall)
    for day_label, text in (summary.get("day_summaries", {}) or {}).items():
        text = str(text or "").strip()
        if text:
            parts.append(f"{day_label}: {text}")
    return "\n\n".join(parts)


# ===================================================================
# Main entry point
# ===================================================================

def explainability_agent(state: dict) -> dict:
    itineraries = _collect_itineraries(state)
    option_key = _selected_option(state, itineraries)
    days = itineraries.get(option_key, []) if option_key else []
    planner_chain_of_thought = state.get(
        "planner_chain_of_thought",
        state.get("chain_of_thought", ""),
    )
    tool_log = state.get("tool_log", []) or []
    destination = _first_present(state, "destination", default="")

    if not option_key or not days:
        empty_summary = {
            "overall_summary": "",
            "day_summaries": {},
            "generation_mode": "llm_trace",
            "language": _summary_language(state),
        }
        return {
            "explain_option": option_key or "A",
            "selected_option": option_key or "A",
            "summary": empty_summary,
            "item_explanations": {"by_key": {}, "by_occurrence": {}},
            "evidence": {},
            "explain_data": {},
            "explain_data_by_occurrence": {},
            "planner_chain_of_thought": planner_chain_of_thought,
            "explainability_chain_of_thought": "",
            "combined_chain_of_thought": planner_chain_of_thought,
            "chain_of_thought": planner_chain_of_thought,
            "agent_steps": _build_agent_steps(tool_log, destination, option_key or "A"),
        }

    place_lookup = _build_place_lookup(state)
    hotel_lookup = _build_hotel_lookup(state)

    # Build item explanations
    explain_data: dict[str, dict] = {}
    explain_data_by_occurrence: dict[str, dict] = {}

    for day_index, day in enumerate(days):
        items = day.get("items", []) or []
        day_label = str(day.get("day") or f"Day {day_index + 1}")
        for item_index, item in enumerate(items):
            explanation = _item_explanation(
                option_key=option_key,
                day_label=day_label,
                day_index=day_index,
                item_index=item_index,
                item=item,
                place_lookup=place_lookup,
                hotel_lookup=hotel_lookup,
            )
            occ_id = explanation["occurrence_id"]
            explain_data_by_occurrence[occ_id] = explanation
            item_key = str(item.get("key") or occ_id)
            if item_key not in explain_data:
                explain_data[item_key] = explanation

    # Get the decision trace and align it to the final itinerary before prompting the LLM.
    decision_trace = _decision_trace_for_itinerary(
        days,
        _get_decision_trace(state, option_key),
    )

    # Generate summary from trace
    summary = _generate_summary(
        state=state,
        option_key=option_key,
        itinerary=days,
        decision_trace=decision_trace,
        option_meta=state.get("option_meta", {}) or {},
        pref_tags=_collect_pref_tags(state),
    )

    explainability_cot = _summary_text(summary)
    combined_cot = "\n\n".join(
        part for part in (planner_chain_of_thought, explainability_cot) if part
    )

    item_explanations = {
        "by_key": {k: _item_display(v) for k, v in explain_data.items()},
        "by_occurrence": {k: _item_display(v) for k, v in explain_data_by_occurrence.items()},
    }

    evidence = {
        "option_key": option_key,
        "decision_trace": decision_trace,
        "items": {
            "by_key": explain_data,
            "by_occurrence": explain_data_by_occurrence,
        },
        "summary_generation": {
            "mode": summary.get("generation_mode", ""),
            "language": summary.get("language", ""),
            "error": summary.get("error", ""),
            "source": summary.get("source", {}),
        },
    }

    print(
        f"[Agent6] Trace-based explainability generated for option {option_key} "
        f"with {len(explain_data_by_occurrence)} items, "
        f"{len(decision_trace)} trace entries"
    )

    return {
        "explain_option": option_key,
        "selected_option": option_key,
        "summary": {
            "overall_summary": summary.get("overall_summary", ""),
            "day_summaries": summary.get("day_summaries", {}) or {},
        },
        "item_explanations": item_explanations,
        "evidence": evidence,
        "explain_data": explain_data,
        "explain_data_by_occurrence": explain_data_by_occurrence,
        "planner_chain_of_thought": planner_chain_of_thought,
        "explainability_chain_of_thought": explainability_cot,
        "combined_chain_of_thought": combined_cot,
        "chain_of_thought": combined_cot,
        "agent_steps": _build_agent_steps(tool_log, destination, option_key),
    }


def _build_agent_steps(tool_log: list, dest: str, option_key: str = "A") -> list:
    tool_count = len([entry for entry in tool_log if str(entry).startswith("[")])
    return [
        {"icon": "research", "name": "Research Agent", "detail": f"Reused {tool_count} logged research/planner steps for {dest}."},
        {"icon": "plan", "name": "Planner Agent", "detail": f"Built itinerary for option {option_key} with LLM-guided day themes."},
        {"icon": "why", "name": "Explainability Agent", "detail": "Generated trace-based explanations from planner decision log."},
    ]
