"""
agents/specialists/explainability_agent.py — Agent6: Explainability

Takes 3 itinerary options from Agent3 and adds:
  1. Per-item "Why?" explanation  → shown in Plan + My Trip "Why? →" button
  2. Chain of thought text        → shown in My Trip side panel
  3. Tool call log                → shown as agent activity trace

Input  (state dict):
    itineraries, option_meta, research, planner_chain_of_thought, tool_log,
    user_profile, preferences, budget, destination

Output (dict):
    explain_data     : {item_key: {name, matches, similar, scores, chain_of_thought}}
    planner_chain_of_thought        : planner reasoning text
    explainability_chain_of_thought : explainability reasoning text
    combined_chain_of_thought       : planner + explainability reasoning text
    chain_of_thought                : compatibility alias of combined_chain_of_thought
    agent_steps      : list for frontend Agent Activity panel
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..llm_config import OPENAI_MODEL


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)


def _legacy_llm_explainability_agent(state: dict) -> dict:
    """
    Agent6: Explainability.
    Generates per-item reasoning traces and an overall chain of thought.
    """
    itineraries      = state.get("itineraries", {})
    research         = state.get("research", {})
    planner_chain_of_thought = state.get(
        "planner_chain_of_thought",
        state.get("chain_of_thought", ""),
    )
    tool_log         = state.get("tool_log", [])
    user_profile     = state.get("user_profile", {})
    preferences      = state.get("preferences", "")
    budget           = state.get("budget") or ""
    dest             = state.get("destination") or ""

    user_prefs = ", ".join(user_profile.get("prefs", [])) if user_profile else preferences

    # ── Collect UNIQUE items that need a Why? explanation ────────────
    unique_items: dict[str, dict] = {}
    seen_names: set[str] = set()

    for opt_key, days in itineraries.items():
        for day in days:
            for item in day.get("items", []):
                name = item.get("name")
                key = item.get("key")
                if not name or not key: continue

                # Deduplicate by name (ignore the day/time suffix in the key for logic)
                name_norm = name.lower().strip()
                if name_norm in seen_names:
                    continue
                
                # Logic: We only need to explain a hotel once, a flight once, and each attraction once.
                seen_names.add(name_norm)
                unique_items[key] = {
                    "name":   name,
                    "cost":   item.get("cost", ""),
                    "type":   "flight" if "flight" in key else ("hotel" if "hotel" in key else "activity")
                }

    if not unique_items:
        return {
            "explain_data": {},
            "planner_chain_of_thought": planner_chain_of_thought,
            "explainability_chain_of_thought": "",
            "combined_chain_of_thought": planner_chain_of_thought,
            "chain_of_thought": planner_chain_of_thought,
            "agent_steps": _build_agent_steps(tool_log, dest),
        }

    # ── Single LLM call for UNIQUE items ONLY ────────────────────────────
    system_prompt = f"""You are Explainability Agent.
Explain WHY these unique travel items were chosen for a trip to {dest}.

User Profile: {user_prefs or "general traveller"}
Budget: {budget}

Research Context:
{str(research.get("hotel_reviews",  "—"))[:500]}
{str(research.get("place_reviews",  "—"))[:500]}

Items (JSON):
{json.dumps(unique_items)}

Return ONLY valid JSON:
{{
  "explain_data": {{
    "<item_key>": {{
      "matches": ["reason 1", "reason 2"],
      "rating": "⭐X.X / 5",
      "review_highlights": ["quote 1"],
      "chain_of_thought": "Short reasoning summary."
    }}
  }},
  "overall_chain_of_thought": "Brief summary of the 3 themes (Options A, B, C)."
}}

Rules:
- Be concise. 1-2 reasons per item.
- Cite REAL ratings/reviews from the context if they exist.
- If no data, return empty string for rating/reviews.
"""

    try:
        llm      = _llm()
        response = llm.invoke(
            [SystemMessage(content=system_prompt)],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.content)

        raw_explain_data = result.get("explain_data", {})
        explainability_chain_of_thought = result.get("overall_chain_of_thought", "")

        # ── Map unique explanations back to ALL itinerary items ──────────
        final_explain_data = {}
        # Create a name-to-explanation map
        name_to_expl = {unique_items[k]["name"].lower().strip(): v for k, v in raw_explain_data.items() if k in unique_items}

        for opt_key, days in itineraries.items():
            for day in days:
                for item in day.get("items", []):
                    key = item.get("key")
                    name = item.get("name", "").lower().strip()
                    if key and name in name_to_expl:
                        final_explain_data[key] = name_to_expl[name]

        combined_chain_of_thought = "\n\n".join(
            filter(None, [planner_chain_of_thought, explainability_chain_of_thought])
        )
        print(f"[Agent6] Speed-optimized — explained {len(final_explain_data)} occurrences via {len(raw_explain_data)} unique items")

        return {
            "explain_data": final_explain_data,
            "planner_chain_of_thought": planner_chain_of_thought,
            "explainability_chain_of_thought": explainability_chain_of_thought,
            "combined_chain_of_thought": combined_chain_of_thought,
            "chain_of_thought": combined_chain_of_thought,
            "agent_steps": _build_agent_steps(tool_log, dest),
        }


    except Exception as e:
        print(f"[Agent6] Error: {e}")
        return {
            "explain_data": {},
            "planner_chain_of_thought": planner_chain_of_thought,
            "explainability_chain_of_thought": "",
            "combined_chain_of_thought": planner_chain_of_thought,
            "chain_of_thought": planner_chain_of_thought,
            "agent_steps": _build_agent_steps(tool_log, dest),
        }


def _legacy_build_agent_steps(tool_log: list, dest: str) -> list:
    """Build the agent activity list shown in the Plan page right panel."""
    tool_count = len([t for t in tool_log if t.startswith("[")])
    return [
        {"icon": "🔍", "name": "Research Agent",        "detail": f"Ran {tool_count} tool calls for {dest}"},
        {"icon": "📅", "name": "Planner Agent",    "detail": "Generated 3 itinerary variants via research data"},
        {"icon": "💡", "name": "Explainability",   "detail": "Reasoning traces + chain of thought generated"},
    ]


import math
import re
import unicodedata
from typing import Any


def _exp_first_present(state: dict, *keys: str, default: Any = "") -> Any:
    for scope in (state, state.get("state", {})):
        if not isinstance(scope, dict):
            continue
        for key in keys:
            value = scope.get(key)
            if value not in (None, "", [], {}):
                return value
    return default


def _exp_collect_itineraries(state: dict) -> dict:
    for key in ("final_itineraries", "validated_itineraries", "itineraries"):
        value = state.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _exp_selected_option(state: dict, itineraries: dict) -> str:
    candidates = [
        _exp_first_present(state, "explain_option"),
        _exp_first_present(state, "selected_option"),
        _exp_first_present(state, "confirmed_option"),
        _exp_first_present(state, "my_trip_option"),
        _exp_first_present(state, "selected_itinerary_option"),
        _exp_first_present(state, "option_key"),
        _exp_first_present(state, "option"),
    ]
    for candidate in candidates:
        option_key = str(candidate or "").strip().upper()
        if option_key in itineraries:
            return option_key
    if "A" in itineraries:
        return "A"
    return next(iter(itineraries), "")


def _exp_normalized_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _exp_numeric(value: Any) -> float | None:
    try:
        text = str(value).replace(",", "").strip()
        return float(text) if text else None
    except Exception:
        return None


def _exp_integer(value: Any) -> int | None:
    match = re.search(r"\d+", str(value or "").replace(",", ""))
    return int(match.group()) if match else None


def _exp_minutes(value: str) -> int | None:
    match = re.search(r"(\d{1,2}):(\d{2})", str(value or ""))
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _exp_slot_label(value: str, icon: str) -> str:
    minutes = _exp_minutes(value)
    if minutes is None:
        return "scheduled"
    if icon == "restaurant":
        if minutes < 15 * 60:
            return "lunch"
        return "dinner"
    if minutes < 12 * 60:
        return "morning"
    if minutes < 17 * 60:
        return "afternoon"
    if minutes < 21 * 60:
        return "evening"
    return "night"


def _exp_text_blob(metadata: dict) -> str:
    parts = [
        metadata.get("name", ""),
        metadata.get("type", ""),
        metadata.get("category", ""),
        metadata.get("address", ""),
        metadata.get("description", ""),
        metadata.get("search_query", ""),
    ]
    return " ".join(str(part or "") for part in parts).casefold()


def _exp_haversine_km(a_lat: Any, a_lng: Any, b_lat: Any, b_lng: Any) -> float | None:
    try:
        a_lat = float(a_lat)
        a_lng = float(a_lng)
        b_lat = float(b_lat)
        b_lng = float(b_lng)
    except Exception:
        return None

    radius_km = 6371.0
    d_lat = math.radians(b_lat - a_lat)
    d_lng = math.radians(b_lng - a_lng)
    a_term = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(a_lat))
        * math.cos(math.radians(b_lat))
        * math.sin(d_lng / 2) ** 2
    )
    c_term = 2 * math.atan2(math.sqrt(a_term), math.sqrt(1 - a_term))
    return radius_km * c_term


def _exp_metadata_score(metadata: dict) -> int:
    score = 0
    for key in ("rating", "reviews", "hours", "weekday_descriptions", "lat", "lng", "address"):
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            score += 1
    return score


def _exp_register_place(lookup: dict, item: dict, kind: str) -> None:
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
        "hours": item.get("hours") or "",
        "weekday_descriptions": item.get("weekday_descriptions") or [],
        "description": item.get("description") or "",
        "search_query": item.get("search_query") or "",
        "source": item.get("source") or "",
    }
    key = _exp_normalized_name(name)
    current = lookup.get(key)
    if current is None or _exp_metadata_score(normalized) > _exp_metadata_score(current):
        lookup[key] = normalized


def _exp_build_place_lookup(state: dict) -> dict:
    lookup: dict[str, dict] = {}
    research = state.get("research", {}) or {}
    inventory = state.get("inventory", {}) or {}

    for key in ("maps_attractions", "ta_attractions"):
        for item in research.get(key, []) or []:
            _exp_register_place(lookup, item, "activity")

    for key in ("maps_restaurants", "ta_restaurants"):
        for item in research.get(key, []) or []:
            _exp_register_place(lookup, item, "restaurant")

    for item in inventory.get("attractions", []) or []:
        _exp_register_place(lookup, item, "activity")

    for item in inventory.get("restaurants", []) or []:
        _exp_register_place(lookup, item, "restaurant")

    return lookup


def _exp_build_hotel_lookup(state: dict) -> dict:
    lookup: dict[str, dict] = {}
    for item in state.get("hotel_options", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        lookup[_exp_normalized_name(name)] = item
    return lookup


def _exp_find_place_metadata(item: dict, place_lookup: dict) -> dict:
    name = item.get("name", "")
    return place_lookup.get(_exp_normalized_name(name), {})


def _exp_find_hotel_metadata(item: dict, hotel_lookup: dict) -> dict:
    name = str(item.get("name") or "")
    if name.lower().startswith("checkout from "):
        name = name[len("Checkout from ") :]
    return hotel_lookup.get(_exp_normalized_name(name), {})


def _exp_find_flight_metadata(item: dict, state: dict) -> dict:
    key = str(item.get("key") or "")
    display_name = str(item.get("name") or "")
    if "return" in key:
        options = state.get("flight_options_return", []) or []
    else:
        options = state.get("flight_options_outbound", []) or []
    for option in options:
        if str(option.get("display") or "") == display_name:
            return option
    return options[0] if options else {}


def _exp_collect_pref_tags(state: dict) -> list[str]:
    tags: list[str] = []
    for scope in (state, state.get("state", {})):
        if not isinstance(scope, dict):
            continue
        raw_preferences = scope.get("preferences")
        if raw_preferences:
            tags.extend(part.strip() for part in str(raw_preferences).split(",") if part.strip())

        user_profile = scope.get("user_profile") or {}
        if isinstance(user_profile, dict):
            tags.extend(str(item).strip() for item in user_profile.get("prefs", []) or [] if str(item).strip())

        soft_preferences = scope.get("soft_preferences") or {}
        if isinstance(soft_preferences, dict):
            tags.extend(str(item).strip() for item in soft_preferences.get("interest_tags", []) or [] if str(item).strip())

    ordered: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = tag.casefold()
        if normalized and normalized not in seen:
            ordered.append(tag)
            seen.add(normalized)
    return ordered


def _exp_rating_text(metadata: dict) -> str:
    rating = _exp_numeric(metadata.get("rating"))
    reviews = _exp_integer(metadata.get("reviews"))
    if rating is None:
        return ""
    if reviews:
        return f"{rating:.1f} / 5 ({reviews:,} reviews)"
    return f"{rating:.1f} / 5"


def _exp_compact_text(value: Any, max_len: int = 96) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _exp_tokenize(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").casefold())


def _exp_data_point(kind: str, **fields: Any) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value in (None, "", [], {}, ()):
            continue
        if isinstance(value, float):
            rendered = f"{value:.2f}".rstrip("0").rstrip(".")
        elif isinstance(value, list):
            rendered = ",".join(str(item) for item in value if str(item))
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    if not parts:
        return kind
    return f"{kind}[{'; '.join(parts)}]"


def _exp_quality_record(metadata: dict, icon: str) -> str:
    rating = _exp_numeric(metadata.get("rating"))
    reviews = _exp_integer(metadata.get("reviews"))
    if rating is None:
        return ""
    return _exp_data_point(
        "quality_signal",
        icon=icon,
        rating=rating,
        reviews=reviews,
    )


def _exp_hours_record(metadata: dict, icon: str, slot_label: str) -> str:
    hours_text = metadata.get("hours") or metadata.get("weekday_descriptions") or ""
    if not hours_text:
        return ""
    return _exp_data_point(
        "hours_signal",
        icon=icon,
        slot=slot_label,
        hours_available=1,
        hours_text=_exp_compact_text(hours_text, max_len=72),
    )


def _exp_preference_records(metadata: dict, pref_tags: list[str], icon: str) -> list[str]:
    metadata_fields = {
        "name": _exp_tokenize(metadata.get("name")),
        "type": _exp_tokenize(metadata.get("type")),
        "category": _exp_tokenize(metadata.get("category")),
        "search_query": _exp_tokenize(metadata.get("search_query")),
    }
    records: list[str] = []
    seen: set[str] = set()

    for pref in pref_tags:
        pref_tokens = [token for token in _exp_tokenize(pref) if len(token) >= 3]
        overlap: dict[str, list[str]] = {}
        for source_name, source_tokens in metadata_fields.items():
            matched = sorted(set(pref_tokens).intersection(source_tokens))
            if matched:
                overlap[source_name] = matched
        if not overlap:
            continue
        record = _exp_data_point(
            "pref_match",
            pref=pref,
            icon=icon,
            overlap="|".join(f"{source}:{','.join(tokens)}" for source, tokens in overlap.items()),
        )
        if record not in seen:
            records.append(record)
            seen.add(record)

    profile_record = _exp_data_point(
        "content_profile",
        icon=icon,
        type=_exp_compact_text(metadata.get("type"), 32),
        category=_exp_compact_text(metadata.get("category"), 32),
        query=_exp_compact_text(metadata.get("search_query"), 48),
    )
    if profile_record not in seen:
        records.append(profile_record)
    return records[:2]


def _exp_route_record(current_meta: dict, previous_meta: dict, next_meta: dict, icon: str) -> str:
    current_lat = current_meta.get("lat")
    current_lng = current_meta.get("lng")
    if current_lat in (None, "") or current_lng in (None, ""):
        return _exp_data_point(
            "route_signal",
            icon=icon,
            has_coordinates=0,
            prev_present=1 if previous_meta else 0,
            next_present=1 if next_meta else 0,
        )

    prev_distance = None
    next_distance = None

    if previous_meta:
        prev_distance = _exp_haversine_km(current_lat, current_lng, previous_meta.get("lat"), previous_meta.get("lng"))
    if next_meta:
        next_distance = _exp_haversine_km(current_lat, current_lng, next_meta.get("lat"), next_meta.get("lng"))
    total_distance = None
    if prev_distance is not None and next_distance is not None:
        total_distance = prev_distance + next_distance
    return _exp_data_point(
        "route_signal",
        icon=icon,
        prev_km=prev_distance,
        next_km=next_distance,
        total_km=total_distance,
    )


def _exp_hotel_highlights(review_blob: str, hotel_name: str, hotel_meta: dict) -> list[str]:
    highlights: list[str] = []
    if review_blob and hotel_name:
        pattern = re.compile(
            rf"===\s*{re.escape(hotel_name)}\s*===\s*(.*?)(?=\n===|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(review_blob)
        if match:
            segment = match.group(1)
            quoted = [text.strip() for text in re.findall(r'\"([^\"]+)\"', segment) if text.strip()]
            highlights.extend(quoted[:2])

    if highlights:
        return highlights[:2]

    rating = _exp_numeric(hotel_meta.get("rating"))
    reviews = _exp_integer(hotel_meta.get("reviews"))
    if rating is not None:
        highlights.append(_exp_data_point("rating_signal", rating=rating, reviews=reviews))
    return highlights[:2]


def _exp_place_highlights(metadata: dict, slot_label: str) -> list[str]:
    highlights: list[str] = []
    rating = _exp_numeric(metadata.get("rating"))
    reviews = _exp_integer(metadata.get("reviews"))

    if rating is not None:
        highlights.append(_exp_data_point("rating_signal", rating=rating, reviews=reviews))

    hours_record = _exp_hours_record(metadata, metadata.get("kind", ""), slot_label)
    if hours_record:
        highlights.append(hours_record)

    return highlights[:2]


def _exp_occurrence_id(option_key: str, day_index: int, item_index: int, item: dict) -> str:
    item_key = str(item.get("key") or "item")
    return f"{option_key}__d{day_index + 1:02d}__i{item_index + 1:02d}__{item_key}"


def _exp_item_chain(day_label: str, item: dict, slot_label: str, matches: list[str]) -> str:
    time_text = str(item.get("time") or "").strip()
    name = str(item.get("name") or "stop").strip()
    icon = str(item.get("icon") or "").strip()
    return _exp_data_point(
        "item_signal",
        day=day_label,
        time=time_text,
        icon=icon,
        slot=slot_label,
        name=_exp_compact_text(name, 80),
        evidence=matches,
    )


def _exp_item_explanation(
    state: dict,
    option_key: str,
    day_label: str,
    day_index: int,
    item_index: int,
    items: list[dict],
    item: dict,
    place_lookup: dict,
    hotel_lookup: dict,
    hotel_review_blob: str,
    pref_tags: list[str],
) -> dict:
    icon = str(item.get("icon") or "")
    slot_label = _exp_slot_label(item.get("time", ""), icon)
    previous_item = items[item_index - 1] if item_index > 0 else {}
    next_item = items[item_index + 1] if item_index + 1 < len(items) else {}

    if icon == "flight":
        metadata = _exp_find_flight_metadata(item, state)
        review_highlights: list[str] = []
        matches: list[str] = []
        key = str(item.get("key") or "")
        display_text = str(metadata.get("display") or item.get("name") or "")
        nonstop = 1 if "nonstop" in display_text.casefold() else 0
        dep_time = str(metadata.get("departure_time") or "")
        arr_time = str(metadata.get("arrival_time") or "")

        if metadata.get("display"):
            review_highlights.append(metadata["display"])
        matches.append(
            _exp_data_point(
                "flight_signal",
                segment="return" if "return" in key else "outbound",
                nonstop=nonstop,
                dep=dep_time,
                arr=arr_time,
                duration_min=metadata.get("duration_min"),
            )
        )
        if "return" in key:
            return_pref = str(_exp_first_present(state, "return_time_pref")).strip()
            if return_pref:
                matches.append(_exp_data_point("time_pref_signal", pref=return_pref, dep=dep_time))
        else:
            outbound_pref = str(_exp_first_present(state, "outbound_time_pref")).strip()
            if outbound_pref:
                matches.append(_exp_data_point("time_pref_signal", pref=outbound_pref, dep=dep_time))
        chain = _exp_item_chain(day_label, item, slot_label, matches)

        return {
            "item_key": item.get("key", ""),
            "name": item.get("name", ""),
            "time": item.get("time", ""),
            "icon": icon,
            "slot": slot_label,
            "matches": matches[:3],
            "all_matches": matches,
            "rating": "",
            "review_highlights": review_highlights[:2],
            "chain_of_thought": chain,
            "occurrence_id": _exp_occurrence_id(option_key, day_index, item_index, item),
            "option": option_key,
            "day": day_label,
        }

    if icon == "hotel":
        metadata = _exp_find_hotel_metadata(item, hotel_lookup)
        rating_text = _exp_rating_text(metadata)
        review_highlights = _exp_hotel_highlights(
            review_blob=hotel_review_blob,
            hotel_name=metadata.get("name", ""),
            hotel_meta=metadata,
        )
        matches = []

        price_per_night = _exp_numeric(metadata.get("price_per_night_usd"))
        budget_text = str(_exp_first_present(state, "budget", default=""))
        if price_per_night is not None and budget_text:
            matches.append(_exp_data_point("budget_signal", price_per_night_usd=price_per_night, budget=budget_text))
        quality_record = _exp_quality_record(metadata, icon)
        if quality_record:
            matches.append(quality_record)
        matches.append(
            _exp_data_point(
                "hotel_signal",
                key=item.get("key"),
                time=item.get("time"),
                name=_exp_compact_text(metadata.get("name") or item.get("name"), 64),
            )
        )
        chain = _exp_item_chain(day_label, item, slot_label, matches)

        return {
            "item_key": item.get("key", ""),
            "name": item.get("name", ""),
            "time": item.get("time", ""),
            "icon": icon,
            "slot": slot_label,
            "matches": matches[:3],
            "all_matches": matches,
            "rating": rating_text,
            "review_highlights": review_highlights[:2],
            "chain_of_thought": chain,
            "occurrence_id": _exp_occurrence_id(option_key, day_index, item_index, item),
            "option": option_key,
            "day": day_label,
        }

    metadata = _exp_find_place_metadata(item, place_lookup)
    previous_meta = _exp_find_place_metadata(previous_item, place_lookup) if previous_item else {}
    next_meta = _exp_find_place_metadata(next_item, place_lookup) if next_item else {}
    rating_text = _exp_rating_text(metadata)
    review_highlights = _exp_place_highlights(metadata, slot_label)

    matches = _exp_preference_records(metadata, pref_tags, icon)
    has_pref_match = any(_exp_parse_data_point(reason)[0] == "pref_match" for reason in matches)
    pref_tags_lower = [str(tag or "").casefold() for tag in pref_tags]
    metadata_blob = _exp_text_blob(metadata)

    if icon == "restaurant" and not has_pref_match:
        if slot_label == "dinner" and any("fine dining" in tag for tag in pref_tags_lower):
            matches.insert(0, _exp_data_point("pref_match", pref="fine dining", icon=icon, overlap="slot:dinner"))
            has_pref_match = True
        elif any(("traditional" in tag) or ("cultural" in tag) for tag in pref_tags_lower):
            cuisine_hint = ""
            if any(token in metadata_blob for token in ("japanese", "sushi", "tonkatsu", "ramen", "wagyu")):
                cuisine_hint = f"type:{_exp_compact_text(metadata.get('type') or metadata.get('category') or 'restaurant', 32)}"
            elif slot_label == "lunch":
                cuisine_hint = "slot:lunch"
            if cuisine_hint:
                matches.insert(0, _exp_data_point("pref_match", pref="traditional cultural", icon=icon, overlap=cuisine_hint))

    quality_record = _exp_quality_record(metadata, icon)
    if quality_record:
        matches.append(quality_record)

    hours_record = _exp_hours_record(metadata, icon, slot_label)
    if hours_record:
        matches.append(hours_record)

    route_record = _exp_route_record(metadata, previous_meta, next_meta, icon)
    if route_record:
        matches.append(route_record)

    chain = _exp_item_chain(day_label, item, slot_label, matches)

    return {
        "item_key": item.get("key", ""),
        "name": item.get("name", ""),
        "time": item.get("time", ""),
        "icon": icon,
        "slot": slot_label,
        "matches": matches[:3],
        "all_matches": matches,
        "rating": rating_text,
        "review_highlights": review_highlights[:2],
        "chain_of_thought": chain,
        "occurrence_id": _exp_occurrence_id(option_key, day_index, item_index, item),
        "option": option_key,
        "day": day_label,
    }


def _exp_item_display(explanation: dict) -> dict:
    return {
        "item_key": explanation.get("item_key", ""),
        "occurrence_id": explanation.get("occurrence_id", ""),
        "day": explanation.get("day", ""),
        "time": explanation.get("time", ""),
        "icon": explanation.get("icon", ""),
        "slot": explanation.get("slot", ""),
        "name": explanation.get("name", ""),
        "why": explanation.get("matches", []) or [],
        "rating": explanation.get("rating", ""),
        "review_highlights": explanation.get("review_highlights", []) or [],
    }


def _exp_summary_stop_name(item: dict) -> str:
    icon = str(item.get("icon") or "")
    item_key = str(item.get("item_key") or item.get("key") or "")
    name = str(item.get("name") or "stop").strip()
    if icon == "flight":
        return "the return nonstop flight" if "return" in item_key else "the outbound nonstop flight"
    if icon == "hotel" and item_key.startswith("hotel_checkout"):
        hotel_name = name[len("Checkout from ") :].strip() if name.lower().startswith("checkout from ") else name
        return f"hotel checkout from {hotel_name}".strip()
    return name


def _exp_transition_link(current_item: dict, next_item: dict) -> str:
    current_icon = str(current_item.get("icon") or "")
    next_icon = str(next_item.get("icon") or "")
    current_slot = str(current_item.get("slot") or "")
    next_slot = str(next_item.get("slot") or "")
    current_key = str(current_item.get("item_key") or current_item.get("key") or "")
    next_key = str(next_item.get("item_key") or next_item.get("key") or "")

    if current_icon == "flight" and next_icon == "restaurant":
        return "arrival_to_meal"
    if current_icon == "restaurant" and next_icon == "hotel":
        return "meal_to_overnight"
    if current_key.startswith("hotel_checkout") and next_icon == "activity":
        return "checkout_to_last_visit"
    if current_icon == "activity" and next_icon == "restaurant" and next_slot == "lunch":
        return "morning_to_lunch_bridge"
    if current_icon == "restaurant" and current_slot == "lunch" and next_icon == "activity":
        return "lunch_to_afternoon_bridge"
    if current_icon == "activity" and next_icon == "restaurant" and next_slot == "dinner":
        return "activity_to_dinner_close"
    if current_icon == "activity" and next_icon == "activity":
        return "same_day_sightseeing_chain"
    if next_icon == "flight":
        return "last_stop_to_airport"
    return "sequence_step"


def _exp_transition_reason_hint(current_row: dict, next_row: dict, current_explanation: dict, next_explanation: dict) -> str:
    current_raw = current_explanation.get("all_matches", current_explanation.get("matches", [])) or []
    next_raw = next_explanation.get("all_matches", next_explanation.get("matches", [])) or []
    current_route = _exp_match_fields(current_raw, "route_signal")
    next_route = _exp_match_fields(next_raw, "route_signal")
    distance = _exp_numeric(next_route.get("prev_km"))
    if distance is None:
        distance = _exp_numeric(current_route.get("next_km"))

    from_name = _exp_summary_stop_name(current_row)
    to_name = _exp_summary_stop_name(next_row)
    link = _exp_transition_link(current_row, next_row)
    distance_text = f" over about {distance:.1f} km" if distance is not None else ""

    templates = {
        "arrival_to_meal": f"{to_name} follows arrival so the evening starts with one manageable ground stop{distance_text}",
        "meal_to_overnight": f"{to_name} comes after {from_name} so the day closes into the overnight instead of reopening the route{distance_text}",
        "checkout_to_last_visit": f"{to_name} is kept right after checkout so the last sightseeing piece still connects to the departure day{distance_text}",
        "morning_to_lunch_bridge": f"{to_name} is used as the lunch bridge after {from_name} instead of creating a separate detour{distance_text}",
        "lunch_to_afternoon_bridge": f"{to_name} resumes the afternoon block after lunch so the same-day run stays intact{distance_text}",
        "activity_to_dinner_close": f"{to_name} closes the day after {from_name} rather than adding another activity{distance_text}",
        "same_day_sightseeing_chain": f"{to_name} stays on the same day as {from_name} to keep the sightseeing chain continuous{distance_text}",
        "last_stop_to_airport": f"{to_name} remains last so the airport leg stays anchored at the end of the day{distance_text}",
    }
    return templates.get(link, f"{to_name} follows {from_name} to keep the day in one sequence{distance_text}")


def _exp_day_evidence(option_key: str, day_index: int, day: dict, explain_data_by_occurrence: dict) -> dict:
    day_label = str(day.get("day") or "Day")
    items = day.get("items", []) or []
    sequence_segments: list[str] = []
    item_rows: list[dict] = []

    for item_index, item in enumerate(items):
        occurrence_id = _exp_occurrence_id(option_key, day_index, item_index, item)
        explanation = explain_data_by_occurrence.get(occurrence_id, {}) or {}
        time_text = str(item.get("time") or "").strip()
        name = str(item.get("name") or "stop").strip()
        icon = str(item.get("icon") or "").strip()
        matches = explanation.get("matches", []) or []
        sequence_segments.append(
            _exp_data_point(
                "stop",
                time=time_text,
                icon=icon,
                name=_exp_compact_text(name, 64),
                evidence=matches[:1],
            )
        )
        item_rows.append(
            {
                "item_key": explanation.get("item_key", item.get("key", "")),
                "occurrence_id": occurrence_id,
                "time": time_text,
                "icon": icon,
                "name": name,
                "slot": explanation.get("slot", ""),
                "top_why": matches[:2],
                "top_why_raw": explanation.get("all_matches", matches),
                "rating": explanation.get("rating", ""),
                "review_highlights": explanation.get("review_highlights", [])[:1],
            }
        )

    transitions: list[dict] = []
    for item_index in range(len(item_rows) - 1):
        current_row = item_rows[item_index]
        next_row = item_rows[item_index + 1]
        current_explanation = explain_data_by_occurrence.get(current_row.get("occurrence_id", ""), {}) or {}
        next_explanation = explain_data_by_occurrence.get(next_row.get("occurrence_id", ""), {}) or {}
        transitions.append(
            {
                "from_name": _exp_summary_stop_name(current_row),
                "to_name": _exp_summary_stop_name(next_row),
                "link": _exp_transition_link(current_row, next_row),
                "reason_hint": _exp_transition_reason_hint(current_row, next_row, current_explanation, next_explanation),
            }
        )

    return {
        "day": day_label,
        "record": _exp_data_point(
            "day_sequence",
            day=day_label,
            item_count=len(sequence_segments),
            sequence=sequence_segments if sequence_segments else None,
        ),
        "transitions": transitions,
        "items": item_rows,
    }


def _exp_build_option_evidence(
    option_key: str,
    days: list[dict],
    explain_data_by_occurrence: dict,
    option_meta: dict,
    planner_chain_of_thought: str,
    pref_tags: list[str],
) -> dict:
    meta = option_meta.get(option_key, {}) if isinstance(option_meta, dict) else {}
    day_evidence_by_label: dict[str, dict] = {}
    day_order: list[str] = []

    for day_index, day in enumerate(days):
        day_evidence = _exp_day_evidence(option_key, day_index, day, explain_data_by_occurrence)
        day_label = day_evidence["day"]
        day_evidence_by_label[day_label] = day_evidence
        day_order.append(day_label)

    return {
        "option": {
            "option_key": option_key,
            "option_meta": meta,
            "preference_tags": pref_tags,
            "option_record": _exp_data_point(
                "option_summary",
                option=option_key,
                label=meta.get("label") or option_key,
                style=meta.get("style") or "",
                prefs=pref_tags[:5],
            ),
            "planner_context": _exp_data_point(
                "planner_context",
                summary=_exp_compact_text(planner_chain_of_thought, 220),
            ) if planner_chain_of_thought else "",
        },
        "day_order": day_order,
        "days": day_evidence_by_label,
    }


def _exp_summary_language(state: dict) -> str:
    raw = str(_exp_first_present(state, "summary_language", "language", "ui_language", "locale", default="")).strip()
    if not raw:
        return "English"
    if raw.casefold().startswith("en"):
        return raw
    return "English"


def _exp_summary_prompt_input(
    option_key: str,
    itinerary: list[dict],
    option_evidence: dict,
) -> dict:
    prompt_days: list[dict] = []
    evidence_days = option_evidence.get("days", {}) or {}

    for day in itinerary:
        day_label = str(day.get("day") or "Day")
        day_evidence = evidence_days.get(day_label, {}) or {}
        day_items = day_evidence.get("items", []) or []
        prompt_days.append(
            {
                "day": day_label,
                "items": [
                    {
                        **item,
                        "reason_hints": _exp_reason_hints(item.get("top_why", []) or []),
                        "top_why_raw": item.get("top_why_raw", []) or [],
                    }
                    for item in day_items
                ],
                "day_evidence": day_evidence.get("record", ""),
                "transitions": day_evidence.get("transitions", []) or [],
                "day_reason_hints": _exp_collect_signal_labels(day_items, "English"),
            }
        )

    option_payload = option_evidence.get("option", {}) or {}
    return {
        "selected_option": option_key,
        "option_meta": option_payload.get("option_meta", {}),
        "preference_tags": option_payload.get("preference_tags", []),
        "option_evidence": option_payload.get("option_record", ""),
        "option_reason_hints": _exp_collect_signal_labels(
            [
                item
                for day in prompt_days
                for item in (day.get("items", []) or [])
            ],
            "English",
        ),
        "planner_context": option_payload.get("planner_context", ""),
        "days": prompt_days,
    }


def _exp_summary_text(summary: dict) -> str:
    overall = str(summary.get("overall_summary") or "").strip()
    day_summaries = summary.get("day_summaries", {}) or {}
    parts: list[str] = []
    if overall:
        parts.append(overall)
    for day_label, text in day_summaries.items():
        text = str(text or "").strip()
        if text:
            parts.append(f"{day_label}: {text}")
    return "\n\n".join(parts)


def _exp_summary_signal_label(reason: str, language: str) -> str:
    prefix = str(reason or "").split("[", 1)[0]
    label_map = {
        "pref_match": "preference fit",
        "content_profile": "content type",
        "quality_signal": "quality signal",
        "hours_signal": "opening hours fit",
        "route_signal": "route order",
        "budget_signal": "budget fit",
        "flight_signal": "flight fit",
        "time_pref_signal": "time preference",
        "hotel_signal": "hotel fit",
    }
    return label_map.get(prefix, prefix or "evidence")


def _exp_collect_signal_labels(items: list[dict], language: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        for reason in item.get("top_why", []) or []:
            label = _exp_summary_signal_label(reason, language)
            if label and label not in seen:
                ordered.append(label)
                seen.add(label)
    return ordered[:3]


def _exp_join_labels(labels: list[str], language: str) -> str:
    labels = [str(label).strip() for label in labels if str(label).strip()]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _exp_parse_data_point(value: str) -> tuple[str, dict[str, str]]:
    text = str(value or "").strip()
    if "[" not in text or not text.endswith("]"):
        return text, {}
    kind, raw_fields = text.split("[", 1)
    fields: dict[str, str] = {}
    for part in raw_fields[:-1].split("; "):
        if "=" not in part:
            continue
        key, field_value = part.split("=", 1)
        fields[key.strip()] = field_value.strip()
    return kind.strip(), fields


def _exp_match_fields(reasons: list[str], target_kind: str) -> dict[str, str]:
    for reason in reasons or []:
        kind, fields = _exp_parse_data_point(reason)
        if kind == target_kind:
            return fields
    return {}


def _exp_reason_hint(reason: str) -> str:
    kind, fields = _exp_parse_data_point(reason)
    if kind == "pref_match":
        pref = fields.get("pref", "")
        overlap = fields.get("overlap", "")
        if pref and overlap:
            return f"it matches your {pref} preference via {overlap}"
        return f"it matches your {pref} preference" if pref else "it matches a stated preference"
    if kind == "content_profile":
        item_type = fields.get("type") or fields.get("category") or "relevant stop type"
        icon = fields.get("icon", "")
        if icon == "restaurant":
            return f"it contributes a {item_type.lower()} meal stop"
        if icon == "activity":
            return f"it contributes a {item_type.lower()} activity"
        return f"it contributes a {item_type.lower()} stop"
    if kind == "quality_signal":
        rating = fields.get("rating", "")
        reviews = fields.get("reviews", "")
        if rating and reviews:
            return f"it has a strong quality signal ({rating}/5 from {reviews} reviews)"
        if rating:
            return f"it has a strong quality signal ({rating}/5)"
        return "it has a strong quality signal"
    if kind == "hours_signal":
        slot = fields.get("slot", "")
        return f"it is open for the {slot} slot" if slot else "it is open at the scheduled time"
    if kind == "route_signal":
        prev_km = _exp_numeric(fields.get("prev_km"))
        next_km = _exp_numeric(fields.get("next_km"))
        if prev_km is not None and next_km is not None:
            return f"it sits centrally between stops ({prev_km:.1f} km from the previous stop, {next_km:.1f} km to the next)"
        prev_present = fields.get("prev_present")
        next_present = fields.get("next_present")
        if prev_km is not None or next_km is not None:
            distance = prev_km if prev_km is not None else next_km
            direction = "from the previous stop" if prev_km is not None else "to the next stop"
            return f"it is {distance:.1f} km {direction}, keeping that transfer shorter"
        if prev_present == "1" or next_present == "1":
            return "it still acts as a route connector even without full coordinates"
        return "it still fits the route order for the day"
    if kind == "budget_signal":
        budget = fields.get("budget", "")
        return f"it stays aligned with the stated budget ({budget})" if budget else "it stays aligned with the stated budget"
    if kind == "flight_signal":
        segment = fields.get("segment", "flight")
        nonstop = fields.get("nonstop", "")
        if nonstop == "1":
            return f"it uses a nonstop {segment} flight"
        return f"it uses the selected {segment} flight"
    if kind == "time_pref_signal":
        pref = fields.get("pref", "")
        return f"it matches the preferred {pref} timing" if pref else "it matches the preferred timing"
    if kind == "hotel_signal":
        key = fields.get("key", "")
        if "checkout" in key:
            return "it anchors the checkout segment before the final transfer"
        return "it anchors the arrival and overnight segment"
    if kind:
        return f"it is supported by {kind.replace('_', ' ')}"
    return "it is supported by the collected evidence"


def _exp_reason_hints(reasons: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        hint = _exp_reason_hint(reason)
        if hint and hint not in seen:
            ordered.append(hint)
            seen.add(hint)
    return ordered[:3]


def _exp_strip_frontend_summary_artifacts(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"\b\d+\.\d+\s*/\s*5\b", "", cleaned)
    cleaned = re.sub(r"\(\s*[\d,]+\s+reviews?\s*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b[\d,]+\s+reviews?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b[a-z_]+(?:_signal|_match)\s*(?:\([^)]*\)|\[[^\]]*\])",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _exp_sanitize_summary_output(overall_summary: str, day_summaries: dict[str, str]) -> tuple[str, dict[str, str]]:
    sanitized_overall = _exp_strip_frontend_summary_artifacts(overall_summary)
    sanitized_days = {
        str(day_label): _exp_strip_frontend_summary_artifacts(text)
        for day_label, text in (day_summaries or {}).items()
    }
    return sanitized_overall, sanitized_days


def _exp_fallback_summary(prompt_payload: dict, language: str) -> dict:
    days = prompt_payload.get("days", []) or []
    option_meta = prompt_payload.get("option_meta", {}) or {}
    option_key = str(prompt_payload.get("selected_option") or "").strip() or "A"
    prefs = [str(item).strip() for item in (prompt_payload.get("preference_tags", []) or []) if str(item).strip()]
    pref_text = _exp_join_labels(prefs[:4], language)
    all_items: list[dict] = []
    for day in days:
        all_items.extend(day.get("items", []) or [])
    top_signals = _exp_collect_signal_labels(all_items, language)
    signal_text = _exp_join_labels(top_signals, language)
    label = str(option_meta.get("label") or f"Option {option_key}").strip()
    style = str(option_meta.get("style") or "").strip()

    overall = (
        f"{label} is arranged as a {style or 'selected'} route that keeps the trip centered on {pref_text or 'the stated trip brief'}."
        f" The ordering is mainly driven by {signal_text or 'route fit and timing feasibility'} across flights, meals, activities, and hotel stays."
    ).strip()

    day_summaries: dict[str, str] = {}
    for day in days:
        day_label = str(day.get("day") or "Day").strip()
        items = day.get("items", []) or []
        names = [str(item.get("name") or "").strip() for item in items if str(item.get("name") or "").strip()]
        day_signal_text = _exp_join_labels(_exp_collect_signal_labels(items, language), language)
        if names:
            sequence_text = " -> ".join(names[:4])
            day_summaries[day_label] = (
                f"{day_label} follows the sequence {sequence_text}."
                f" This day is mainly held together by {day_signal_text or 'timing and route fit'}."
            ).strip()
        else:
            day_summaries[day_label] = f"{day_label} is organized around timing and route fit."

    overall, day_summaries = _exp_sanitize_summary_output(overall, day_summaries)
    return {
        "overall_summary": overall,
        "day_summaries": day_summaries,
        "generation_mode": "fallback",
        "language": language,
        "source": prompt_payload,
    }


def _exp_normalize_day_summaries(day_summaries_raw: Any, day_order: list[str]) -> dict[str, str]:
    if isinstance(day_summaries_raw, dict):
        return {day_label: str(day_summaries_raw.get(day_label) or "").strip() for day_label in day_order}

    mapped: dict[str, str] = {}
    if isinstance(day_summaries_raw, list):
        for index, entry in enumerate(day_summaries_raw):
            if isinstance(entry, dict):
                day_label = str(
                    entry.get("day")
                    or entry.get("day_label")
                    or entry.get("label")
                    or (day_order[index] if index < len(day_order) else "")
                ).strip()
                text = str(
                    entry.get("summary")
                    or entry.get("text")
                    or entry.get("reason")
                    or entry.get("content")
                    or ""
                ).strip()
                if day_label:
                    mapped[day_label] = text
            else:
                if index < len(day_order):
                    mapped[day_order[index]] = str(entry or "").strip()
    return {day_label: mapped.get(day_label, "") for day_label in day_order}


def _exp_generate_summary(
    state: dict,
    option_key: str,
    itinerary: list[dict],
    option_evidence: dict,
) -> dict:
    language = _exp_summary_language(state)
    prompt_payload = _exp_summary_prompt_input(option_key, itinerary, option_evidence)
    day_order = option_evidence.get("day_order", []) or []

    system_prompt = (
        "You are a travel itinerary explainability summarizer. "
        "Convert structured evidence for ONE selected option into frontend-ready natural-language summaries. "
        f"Write in {language}. "
        "Use only the provided evidence. "
        "Do not mention internal labels like pref_match, route_signal, hours_signal, item_signal, day_sequence, or option_summary. "
        "Do not mention other options. "
        "Do not invent facts, places, or reasons not supported by the evidence. "
        "Avoid exact clock times and raw flight booking strings unless they are essential to the explanation. "
        "Use the raw evidence in top_why_raw when available. quality_signal contains rating/review strength, pref_match contains preference fit, and route_signal contains transfer distances. "
        "Explain same-day grouping and ordering first, then use quality or preference evidence to explain why a specific stop was chosen. "
        "Each meal explanation must be unique and should reference the specific restaurant's rating, cuisine type, or preference match when the evidence supports it. "
        "Do not use the same generic sentence for every lunch or dinner. "
        "Do not rely on 'bridge' language alone; explain why this specific stop works in that position. "
        "When transitions are provided, use them to explain why A leads to B and why B stays with C on the same day. "
        "You may use ratings and review counts from the evidence to inform your reasoning, but do NOT include any numeric ratings, scores, or review counts in the output text. "
        "Describe quality using words only, such as 'highly rated', 'well reviewed', or 'a local favourite'. "
        "Do not surface raw evidence snippets or internal payload text such as price_per_night_usd, budget_signal, quality_signal, pref_match, or flight_signal in the output. "
        "Keep the overall summary to 2-4 sentences and each day summary to 1-3 sentences. "
        "Return only valid JSON with keys overall_summary and day_summaries."
    )

    human_prompt = (
        "Summarize why this selected itinerary option is arranged this way overall and day by day. "
        "Focus on why the stops belong together on the same day and why each restaurant/activity was chosen in that specific position.\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False)}"
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
        payload = parsed if isinstance(parsed, dict) else {"day_summaries": parsed}
        overall_summary = ""
        for key in ("overall_summary", "overall", "option_summary", "summary"):
            if isinstance(payload, dict) and payload.get(key) not in (None, "", [], {}):
                overall_summary = str(payload.get(key) or "").strip()
                break
        day_summaries_raw = payload.get("day_summaries", payload.get("days", [])) if isinstance(payload, dict) else []
        day_summaries = _exp_normalize_day_summaries(day_summaries_raw, day_order)
        overall_summary, day_summaries = _exp_sanitize_summary_output(overall_summary, day_summaries)
        if not overall_summary and not any(day_summaries.values()):
            fallback = _exp_fallback_summary(prompt_payload, language)
            fallback["generation_mode"] = "llm_empty_fallback"
            return fallback
        return {
            "overall_summary": overall_summary,
            "day_summaries": day_summaries,
            "generation_mode": "llm",
            "language": language,
            "source": prompt_payload,
        }
    except Exception as exc:
        fallback = _exp_fallback_summary(prompt_payload, language)
        fallback["generation_mode"] = "llm_error_fallback"
        fallback["error"] = str(exc)
        return fallback


def explainability_agent(state: dict) -> dict:
    itineraries = _exp_collect_itineraries(state)
    option_key = _exp_selected_option(state, itineraries)
    days = itineraries.get(option_key, []) if option_key else []
    planner_chain_of_thought = state.get(
        "planner_chain_of_thought",
        state.get("chain_of_thought", ""),
    )
    tool_log = state.get("tool_log", []) or []
    destination = _exp_first_present(state, "destination", default="")
    option_meta = state.get("option_meta", {}) or {}

    if not option_key or not days:
        empty_summary = {
            "overall_summary": "",
            "day_summaries": {},
            "generation_mode": "llm",
            "language": _exp_summary_language(state),
        }
        empty_item_explanations = {"by_key": {}, "by_occurrence": {}}
        empty_evidence = {
            "option": {
                "option_key": option_key or "A",
                "option_meta": (option_meta.get(option_key or "A", {}) if isinstance(option_meta, dict) else {}),
                "preference_tags": _exp_collect_pref_tags(state),
                "option_record": "",
                "planner_context": "",
            },
            "day_order": [],
            "days": {},
            "items": {"by_key": {}, "by_occurrence": {}},
            "summary_generation": {
                "mode": empty_summary["generation_mode"],
                "language": empty_summary["language"],
                "source": {},
            },
        }
        return {
            "explain_option": option_key or "A",
            "selected_option": option_key or "A",
            "summary": empty_summary,
            "item_explanations": empty_item_explanations,
            "evidence": empty_evidence,
            "explain_data": {},
            "explain_data_by_occurrence": {},
            "planner_chain_of_thought": planner_chain_of_thought,
            "explainability_chain_of_thought": "",
            "combined_chain_of_thought": planner_chain_of_thought,
            "chain_of_thought": planner_chain_of_thought,
            "agent_steps": _build_agent_steps(tool_log, destination, option_key or "A"),
        }

    place_lookup = _exp_build_place_lookup(state)
    hotel_lookup = _exp_build_hotel_lookup(state)
    hotel_review_blob = str((state.get("research", {}) or {}).get("hotel_reviews", "") or "")
    pref_tags = _exp_collect_pref_tags(state)

    explain_data: dict[str, dict] = {}
    explain_data_by_occurrence: dict[str, dict] = {}

    for day_index, day in enumerate(days):
        items = day.get("items", []) or []
        day_label = str(day.get("day") or f"Day {day_index + 1}")
        for item_index, item in enumerate(items):
            explanation = _exp_item_explanation(
                state=state,
                option_key=option_key,
                day_label=day_label,
                day_index=day_index,
                item_index=item_index,
                items=items,
                item=item,
                place_lookup=place_lookup,
                hotel_lookup=hotel_lookup,
                hotel_review_blob=hotel_review_blob,
                pref_tags=pref_tags,
            )
            occurrence_id = explanation["occurrence_id"]
            explain_data_by_occurrence[occurrence_id] = explanation

            item_key = str(item.get("key") or occurrence_id)
            if item_key not in explain_data:
                explain_data[item_key] = explanation

    option_evidence = _exp_build_option_evidence(
        option_key=option_key,
        days=days,
        explain_data_by_occurrence=explain_data_by_occurrence,
        option_meta=option_meta,
        planner_chain_of_thought=planner_chain_of_thought,
        pref_tags=pref_tags,
    )
    summary = _exp_generate_summary(
        state=state,
        option_key=option_key,
        itinerary=days,
        option_evidence=option_evidence,
    )
    explainability_chain_of_thought = _exp_summary_text(summary)
    combined_chain_of_thought = "\n\n".join(
        part for part in (planner_chain_of_thought, explainability_chain_of_thought) if part
    )
    item_explanations = {
        "by_key": {item_key: _exp_item_display(explanation) for item_key, explanation in explain_data.items()},
        "by_occurrence": {
            occurrence_id: _exp_item_display(explanation)
            for occurrence_id, explanation in explain_data_by_occurrence.items()
        },
    }
    evidence = {
        **option_evidence,
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

    print(f"[Agent6] Deterministic explainability generated for option {option_key} with {len(explain_data_by_occurrence)} items")

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
        "explainability_chain_of_thought": explainability_chain_of_thought,
        "combined_chain_of_thought": combined_chain_of_thought,
        "chain_of_thought": combined_chain_of_thought,
        "agent_steps": _build_agent_steps(tool_log, destination, option_key),
    }


def _build_agent_steps(tool_log: list, dest: str, option_key: str = "A") -> list:
    tool_count = len([entry for entry in tool_log if str(entry).startswith("[")])
    return [
        {"icon": "research", "name": "Research Agent", "detail": f"Reused {tool_count} logged research/planner steps for {dest}."},
        {"icon": "plan", "name": "Planner Agent", "detail": f"Explaining the final itinerary for option {option_key}."},
        {"icon": "why", "name": "Explainability Agent", "detail": "Generated deterministic reasons for route, timing, and fit."},
    ]
