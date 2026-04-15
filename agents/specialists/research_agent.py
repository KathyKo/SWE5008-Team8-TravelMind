"""
Research-only pipeline for itinerary planning.

This module merges the search/data-collection responsibilities that were
previously split between the legacy planner flow and `research_agent.py`.
It intentionally does not schedule itineraries.
"""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from ..llm_config import OPENAI_MODEL


_DEST_TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "best",
    "top",
    "city",
    "near",
    "around",
    "restaurant",
    "restaurants",
    "attraction",
    "attractions",
    "museum",
    "museums",
    "garden",
    "gardens",
}


def _places_debug_enabled() -> bool:
    raw = os.getenv("GOOGLE_PLACES_DEBUG_SAVE", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _places_debug_dir() -> Path:
    root = Path(os.getenv("GOOGLE_PLACES_DEBUG_DIR", "debug/google_places_raw"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip())
    return cleaned.strip("_").lower()[:80] or "item"


def _candidate_id(item: dict) -> str:
    return str(
        item.get("place_id")
        or item.get("id")
        or item.get("resource_name")
        or item.get("name")
        or ""
    ).strip()


def _merge_place_candidate(base: dict, incoming: dict) -> dict:
    for field in (
        "price",
        "description",
        "address",
        "lat",
        "lng",
        "hours",
        "weekday_descriptions",
        "open_now",
        "search_query",
        "type",
        "category",
        "rating",
        "reviews",
        "google_maps_uri",
    ):
        if base.get(field) in (None, "", []) and incoming.get(field) not in (None, "", []):
            base[field] = incoming[field]
    return base


def _dedupe_place_candidates(candidates: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    ordered: list[dict] = []
    for item in candidates:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        name_l = name.lower()
        existing = next(
            (existing_name for existing_name in seen if name_l in existing_name or existing_name in name_l),
            None,
        )
        if existing:
            _merge_place_candidate(seen[existing], item)
            continue
        clone = dict(item)
        seen[name_l] = clone
        ordered.append(clone)
    return ordered


def _save_places_group_debug(
    *,
    place_kind: str,
    city: str,
    preferences_text: str,
    queries: list[str],
    max_output: int,
    min_keep: int,
    raw_candidates: list[dict],
    llm_filtered_candidates: list[dict],
    removed_ids: list[str],
    skipped_llm_filter: bool,
) -> None:
    if not _places_debug_enabled():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    payload = {
        "saved_at": ts,
        "place_kind": place_kind,
        "city": city,
        "preferences": preferences_text,
        "queries": queries,
        "max_output": max_output,
        "min_keep_after_filter": min_keep,
        "skipped_llm_filter": skipped_llm_filter,
        "removed_ids": removed_ids,
        "counts": {
            "raw_candidates": len(raw_candidates),
            "llm_filtered_candidates": len(llm_filtered_candidates),
        },
        "raw_candidates": raw_candidates,
        "llm_filtered_candidates": llm_filtered_candidates,
    }
    out_path = _places_debug_dir() / f"{ts}_{place_kind}_{_slugify(city)}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Research1] saved grouped places debug -> {out_path}")


def _llm_filter_places(
    *,
    place_kind: str,
    candidates: list[dict],
    queries: list[str],
    preferences_text: str,
    city: str,
    max_output: int,
    tool_log: list[str],
) -> tuple[list[dict], list[str], bool]:
    raw_count = len(candidates)
    min_keep = math.ceil(max_output * 0.8)
    if raw_count < min_keep:
        tool_log.append(
            f"[places_llm_filter {place_kind} skipped raw={raw_count} min_keep={min_keep} max_output={max_output}]"
        )
        return candidates, [], True

    max_remove = max(0, raw_count - min_keep)
    if max_remove <= 0:
        tool_log.append(f"[places_llm_filter {place_kind} skipped raw={raw_count} nothing_to_remove]")
        return candidates, [], True

    numbered = []
    for idx, item in enumerate(candidates, start=1):
        numbered.append(
            {
                "rank": idx,
                "id": _candidate_id(item),
                "name": item.get("name"),
                "category": item.get("category") or item.get("type"),
                "rating": item.get("rating"),
                "address": item.get("address"),
                "description": item.get("description"),
                "query": item.get("search_query"),
            }
        )

    from collections import Counter

    category_counts = Counter(
        (c.get("category") or c.get("type") or "unknown").lower().strip()
        for c in candidates
    )
    dominant = [cat for cat, cnt in category_counts.items() if cnt / max(raw_count, 1) > 0.35]

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    prompt = (
        f"You are filtering {place_kind} candidates for a travel planner.\n"
        f"Destination context: {city}\n"
        f"User preferences: {preferences_text}\n"
        f"Search queries used:\n"
        + "\n".join(f"- {q}" for q in queries)
        + "\n\n"
        + 'Return ONLY JSON: {"remove_ids":["id1","id2"]}.\n'
        + "Rules:\n"
        + "- Prefer candidates that fit the queries and user preferences.\n"
        + f"- Keep category variety. Over-represented categories now: {dominant or 'none'}.\n"
        + f"- Do not remove more than {max_remove} candidates.\n"
        + "- Only use IDs that appear in the candidate list.\n\n"
        + f"Candidates:\n{json.dumps(numbered, ensure_ascii=False, indent=2)}"
    )

    remove_ids: list[str] = []
    try:
        resp = llm.invoke([SystemMessage(content=prompt)])
        match = re.search(r"\{.*\}", getattr(resp, "content", ""), re.DOTALL)
        if match:
            data = json.loads(match.group())
            if isinstance(data, dict) and isinstance(data.get("remove_ids"), list):
                remove_ids = [str(x).strip() for x in data["remove_ids"] if str(x).strip()]
    except Exception as exc:
        tool_log.append(f"[places_llm_filter {place_kind} error: {exc}]")
        return candidates, [], True

    valid_ids = {_candidate_id(item) for item in candidates if _candidate_id(item)}
    bounded_remove_ids: list[str] = []
    seen_ids: set[str] = set()
    for candidate_id in remove_ids:
        if candidate_id in valid_ids and candidate_id not in seen_ids:
            bounded_remove_ids.append(candidate_id)
            seen_ids.add(candidate_id)
        if len(bounded_remove_ids) >= max_remove:
            break

    filtered = [
        item for item in candidates if _candidate_id(item) not in set(bounded_remove_ids)
    ]
    tool_log.append(
        f"[places_llm_filter {place_kind} raw={raw_count} removed={len(bounded_remove_ids)} "
        f"kept={len(filtered)} min_keep={min_keep} max_output={max_output}]"
    )
    return filtered, bounded_remove_ids, False


_FOOD_TYPE_KEYWORDS = {
    "restaurant",
    "food",
    "hawker",
    "cafe",
    "coffee",
    "bakery",
    "bar",
    "eatery",
    "bistro",
    "diner",
    "canteen",
    "food court",
    "foodcourt",
    "buffet",
    "catering",
    "market stall",
    "sushi",
    "grill",
    "pub",
    "lounge",
}

_INVALID_ATTRACTION_KEYWORDS = {
    "pte ltd",
    "corp",
    "corporation",
    "inc",
    "co.",
    "group pte",
    "office",
    "agency",
    "consultancy",
    "private limited",
    "holding",
    "management",
    "services",
    "logistics",
    "shipping",
    "construction",
    "industrial",
    "associates",
    "partners",
    "hq",
    "headquarters",
}

_FX_TO_SGD = {
    "usd": 1.35,
    "us$": 1.35,
    "aud": 0.91,
    "au$": 0.91,
    "eur": 1.45,
    "gbp": 1.71,
    "hkd": 0.17,
    "jpy": 0.009,
    "twd": 0.042,
    "myr": 0.30,
}


def _is_food_place(item: dict) -> bool:
    combined = " ".join(
        [
            item.get("type", "") or "",
            item.get("category", "") or "",
            item.get("name", "") or "",
        ]
    ).lower()
    return any(keyword in combined for keyword in _FOOD_TYPE_KEYWORDS)


def _is_corporate_noise(item: dict) -> bool:
    name = (item.get("name") or item.get("title") or "").lower()
    if any(keyword in name for keyword in _INVALID_ATTRACTION_KEYWORDS):
        return True
    return name.strip().isdigit()


def _strip_dollar(value) -> str:
    return str(value or "").strip().lstrip("$")


def _to_sgd(price_str: str) -> str:
    if not price_str:
        return price_str

    original = str(price_str).strip()
    lowered = original.lower()
    if lowered in {"free", "tbc", "$", "$$", "$$$", "$$$$"}:
        return original
    if lowered.startswith("sgd "):
        return original

    cleaned = re.sub(r"(?i)^from\s+", "", original)
    cleaned = re.sub(r"(?i)\s+per\s+(adult|person|pax).*$", "", cleaned).strip()

    if cleaned.upper().startswith("S$"):
        try:
            return f"SGD {float(cleaned[2:].strip().replace(',', '')):g}"
        except ValueError:
            return original

    if cleaned.startswith("$") and not cleaned.startswith("$$"):
        num_part = cleaned[1:].strip().replace(",", "")
        if "-" in num_part:
            pieces = re.split(r"-", num_part)
            nums = []
            for piece in pieces:
                try:
                    nums.append(f"{float(piece.strip().lstrip('$')):g}")
                except ValueError:
                    pass
            return f"SGD {' - '.join(nums)}" if nums else original
        try:
            return f"SGD {float(num_part):g}"
        except ValueError:
            return original

    yen_match = re.match(r"^JPY\s*([\d,]+(?:\.\d+)?)$", cleaned, re.IGNORECASE)
    if yen_match:
        return f"SGD {round(float(yen_match.group(1).replace(',', '')) * 0.009)}"

    for prefix, rate in _FX_TO_SGD.items():
        if cleaned.lower().startswith(prefix):
            try:
                numeric = cleaned[len(prefix) :].strip().lstrip("$").replace(",", "")
                return f"SGD {round(float(numeric) * rate)}"
            except ValueError:
                return original

    return original


def _safe_price(raw_price: str) -> str:
    if not raw_price:
        return raw_price
    converted = _to_sgd(str(raw_price))
    match = re.search(r"SGD\s*([\d.]+)", converted)
    if match:
        try:
            if float(match.group(1)) > 500:
                return "TBC"
        except Exception:
            pass
    return converted


def _duration_days(duration) -> int:
    match = re.search(r"\d+", str(duration or "").strip())
    return max(1, int(match.group())) if match else 1


def _detect_dietary_style(*texts: str) -> str:
    combined = " ".join(str(text or "") for text in texts).lower()
    if any(token in combined for token in ("vegan", "plant-based", "plant based")):
        return "vegan"
    if any(token in combined for token in ("vegetarian", "veg-friendly", "veg friendly")):
        return "vegetarian"
    if "halal" in combined:
        return "halal"
    if "kosher" in combined:
        return "kosher"
    if "gluten-free" in combined or "gluten free" in combined:
        return "gluten-free"
    return ""


def _infer_query_city(queries: list[str], fallback_city: str) -> str:
    for query in queries:
        match = re.search(
            r"\bin\s+([A-Za-z][A-Za-z\s,\-]+?)(?:\s+(?:for|with|near|around)\b|$)",
            str(query or "").strip(),
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" ,")
    return fallback_city


def _safe_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _place_latlng(item: dict) -> tuple[float | None, float | None]:
    return _safe_float(item.get("lat")), _safe_float(item.get("lng"))


def _destination_tokens(*parts: str) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        for token in re.findall(r"[A-Za-z]{3,}", str(part or "").lower()):
            if token not in _DEST_TOKEN_STOPWORDS:
                tokens.add(token)
    return tokens


def _looks_like_destination_text(item: dict, dest_tokens: set[str]) -> bool:
    haystack = " ".join(
        [
            str(item.get("name") or ""),
            str(item.get("address") or ""),
            str(item.get("search_query") or ""),
        ]
    ).lower()
    return any(token in haystack for token in dest_tokens)


def _dominant_geo_center(
    candidates: list[dict],
    *,
    cluster_radius_km: float = 80.0,
) -> tuple[float, float] | None:
    points: list[tuple[float, float]] = []
    for item in candidates:
        lat, lng = _place_latlng(item)
        if lat is None or lng is None:
            continue
        points.append((lat, lng))
    if not points:
        return None

    best_point = points[0]
    best_count = 0
    for lat, lng in points:
        count = sum(
            1
            for lat2, lng2 in points
            if _haversine_km(lat, lng, lat2, lng2) <= cluster_radius_km
        )
        if count > best_count:
            best_point = (lat, lng)
            best_count = count
    return best_point


def _filter_candidates_to_destination_cluster(
    candidates: list[dict],
    *,
    place_kind: str,
    target_city: str,
    destination: str,
    tool_log: list[str],
    reference_candidates: list[dict] | None = None,
    cluster_radius_km: float = 80.0,
) -> list[dict]:
    if not candidates:
        return candidates

    dest_tokens = _destination_tokens(target_city, destination)
    center = _dominant_geo_center(reference_candidates or candidates, cluster_radius_km=cluster_radius_km)
    filtered: list[dict] = []
    removed: list[str] = []

    for item in candidates:
        name = str(item.get("name") or "").strip()
        lat, lng = _place_latlng(item)
        keep = True

        if center and lat is not None and lng is not None:
            distance = _haversine_km(lat, lng, center[0], center[1])
            keep = distance <= cluster_radius_km
        elif lat is None or lng is None:
            keep = _looks_like_destination_text(item, dest_tokens)
        elif center is None:
            keep = _looks_like_destination_text(item, dest_tokens)

        if keep:
            filtered.append(item)
        else:
            removed.append(name or _candidate_id(item) or "unknown")

    if removed:
        sample = ", ".join(removed[:5])
        tool_log.append(
            f"[destination_geofence {place_kind} city={target_city} removed={len(removed)} sample={sample}]"
        )
    return filtered


def _allocate_query_plan(
    base_queries: list[str],
    *,
    total_quota: int,
    max_per_query: int,
    fallback_query_builder,
) -> list[tuple[str, int]]:
    queries = [q.strip() for q in base_queries if str(q).strip()]
    remaining_quota = max(0, total_quota)

    while queries and remaining_quota > len(queries) * max_per_query:
        queries.append(fallback_query_builder(len(queries)))

    if not queries:
        queries = [fallback_query_builder(0)]
        while remaining_quota > len(queries) * max_per_query:
            queries.append(fallback_query_builder(len(queries)))

    query_count = max(1, len(queries))
    base = remaining_quota // query_count
    remainder = remaining_quota % query_count

    plan: list[tuple[str, int]] = []
    for idx, query in enumerate(queries):
        quota = base + (1 if idx < remainder else 0)
        quota = max(1, min(max_per_query, quota))
        plan.append((query, quota))
    return plan


def _normalize_trip_state(state: dict) -> dict:
    hard = state.get("hard_constraints", {}) or {}
    soft = state.get("soft_preferences", {}) or {}
    budget_raw = state.get("budget")
    budget_struct = hard.get("budget", {}) if isinstance(hard.get("budget"), dict) else {}

    start_date = hard.get("start_date", "")
    end_date = hard.get("end_date", "")
    dates = state.get("dates") or (
        f"{start_date} to {end_date}" if start_date and end_date else start_date or end_date
    )

    duration = state.get("duration")
    if not duration and start_date and end_date:
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
            duration = f"{(end - start).days + 1} days"
        except Exception:
            duration = ""

    if isinstance(budget_raw, dict):
        amount = budget_raw.get("amount")
        currency = budget_raw.get("currency", "")
        flexibility = budget_raw.get("flexibility", "")
        budget = " ".join(str(x) for x in (currency, amount, flexibility) if x).strip()
    elif budget_raw:
        budget = str(budget_raw)
    else:
        amount = budget_struct.get("amount")
        currency = budget_struct.get("currency", "")
        flexibility = budget_struct.get("flexibility", "")
        budget = " ".join(str(x) for x in (currency, amount, flexibility) if x).strip()

    preferences = state.get("preferences", "")
    if not preferences:
        parts = []
        parts.extend(hard.get("requirements", []) or [])
        parts.extend(soft.get("interest_tags", []) or [])
        for scalar in (soft.get("travel_style"), soft.get("vibe")):
            if scalar:
                parts.append(str(scalar))
        preferences = ", ".join(str(x) for x in parts if x)

    return {
        **state,
        "origin": state.get("origin") or hard.get("origin"),
        "destination": state.get("destination") or hard.get("destination"),
        "dates": dates,
        "duration": duration,
        "budget": budget,
        "preferences": preferences,
        "search_queries": state.get("search_queries", []) or [],
        "hard_constraints": hard,
        "soft_preferences": soft,
    }


def _build_flight_display(flight: dict) -> str:
    display = str(flight.get("display") or "").strip()
    if display:
        return display
    parts = [
        f"{flight.get('airline', '')} {flight.get('flight_number', '')}".strip(),
        f"{flight.get('departure_airport', '')} -> {flight.get('arrival_airport', '')}".strip(),
        f"dep {flight.get('departure_time', '')} -> arr {flight.get('arrival_time', '')}".strip(),
        f"{flight.get('duration_min', '')} min".strip(),
        str(flight.get("travel_class") or "").strip(),
        f"USD {flight.get('price_usd', '')}".strip(),
    ]
    return " | ".join(part for part in parts if part and part != "USD")


def _flight_stop_rank(flight: dict) -> int:
    display = str(flight.get("display") or "").lower()
    if "nonstop" in display or "direct" in display:
        return 0
    match = re.search(r"(\d+)\s*stop", display)
    if match:
        return int(match.group(1))
    return 9


def _time_pref_penalty(flight: dict, time_pref: str) -> int:
    pref = str(time_pref or "").lower().strip()
    departure = str(flight.get("departure_time") or "")
    try:
        hour = int(departure[-5:-3]) if len(departure) >= 5 else None
    except Exception:
        hour = None
    if hour is None or not pref:
        return 0
    if pref == "morning":
        return 0 if 5 <= hour < 12 else 1
    if pref == "afternoon":
        return 0 if 12 <= hour < 18 else 1
    if pref == "evening":
        return 0 if 18 <= hour < 24 else 1
    return 0


def _flight_sort_key(flight: dict, time_pref: str) -> tuple:
    try:
        price = float(flight.get("price_usd"))
    except Exception:
        price = 999999
    try:
        duration = int(flight.get("duration_min"))
    except Exception:
        duration = 999999
    return (
        _flight_stop_rank(flight),
        _time_pref_penalty(flight, time_pref),
        price,
        duration,
        str(flight.get("departure_time") or ""),
    )


def _build_hotel_display(hotel: dict) -> str:
    display = str(hotel.get("display") or "").strip()
    if display:
        return display
    parts = [str(hotel.get("name") or "").strip()]
    hotel_class = str(hotel.get("hotel_class") or "").strip()
    rating = hotel.get("rating")
    price = hotel.get("price_per_night_usd")
    if hotel_class:
        parts.append(hotel_class)
    if rating not in (None, ""):
        parts.append(f"rating {rating}")
    if price not in (None, ""):
        parts.append(f"USD {price}/night")
    return " | ".join(part for part in parts if part)


def research_agent_1(state: dict, tools: dict | None = None) -> dict:
    state = _normalize_trip_state(state)

    dest = state.get("destination")
    origin = state.get("origin")
    dates = state.get("dates")
    budget = state.get("budget")
    preferences = state.get("preferences", "")
    duration = state.get("duration")
    user_profile = state.get("user_profile", {})
    outbound_time_pref = state.get("outbound_time_pref", "")
    return_time_pref = state.get("return_time_pref", "")
    search_queries = state.get("search_queries", []) or []

    date_parts = [d.strip() for d in dates.split(" to ")] if " to " in str(dates) else [str(dates).strip()]
    outbound_date = date_parts[0]
    return_date = date_parts[-1]

    tools = tools or {}
    search_weather = tools.get("search_weather")
    search_flights = tools.get("search_flights")
    search_hotels = tools.get("search_hotels")
    web_search = tools.get("web_search")

    profile_prefs = ", ".join(user_profile.get("prefs", [])) if user_profile else ""
    user_prefs = ", ".join(filter(None, [preferences, profile_prefs])) or "general traveller"

    research: dict = {}
    tool_log: list[str] = []

    try:
        from tools.serp_search import (
            _flight_options_outbound,
            _flight_options_return,
            _hotel_options,
            _hotel_tokens,
            _place_data_ids,
        )

        _hotel_tokens.clear()
        _place_data_ids.clear()
        _flight_options_outbound.clear()
        _flight_options_return.clear()
        _hotel_options.clear()
    except Exception:
        pass

    from tools.google_places_search import google_places_search_bundle
    from tools.serp_search import serp_local_structured, serp_tripadvisor_structured

    typed_prefs = [p.strip() for p in preferences.split(",") if p.strip()]
    attraction_queries = [
        str(q.get("query", "")).strip()
        for q in search_queries
        if isinstance(q, dict)
        and str(q.get("type", "")).strip().lower() == "rag_attraction"
        and str(q.get("query", "")).strip()
    ]
    restaurant_queries = [
        str(q.get("query", "")).strip()
        for q in search_queries
        if isinstance(q, dict)
        and str(q.get("type", "")).strip().lower() == "rag_restaurant"
        and str(q.get("query", "")).strip()
    ]

    trip_days = _duration_days(duration)
    target_city = _infer_query_city(attraction_queries + restaurant_queries, dest)
    dietary_style = _detect_dietary_style(
        preferences,
        profile_prefs,
        " ".join((state.get("hard_constraints", {}) or {}).get("requirements", []) or []),
        " ".join((state.get("soft_preferences", {}) or {}).get("interest_tags", []) or []),
    )

    attraction_total_quota = trip_days * 12
    restaurant_total_quota = trip_days * 8
    restaurant_floor = max(8, min(14, trip_days * 2))

    def _fallback_attraction_query(_: int) -> str:
        return f"Best attractions in {target_city}"

    def _fallback_restaurant_query(idx: int) -> str:
        candidates = []
        if dietary_style:
            candidates.append(f"Best {dietary_style} restaurants in {target_city}")
        candidates.extend(
            [
                f"Best local restaurants in {target_city}",
                f"Best traditional restaurants in {target_city}",
                f"Best street food and local restaurants in {target_city}",
                f"Best restaurants in {target_city}",
            ]
        )
        return candidates[idx % len(candidates)]

    attraction_query_plan = _allocate_query_plan(
        attraction_queries,
        total_quota=attraction_total_quota,
        max_per_query=12,
        fallback_query_builder=_fallback_attraction_query,
    )
    restaurant_query_plan = _allocate_query_plan(
        restaurant_queries,
        total_quota=restaurant_total_quota,
        max_per_query=8,
        fallback_query_builder=_fallback_restaurant_query,
    )
    use_explicit_place_queries = bool(attraction_queries or restaurant_queries)

    def _fetch_flights_out():
        return (
            search_flights(origin, dest, outbound_date, outbound_time_pref, "outbound")
            if search_flights
            else None
        )

    def _fetch_flights_ret():
        return (
            search_flights(dest, origin, return_date, return_time_pref, "return")
            if search_flights
            else None
        )

    def _fetch_hotels():
        return search_hotels(dest, budget, dates) if search_hotels else None

    def _fetch_maps_att():
        return [] if use_explicit_place_queries else serp_local_structured(dest, "things to do attractions")

    def _fetch_maps_rest():
        return [] if use_explicit_place_queries else serp_local_structured(dest, "restaurants")

    def _fetch_places_att():
        if use_explicit_place_queries:
            bundles = []
            for query, quota in attraction_query_plan:
                bundle = google_places_search_bundle(query, "", place_kind="attraction", max_results=quota)
                for item in bundle.get("selected_places", []):
                    item["search_query"] = query
                bundles.append(bundle)
            return bundles
        return serp_tripadvisor_structured("things to do attractions", dest)

    def _fetch_places_rest():
        if use_explicit_place_queries:
            bundles = []
            for query, quota in restaurant_query_plan:
                bundle = google_places_search_bundle(query, "", place_kind="restaurant", max_results=quota)
                for item in bundle.get("selected_places", []):
                    item["search_query"] = query
                bundles.append(bundle)
            return bundles
        return serp_tripadvisor_structured("restaurants", dest)

    def _fetch_weather():
        if not search_weather:
            return ""
        try:
            return search_weather(dest, dates) or ""
        except Exception:
            return ""

    def _fetch_web():
        if not web_search:
            return ""
        try:
            return web_search(f"{dest} travel guide {dates}") or ""
        except Exception:
            return ""

    def _fetch_pref(pref: str):
        if use_explicit_place_queries:
            return []
        try:
            return (serp_local_structured(dest, pref) or []) + (serp_tripadvisor_structured(pref, dest) or [])
        except Exception:
            return []

    tasks = {
        "flights_out": _fetch_flights_out,
        "flights_ret": _fetch_flights_ret,
        "hotels": _fetch_hotels,
        "places_att": _fetch_places_att,
        "places_rest": _fetch_places_rest,
        "weather": _fetch_weather,
        "web_general": _fetch_web,
    }
    if not use_explicit_place_queries:
        tasks["maps_att"] = _fetch_maps_att
        tasks["maps_rest"] = _fetch_maps_rest
        for idx, pref in enumerate(typed_prefs[:2]):
            tasks[f"pref_{idx}"] = lambda p=pref: _fetch_pref(p)

    results_map: dict = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        future_to_key = {pool.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results_map[key] = future.result()
            except Exception as exc:
                results_map[key] = None
                tool_log.append(f"[{key} error: {exc}]")

    research["maps_attractions"] = _filter_candidates_to_destination_cluster(
        results_map.get("maps_att") or [],
        place_kind="attractions",
        target_city=target_city,
        destination=dest,
        tool_log=tool_log,
    )
    research["maps_restaurants"] = _filter_candidates_to_destination_cluster(
        results_map.get("maps_rest") or [],
        place_kind="restaurants",
        target_city=target_city,
        destination=dest,
        tool_log=tool_log,
        reference_candidates=(results_map.get("maps_att") or []) + (results_map.get("maps_rest") or []),
    )

    attraction_bundles = results_map.get("places_att") or []
    restaurant_bundles = results_map.get("places_rest") or []
    if use_explicit_place_queries:
        raw_attraction_candidates: list[dict] = []
        raw_restaurant_candidates: list[dict] = []
        for bundle in attraction_bundles:
            query = bundle.get("original_query", "")
            for item in bundle.get("selected_places", []):
                item["search_query"] = item.get("search_query") or query
                raw_attraction_candidates.append(item)
        for bundle in restaurant_bundles:
            query = bundle.get("original_query", "")
            for item in bundle.get("selected_places", []):
                item["search_query"] = item.get("search_query") or query
                raw_restaurant_candidates.append(item)
        raw_attraction_candidates = _filter_candidates_to_destination_cluster(
            raw_attraction_candidates,
            place_kind="attractions",
            target_city=target_city,
            destination=dest,
            tool_log=tool_log,
        )
        raw_restaurant_candidates = _filter_candidates_to_destination_cluster(
            raw_restaurant_candidates,
            place_kind="restaurants",
            target_city=target_city,
            destination=dest,
            tool_log=tool_log,
            reference_candidates=raw_attraction_candidates or raw_restaurant_candidates,
        )
        unique_restaurant_candidates = _dedupe_place_candidates(raw_restaurant_candidates)

        filtered_attraction_candidates, removed_att_ids, skipped_att_filter = _llm_filter_places(
            place_kind="attractions",
            candidates=raw_attraction_candidates,
            queries=[q for q, _ in attraction_query_plan],
            preferences_text=user_prefs,
            city=target_city,
            max_output=attraction_total_quota,
            tool_log=tool_log,
        )
        if len(unique_restaurant_candidates) <= restaurant_floor + 2:
            filtered_restaurant_candidates = unique_restaurant_candidates
            removed_rest_ids = []
            skipped_rest_filter = True
            tool_log.append(
                f"[places_llm_filter restaurants skipped unique={len(unique_restaurant_candidates)} "
                f"restaurant_floor={restaurant_floor}]"
            )
        else:
            filtered_restaurant_candidates, removed_rest_ids, skipped_rest_filter = _llm_filter_places(
                place_kind="restaurants",
                candidates=unique_restaurant_candidates,
                queries=[q for q, _ in restaurant_query_plan],
                preferences_text=user_prefs,
                city=target_city,
                max_output=restaurant_total_quota,
                tool_log=tool_log,
            )

        _save_places_group_debug(
            place_kind="attractions",
            city=target_city,
            preferences_text=user_prefs,
            queries=[q for q, _ in attraction_query_plan],
            max_output=attraction_total_quota,
            min_keep=math.ceil(attraction_total_quota * 0.8),
            raw_candidates=raw_attraction_candidates,
            llm_filtered_candidates=filtered_attraction_candidates,
            removed_ids=removed_att_ids,
            skipped_llm_filter=skipped_att_filter,
        )
        _save_places_group_debug(
            place_kind="restaurants",
            city=target_city,
            preferences_text=user_prefs,
            queries=[q for q, _ in restaurant_query_plan],
            max_output=restaurant_total_quota,
            min_keep=math.ceil(restaurant_total_quota * 0.8),
            raw_candidates=raw_restaurant_candidates,
            llm_filtered_candidates=filtered_restaurant_candidates,
            removed_ids=removed_rest_ids,
            skipped_llm_filter=skipped_rest_filter,
        )
        research["ta_attractions"] = filtered_attraction_candidates
        research["ta_restaurants"] = filtered_restaurant_candidates
    else:
        research["ta_attractions"] = _filter_candidates_to_destination_cluster(
            attraction_bundles or [],
            place_kind="attractions",
            target_city=target_city,
            destination=dest,
            tool_log=tool_log,
        )
        research["ta_restaurants"] = _filter_candidates_to_destination_cluster(
            restaurant_bundles or [],
            place_kind="restaurants",
            target_city=target_city,
            destination=dest,
            tool_log=tool_log,
            reference_candidates=(attraction_bundles or []) + (restaurant_bundles or []),
        )

    research["weather"] = results_map.get("weather") or ""
    research["web_general"] = results_map.get("web_general") or ""

    pref_structured: list[dict] = []
    if not use_explicit_place_queries:
        for idx in range(len(typed_prefs[:2])):
            pref_structured.extend(results_map.get(f"pref_{idx}") or [])
    research["pref_structured"] = _filter_candidates_to_destination_cluster(
        pref_structured,
        place_kind="attractions",
        target_city=target_city,
        destination=dest,
        tool_log=tool_log,
    )

    tool_log.extend(
        [
            f"[search_flights({origin}->{dest})]",
            f"[search_flights({dest}->{origin}) return]",
            f"[search_hotels({dest})]",
            f"[google_places_attractions({dest if not attraction_queries else 'explicit_queries'})]",
            f"[google_places_restaurants({dest if not restaurant_queries else 'explicit_queries'})]",
            f"[search_weather({dest})]",
        ]
    )
    if not use_explicit_place_queries:
        tool_log.insert(3, f"[maps_structured({dest})]")
        tool_log.insert(4, f"[maps_restaurants({dest})]")
    if use_explicit_place_queries:
        tool_log.append(
            f"[places_quota attractions={attraction_total_quota} restaurants={restaurant_total_quota} "
            f"days={trip_days} city={target_city} dietary={dietary_style or 'none'}]"
        )
        for query, quota in attraction_query_plan:
            tool_log.append(f"[places_query_attraction({query}) quota={quota}]")
        for query, quota in restaurant_query_plan:
            tool_log.append(f"[places_query_restaurant({query}) quota={quota}]")
    else:
        for pref in typed_prefs[:2]:
            tool_log.append(f"[pref_search({pref}|{dest})]")

    try:
        from tools.serp_search import _flight_options_outbound, _flight_options_return, _hotel_options

        flights_out = sorted(
            list(_flight_options_outbound),
            key=lambda flight: _flight_sort_key(flight, outbound_time_pref),
        )[:5]
        flights_ret = sorted(
            list(_flight_options_return),
            key=lambda flight: _flight_sort_key(flight, return_time_pref),
        )[:5]
        hotels_list = list(_hotel_options)[:6]
    except Exception:
        flights_out = []
        flights_ret = []
        hotels_list = []
    if flights_out or flights_ret:
        tool_log.append("[flight_ranking] prioritized nonstop flights, then time preference, then price")

    try:
        from tools.serp_search import _hotel_tokens, serp_hotel_details, serp_hotel_reviews

        hotel_review_parts = []
        for name, token in list(_hotel_tokens.items())[:2]:
            details = serp_hotel_details(token, outbound_date, return_date)
            reviews = serp_hotel_reviews(token, num=3)
            parts = [part for part in [details, reviews] if part]
            if parts:
                hotel_review_parts.append(f"=== {name} ===\n" + "\n".join(parts))
        if hotel_review_parts:
            research["hotel_reviews"] = "\n\n".join(hotel_review_parts)
    except Exception as exc:
        tool_log.append(f"[hotel_reviews skipped: {exc}]")

    try:
        from tools.serp_search import _place_data_ids, serp_maps_reviews

        place_review_parts = []
        reviewed = 0
        for item in (research.get("ta_attractions") or []) + (research.get("maps_attractions") or []):
            if reviewed >= 5:
                break
            name = item.get("name", "")
            if not name:
                continue
            if not _place_data_ids.get(name.lower().strip()):
                continue
            reviews = serp_maps_reviews(name, num=3)
            if reviews:
                place_review_parts.append(f"=== {name} ===\n{reviews}")
                reviewed += 1
        if place_review_parts:
            research["place_reviews"] = "\n\n".join(place_review_parts)
            tool_log.append(f"[place_reviews] fetched {len(place_review_parts)} attraction(s)")
    except Exception as exc:
        tool_log.append(f"[place_reviews skipped: {exc}]")

    all_attractions: list[dict] = []
    seen_att: dict[str, dict] = {}

    raw_att_sources = (
        research["ta_attractions"]
        if use_explicit_place_queries
        else research["ta_attractions"] + research["pref_structured"] + research["maps_attractions"]
    )

    for item in raw_att_sources:
        name = item.get("name", "")
        if not name:
            continue
        name_l = name.lower().strip()
        if _is_corporate_noise(item) or _is_food_place(item):
            continue
        if re.search(r"\btours?\b", name_l):
            continue
        existing = next(
            (existing_name for existing_name in seen_att if name_l in existing_name or existing_name in name_l),
            None,
        )
        if existing:
            existing_item = seen_att[existing]
            if not existing_item.get("price") and item.get("price"):
                existing_item["price"] = item["price"]
            if not existing_item.get("hours") and item.get("hours"):
                existing_item["hours"] = item["hours"]
            if not existing_item.get("weekday_descriptions") and item.get("weekday_descriptions"):
                existing_item["weekday_descriptions"] = item["weekday_descriptions"]
            if existing_item.get("open_now") in (None, "") and item.get("open_now") not in (None, ""):
                existing_item["open_now"] = item["open_now"]
            continue
        seen_att[name_l] = dict(item)
        for pref in typed_prefs:
            pref_l = pref.lower()
            if (
                pref_l in name_l
                or pref_l in (item.get("type", "") or "").lower()
                or pref_l in (item.get("category", "") or "").lower()
            ):
                seen_att[name_l]["matches_preference"] = pref
                break
        all_attractions.append(seen_att[name_l])

    hotel_names_lower = {hotel.get("name", "").lower().strip() for hotel in hotels_list}
    all_restaurants: list[dict] = []
    seen_rest: set[str] = set()

    def _merge_restaurant_item(item: dict) -> None:
        name_key = item.get("name", "").lower().strip()
        if not name_key or name_key in seen_rest:
            return
        if any(name_key in hotel_name or hotel_name in name_key for hotel_name in hotel_names_lower):
            return
        seen_rest.add(name_key)
        all_restaurants.append(item)

    restaurant_sources = (
        research["ta_restaurants"]
        if use_explicit_place_queries
        else research["ta_restaurants"] + research["maps_restaurants"]
    )
    for item in restaurant_sources:
        _merge_restaurant_item(item)

    if use_explicit_place_queries and len(all_restaurants) < restaurant_floor:
        seen_queries = {
            str(query).strip().lower()
            for query in restaurant_queries
            if str(query).strip()
        }
        supplement_queries: list[str] = []
        generic_candidates = [
            f"Best local restaurants in {target_city}",
            f"Best traditional restaurants in {target_city}",
            f"Best street food and local restaurants in {target_city}",
            f"Best restaurants in {target_city}",
        ]
        for query in generic_candidates:
            if query.lower() not in seen_queries and query not in supplement_queries:
                supplement_queries.append(query)
            if len(supplement_queries) >= 3:
                break

        supplemented = 0
        for query in supplement_queries:
            try:
                bundle = google_places_search_bundle(query, "", place_kind="restaurant", max_results=8)
            except Exception:
                continue
            supplemented_candidates = _filter_candidates_to_destination_cluster(
                bundle.get("selected_places", []),
                place_kind="restaurants",
                target_city=target_city,
                destination=dest,
                tool_log=tool_log,
                reference_candidates=all_attractions or all_restaurants,
            )
            for item in supplemented_candidates:
                before = len(all_restaurants)
                item["search_query"] = item.get("search_query") or query
                _merge_restaurant_item(item)
                if len(all_restaurants) > before:
                    supplemented += 1
            if len(all_restaurants) >= restaurant_floor:
                break
        if supplemented:
            tool_log.append(
                f"[restaurant_supplement] added {supplemented} extra restaurant(s) to reach {len(all_restaurants)} candidates]"
            )

    def _needs_price(price_value):
        value = str(price_value or "").strip()
        return not value or value.lower() in {"tbc", "unknown", "n/a", "$", "$$", "$$$", "$$$$"}

    no_price_atts = [item for item in all_attractions if _needs_price(item.get("price")) and item.get("name")]
    if no_price_atts:
        price_queries = [
            f"{dest} top tourist attractions admission fee ticket price 2026",
            f"{dest} theme parks zoo aquarium wildlife ticket price adult 2026",
            f"{dest} museum heritage gallery entry fee 2026",
            f"{dest} free things to do attractions admission list",
        ]
        for idx in range(0, len(no_price_atts), 5):
            batch_names = " and ".join(item["name"] for item in no_price_atts[idx : idx + 5])
            price_queries.append(f"{dest} ticket prices for {batch_names} 2026")

        def _price_fetch(query: str) -> str:
            parts: list[str] = []

            def _extract(raw: str) -> str:
                if not raw:
                    return ""
                if raw.strip().startswith("[") or raw.strip().startswith("{"):
                    try:
                        data = json.loads(raw)
                        if isinstance(data, list):
                            return " ".join(
                                f"{entry.get('title', '')} {entry.get('snippet', '') or entry.get('content', '')}"
                                for entry in data
                                if isinstance(entry, dict)
                            )
                        if isinstance(data, dict):
                            return f"{data.get('answer', '')} {data}"
                    except Exception:
                        return raw
                return raw

            try:
                from tools.web_search import web_search as tavily_search

                result = tavily_search(query)
                if result and "error" not in result:
                    parts.append(_extract(result))
            except Exception:
                pass

            try:
                from tools.google_search import google_search

                result = google_search(query)
                if result and "error" not in result:
                    parts.append(_extract(result))
            except Exception:
                pass

            return "\n\n".join(parts)

        with ThreadPoolExecutor(max_workers=10) as pool:
            price_corpora = list(pool.map(_price_fetch, price_queries))

        combined_corpus = "\n\n===\n\n".join(
            re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", corpus[:5000])
            for corpus in price_corpora
            if corpus
        )
        tool_log.append(f"[price_search] corpus {len(combined_corpus)} chars")
        if combined_corpus:
            price_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
            enriched_count = 0
            for idx in range(0, len(no_price_atts), 15):
                batch = no_price_atts[idx : idx + 15]
                names_str = "\n".join(f"- {item['name']}" for item in batch)
                try:
                    response = price_llm.invoke(
                        [
                            SystemMessage(
                                content=(
                                    f"From the search results below, extract admission prices for attractions in {dest}.\n"
                                    'Return ONLY JSON: {"Attraction Name": "price string"}.\n'
                                    f"Attractions:\n{names_str}\n\n"
                                    f"Search results:\n{combined_corpus[:48000]}"
                                )
                            )
                        ]
                    )
                    match = re.search(r"\{.*\}", response.content, re.DOTALL)
                    if match:
                        price_map = json.loads(match.group())
                        for attraction in batch:
                            attraction_name = attraction["name"].lower()
                            for found_name, found_price in price_map.items():
                                found_l = str(found_name).lower()
                                if found_l and (found_l in attraction_name or attraction_name in found_l):
                                    value = str(found_price).strip()
                                    if value and value.lower() not in {"tbc", "unknown", "n/a"}:
                                        attraction["price"] = value
                                        enriched_count += 1
                                        break
                except Exception:
                    continue
            tool_log.append(f"[price_enrichment] {enriched_count}/{len(no_price_atts)} priced")

    def _compact_place(place: dict) -> dict:
        raw_price = place.get("price") or None
        return {
            key: value
            for key, value in {
                "name": place.get("name"),
                "type": place.get("type") or place.get("category"),
                "rating": place.get("rating"),
                "price": _safe_price(raw_price) if raw_price else None,
                "address": place.get("address"),
                "lat": place.get("lat"),
                "lng": place.get("lng"),
                "hours": place.get("hours"),
                "weekday_descriptions": place.get("weekday_descriptions"),
                "open_now": place.get("open_now"),
                "matches_preference": place.get("matches_preference"),
            }.items()
            if value not in (None, "", [])
        }

    def _compact_hotel(hotel: dict) -> dict:
        return {
            key: value
            for key, value in {
                "name": hotel.get("name"),
                "hotel_class": hotel.get("hotel_class"),
                "rating": hotel.get("rating"),
                "price_per_night_usd": _strip_dollar(hotel.get("price_per_night_usd")),
                "description": (hotel.get("description") or "")[:120] or None,
                "display": hotel.get("display"),
            }.items()
            if value not in (None, "", [])
        }

    def _compact_flight(flight: dict) -> dict:
        return {
            key: value
            for key, value in {
                "airline": flight.get("airline"),
                "flight_number": flight.get("flight_number"),
                "departure_airport": flight.get("departure_airport"),
                "arrival_airport": flight.get("arrival_airport"),
                "departure_time": flight.get("departure_time"),
                "arrival_time": flight.get("arrival_time"),
                "duration_min": flight.get("duration_min"),
                "travel_class": flight.get("travel_class"),
                "price_usd": flight.get("price_usd"),
                "display": flight.get("display"),
            }.items()
            if value not in (None, "", [])
        }

    compact_attractions = [_compact_place(item) for item in all_attractions]
    compact_restaurants = [_compact_place(item) for item in all_restaurants]
    compact_hotels = [_compact_hotel(item) for item in hotels_list]
    compact_flights_out = [_compact_flight(item) for item in flights_out]
    compact_flights_ret = [_compact_flight(item) for item in flights_ret]

    def _att_line(attraction: dict) -> str:
        parts = [attraction.get("name", "")]
        if attraction.get("lat") and attraction.get("lng"):
            parts.append(f"[{round(attraction['lat'], 4)},{round(attraction['lng'], 4)}]")
        elif attraction.get("address"):
            parts.append(str(attraction["address"])[:40])
        parts.append(attraction.get("price") or "admission: TBC")
        if attraction.get("hours"):
            parts.append(str(attraction.get("hours"))[:80])
        line = "  - " + " | ".join(part for part in parts if part)
        if attraction.get("matches_preference"):
            line += f" (pref:{attraction['matches_preference']})"
        return line

    def _rest_line(restaurant: dict) -> str:
        parts = [restaurant.get("name", "")]
        if restaurant.get("address"):
            parts.append(str(restaurant["address"])[:35])
        elif restaurant.get("lat") and restaurant.get("lng"):
            parts.append(f"[{round(restaurant['lat'], 4)},{round(restaurant['lng'], 4)}]")
        if restaurant.get("price"):
            parts.append(restaurant["price"])
        if restaurant.get("hours"):
            parts.append(str(restaurant.get("hours"))[:80])
        return "  - " + " | ".join(part for part in parts if part)

    att_list_text = "\n".join(_att_line(item) for item in compact_attractions)
    rest_list_text = "\n".join(_rest_line(item) for item in compact_restaurants[:20])
    hotel_list_text = "\n".join(f"  - {_build_hotel_display(item)}" for item in compact_hotels)
    flight_out_text = "\n".join(f"  - {_build_flight_display(item)}" for item in compact_flights_out)
    flight_ret_text = "\n".join(f"  - {_build_flight_display(item)}" for item in compact_flights_ret)

    ret_dep_times = [
        flight.get("departure_time", "")
        for flight in compact_flights_ret
        if flight.get("departure_time")
    ]
    earliest_ret_dep = min(ret_dep_times) if ret_dep_times else ""
    if earliest_ret_dep:
        cutoff_note = f"Earliest return flight departs {earliest_ret_dep}; airport cutoff is 3 hours before."
    else:
        cutoff_note = "Return departure time unknown; assume a conservative airport cutoff."

    tool_log.append("[research_agent_1] Data collection complete")

    inventory = {
        "attractions": compact_attractions,
        "restaurants": compact_restaurants,
        "hotels": compact_hotels,
        "flights_outbound": compact_flights_out,
        "flights_return": compact_flights_ret,
        "att_list_text": att_list_text,
        "rest_list_text": rest_list_text,
        "hotel_list_text": hotel_list_text,
        "flight_out_text": flight_out_text,
        "flight_ret_text": flight_ret_text,
        "ret_cutoff_note": cutoff_note,
    }

    return {
        "state": state,
        "research": research,
        "inventory": inventory,
        "tool_log": tool_log,
        "user_prefs": user_prefs,
        "compact_attractions": compact_attractions,
        "compact_restaurants": compact_restaurants,
        "compact_hotels": compact_hotels,
        "compact_flights_out": compact_flights_out,
        "compact_flights_ret": compact_flights_ret,
        "att_list_text": att_list_text,
        "rest_list_text": rest_list_text,
        "hotel_list_text": hotel_list_text,
        "flight_out_text": flight_out_text,
        "flight_ret_text": flight_ret_text,
        "ret_cutoff_note": cutoff_note,
        "hotel_opts": hotels_list,
    }
