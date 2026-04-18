"""
Planner-only pipeline that consumes `research_agent`.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta
import json
import math
import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..llm_config import OPENAI_MODEL
from .research_agent import _normalize_trip_state, research_agent


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.5)


def _has_expected_option_schema(options: dict) -> bool:
    if not isinstance(options, dict):
        return False
    for key in ("A", "B", "C"):
        value = options.get(key)
        if not isinstance(value, dict):
            return False
        if not isinstance(value.get("days"), list):
            return False
    return True


_inventory_cache: dict = {}


# Shared formatting / parsing helpers
def _usd_to_sgd_str(usd_val) -> str:
    try:
        return f"SGD {round(float(str(usd_val).replace(',', '')) * 1.35)}"
    except Exception:
        return f"USD {usd_val}" if usd_val else "TBC"


def _safe_price(raw_price: str) -> str:
    if not raw_price:
        return raw_price
    s = str(raw_price).strip()
    if s.lower().startswith("sgd "):
        return s
    match = re.search(r"usd\s*([\d.]+)", s, re.IGNORECASE)
    if match:
        return _usd_to_sgd_str(match.group(1))
    return s


def _flight_display(flight: dict) -> str:
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


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return text or "item"


def _stable_item_key(prefix: str, idx: int, name: str) -> str:
    return f"{prefix}_{idx:02d}_{_slugify(name)[:40]}"


def _parse_time_value(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%H:%M"):
        try:
            parsed = datetime.strptime(raw, fmt)
            if fmt == "%H:%M":
                return datetime(2000, 1, 1, parsed.hour, parsed.minute)
            return parsed
        except ValueError:
            continue
    return None


def _clock_time(value: str, fallback: str = "") -> str:
    parsed = _parse_time_value(value)
    if parsed:
        return parsed.strftime("%H:%M")
    return fallback


def _shift_clock(value: str, minutes: int, fallback: str = "") -> str:
    parsed = _parse_time_value(value)
    if not parsed:
        return fallback
    return (parsed + timedelta(minutes=minutes)).strftime("%H:%M")


def _sort_time_value(value: str) -> tuple[int, str]:
    parsed = _parse_time_value(value)
    if parsed:
        return (parsed.hour * 60 + parsed.minute, str(value or ""))
    return (24 * 60 + 59, str(value or ""))


def _minutes_since_midnight(value: str) -> int | None:
    parsed = _parse_time_value(value)
    if not parsed:
        return None
    return parsed.hour * 60 + parsed.minute


def _hhmm(minutes: int) -> str:
    minutes = max(0, min(minutes, 23 * 60 + 59))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _duration_days(duration) -> int:
    match = re.search(r"\d+", str(duration or "").strip())
    return max(1, int(match.group())) if match else 1


def _trip_start_date(state: dict) -> date | None:
    hard = state.get("hard_constraints", {}) or {}
    candidates = [
        hard.get("start_date"),
        str(state.get("dates") or "").split(" to ")[0].strip() if state.get("dates") else "",
    ]
    for raw in candidates:
        raw = str(raw or "").strip()
        if not raw:
            continue
        try:
            return date.fromisoformat(raw)
        except ValueError:
            continue
    return None


def _service_date(trip_start: date | None, day_offset: int) -> date | None:
    if not trip_start:
        return None
    return trip_start + timedelta(days=max(0, day_offset))


def _normalize_hours_text(value: str) -> str:
    return (
        str(value or "")
        .replace("\u202f", " ")
        .replace("\u2009", " ")
        .replace("\u2007", " ")
        .replace("\u2002", " ")
        .replace("\u2003", " ")
        .replace("\u00a0", " ")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .strip()
    )


def _parse_ampm_minutes(token: str) -> int | None:
    cleaned = _normalize_hours_text(token).upper().replace(".", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for fmt in ("%I:%M %p", "%I %p"):
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.hour * 60 + parsed.minute
        except ValueError:
            continue
    return None


def _latest_close_from_hours_text(hours_text: str) -> int | None:
    normalized = _normalize_hours_text(hours_text)
    lowered = normalized.lower()
    if not normalized:
        return None
    if "closed" in lowered and "open 24 hours" not in lowered and "opens" not in lowered:
        return -1
    if "open 24 hours" in lowered:
        return 23 * 60 + 59

    time_tokens = re.findall(r"\b\d{1,2}(?::\d{2})?\s*[APap]\.?M\.?\b", normalized)
    if len(time_tokens) < 2:
        close_match = re.search(r"closes?\s+(\d{1,2}(?::\d{2})?\s*[APap]\.?M\.?)", normalized, re.IGNORECASE)
        if close_match:
            return _parse_ampm_minutes(close_match.group(1))
        return None

    parsed_times = [_parse_ampm_minutes(token) for token in time_tokens]
    parsed_times = [minutes for minutes in parsed_times if minutes is not None]
    if len(parsed_times) < 2:
        return None

    latest_close: int | None = None
    for idx in range(0, len(parsed_times) - 1, 2):
        open_minutes = parsed_times[idx]
        close_minutes = parsed_times[idx + 1]
        if close_minutes <= open_minutes:
            close_minutes += 24 * 60
        latest_close = close_minutes if latest_close is None else max(latest_close, close_minutes)
    if latest_close is None:
        return None
    return min(latest_close, 23 * 60 + 59)


def _service_windows_from_hours_text(hours_text: str) -> list[tuple[int, int]]:
    normalized = _normalize_hours_text(hours_text)
    lowered = normalized.lower()
    if not normalized:
        return []
    if "closed" in lowered and "open 24 hours" not in lowered and "opens" not in lowered:
        return []
    if "open 24 hours" in lowered:
        return [(0, 23 * 60 + 59)]

    time_tokens = re.findall(r"\b\d{1,2}(?::\d{2})?\s*[APap]\.?M\.?\b", normalized)
    parsed_times = [_parse_ampm_minutes(token) for token in time_tokens]
    parsed_times = [minutes for minutes in parsed_times if minutes is not None]
    windows: list[tuple[int, int]] = []
    for idx in range(0, len(parsed_times) - 1, 2):
        open_minutes = parsed_times[idx]
        close_minutes = parsed_times[idx + 1]
        if close_minutes <= open_minutes:
            close_minutes += 24 * 60
        windows.append((open_minutes, min(close_minutes, 23 * 60 + 59)))
    return windows


def _latest_close_from_weekday_descriptions(descriptions: list[str], service_date: date | None) -> int | None:
    if not descriptions:
        return None
    weekday = service_date.strftime("%A").lower() if service_date else ""
    fallback_text = ""
    for description in descriptions:
        normalized = _normalize_hours_text(description)
        if not normalized:
            continue
        if ":" in normalized:
            day_label, hours_text = normalized.split(":", 1)
            if weekday and day_label.strip().lower() != weekday:
                continue
            return _latest_close_from_hours_text(hours_text)
        fallback_text = normalized
    if fallback_text:
        return _latest_close_from_hours_text(fallback_text)
    return None


def _service_windows_for_item(item: dict, service_date: date | None) -> list[tuple[int, int]]:
    descriptions = item.get("weekday_descriptions") or []
    weekday = service_date.strftime("%A").lower() if service_date else ""
    fallback_text = ""
    for description in descriptions:
        normalized = _normalize_hours_text(description)
        if not normalized:
            continue
        if ":" in normalized:
            day_label, hours_text = normalized.split(":", 1)
            if weekday and day_label.strip().lower() != weekday:
                continue
            return _service_windows_from_hours_text(hours_text)
        fallback_text = normalized
    if fallback_text:
        return _service_windows_from_hours_text(fallback_text)
    return _service_windows_from_hours_text(str(item.get("hours") or ""))


def _fit_service_start(
    item: dict,
    preferred_start: int,
    duration_minutes: int,
    service_date: date | None,
    latest_finish: int | None = None,
) -> int | None:
    windows = _service_windows_for_item(item, service_date)
    if not windows:
        if item.get("weekday_descriptions") or item.get("hours"):
            return None
        if latest_finish is not None and preferred_start + duration_minutes > latest_finish:
            return None
        return preferred_start
    for open_minutes, close_minutes in windows:
        candidate = max(preferred_start, open_minutes)
        if candidate + duration_minutes > close_minutes:
            continue
        if latest_finish is not None and candidate + duration_minutes > latest_finish:
            continue
        return candidate
    return None


def _haversine_km(lat1, lng1, lat2, lng2) -> float | None:
    try:
        from math import asin, cos, radians, sin, sqrt

        lat1 = float(lat1)
        lng1 = float(lng1)
        lat2 = float(lat2)
        lng2 = float(lng2)
    except Exception:
        return None

    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


def _build_activity_entries(items: list[dict], prefix: str) -> list[dict]:
    entries: list[dict] = []
    for idx, item in enumerate(items, 1):
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        entry = {
            "key": _stable_item_key(prefix, idx, name),
            "name": name,
        }
        for field in (
            "type",
            "rating",
            "price",
            "address",
            "lat",
            "lng",
            "matches_preference",
            "hours",
            "weekday_descriptions",
            "open_now",
        ):
            value = item.get(field)
            if value not in (None, "", []):
                entry[field] = value
        entries.append(entry)
    return entries


def _ensure_inventory_keys(items: list[dict], prefix: str) -> list[dict]:
    normalized: list[dict] = []
    for idx, item in enumerate(items, 1):
        clone = deepcopy(item)
        clone.setdefault("key", _stable_item_key(prefix, idx, clone.get("name", "")))
        normalized.append(clone)
    return normalized


def _build_prompt_inventory(
    compact_attractions: list[dict],
    compact_restaurants: list[dict],
    compact_hotels: list[dict],
    compact_flights_out: list[dict],
    compact_flights_ret: list[dict],
) -> dict:
    attraction_entries = _build_activity_entries(compact_attractions, "attraction")
    restaurant_entries = _build_activity_entries(compact_restaurants, "restaurant")
    hotel_entries = []
    for idx, hotel in enumerate(compact_hotels, 1):
        name = str(hotel.get("name") or "").strip()
        if not name:
            continue
        hotel_entries.append(
            {
                "key": _stable_item_key("hotel", idx, name),
                "name": name,
                "nightly_cost": (
                    _usd_to_sgd_str(hotel.get("price_per_night_usd"))
                    if hotel.get("price_per_night_usd")
                    else "TBC"
                ),
                "rating": hotel.get("rating"),
                "description": hotel.get("description"),
            }
        )

    outbound_flights = []
    for idx, flight in enumerate(compact_flights_out, 1):
        outbound_flights.append(
            {
                "option_key": _stable_item_key("outbound", idx, flight.get("flight_number") or flight.get("airline")),
                "name": _flight_display(flight),
                "departure_time": flight.get("departure_time"),
                "arrival_time": flight.get("arrival_time"),
                "cost": _usd_to_sgd_str(flight.get("price_usd")),
            }
        )

    return_flights = []
    for idx, flight in enumerate(compact_flights_ret, 1):
        return_flights.append(
            {
                "option_key": _stable_item_key("return", idx, flight.get("flight_number") or flight.get("airline")),
                "name": _flight_display(flight),
                "departure_time": flight.get("departure_time"),
                "arrival_time": flight.get("arrival_time"),
                "cost": _usd_to_sgd_str(flight.get("price_usd")),
            }
        )

    return {
        "outbound_flights": outbound_flights,
        "return_flights": return_flights,
        "hotels": hotel_entries,
        "attractions": attraction_entries,
        "restaurants": restaurant_entries,
    }


def _cache_inventory_snapshot(
    research: dict,
    research_result: dict,
    tool_log: list[str],
    compact_attractions: list[dict],
    compact_restaurants: list[dict],
    compact_hotels: list[dict],
    compact_flights_out: list[dict],
    compact_flights_ret: list[dict],
    prompt_inventory: dict,
) -> None:
    _inventory_cache.clear()
    _inventory_cache.update(
        {
            "compact_attractions": compact_attractions,
            "compact_restaurants": compact_restaurants,
            "compact_hotels": compact_hotels,
            "compact_flights_out": compact_flights_out,
            "compact_flights_ret": compact_flights_ret,
            "att_list_text": research_result.get("att_list_text") or (research_result.get("inventory") or {}).get("att_list_text", ""),
            "rest_list_text": research_result.get("rest_list_text") or (research_result.get("inventory") or {}).get("rest_list_text", ""),
            "hotel_list_text": research_result.get("hotel_list_text") or (research_result.get("inventory") or {}).get("hotel_list_text", ""),
            "flight_out_text": research_result.get("flight_out_text") or (research_result.get("inventory") or {}).get("flight_out_text", ""),
            "flight_ret_text": research_result.get("flight_ret_text") or (research_result.get("inventory") or {}).get("flight_ret_text", ""),
            "ret_cutoff_note": research_result.get("ret_cutoff_note") or (research_result.get("inventory") or {}).get("ret_cutoff_note", ""),
            "prompt_inventory": prompt_inventory,
            "research": research,
            "hotel_opts": research_result.get("hotel_opts") or research_result.get("hotel_options", []) or (research_result.get("inventory") or {}).get("hotels", []),
            "tool_log": list(tool_log),
        }
    )


def _extract_research_payload(state: dict, research_result: dict) -> dict:
    inventory = research_result.get("inventory") or {}
    return {
        "research": research_result.get("research", {}),
        "tool_log": list(research_result.get("tool_log", [])),
        "user_prefs": research_result.get("user_prefs") or state.get("preferences", ""),
        "compact_attractions": research_result.get("compact_attractions") or inventory.get("attractions", []),
        "compact_restaurants": research_result.get("compact_restaurants") or inventory.get("restaurants", []),
        "compact_hotels": research_result.get("compact_hotels") or research_result.get("hotel_options", []) or inventory.get("hotels", []),
        "compact_flights_out": research_result.get("compact_flights_out") or research_result.get("flight_options_outbound", []) or inventory.get("flights_outbound", []),
        "compact_flights_ret": research_result.get("compact_flights_ret") or research_result.get("flight_options_return", []) or inventory.get("flights_return", []),
        "att_list_text": research_result.get("att_list_text") or inventory.get("att_list_text", ""),
        "rest_list_text": research_result.get("rest_list_text") or inventory.get("rest_list_text", ""),
        "hotel_list_text": research_result.get("hotel_list_text") or inventory.get("hotel_list_text", ""),
        "flight_out_text": research_result.get("flight_out_text") or inventory.get("flight_out_text", ""),
        "flight_ret_text": research_result.get("flight_ret_text") or inventory.get("flight_ret_text", ""),
        "ret_cutoff_note": research_result.get("ret_cutoff_note") or inventory.get("ret_cutoff_note", ""),
        "hotel_opts": research_result.get("hotel_opts") or research_result.get("hotel_options", []) or inventory.get("hotels", []),
    }


def _match_flight(name: str, candidates: list[dict]) -> dict | None:
    name_l = str(name or "").lower()
    for flight in candidates:
        flight_number = str(flight.get("flight_number") or "").lower()
        display = str(flight.get("display") or "").lower()
        if flight_number and flight_number in name_l:
            return flight
        if display and (name_l in display or display in name_l):
            return flight
        if "/" in flight_number:
            segments = [segment.strip() for segment in flight_number.split("/") if segment.strip()]
            if any(segment in name_l for segment in segments):
                return flight
    return None


def _match_hotel(name: str, hotels: list[dict]) -> dict | None:
    name_l = str(name or "").lower().strip()
    for hotel in hotels:
        hotel_name = str(hotel.get("name") or "").lower().strip()
        if not hotel_name:
            continue
        if hotel_name == name_l or hotel_name in name_l or name_l in hotel_name:
            return hotel
    return None


def _looks_like_hotel_item(name: str, key: str, hotels: list[dict]) -> bool:
    if "hotel" in (key or "").lower():
        return True
    if _match_hotel(name, hotels):
        return True
    return bool(re.search(r"\b(hotel|hostel|motel|resort|lodge|inn)\b", str(name or "").lower()))


def _ensure_boundary_days(
    days: list,
    compact_hotels: list,
    compact_flights_out: list,
    compact_flights_ret: list,
) -> tuple[list, list[str]]:
    if not days:
        return days, []

    days = deepcopy(days)
    repairs: list[str] = []

    first_day = days[0]
    last_day = days[-1]
    selected_hotel = next(
        (
            item for day in days for item in day.get("items", [])
            if str(item.get("icon") or "").lower() == "hotel"
        ),
        None,
    )
    selected_outbound_item = next(
        (item for item in first_day.get("items", []) if str(item.get("key", "")) == "flight_outbound"),
        None,
    )
    selected_return_item = next(
        (item for item in last_day.get("items", []) if str(item.get("key", "")) == "flight_return"),
        None,
    )

    primary_hotel = (
        _match_hotel(selected_hotel.get("name", ""), compact_hotels) if selected_hotel else None
    ) or (compact_hotels[0] if compact_hotels else None)
    outbound_flight = (
        _match_flight(selected_outbound_item.get("name", ""), compact_flights_out) if selected_outbound_item else None
    ) or (compact_flights_out[0] if compact_flights_out else None)
    return_flight = (
        _match_flight(selected_return_item.get("name", ""), compact_flights_ret) if selected_return_item else None
    ) or (compact_flights_ret[0] if compact_flights_ret else None)

    if outbound_flight:
        has_outbound = any(str(item.get("key", "")) == "flight_outbound" for item in first_day.get("items", []))
        if not has_outbound:
            first_day.setdefault("items", []).insert(
                0,
                {
                    "time": _clock_time(
                        outbound_flight.get("arrival_time") or outbound_flight.get("departure_time"),
                        fallback="",
                    ),
                    "icon": "flight",
                    "name": _flight_display(outbound_flight),
                    "cost": _usd_to_sgd_str(outbound_flight.get("price_usd")),
                    "key": "flight_outbound",
                },
            )
            repairs.append("inserted Day 1 outbound flight from research inventory")

    if primary_hotel:
        has_first_hotel = any("hotel" in str(item.get("key", "")).lower() for item in first_day.get("items", []))
        if not has_first_hotel:
            first_day.setdefault("items", []).append(
                {
                    "time": _shift_clock(
                        outbound_flight.get("arrival_time") if outbound_flight else "",
                        90,
                        fallback="20:30",
                    ),
                    "icon": "hotel",
                    "name": primary_hotel.get("name", ""),
                    "cost": (
                        _usd_to_sgd_str(primary_hotel.get("price_per_night_usd")) + "/night"
                        if primary_hotel.get("price_per_night_usd")
                        else "TBC"
                    ),
                    "key": "d1_hotel",
                }
            )
            repairs.append("inserted Day 1 hotel stay from research inventory")

    if primary_hotel:
        has_last_hotel = any("hotel" in str(item.get("key", "")).lower() for item in last_day.get("items", []))
        if not has_last_hotel:
            departure_minutes = _minutes_since_midnight(return_flight.get("departure_time", "")) if return_flight else None
            suggested_hotel_time = (
                _hhmm(departure_minutes - 240) if departure_minutes is not None and departure_minutes <= 14 * 60
                else "09:00"
            )
            last_day.setdefault("items", []).append(
                {
                    "time": suggested_hotel_time,
                    "icon": "hotel",
                    "name": primary_hotel.get("name", ""),
                    "cost": (
                        _usd_to_sgd_str(primary_hotel.get("price_per_night_usd")) + "/night"
                        if primary_hotel.get("price_per_night_usd")
                        else "TBC"
                    ),
                    "key": f"d{len(days)}_hotel",
                }
            )
            repairs.append("inserted final-day hotel checkout anchor from research inventory")

    if return_flight:
        has_return = any(str(item.get("key", "")) == "flight_return" for item in last_day.get("items", []))
        if not has_return:
            last_day.setdefault("items", []).append(
                {
                    "time": _clock_time(return_flight.get("departure_time"), fallback=""),
                    "icon": "flight",
                    "name": _flight_display(return_flight),
                    "cost": _usd_to_sgd_str(return_flight.get("price_usd")),
                    "key": "flight_return",
                }
            )
            repairs.append("inserted final-day return flight from research inventory")

    for day in (first_day, last_day):
        day["items"] = sorted(day.get("items", []), key=lambda item: _sort_time_value(item.get("time", "")))

    return days, repairs


# Legacy LLM cleanup path kept for revision mode only.
def _post_process_options(
    options: dict,
    compact_attractions: list,
    compact_restaurants: list,
    compact_hotels: list,
    compact_flights_out: list,
    compact_flights_ret: list,
    tool_log: list,
) -> tuple[dict, dict, dict, dict]:
    attraction_entries = _build_activity_entries(compact_attractions, "attraction")
    restaurant_entries = _build_activity_entries(compact_restaurants, "restaurant")
    attraction_by_key = {item["key"]: item for item in attraction_entries}
    restaurant_by_key = {item["key"]: item for item in restaurant_entries}
    attraction_by_name = {item["name"].lower().strip(): item for item in attraction_entries}
    restaurant_by_name = {item["name"].lower().strip(): item for item in restaurant_entries}

    def _is_evening_safe_item(item: dict) -> bool:
        name_l = str(item.get("name") or "").lower()
        kind = str(item.get("icon") or "").lower()
        safe_keywords = (
            "yokocho",
            "sky",
            "tower",
            "night",
            "street",
            "district",
            "market",
            "observatory",
            "view",
            "teamlab",
        )
        daytime_keywords = (
            "garden",
            "museum",
            "palace",
            "park",
            "temple",
            "shrine",
            "zoo",
            "aquarium",
            "national",
        )
        if kind in {"restaurant", "hotel", "flight"}:
            return True
        if any(keyword in name_l for keyword in safe_keywords):
            return True
        if any(keyword in name_l for keyword in daytime_keywords):
            return False
        return False

    def _calc_day_centroid(items: list[dict]) -> tuple[float, float] | None:
        coords = []
        for item in items:
            lat = item.get("_lat")
            lng = item.get("_lng")
            if lat in (None, "") or lng in (None, ""):
                continue
            try:
                coords.append((float(lat), float(lng)))
            except Exception:
                continue
        if not coords:
            return None
        lat = sum(pair[0] for pair in coords) / len(coords)
        lng = sum(pair[1] for pair in coords) / len(coords)
        return lat, lng

    def _pick_restaurant(day_items: list[dict], used_names: set[str]) -> dict | None:
        centroid = _calc_day_centroid(day_items)
        best_entry = None
        best_score = None
        for entry in restaurant_entries:
            name_l = entry["name"].lower().strip()
            if name_l in used_names:
                continue
            if centroid and entry.get("lat") not in (None, "") and entry.get("lng") not in (None, ""):
                score = _haversine_km(centroid[0], centroid[1], entry.get("lat"), entry.get("lng"))
                if score is None:
                    score = 9999
            else:
                score = 9999
            if best_score is None or score < best_score:
                best_entry = entry
                best_score = score
        return best_entry

    def _pick_restaurant_near(
        target_lat,
        target_lng,
        used_names: set[str],
        current_name: str = "",
    ) -> dict | None:
        best_entry = None
        best_score = None
        current_name_l = str(current_name or "").lower().strip()
        for entry in restaurant_entries:
            name_l = entry["name"].lower().strip()
            if name_l in used_names and name_l != current_name_l:
                continue
            if target_lat in (None, "") or target_lng in (None, ""):
                score = 9999
            else:
                score = _haversine_km(target_lat, target_lng, entry.get("lat"), entry.get("lng"))
                if score is None:
                    score = 9999
            if best_score is None or score < best_score:
                best_entry = entry
                best_score = score
        return best_entry

    def _insert_meal(
        day: dict,
        meal_label: str,
        target_time: str,
        used_names: set[str],
        stats: dict,
    ) -> None:
        restaurant = _pick_restaurant(day.get("items", []), used_names)
        if not restaurant:
            stats.setdefault("meal_repairs", []).append(f"{day.get('day', '')}: unable to add {meal_label}")
            return
        item = {
            "time": target_time,
            "icon": "restaurant",
            "key": restaurant["key"],
            "name": restaurant["name"],
            "cost": restaurant.get("price") or "TBC",
            "_lat": restaurant.get("lat"),
            "_lng": restaurant.get("lng"),
            "_meal_slot": meal_label,
        }
        used_names.add(restaurant["name"].lower().strip())
        day.setdefault("items", []).append(item)
        day["items"] = sorted(day.get("items", []), key=lambda current: _sort_time_value(current.get("time", "")))
        stats.setdefault("meal_repairs", []).append(f"{day.get('day', '')}: added {meal_label} at {target_time}")

    def _enforce_boundary_feasibility(days: list, used_names: set[str], stats: dict) -> list:
        if not days:
            return days
        days = deepcopy(days)

        first_day = days[0]
        selected_outbound = next(
            (item for item in first_day.get("items", []) if str(item.get("key") or "") == "flight_outbound"),
            None,
        )
        arrival_minutes = _minutes_since_midnight(selected_outbound.get("time", "")) if selected_outbound else (
            _minutes_since_midnight(compact_flights_out[0].get("arrival_time", "")) if compact_flights_out else None
        )
        filtered_first_items = []
        for item in first_day.get("items", []):
            item_minutes = _minutes_since_midnight(item.get("time", ""))
            kind = str(item.get("icon") or "").lower()
            if (
                arrival_minutes is not None
                and arrival_minutes >= 18 * 60
                and kind not in {"flight", "hotel", "restaurant"}
                and item_minutes is not None
                and item_minutes >= arrival_minutes
                and not _is_evening_safe_item(item)
            ):
                stats.setdefault("boundary_filtered", []).append(
                    f"removed late Day 1 item '{item.get('name', '')}' at {item.get('time', '')}"
                )
                stats["removed_items"] += 1
                stats["invalid_items"] += 1
                continue
            filtered_first_items.append(item)
        first_day["items"] = sorted(filtered_first_items, key=lambda item: _sort_time_value(item.get("time", "")))

        if arrival_minutes is not None:
            has_dinner = any(
                str(item.get("icon") or "").lower() == "restaurant"
                and (_minutes_since_midnight(item.get("time", "")) or 0) >= 17 * 60
                for item in first_day.get("items", [])
            )
            if arrival_minutes <= 21 * 60 and not has_dinner:
                dinner_time = _shift_clock(selected_outbound.get("time", "") if selected_outbound else "", 120, fallback="20:30")
                _insert_meal(first_day, "dinner", dinner_time, used_names, stats)

        last_day = days[-1]
        selected_return = next(
            (item for item in last_day.get("items", []) if str(item.get("key") or "") == "flight_return"),
            None,
        )
        return_departure_minutes = _minutes_since_midnight(selected_return.get("time", "")) if selected_return else (
            _minutes_since_midnight(compact_flights_ret[0].get("departure_time", "")) if compact_flights_ret else None
        )
        if return_departure_minutes is not None:
            airport_cutoff = return_departure_minutes - 180
            filtered_last_items = []
            for item in last_day.get("items", []):
                kind = str(item.get("icon") or "").lower()
                item_minutes = _minutes_since_midnight(item.get("time", ""))
                if kind == "flight":
                    filtered_last_items.append(item)
                    continue
                if item_minutes is not None and item_minutes > airport_cutoff:
                    stats.setdefault("boundary_filtered", []).append(
                        f"removed pre-flight item '{item.get('name', '')}' at {item.get('time', '')}"
                    )
                    stats["removed_items"] += 1
                    stats["invalid_items"] += 1
                    continue
                filtered_last_items.append(item)
            last_day["items"] = sorted(filtered_last_items, key=lambda item: _sort_time_value(item.get("time", "")))

            has_lunch = any(
                str(item.get("icon") or "").lower() == "restaurant"
                and 11 * 60 <= (_minutes_since_midnight(item.get("time", "")) or 0) <= 15 * 60
                for item in last_day.get("items", [])
            )
            has_dinner = any(
                str(item.get("icon") or "").lower() == "restaurant"
                and (_minutes_since_midnight(item.get("time", "")) or 0) >= 17 * 60
                for item in last_day.get("items", [])
            )
            if return_departure_minutes >= 14 * 60 + 30 and not has_lunch:
                lunch_time = f"{max(11 * 60 + 30, return_departure_minutes - 300) // 60:02d}:{max(11 * 60 + 30, return_departure_minutes - 300) % 60:02d}"
                _insert_meal(last_day, "lunch", lunch_time, used_names, stats)
            if return_departure_minutes >= 20 * 60 and not has_dinner:
                dinner_minutes = max(17 * 60 + 30, return_departure_minutes - 240)
                dinner_time = f"{dinner_minutes // 60:02d}:{dinner_minutes % 60:02d}"
                _insert_meal(last_day, "dinner", dinner_time, used_names, stats)

        for day in days[1:-1]:
            lunch_exists = any(
                str(item.get("icon") or "").lower() == "restaurant"
                and 11 * 60 <= (_minutes_since_midnight(item.get("time", "")) or 0) <= 15 * 60
                for item in day.get("items", [])
            )
            dinner_exists = any(
                str(item.get("icon") or "").lower() == "restaurant"
                and (_minutes_since_midnight(item.get("time", "")) or 0) >= 17 * 60
                for item in day.get("items", [])
            )
            if not lunch_exists:
                _insert_meal(day, "lunch", "12:30", used_names, stats)
            if not dinner_exists:
                _insert_meal(day, "dinner", "18:30", used_names, stats)

        return days

    def _repair_routes(days: list, used_names: set[str], stats: dict) -> list:
        days = deepcopy(days)
        route_repairs: list[str] = []
        for day in days:
            items = sorted(day.get("items", []), key=lambda item: _sort_time_value(item.get("time", "")))
            for idx in range(1, len(items)):
                prev = items[idx - 1]
                current = items[idx]
                prev_kind = str(prev.get("icon") or "").lower()
                curr_kind = str(current.get("icon") or "").lower()
                if prev_kind not in {"activity", "restaurant"} or curr_kind not in {"activity", "restaurant"}:
                    continue
                dist = _haversine_km(prev.get("_lat"), prev.get("_lng"), current.get("_lat"), current.get("_lng"))
                if dist is None or dist <= 10:
                    continue

                repaired = False
                if curr_kind == "restaurant":
                    replacement = _pick_restaurant_near(
                        prev.get("_lat"),
                        prev.get("_lng"),
                        used_names,
                        current_name=current.get("name", ""),
                    )
                    if replacement:
                        current_name_l = str(current.get("name", "")).lower().strip()
                        replacement_name_l = replacement["name"].lower().strip()
                        replacement_dist = _haversine_km(
                            prev.get("_lat"),
                            prev.get("_lng"),
                            replacement.get("lat"),
                            replacement.get("lng"),
                        )
                        if (
                            replacement_name_l != current_name_l
                            and replacement_dist is not None
                            and replacement_dist + 2 < dist
                        ):
                            used_names.discard(current_name_l)
                            used_names.add(replacement_name_l)
                            items[idx] = {
                                **current,
                                "key": replacement["key"],
                                "name": replacement["name"],
                                "cost": replacement.get("price") or current.get("cost") or "TBC",
                                "_lat": replacement.get("lat"),
                                "_lng": replacement.get("lng"),
                            }
                            route_repairs.append(
                                f"{day.get('day', '')}: replaced restaurant '{current.get('name', '')}' "
                                f"with '{replacement['name']}' to reduce routing jump"
                            )
                            repaired = True

                if not repaired and prev_kind == "restaurant":
                    replacement = _pick_restaurant_near(
                        current.get("_lat"),
                        current.get("_lng"),
                        used_names,
                        current_name=prev.get("name", ""),
                    )
                    if replacement:
                        prev_name_l = str(prev.get("name", "")).lower().strip()
                        replacement_name_l = replacement["name"].lower().strip()
                        replacement_dist = _haversine_km(
                            current.get("_lat"),
                            current.get("_lng"),
                            replacement.get("lat"),
                            replacement.get("lng"),
                        )
                        if (
                            replacement_name_l != prev_name_l
                            and replacement_dist is not None
                            and replacement_dist + 2 < dist
                        ):
                            used_names.discard(prev_name_l)
                            used_names.add(replacement_name_l)
                            items[idx - 1] = {
                                **prev,
                                "key": replacement["key"],
                                "name": replacement["name"],
                                "cost": replacement.get("price") or prev.get("cost") or "TBC",
                                "_lat": replacement.get("lat"),
                                "_lng": replacement.get("lng"),
                            }
                            route_repairs.append(
                                f"{day.get('day', '')}: replaced restaurant '{prev.get('name', '')}' "
                                f"with '{replacement['name']}' to reduce routing jump"
                            )
                            repaired = True

                if not repaired and prev_kind == "activity" and curr_kind == "activity" and idx + 1 < len(items):
                    best_idx = idx
                    best_dist = dist
                    for candidate_idx in range(idx + 1, len(items)):
                        candidate = items[candidate_idx]
                        candidate_kind = str(candidate.get("icon") or "").lower()
                        if candidate_kind != "activity":
                            continue
                        candidate_dist = _haversine_km(
                            prev.get("_lat"),
                            prev.get("_lng"),
                            candidate.get("_lat"),
                            candidate.get("_lng"),
                        )
                        if candidate_dist is not None and candidate_dist < best_dist:
                            best_idx = candidate_idx
                            best_dist = candidate_dist
                    if best_idx != idx:
                        items[idx], items[best_idx] = items[best_idx], items[idx]
                        route_repairs.append(
                            f"{day.get('day', '')}: reordered activities to keep "
                            f"'{items[idx - 1].get('name', '')}' closer to the next stop"
                        )

            day["items"] = sorted(items, key=lambda item: _sort_time_value(item.get("time", "")))
        if route_repairs:
            stats["route_repairs"] = route_repairs
        return days

    def _reslot_days(days: list) -> list:
        days = deepcopy(days)
        for day_idx, day in enumerate(days):
            items = sorted(day.get("items", []), key=lambda item: _sort_time_value(item.get("time", "")))
            if not items:
                continue

            if day_idx == 0:
                outbound = next((item for item in items if str(item.get("key", "")) == "flight_outbound"), None)
                hotel = next((item for item in items if str(item.get("icon") or "").lower() == "hotel"), None)
                other_items = [item for item in items if item is not outbound and item is not hotel]
                if outbound:
                    arrival_minutes = _minutes_since_midnight(outbound.get("time", "")) or 17 * 60 + 30
                    next_minutes = arrival_minutes + 90
                    for item in other_items:
                        kind = str(item.get("icon") or "").lower()
                        if kind == "restaurant":
                            item["time"] = _hhmm(max(next_minutes, 19 * 60))
                            next_minutes = _minutes_since_midnight(item["time"]) + 90
                        else:
                            item["time"] = _hhmm(next_minutes)
                            next_minutes += 90
                    if hotel:
                        hotel["time"] = _hhmm(max(next_minutes, arrival_minutes + 180))
                day["items"] = sorted(items, key=lambda item: _sort_time_value(item.get("time", "")))
                continue

            if day_idx == len(days) - 1:
                hotel = next((item for item in items if str(item.get("icon") or "").lower() == "hotel"), None)
                flight = next((item for item in items if str(item.get("key", "")) == "flight_return"), None)
                preflight = [item for item in items if item is not hotel and item is not flight]
                departure_minutes = _minutes_since_midnight(flight.get("time", "")) if flight else None
                airport_cutoff = departure_minutes - 180 if departure_minutes is not None else None
                if hotel:
                    hotel_time = (
                        _hhmm(max(7 * 60 + 30, airport_cutoff - 90))
                        if airport_cutoff is not None and airport_cutoff <= 11 * 60
                        else "09:00"
                    )
                    hotel["time"] = hotel_time
                if preflight:
                    activity_items = [item for item in preflight if str(item.get("icon") or "").lower() == "activity"]
                    restaurant_items = [item for item in preflight if str(item.get("icon") or "").lower() == "restaurant"]
                    if activity_items:
                        activity_items[0]["time"] = "10:00" if airport_cutoff is None or airport_cutoff >= 12 * 60 else _hhmm(max(8 * 60 + 45, airport_cutoff - 120))
                    if restaurant_items:
                        lunch_target = 11 * 60 + 30
                        if airport_cutoff is not None:
                            lunch_target = min(lunch_target, airport_cutoff - 60)
                        restaurant_items[0]["time"] = _hhmm(max(10 * 60 + 45, lunch_target))
                day["items"] = sorted(items, key=lambda item: _sort_time_value(item.get("time", "")))
                continue

            activity_items = [item for item in items if str(item.get("icon") or "").lower() == "activity"]
            restaurant_items = [item for item in items if str(item.get("icon") or "").lower() == "restaurant"]

            if len(activity_items) >= 1:
                activity_items[0]["time"] = "10:00"
            if len(restaurant_items) >= 1:
                restaurant_items[0]["time"] = "12:30"
            if len(activity_items) >= 2:
                activity_items[1]["time"] = "15:00"
            if len(restaurant_items) >= 2:
                restaurant_items[1]["time"] = "18:30"
            if len(activity_items) >= 3:
                activity_items[2]["time"] = "20:00"

            day["items"] = sorted(items, key=lambda item: _sort_time_value(item.get("time", "")))
        return days

    def _route_warnings(days: list) -> list[str]:
        warnings = []
        for day in days:
            prev = None
            for item in day.get("items", []):
                if str(item.get("icon") or "").lower() not in {"activity", "restaurant"}:
                    continue
                if prev is None:
                    prev = item
                    continue
                dist = _haversine_km(prev.get("_lat"), prev.get("_lng"), item.get("_lat"), item.get("_lng"))
                if dist is not None and dist > 10:
                    warnings.append(
                        f"{day.get('day', '')}: {prev.get('name', '')} -> {item.get('name', '')} is {dist:.1f} km apart"
                    )
                prev = item
        return warnings

    def _strip_internal_fields(days: list) -> list:
        cleaned_days = []
        for day in days:
            cleaned_items = []
            for item in day.get("items", []):
                cleaned_items.append({k: v for k, v in item.items() if not str(k).startswith("_")})
            cleaned_days.append({**day, "items": cleaned_items})
        return cleaned_days

    def _resolve_catalog_item(name: str, key: str) -> tuple[dict | None, str | None]:
        key = str(key or "").strip()
        name_l = str(name or "").lower().strip()
        if key in attraction_by_key:
            return attraction_by_key[key], "activity"
        if key in restaurant_by_key:
            return restaurant_by_key[key], "restaurant"
        if name_l in attraction_by_name:
            return attraction_by_name[name_l], "activity"
        if name_l in restaurant_by_name:
            return restaurant_by_name[name_l], "restaurant"

        for lookup, kind in ((attraction_by_name, "activity"), (restaurant_by_name, "restaurant")):
            for valid_name, entry in lookup.items():
                if name_l and (name_l in valid_name or valid_name in name_l):
                    return entry, kind
        return None, None

    def _process_days(days: list) -> tuple[list, dict]:
        stats = {
            "removed_items": 0,
            "normalized_items": 0,
            "duplicate_items": 0,
            "invalid_items": 0,
            "sparse_days": [],
        }
        seen_items: set[str] = set()
        normalized_days: list = []

        for day_idx, raw_day in enumerate(days):
            if isinstance(raw_day, dict):
                day = deepcopy(raw_day)
            elif isinstance(raw_day, list):
                day = {"day": f"Day {day_idx + 1}", "items": raw_day}
            else:
                stats["removed_items"] += 1
                stats["invalid_items"] += 1
                continue

            clean_items = []
            is_last_day = day_idx == len(days) - 1

            for item_idx, item in enumerate(day.get("items", [])):
                if not isinstance(item, dict):
                    stats["removed_items"] += 1
                    stats["invalid_items"] += 1
                    continue

                item = deepcopy(item)
                name = item.get("name", "")
                key = item.get("key", "") or ""
                before = json.dumps(item, sort_keys=True, ensure_ascii=False)

                matched_outbound = _match_flight(name, compact_flights_out)
                matched_return = _match_flight(name, compact_flights_ret)
                matched_hotel = _match_hotel(name, compact_hotels)

                is_flight_candidate = (
                    "flight" in key.lower()
                    or str(item.get("icon") or "").lower() == "flight"
                    or matched_outbound is not None
                    or matched_return is not None
                )
                if is_flight_candidate:
                    if day_idx == 0:
                        is_outbound = True
                    elif is_last_day:
                        is_outbound = False
                    elif matched_outbound and not matched_return:
                        is_outbound = True
                    elif matched_return and not matched_outbound:
                        is_outbound = False
                    else:
                        is_outbound = item_idx == 0

                    item["key"] = "flight_outbound" if is_outbound else "flight_return"
                    matched_flight = matched_outbound if is_outbound else matched_return
                    if not matched_flight:
                        candidates = compact_flights_out if is_outbound else compact_flights_ret
                        matched_flight = _match_flight(name, candidates) or _match_flight(
                            name, compact_flights_out + compact_flights_ret
                        )
                    if not matched_flight:
                        stats["removed_items"] += 1
                        stats["invalid_items"] += 1
                        continue
                    item["name"] = _flight_display(matched_flight)
                    item["time"] = (
                        _clock_time(matched_flight.get("arrival_time"))
                        if is_outbound
                        else _clock_time(matched_flight.get("departure_time"))
                    ) or item.get("time", "")
                    item["cost"] = _usd_to_sgd_str(matched_flight.get("price_usd"))
                    item["icon"] = "flight"
                    if json.dumps(item, sort_keys=True, ensure_ascii=False) != before:
                        stats["normalized_items"] += 1
                    clean_items.append(item)
                    continue

                if _looks_like_hotel_item(name, key, compact_hotels):
                    if not matched_hotel:
                        stats["removed_items"] += 1
                        stats["invalid_items"] += 1
                        continue
                    item["name"] = matched_hotel.get("name", name)
                    item["cost"] = (
                        _usd_to_sgd_str(matched_hotel.get("price_per_night_usd")) + "/night"
                        if matched_hotel.get("price_per_night_usd")
                        else "TBC"
                    )
                    item["key"] = "hotel_stay"
                    item["icon"] = "hotel"
                    if not item.get("time"):
                        item["time"] = "20:30" if day_idx == 0 else "09:30"
                    if json.dumps(item, sort_keys=True, ensure_ascii=False) != before:
                        stats["normalized_items"] += 1
                    clean_items.append(item)
                    continue

                if " | " in name:
                    item["name"] = name.split(" | ")[0].strip()
                    name = item["name"]

                matched_catalog, kind = _resolve_catalog_item(name, key)
                if not matched_catalog:
                    stats["removed_items"] += 1
                    stats["invalid_items"] += 1
                    continue

                item["name"] = matched_catalog.get("name", name)
                item["key"] = matched_catalog.get("key", key) or key
                item["icon"] = kind
                item["cost"] = matched_catalog.get("price") or _safe_price(item.get("cost")) or "TBC"
                item["_lat"] = matched_catalog.get("lat")
                item["_lng"] = matched_catalog.get("lng")
                item["_type"] = matched_catalog.get("type")

                name_norm = item["name"].lower().strip()
                if name_norm in seen_items:
                    stats["removed_items"] += 1
                    stats["duplicate_items"] += 1
                    continue
                seen_items.add(name_norm)

                if json.dumps(item, sort_keys=True, ensure_ascii=False) != before:
                    stats["normalized_items"] += 1
                clean_items.append(item)

            day["items"] = sorted(clean_items, key=lambda item: _sort_time_value(item.get("time", "")))
            activity_count = sum(
                1 for item in day["items"] if str(item.get("icon") or "").lower() in {"activity", "restaurant"}
            )
            if 0 < day_idx < len(days) - 1 and activity_count < 2:
                stats["sparse_days"].append(
                    {
                        "day_index": day_idx + 1,
                        "activity_count": activity_count,
                    }
                )
            normalized_days.append(day)

        return normalized_days, stats, seen_items

    final_itineraries: dict = {}
    normalized_itineraries: dict = {}
    option_meta: dict = {}
    validation_report: dict = {}
    for option_key, option in options.items():
        option_meta[option_key] = {
            "label": option.get("label", f"Option {option_key}"),
            "desc": option.get("desc", ""),
            "budget": option.get("budget", ""),
            "style": option.get("style", ""),
            "badge": option.get("badge", f"Option {option_key}"),
        }
        normalized_days, stats, used_names = _process_days(option.get("days", []))
        boundary_ready_days = _enforce_boundary_feasibility(normalized_days, used_names, stats)
        route_repaired_days = _repair_routes(boundary_ready_days, used_names, stats)
        final_days, boundary_repairs = _ensure_boundary_days(
            route_repaired_days,
            compact_hotels,
            compact_flights_out,
            compact_flights_ret,
        )
        final_days = _reslot_days(final_days)
        stats["boundary_repairs"] = boundary_repairs
        stats["route_warnings"] = _route_warnings(final_days)
        normalized_itineraries[option_key] = _strip_internal_fields(normalized_days)
        final_itineraries[option_key] = _strip_internal_fields(final_days)
        validation_report[option_key] = stats

        if stats["removed_items"]:
            tool_log.append(
                f"[validator] Option {option_key}: removed {stats['removed_items']} item(s) "
                f"(invalid={stats['invalid_items']}, duplicate={stats['duplicate_items']})"
            )
        if stats["normalized_items"]:
            tool_log.append(
                f"[validator] Option {option_key}: normalized {stats['normalized_items']} item(s) to research inventory"
            )
        if stats["sparse_days"]:
            sparse_desc = ", ".join(f"Day {entry['day_index']} ({entry['activity_count']} activity)" for entry in stats["sparse_days"])
            tool_log.append(f"[validator] Option {option_key}: sparse middle days detected -> {sparse_desc}")
        if boundary_repairs:
            tool_log.append(f"[boundary] Option {option_key}: " + "; ".join(boundary_repairs))
        if stats.get("meal_repairs"):
            tool_log.append(f"[meals] Option {option_key}: " + "; ".join(stats["meal_repairs"]))
        if stats.get("boundary_filtered"):
            tool_log.append(f"[timing] Option {option_key}: " + "; ".join(stats["boundary_filtered"]))
        if stats.get("route_repairs"):
            tool_log.append(f"[routing] Option {option_key}: " + "; ".join(stats["route_repairs"][:3]))
        if stats["route_warnings"]:
            tool_log.append(f"[routing-warn] Option {option_key}: " + "; ".join(stats["route_warnings"][:3]))

    return final_itineraries, option_meta, normalized_itineraries, validation_report


def revise_itinerary(state: dict, critique: str, current_result: dict) -> dict:
    if not _inventory_cache:
        return {"error": "No cached inventory - run planner_agent_1() first"}

    inv = _inventory_cache
    dest = state.get("destination", "")
    origin = state.get("origin", "")
    dates = state.get("dates", "")
    duration = state.get("duration", "")
    budget = state.get("budget", "")

    current_json = json.dumps(current_result.get("itineraries", {}), indent=2, ensure_ascii=False)
    prompt = f"""You are Agent3 Planner in revision mode.

Trip: {origin} -> {dest} | {dates} | {duration} | budget: {budget}

Current itineraries:
{current_json}

Critique to fix:
{critique}

Structured inventory JSON:
{json.dumps(inv.get("prompt_inventory", {}), indent=2, ensure_ascii=False)}

Readable inventory summary:

Outbound flights:
{inv["flight_out_text"]}

Return flights:
{inv["flight_ret_text"]}

Hotels:
{inv["hotel_list_text"]}

Attractions:
{inv["att_list_text"]}

Restaurants:
{inv["rest_list_text"]}

Rules:
- Keep 3 materially different options.
- Use ONLY names and keys from the structured inventory JSON.
- Keep outbound flight key as "flight_outbound", return flight key as "flight_return", hotel key as "hotel_stay".
- For attractions and restaurants, preserve the exact key and exact name from the structured inventory JSON.
- Day 1 flight time must use arrival_time. Last-day flight time must use departure_time.
- Keep restaurants near the surrounding activities and avoid cross-city zig-zagging.
- Respect the 3-hour airport buffer on the last day.
- Use realistic local times, not a fixed repeating template.
- Prefer fixing the critique by rearranging or swapping inventory items, not by inventing new items.

Return ONLY valid JSON:
{{
  "chain_of_thought": "Short planning summary.",
  "options": {{
    "A": {{"label":"...","desc":"...","budget":"...","style":"...","badge":"...","days":[]}},
    "B": {{"label":"...","desc":"...","budget":"...","style":"...","badge":"...","days":[]}},
    "C": {{"label":"...","desc":"...","budget":"...","style":"...","badge":"...","days":[]}}
  }}
}}
"""

    tool_log = [f"[planner_agent_1] Revision mode critique received ({len(critique)} chars)"]
    try:
        llm = _llm()
        response = llm.invoke([SystemMessage(content=prompt)], response_format={"type": "json_object"})
        result = json.loads(response.content)
        options = result.get("options", {})
        itineraries, option_meta, normalized_itineraries, validation_report = _post_process_options(
            options,
            inv["compact_attractions"],
            inv["compact_restaurants"],
            inv["compact_hotels"],
            inv["compact_flights_out"],
            inv["compact_flights_ret"],
            tool_log,
        )
        tool_log.append(f"[planner_agent_1] Revision complete - {list(options.keys())}")
        return {
            "itineraries": itineraries,
            "final_itineraries": itineraries,
            "normalized_itineraries": normalized_itineraries,
            "validated_itineraries": normalized_itineraries,
            "raw_planner_output": result,
            "validation_report": validation_report,
            "option_meta": option_meta,
            "planner_chain_of_thought": result.get("chain_of_thought", ""),
            "chain_of_thought": result.get("chain_of_thought", ""),
            "research": inv.get("research", {}),
            "tool_log": tool_log,
            "flight_options_outbound": inv["compact_flights_out"],
            "flight_options_return": inv["compact_flights_ret"],
            "hotel_options": inv.get("hotel_opts", []),
        }
    except Exception as exc:
        print(f"[Planner1] Revision error: {exc}")
        return {"error": str(exc)}


# Deterministic planner profiles used by the main planner path.
_OPTION_PROFILES = {
    "A": {
        "label": "Balanced Highlights",
        "desc": "A balanced plan that covers the strongest relevant picks for the user's stated interests with a steady full-day pace.",
        "budget": "SGD 5000 (flight + hotel; meals/transport TBC)",
        "style": "Balanced / top-relevance route",
        "badge": "Best for a broad first pass through the destination",
        "selection_mode": "coverage",
        "meal_role": "highlight",
        "arrival_restaurant_offset": 0,
        "start_min": 9 * 60 + 20,
        "target_activity_count": 3,
        "min_activity_count": 2,
        "seed_offsets": [0, 1, 2],
        "focus": {"culture": 1.0, "food": 1.0, "nature": 1.0, "modern": 0.55, "landmark": 0.85, "quirky": -0.8},
        "discouraged_activity_penalty": 18,
        "discouraged_restaurant_penalty": 16,
        "route_penalty": 1.6,
        "fallback_route_penalty": 1.9,
        "restaurant_route_penalty": 1.4,
        "nearby_route_penalty": 1.6,
        "spread_bonus": 1.15,
        "fresh_activity_bonus": 0,
        "fresh_restaurant_bonus": 0,
        "fresh_activity_quota": 0,
        "fresh_restaurant_quota": 0,
        "hotel_mode": "premium",
    },
    "B": {
        "label": "Alternative Mix",
        "desc": "A more exploratory plan that avoids repeating the first option where possible and surfaces different but still relevant alternatives.",
        "budget": "SGD 5000 (flight + hotel; meals/transport TBC)",
        "style": "Alternative / novelty-first route",
        "badge": "Best for travelers who want a different take on the same trip brief",
        "selection_mode": "alternative",
        "meal_role": "local_mix",
        "arrival_restaurant_offset": 1,
        "start_min": 9 * 60 + 50,
        "target_activity_count": 3,
        "min_activity_count": 2,
        "seed_offsets": [2, 4, 6],
        "focus": {"culture": 1.0, "food": 1.0, "nature": 1.0, "modern": 0.55, "landmark": 0.7, "quirky": -0.9},
        "discouraged_activity_penalty": 42,
        "discouraged_restaurant_penalty": 22,
        "route_penalty": 1.5,
        "fallback_route_penalty": 1.8,
        "restaurant_route_penalty": 1.25,
        "nearby_route_penalty": 1.5,
        "fresh_activity_bonus": 12,
        "fresh_restaurant_bonus": 6,
        "fresh_activity_quota": 2,
        "fresh_restaurant_quota": 1,
        "hotel_mode": "budget",
    },
    "C": {
        "label": "Focused Depth",
        "desc": "A coherence-first alternative that keeps the trip compact, protects stronger thematic fit, and trades a little breadth for a more internally consistent route.",
        "budget": "SGD 5000 (flight + hotel; meals/transport TBC)",
        "style": "Focused / lower-friction route",
        "badge": "Best for travelers who want a more coherent version of the same brief",
        "selection_mode": "immersion",
        "meal_role": "district_coherent",
        "arrival_restaurant_offset": 2,
        "start_min": 10 * 60 + 10,
        "target_activity_count": 2,
        "min_activity_count": 2,
        "seed_offsets": [5, 8, 11],
        "focus": {"culture": 1.15, "food": 0.95, "nature": 1.15, "modern": 0.25, "landmark": 0.45, "quirky": -1.0},
        "discouraged_activity_penalty": 34,
        "discouraged_restaurant_penalty": 24,
        "route_penalty": 2.5,
        "fallback_route_penalty": 2.9,
        "restaurant_route_penalty": 2.1,
        "nearby_route_penalty": 2.4,
        "cluster_seed_penalty": 1.6,
        "cluster_radius_km": 5.0,
        "spread_bonus": 0.0,
        "fresh_activity_bonus": 6,
        "fresh_restaurant_bonus": 6,
        "fresh_activity_quota": 2,
        "fresh_restaurant_quota": 1,
        "hotel_mode": "quiet",
    },
}


_PREFERENCE_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "best", "top", "trip", "travel", "traveller", "traveler",
    "place", "places", "spot", "spots", "city", "around", "near", "nearby", "good", "great", "must", "famous",
    "to", "of", "in", "on", "at", "or", "a", "an", "is", "are", "be", "more", "less", "very",
}

_PREFERENCE_FAMILIES = {
    "culture": {"traditional", "cultural", "culture", "heritage", "historical", "history", "temple", "shrine", "authentic", "religious"},
    "food": {"food", "street", "hawker", "snack", "cuisine", "culinary", "market", "eat", "local", "restaurant"},
    "nature": {"garden", "park", "nature", "scenic", "zen", "outdoor", "botanic"},
    "modern": {"modern", "science", "digital", "interactive", "discovery", "skyline", "observation", "deck"},
    "museumish": {"museum", "gallery", "exhibit"},
}

_ITEM_SIGNAL_KEYWORDS = {
    "heritage": {
        "heritage", "historic", "historical", "history", "traditional", "cultural", "former house",
        "relic", "clan", "memorial", "heritage gallery", "cultural centre", "cultural center",
    },
    "temple": {"temple", "shrine", "monastery", "mosque", "church"},
    "museum": {"museum", "gallery", "exhibit", "exhibition"},
    "local_food": {"hawker", "food centre", "food center", "food street", "market", "street food", "snack", "kopitiam"},
    "traditional_cuisine": {
        "traditional restaurant", "authentic", "regional", "local cuisine", "heritage dining",
        "signature menu", "chef special", "cuisine",
    },
    "plant_based": {"vegan", "vegetarian", "plant-based"},
    "fine_dining": {"fine dining", "chef's table", "tasting menu", "michelin", "degustation"},
    "casual_food": {"snack", "market", "street", "food centre", "food center", "food street", "cafe", "eatery", "stall"},
    "nature": {"garden", "park", "grove", "tree", "botanic", "nature", "trail", "tunnel"},
    "scenic": {"tower", "bridge", "skyline", "observation", "deck", "skypark", "view", "waterfront"},
    "modern": {"science", "discovery", "interactive", "digital", "innovation", "artscience"},
    "quirky": {"hell", "trick", "optical", "horror", "immersive", "selfie", "monster"},
    "neighborhood": {"district", "quarter", "street", "lane", "village", "town", "old town"},
}

_AREA_LIKE_KEYWORDS = {
    "yokocho",
    "memory lane",
    "food street",
    "ramen street",
    "old town",
    "market lane",
    "alley",
}

_TOUR_LIKE_KEYWORDS = {
    "tour",
    "tours",
    "walking tour",
    "food tours",
    "guided tour",
}

_NIGHT_ONLY_KEYWORDS = {
    "night & light",
    "night and light",
    "night light",
    "illumination",
}

_PHOTO_SPOT_KEYWORDS = {
    "word mark",
    "monument",
}

_FOOD_ENTITY_KEYWORDS = {
    "restaurant",
    "ramen",
    "sushi",
    "bistro",
    "bar",
    "cafe",
    "eatery",
    "grill",
    "dining",
    "buffet",
    "kitchen",
    "izakaya",
    "paradise",
    "gyukatsu",
    "tonkatsu",
}

_NON_LOCAL_CUISINE_KEYWORDS = {
    "indian",
    "italian",
    "french",
    "spanish",
    "mexican",
    "american",
    "burger",
    "steakhouse",
    "brasserie",
}


def _numeric_rating(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _tokenize_text(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9']+", str(text or "").lower()) if token]


def _build_preference_model(state: dict) -> dict:
    texts: list[str] = [str(state.get("preferences") or "")]
    user_profile = state.get("user_profile") or {}
    texts.extend(str(pref) for pref in user_profile.get("prefs", []) if pref)
    soft_preferences = state.get("soft_preferences") or {}
    texts.extend(str(tag) for tag in soft_preferences.get("interest_tags", []) if tag)
    texts.append(str(soft_preferences.get("vibe") or ""))
    for query in state.get("search_queries") or []:
        texts.append(str(query.get("query") or ""))

    ignore_tokens = set(_tokenize_text(state.get("origin") or "")) | set(_tokenize_text(state.get("destination") or ""))
    token_counts: Counter[str] = Counter()
    for text in texts:
        tokens = [
            token
            for token in _tokenize_text(text)
            if len(token) >= 3 and token not in _PREFERENCE_STOPWORDS and token not in ignore_tokens
        ]
        token_counts.update(tokens)
        token_counts.update(
            f"{tokens[idx]} {tokens[idx + 1]}"
            for idx in range(len(tokens) - 1)
            if tokens[idx] not in _PREFERENCE_STOPWORDS and tokens[idx + 1] not in _PREFERENCE_STOPWORDS
        )

    families = {
        family: sum(weight for token, weight in token_counts.items() if any(keyword == token or keyword in token.split() for keyword in keywords))
        for family, keywords in _PREFERENCE_FAMILIES.items()
    }
    return {"token_counts": token_counts, "families": families}


def _text_blob(item: dict) -> str:
    return " ".join(
        str(item.get(key, "") or "")
        for key in ("name", "type", "description", "address", "matches_preference")
    ).lower()


def _blob_tokens(text: str) -> set[str]:
    tokens = _tokenize_text(text)
    return set(tokens) | {f"{tokens[idx]} {tokens[idx + 1]}" for idx in range(len(tokens) - 1)}


def _preference_overlap_score(blob: str, preference_model: dict, *, cap: int = 6) -> float:
    tokens = _blob_tokens(blob)
    total = 0.0
    for token, weight in preference_model.get("token_counts", {}).items():
        if token in tokens:
            total += min(weight, 3)
    return min(total, cap)


def _item_signal_counts(blob: str) -> dict[str, int]:
    tokens = _blob_tokens(blob)
    signals = {
        signal: sum(1 for keyword in keywords if keyword in tokens or keyword in blob)
        for signal, keywords in _ITEM_SIGNAL_KEYWORDS.items()
    }
    signals["heritage"] += signals["temple"] + min(signals["neighborhood"], 1)
    signals["local_food"] += min(signals["neighborhood"], 1) if "food" in blob or "hawker" in blob else 0
    return signals


def _preference_emphasis(preference_model: dict, token: str) -> int:
    return int(preference_model.get("token_counts", {}).get(token, 0))


def _normalized_name(item: dict | None) -> str:
    if not item:
        return ""
    return str(item.get("name") or "").strip().lower()


def _classify_itinerary_candidate(item: dict) -> str:
    blob = _text_blob(item)
    if any(keyword in blob for keyword in _TOUR_LIKE_KEYWORDS):
        return "tour"
    if any(keyword in blob for keyword in _NIGHT_ONLY_KEYWORDS):
        return "night_only"
    if any(keyword in blob for keyword in _PHOTO_SPOT_KEYWORDS):
        return "photo_spot"
    if any(keyword in blob for keyword in _AREA_LIKE_KEYWORDS):
        return "area"
    if any(keyword in blob for keyword in _FOOD_ENTITY_KEYWORDS):
        return "restaurant"
    return "attraction"


def _is_normal_activity_candidate(item: dict) -> bool:
    return _classify_itinerary_candidate(item) == "attraction"


def _is_normal_restaurant_candidate(item: dict) -> bool:
    return _classify_itinerary_candidate(item) == "restaurant"


def _exclude_same_place(candidates: list[dict], blocked_names: set[str]) -> list[dict]:
    if not blocked_names:
        return candidates
    return [item for item in candidates if _normalized_name(item) not in blocked_names]


def _merge_used_place_names(*name_sets: set[str]) -> set[str]:
    merged: set[str] = set()
    for names in name_sets:
        merged.update(name for name in names if name)
    return merged


def _items_centroid(items: list[dict]) -> tuple[float, float] | None:
    points = [
        (float(item.get("lat")), float(item.get("lng")))
        for item in items
        if item.get("lat") is not None and item.get("lng") is not None
    ]
    if not points:
        return None
    lat = sum(point[0] for point in points) / len(points)
    lng = sum(point[1] for point in points) / len(points)
    return (lat, lng)


def _is_nonlocal_restaurant(item: dict) -> bool:
    blob = _text_blob(item)
    return any(keyword in blob for keyword in _NON_LOCAL_CUISINE_KEYWORDS)


def _activity_relevance_score(item: dict, profile: dict, preference_model: dict) -> float:
    blob = _text_blob(item)
    signals = _item_signal_counts(blob)
    focus = profile.get("focus", {})
    culture_pref = preference_model["families"].get("culture", 0)
    food_pref = preference_model["families"].get("food", 0)
    nature_pref = preference_model["families"].get("nature", 0)
    modern_pref = preference_model["families"].get("modern", 0)
    museum_pref = preference_model["families"].get("museumish", 0)
    traditional_pref = (
        _preference_emphasis(preference_model, "traditional")
        + _preference_emphasis(preference_model, "traditional cultural")
        + _preference_emphasis(preference_model, "heritage")
    )
    local_food_pref = (
        _preference_emphasis(preference_model, "local")
        + _preference_emphasis(preference_model, "street food")
        + _preference_emphasis(preference_model, "hawker")
        + _preference_emphasis(preference_model, "local street")
    )

    score = _preference_overlap_score(blob, preference_model, cap=10) * 1.4
    score += signals["heritage"] * (4.8 + focus.get("culture", 1.0) * 2.6 + culture_pref * 0.65 + traditional_pref * 0.85)
    score += signals["temple"] * (4.0 + focus.get("culture", 1.0) * 2.1 + culture_pref * 0.55 + traditional_pref * 0.65)
    score += signals["neighborhood"] * (1.8 + focus.get("culture", 1.0) * 0.8 + food_pref * 0.25)
    score += signals["nature"] * (2.1 + focus.get("nature", 0.5) * 1.8 + nature_pref * 0.45)
    score += signals["scenic"] * (1.4 + focus.get("landmark", 0.4) * 1.7 + culture_pref * 0.12)
    score += signals["museum"] * (1.2 + museum_pref * 0.35 + focus.get("culture", 1.0) * 0.45)

    if signals["modern"]:
        score -= signals["modern"] * max(0.0, traditional_pref * 1.15 + culture_pref * 0.5 + food_pref * 0.35 - modern_pref * 0.55 - focus.get("modern", 0.0) * 3.0)
    if signals["quirky"]:
        score -= signals["quirky"] * (5.5 + traditional_pref * 0.9 + culture_pref * 0.45 + food_pref * 0.25)
    if "natural history" in blob and traditional_pref:
        score -= 6.0
    if "discovery centre" in blob or "science centre" in blob or "science center" in blob:
        score -= 7.0 + traditional_pref * 0.35
    if signals["museum"] and signals["modern"] and not (signals["heritage"] or signals["temple"]):
        score -= 4.5 + traditional_pref * 0.55
    if local_food_pref and not (signals["heritage"] or signals["temple"] or signals["nature"] or signals["scenic"]):
        score -= 1.8
    return score


def _restaurant_relevance_score(item: dict, profile: dict, preference_model: dict) -> float:
    blob = _text_blob(item)
    signals = _item_signal_counts(blob)
    focus = profile.get("focus", {})
    rating = _numeric_rating(item.get("rating"))
    culture_pref = preference_model["families"].get("culture", 0)
    food_pref = preference_model["families"].get("food", 0)
    modern_pref = preference_model["families"].get("modern", 0)
    traditional_pref = (
        _preference_emphasis(preference_model, "traditional")
        + _preference_emphasis(preference_model, "traditional cultural")
        + _preference_emphasis(preference_model, "historical")
        + _preference_emphasis(preference_model, "authentic")
    )
    fine_pref = (
        _preference_emphasis(preference_model, "fine dining")
        + _preference_emphasis(preference_model, "fine")
    )
    local_food_pref = (
        _preference_emphasis(preference_model, "local")
        + _preference_emphasis(preference_model, "street food")
        + _preference_emphasis(preference_model, "hawker")
        + _preference_emphasis(preference_model, "local street")
    )
    score = _preference_overlap_score(blob, preference_model, cap=10) * 1.6
    score += signals["local_food"] * (5.8 + focus.get("food", 1.0) * 2.4 + food_pref * 0.75 + local_food_pref * 0.85)
    score += signals["neighborhood"] * (1.6 + focus.get("culture", 1.0) * 0.45 + culture_pref * 0.2)
    score += signals["traditional_cuisine"] * (4.6 + traditional_pref * 0.7 + culture_pref * 0.35 + food_pref * 0.25)
    score += signals["fine_dining"] * (3.8 + fine_pref * 0.9 + food_pref * 0.3)
    if (traditional_pref or fine_pref) and not any(keyword in blob for keyword in _NON_LOCAL_CUISINE_KEYWORDS):
        score += 2.2 + max(0.0, rating - 4.0) * 4.0
    if signals["modern"]:
        score -= signals["modern"] * max(0.0, local_food_pref * 0.65 + food_pref * 0.25 - modern_pref * 0.4)
    if local_food_pref and not signals["local_food"]:
        score -= 5.0
    if (traditional_pref or fine_pref) and any(keyword in blob for keyword in _NON_LOCAL_CUISINE_KEYWORDS):
        score -= 8.0 + traditional_pref * 0.35 + fine_pref * 0.4
    if (traditional_pref or local_food_pref) and any(keyword in blob for keyword in _NON_LOCAL_CUISINE_KEYWORDS):
        score -= 5.2 + traditional_pref * 0.35 + local_food_pref * 0.25
    if any(marker in blob for marker in ("italian", "grill", "buffet", "brasserie", "steakhouse")) and not signals["local_food"]:
        score -= 4.5 + local_food_pref * 0.25
    return score


def _hotel_price_sgd(hotel: dict) -> float:
    try:
        return float(hotel.get("price_per_night_usd")) * 1.35
    except Exception:
        return 9999.0


def _flight_stop_rank_local(flight: dict) -> int:
    display = str(flight.get("display") or "").lower()
    if "nonstop" in display or "direct" in display:
        return 0
    match = re.search(r"(\d+)\s*stop", display)
    if match:
        return int(match.group(1))
    return 9


def _pick_outbound_flight(compact_flights_out: list[dict]) -> dict | None:
    if not compact_flights_out:
        return None
    return sorted(
        compact_flights_out,
        key=lambda flight: (
            _flight_stop_rank_local(flight),
            _minutes_since_midnight(flight.get("arrival_time", "")) or 9999,
            flight.get("duration_min") or 9999,
            flight.get("price_usd") or 9999,
        ),
    )[0]


def _pick_return_flight(compact_flights_ret: list[dict]) -> dict | None:
    if not compact_flights_ret:
        return None
    return sorted(
        compact_flights_ret,
        key=lambda flight: (
            _flight_stop_rank_local(flight),
            -(_minutes_since_midnight(flight.get("departure_time", "")) or 0),
            flight.get("price_usd") or 9999,
        ),
    )[0]


def _pick_hotels_by_profile(compact_hotels: list[dict]) -> dict[str, dict | None]:
    remaining = list(compact_hotels)
    picks: dict[str, dict | None] = {}
    for option_key, profile in _OPTION_PROFILES.items():
        if not remaining:
            picks[option_key] = compact_hotels[0] if compact_hotels else None
            continue

        def _score(hotel: dict) -> tuple:
            rating = _numeric_rating(hotel.get("rating"))
            price = _hotel_price_sgd(hotel)
            blob = _text_blob(hotel)
            quiet_bonus = 8 if any(word in blob for word in ("apartment", "suite", "residence")) else 0
            if profile["hotel_mode"] == "budget":
                return (price, -rating)
            if profile["hotel_mode"] == "quiet":
                return (-rating - quiet_bonus, price)
            return (-rating, price)

        chosen = sorted(remaining, key=_score)[0]
        picks[option_key] = chosen
        remaining = [hotel for hotel in remaining if hotel.get("name") != chosen.get("name")]
    return picks


def _activity_score(item: dict, profile: dict, preference_model: dict, discouraged_names: set[str] | None = None) -> float:
    blob = _text_blob(item)
    signals = _item_signal_counts(blob)
    rating = _numeric_rating(item.get("rating")) or (4.1 if (signals["heritage"] or signals["temple"]) else 0.0)
    score = _activity_relevance_score(item, profile, preference_model) + rating * 2.4
    if item.get("matches_preference"):
        score += 3
    if item.get("price") in (None, "", "TBC"):
        score += 0.5
    if discouraged_names:
        if _normalized_name(item) in discouraged_names:
            score -= profile.get("discouraged_activity_penalty", 18)
        else:
            score += profile.get("fresh_activity_bonus", 0)
    return score


def _restaurant_score(item: dict, profile: dict, preference_model: dict, discouraged_names: set[str] | None = None) -> float:
    blob = _text_blob(item)
    rating = _numeric_rating(item.get("rating")) or 4.0
    relevance = _restaurant_relevance_score(item, profile, preference_model)
    score = relevance + rating * 2.2 + _restaurant_role_adjustment(item, profile, preference_model)
    if "vegetarian" in blob or "vegan" in blob:
        score += 3
    if discouraged_names:
        if _normalized_name(item) in discouraged_names:
            penalty = float(profile.get("discouraged_restaurant_penalty", 16))
            if relevance >= 8:
                penalty *= 0.3
            elif relevance >= 2:
                penalty *= 0.55
            score -= penalty
        else:
            score += profile.get("fresh_restaurant_bonus", 0)
    return score


def _travel_minutes(a: dict | None, b: dict | None, base: int = 20) -> int:
    if not a or not b:
        return base + 10
    dist = _haversine_km(a.get("lat") or a.get("_lat"), a.get("lng") or a.get("_lng"), b.get("lat") or b.get("_lat"), b.get("lng") or b.get("_lng"))
    if dist is None:
        return base + 10
    return int(base + min(dist * 8, 45))


def _activity_duration_minutes(item: dict) -> int:
    blob = _text_blob(item)
    if "museum" in blob or "teamlab" in blob:
        return 105
    if "garden" in blob or "park" in blob:
        return 90
    if "temple" in blob or "shrine" in blob or "tower" in blob or "sky" in blob:
        return 75
    return 90


def _activity_latest_close_minutes(item: dict, service_date: date | None = None) -> int | None:
    close_minutes = _latest_close_from_weekday_descriptions(
        item.get("weekday_descriptions") or [],
        service_date,
    )
    if close_minutes is not None:
        return close_minutes
    return _latest_close_from_hours_text(str(item.get("hours") or ""))


def _activity_latest_start_minutes(item: dict, service_date: date | None = None) -> int:
    actual_close = _activity_latest_close_minutes(item, service_date)
    if actual_close is not None:
        if actual_close < 0:
            return -1
        return max(0, actual_close - _activity_duration_minutes(item) - 15)
    blob = _text_blob(item)
    signals = _item_signal_counts(blob)
    if "teamlab" in blob or any(word in blob for word in ("tower", "sky", "observation", "night", "light")):
        return 19 * 60
    if signals.get("museum", 0) > 0:
        return 15 * 60 + 30
    if signals.get("nature", 0) > 0 or any(word in blob for word in ("garden", "park", "botanic")):
        return 16 * 60
    if signals.get("temple", 0) > 0 or any(word in blob for word in ("shrine", "church", "mosque", "palace")):
        return 16 * 60 + 30
    if signals.get("scenic", 0) > 0 or signals.get("neighborhood", 0) > 0:
        return 18 * 60
    return 17 * 60


def _pick_next_activity_for_time(
    remaining: list[tuple[int, dict]],
    current_minutes: int,
    anchor: dict | None,
    service_date: date | None = None,
) -> tuple[int, dict] | None:
    feasible: list[tuple[tuple, tuple[int, dict]]] = []
    fallback: list[tuple[tuple, tuple[int, dict]]] = []
    for original_idx, item in remaining:
        travel = _travel_minutes(anchor, item, base=18 if anchor and str(anchor.get("key", "")).startswith("restaurant") else 15) if anchor else 0
        start_at = current_minutes + travel
        latest_start = _activity_latest_start_minutes(item, service_date)
        if latest_start < 0:
            continue
        lateness = max(0, start_at - latest_start)
        score = (
            lateness > 0,
            lateness,
            travel,
            latest_start,
            original_idx,
        )
        if lateness == 0:
            feasible.append((score, (original_idx, item)))
        else:
            fallback.append((score, (original_idx, item)))
    if feasible:
        feasible.sort(key=lambda pair: pair[0])
        return feasible[0][1]
    if fallback:
        fallback.sort(key=lambda pair: pair[0])
        return fallback[0][1]
    return None


def _meal_duration_minutes(item: dict) -> int:
    blob = _text_blob(item)
    signals = _item_signal_counts(blob)
    if signals.get("fine_dining", 0) > 0:
        return 90
    return 75


def _prepare_seed_candidates(
    pool: list[dict],
    profile: dict,
    preference_model: dict,
    cluster_radius: float = 5.0,
    max_candidates: int = 5,
    discouraged_names: set[str] | None = None,
) -> list[dict]:
    """Build a short-list of seed candidates with their cluster contents
    so the LLM can make an informed thematic choice."""
    if not pool:
        return []

    ranked = sorted(
        pool,
        key=lambda item: _activity_score(item, profile, preference_model, discouraged_names),
        reverse=True,
    )

    seen_seeds: set[str] = set()
    candidates: list[dict] = []

    for seed in ranked[: max_candidates * 3]:
        seed_name = _normalized_name(seed)
        if seed_name in seen_seeds:
            continue
        seen_seeds.add(seed_name)

        cluster_members = [
            item
            for item in ranked
            if (
                item is seed
                or (
                    (_haversine_km(
                        seed.get("lat"), seed.get("lng"),
                        item.get("lat"), item.get("lng"),
                    ) or 999)
                    <= cluster_radius
                )
            )
        ]
        if len(cluster_members) < 2:
            continue

        candidates.append({
            "seed": seed,
            "name": seed.get("name", ""),
            "type": str(seed.get("type") or seed.get("category") or ""),
            "cluster_members": [
                {
                    "name": m.get("name", ""),
                    "type": str(m.get("type") or m.get("category") or ""),
                }
                for m in cluster_members[:6]
                if m is not seed
            ],
            "cluster_size": len(cluster_members),
        })
        if len(candidates) >= max_candidates:
            break

    return candidates


def _llm_select_seed(
    candidates: list[dict],
    day_index: int,
    option_key: str,
    profile: dict,
    preference_model: dict,
    prior_themes: list[str],
) -> tuple[dict | None, str, str]:
    """Ask the LLM to choose the best cluster seed for one day.

    Returns (seed_item_dict, theme_string, reason_string).
    Falls back to the first candidate on any error.
    """
    if not candidates:
        return None, "mixed sightseeing", "no seed candidates available"

    pref_tags = [
        token
        for token, count in sorted(
            preference_model.get("token_counts", {}).items(),
            key=lambda x: -x[1],
        )
        if count >= 1 and len(token) >= 3
    ][:8]

    display_candidates = [
        {
            "name": c["name"],
            "type": c["type"],
            "nearby_attractions": [
                {"name": m["name"], "type": m["type"]}
                for m in c["cluster_members"][:5]
            ],
        }
        for c in candidates
    ]

    prompt_payload = json.dumps(
        {
            "task": "select_day_anchor",
            "day_number": day_index + 2,
            "option_style": profile.get("style", ""),
            "user_preferences": pref_tags,
            "themes_already_used": prior_themes,
            "candidates": display_candidates,
        },
        ensure_ascii=False,
    )

    system_prompt = (
        "You select which attraction should anchor one day of a travel itinerary. "
        "Pick the candidate whose cluster of nearby attractions forms the most "
        "coherent thematic day for the stated user preferences. "
        "Prefer themes not already used on prior days when possible.\n\n"
        "IMPORTANT: The 'theme' must accurately describe ALL attractions in the "
        "cluster (the seed AND its nearby_attractions), not just the seed. "
        "Look at the 'type' field of every item in nearby_attractions. "
        "If the cluster contains a mix (e.g. a garden, a museum, and a digital "
        "art venue), the theme should reflect that mix.\n\n"
        "The 'reason' should explain why this particular combination of "
        "attractions makes a coherent or interesting day, mentioning the "
        "specific types involved.\n\n"
        "Return ONLY valid JSON with exactly three keys:\n"
        '  "selected_name": the exact name string of the chosen candidate,\n'
        '  "theme": a 2-5 word day theme that covers ALL cluster members,\n'
        '  "reason": one sentence explaining why this cluster makes a good day.\n'
    )

    try:
        response = _llm().invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_payload),
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.content)
        selected_name = str(parsed.get("selected_name", "")).strip()
        theme = str(parsed.get("theme", "")).strip() or "mixed sightseeing"
        reason = str(parsed.get("reason", "")).strip() or ""

        for c in candidates:
            if c["name"].strip().lower() == selected_name.lower():
                return c["seed"], theme, reason

        for c in candidates:
            if (
                selected_name.lower() in c["name"].lower()
                or c["name"].lower() in selected_name.lower()
            ):
                return c["seed"], theme, reason

        return (
            candidates[0]["seed"],
            theme,
            f"LLM selected '{selected_name}' which did not match any candidate name; "
            f"using top-ranked candidate '{candidates[0]['name']}' instead",
        )
    except Exception as exc:
        return (
            candidates[0]["seed"],
            "mixed sightseeing",
            f"LLM seed selection failed ({exc}); deterministic fallback",
        )


def _meal_trace(
    restaurant: dict | None,
    meal_kind: str,
    preference_model: dict,
) -> dict:
    """Build a decision-trace entry for one meal pick."""
    if not restaurant:
        return {"meal_kind": meal_kind, "picked": None}

    cuisine = str(
        restaurant.get("type") or restaurant.get("category") or "restaurant"
    ).strip()
    blob = _text_blob(restaurant)
    matched = [
        token
        for token, count in preference_model.get("token_counts", {}).items()
        if count >= 1 and len(token) >= 3 and token in blob
    ][:4]

    return {
        "meal_kind": meal_kind,
        "name": restaurant.get("name", ""),
        "cuisine": cuisine,
        "matched_preferences": matched,
    }


def _name_lookup(items: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = _normalized_name(item)
        if name and name not in lookup:
            lookup[name] = item
    return lookup


def _trace_day_type(day_label: str, day_index: int, total_days: int, original_entry: dict) -> str:
    day_type = str((original_entry or {}).get("day_type") or "").strip()
    if day_type:
        return day_type
    if day_index == 0:
        return "arrival"
    if day_index == max(0, total_days - 1):
        return "departure"
    return "middle"


def _trace_theme_from_items(original_entry: dict, activities: list[dict], day_type: str) -> str:
    original_theme = str((original_entry or {}).get("theme") or "").strip()
    if day_type == "arrival":
        return original_theme or "arrival evening"
    if day_type == "departure":
        return original_theme or "departure day"
    if original_theme:
        return original_theme

    type_blob = " ".join(str(item.get("type") or "").lower() for item in activities)
    if "garden" in type_blob and any(token in type_blob for token in ("temple", "shrine", "museum")):
        return "gardens & culture"
    if "garden" in type_blob:
        return "garden day"
    if any(token in type_blob for token in ("temple", "shrine", "museum")):
        return "culture day"
    return "sightseeing day"


def _trace_activities_from_items(day_items: list[dict], attraction_lookup: dict[str, dict]) -> list[dict]:
    activities: list[dict] = []
    for item in day_items:
        if item.get("icon") != "activity":
            continue
        meta = attraction_lookup.get(_normalized_name(item), {})
        activities.append(
            {
                "name": item.get("name", ""),
                "type": str(meta.get("type") or meta.get("category") or "").strip(),
            }
        )
    return activities


def _trace_meals_from_items(
    day_items: list[dict],
    restaurant_lookup: dict[str, dict],
    preference_model: dict,
) -> tuple[dict | None, dict | None]:
    restaurant_items = [item for item in day_items if item.get("icon") == "restaurant"]
    if not restaurant_items:
        return None, None

    lunch_candidates = [
        item for item in restaurant_items
        if (_minutes_since_midnight(item.get("time", "")) or 0) < 16 * 60
    ]
    dinner_candidates = [
        item for item in restaurant_items
        if (_minutes_since_midnight(item.get("time", "")) or 0) >= 16 * 60
    ]

    lunch_item = lunch_candidates[0] if lunch_candidates else None
    dinner_item = dinner_candidates[-1] if dinner_candidates else None

    def _trace_for_item(item: dict | None, meal_kind: str) -> dict | None:
        if not item:
            return None
        meta = restaurant_lookup.get(_normalized_name(item), {})
        if meta:
            return _meal_trace(meta, meal_kind, preference_model)
        return _meal_trace(
            {
                "name": item.get("name", ""),
                "type": item.get("type") or "restaurant",
            },
            meal_kind,
            preference_model,
        )

    return _trace_for_item(lunch_item, "lunch"), _trace_for_item(dinner_item, "dinner")


def _trace_seed_name(original_entry: dict, activities: list[dict], day_type: str = "") -> str | None:
    if day_type in ("arrival", "departure"):
        return None
    if not activities:
        return None
    original_seed = str((original_entry or {}).get("seed_name") or "").strip()
    activity_names = {str(item.get("name") or "").strip() for item in activities}
    if original_seed and original_seed in activity_names:
        return original_seed
    return str(activities[0].get("name") or "").strip() or None


def _trace_seed_reason(
    day_type: str,
    theme: str,
    activities: list[dict],
    lunch: dict | None,
    dinner: dict | None,
    original_reason: str = "",
) -> str:
    """Return the seed reason.

    If the LLM already produced a reason during seed selection AND
    the activities it mentions are still present in the final itinerary,
    keep the original reason.  Otherwise generate a minimal factual
    description from the final activity types (no template prose).
    """
    if day_type == "arrival":
        return ""
    if day_type == "departure":
        return ""

    if not activities:
        return ""

    # Check whether the original LLM reason is still valid
    if original_reason and original_reason not in (
        "no seed candidates available",
        "pool too small for seed selection",
    ):
        final_names_lower = {
            str(a.get("name") or "").strip().lower() for a in activities
        }
        # Count how many final activities are mentioned in the reason
        mentioned = sum(
            1 for name in final_names_lower
            if name and name in original_reason.lower()
        )
        # If at least half of the final activities appear in the reason,
        # the reason is still relevant enough to keep
        if mentioned >= max(1, len(activities) * 0.5):
            return original_reason

    # Fallback: minimal factual line from the final activity types
    names = [str(a.get("name") or "").strip() for a in activities[:3] if str(a.get("name") or "").strip()]
    types = list(dict.fromkeys(
        str(a.get("type") or "").strip().lower()
        for a in activities
        if str(a.get("type") or "").strip()
    ))
    types_text = ", ".join(types) if types else "sightseeing"
    names_text = ", ".join(names)
    return f"{names_text} ({types_text})" if names_text else ""


def _synchronize_planner_decision_trace(
    days: list[dict],
    draft_trace: list[dict],
    attraction_lookup: dict[str, dict],
    restaurant_lookup: dict[str, dict],
    preference_model: dict,
) -> list[dict]:
    """Enrich the draft trace with actual itinerary data.

    Key principle: activities/lunch/dinner are always rebuilt from the
    final items so the trace never references stops that got dropped.
    theme and seed_reason from the LLM are kept when they are still
    consistent with the final items.
    """
    trace_by_day = {
        str(entry.get("day") or "").strip(): entry
        for entry in draft_trace or []
        if isinstance(entry, dict)
    }
    total_days = len(days)
    synced: list[dict] = []

    for day_index, day in enumerate(days):
        day_label = str(day.get("day") or f"Day {day_index + 1}")
        day_items = day.get("items", []) or []
        original_entry = trace_by_day.get(day_label, {})
        day_type = _trace_day_type(day_label, day_index, total_days, original_entry)

        # Always rebuild from final items (most accurate)
        activities = _trace_activities_from_items(day_items, attraction_lookup)
        lunch, dinner = _trace_meals_from_items(day_items, restaurant_lookup, preference_model)

        # Theme: keep LLM theme for middle days, infer for arrival/departure
        theme = _trace_theme_from_items(original_entry, activities, day_type)

        # Seed name: keep LLM choice if still present, None for arrival/departure
        seed_name = _trace_seed_name(original_entry, activities, day_type)

        # Seed reason: keep LLM reason if it still mentions the final activities
        original_reason = str(original_entry.get("seed_reason") or "").strip()
        seed_reason = _trace_seed_reason(
            day_type, theme, activities, lunch, dinner,
            original_reason=original_reason,
        )

        synced.append(
            {
                "day": day_label,
                "day_type": day_type,
                "theme": theme,
                "seed_name": seed_name,
                "seed_reason": seed_reason,
                "activities": activities,
                "lunch": lunch,
                "dinner": dinner,
            }
        )

    return synced


def _choose_cluster_activities(
    attractions: list[dict],
    used_names: set[str],
    profile: dict,
    preference_model: dict,
    count: int = 2,
    light_only: bool = False,
    day_index: int = 0,
    discouraged_names: set[str] | None = None,
    forced_seed: dict | None = None,
) -> list[dict]:
    available = [
        item for item in attractions
        if item.get("name") and item.get("name").lower().strip() not in used_names
    ]
    if not available:
        return []
    broad_available = list(available)

    strong_preference = (
        preference_model["families"].get("culture", 0)
        + preference_model["families"].get("food", 0)
        >= 3
    )

    if light_only:
        def _light_bonus(item: dict) -> float:
            blob = _text_blob(item)
            bonus = 0
            if any(word in blob for word in ("tower", "sky", "temple", "garden", "park")):
                bonus += 6
            if "museum" in blob:
                bonus -= 4
            return bonus
    else:
        _light_bonus = lambda item: 0  # noqa: E731

    scored_available = [
        (
            item,
            _activity_score(item, profile, preference_model, discouraged_names) + _light_bonus(item),
            _activity_relevance_score(item, profile, preference_model),
        )
        for item in available
    ]
    best_relevance = max(relevance for _, _, relevance in scored_available)
    relevance_cutoff = max(28.0, best_relevance - (38 if strong_preference else 24))
    filtered = [item for item, _, relevance in scored_available if relevance >= relevance_cutoff]
    if filtered:
        available = filtered

    if discouraged_names:
        fresh_quota = max(0, int(profile.get("fresh_activity_quota", 0)))
        novel_available = [item for item in available if _normalized_name(item) not in discouraged_names]
        required_novel = min(count, fresh_quota) if fresh_quota else max(1, min(count, 2))
        if len(novel_available) >= required_novel:
            overall_best = max(
                _activity_score(item, profile, preference_model, discouraged_names) + _light_bonus(item)
                for item in available
            )
            novel_best = max(
                _activity_score(item, profile, preference_model, discouraged_names) + _light_bonus(item)
                for item in novel_available
            )
            if novel_best >= overall_best - 12:
                available = novel_available
        else:
            relaxed_novel_available = [
                item for item in broad_available if _normalized_name(item) not in discouraged_names
            ]
            if len(relaxed_novel_available) >= required_novel:
                available = relaxed_novel_available

    ranked = sorted(
        available,
        key=lambda item: _activity_score(item, profile, preference_model, discouraged_names) + _light_bonus(item),
        reverse=True,
    )

    if forced_seed is not None:
        seed = forced_seed
    else:
        selection_mode = str(profile.get("selection_mode") or "coverage")
        seed_offsets = profile.get("seed_offsets") or [0]
        seed_index = min(seed_offsets[day_index % len(seed_offsets)], len(ranked) - 1)
        seed = ranked[seed_index]

        if selection_mode == "immersion":
            cluster_radius = float(profile.get("cluster_radius_km", 5.0))
            if discouraged_names:
                novel_ranked = [item for item in ranked if _normalized_name(item) not in discouraged_names]
                seed_candidates = novel_ranked[: min(len(novel_ranked), 12)] + ranked[: min(len(ranked), 12)]
                deduped_candidates: list[dict] = []
                seen_candidates: set[str] = set()
                for item in seed_candidates:
                    normalized = _normalized_name(item)
                    if normalized and normalized not in seen_candidates:
                        seen_candidates.add(normalized)
                        deduped_candidates.append(item)
                seed_candidates = deduped_candidates
            else:
                seed_candidates = ranked[: min(len(ranked), 12)]

            best_cluster: list[dict] | None = None
            best_seed = seed
            best_cluster_score = float("-inf")
            min_cluster_size = max(2, min(count, 3))
            for seed_candidate in seed_candidates:
                cluster_items = [
                    item
                    for item in ranked
                    if (
                        item is seed_candidate
                        or ((_haversine_km(seed_candidate.get("lat"), seed_candidate.get("lng"), item.get("lat"), item.get("lng")) or 999) <= cluster_radius)
                    )
                ]
                if len(cluster_items) < min_cluster_size:
                    continue
                cluster_ranked = sorted(
                    cluster_items,
                    key=lambda item: _activity_score(item, profile, preference_model, discouraged_names) + _light_bonus(item),
                    reverse=True,
                )
                if discouraged_names:
                    novel_cluster_items = [
                        item for item in cluster_ranked if _normalized_name(item) not in discouraged_names
                    ]
                    top_cluster_items = novel_cluster_items[: min(count, 2)]
                    for item in cluster_ranked:
                        if item not in top_cluster_items:
                            top_cluster_items.append(item)
                        if len(top_cluster_items) >= max(count, min_cluster_size):
                            break
                else:
                    top_cluster_items = cluster_ranked[: max(count, min_cluster_size)]
                fresh_count = sum(
                    1 for item in top_cluster_items
                    if not discouraged_names or _normalized_name(item) not in discouraged_names
                )
                repeated_count = len(top_cluster_items) - fresh_count
                avg_seed_distance = (
                    sum(
                        (_haversine_km(seed_candidate.get("lat"), seed_candidate.get("lng"), item.get("lat"), item.get("lng")) or cluster_radius)
                        for item in top_cluster_items
                        if item is not seed_candidate
                    ) / max(1, len(top_cluster_items) - 1)
                )
                cluster_score = (
                    sum(
                        _activity_score(item, profile, preference_model, discouraged_names) + _light_bonus(item)
                        for item in top_cluster_items[:count]
                    )
                    + fresh_count * 12.0
                    - repeated_count * 28.0
                    - avg_seed_distance * 3.8
                )
                if cluster_score > best_cluster_score:
                    best_cluster_score = cluster_score
                    best_cluster = top_cluster_items
                    best_seed = seed_candidate
            if best_cluster:
                available = best_cluster
                seed = best_seed

    selected = [seed]
    remaining = [item for item in available if item is not seed]
    while len(selected) < count and remaining:
        anchor = selected[-1]
        seed_distance_penalty = float(profile.get("cluster_seed_penalty", 0.0))
        spread_bonus = float(profile.get("spread_bonus", 0.0))
        candidate = max(
            remaining,
            key=lambda item: (
                _activity_score(item, profile, preference_model)
                + (-0 if not discouraged_names else 0)
                + _light_bonus(item)
                - profile.get("route_penalty", 1.6) * ((_haversine_km(anchor.get("lat"), anchor.get("lng"), item.get("lat"), item.get("lng")) or 6))
                - seed_distance_penalty * ((_haversine_km(seed.get("lat"), seed.get("lng"), item.get("lat"), item.get("lng")) or 0))
                + spread_bonus * min((_haversine_km(seed.get("lat"), seed.get("lng"), item.get("lat"), item.get("lng")) or 0), 6)
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)

    if len(selected) < count:
        fallback_pool = [item for item in broad_available if item not in selected]
        while len(selected) < count and fallback_pool:
            anchor = selected[-1]
            candidate = max(
                fallback_pool,
                key=lambda item: (
                    _activity_score(item, profile, preference_model, discouraged_names)
                    - profile.get("fallback_route_penalty", 1.9) * ((_haversine_km(anchor.get("lat"), anchor.get("lng"), item.get("lat"), item.get("lng")) or 6))
                    - float(profile.get("cluster_seed_penalty", 0.0)) * ((_haversine_km(seed.get("lat"), seed.get("lng"), item.get("lat"), item.get("lng")) or 0))
                    + float(profile.get("spread_bonus", 0.0)) * min((_haversine_km(seed.get("lat"), seed.get("lng"), item.get("lat"), item.get("lng")) or 0), 6)
                ),
            )
            selected.append(candidate)
            fallback_pool.remove(candidate)

    if discouraged_names:
        fresh_quota = max(0, int(profile.get("fresh_activity_quota", 0)))
        fresh_selected = [item for item in selected if _normalized_name(item) not in discouraged_names]
        if len(fresh_selected) < fresh_quota:
            fresh_candidates = sorted(
                [
                    item
                    for item in broad_available
                    if item not in selected and _normalized_name(item) not in discouraged_names
                ],
                key=lambda item: _activity_score(item, profile, preference_model, discouraged_names) + _light_bonus(item),
                reverse=True,
            )
            while len(fresh_selected) < fresh_quota and fresh_candidates:
                discouraged_positions = [
                    idx for idx, item in enumerate(selected)
                    if _normalized_name(item) in discouraged_names
                ]
                if not discouraged_positions:
                    break
                worst_idx = min(
                    discouraged_positions,
                    key=lambda idx: _activity_score(selected[idx], profile, preference_model, discouraged_names) + _light_bonus(selected[idx]),
                )
                selected[worst_idx] = fresh_candidates.pop(0)
                fresh_selected = [item for item in selected if _normalized_name(item) not in discouraged_names]

    for item in selected:
        used_names.add(item.get("name", "").lower().strip())
    return selected


def _build_day_activity_pool(
    attractions: list[dict],
    used_names: set[str],
    profile: dict,
    preference_model: dict,
    *,
    desired_count: int,
    day_index: int,
    discouraged_names: set[str] | None = None,
    used_cluster_seeds: set[str] | None = None,
    discouraged_centroids: list[tuple[float, float]] | None = None,
) -> list[dict]:
    available = [
        item for item in attractions
        if item.get("name") and item.get("name").lower().strip() not in used_names
    ]
    if not available:
        return []

    selection_mode = str(profile.get("selection_mode") or "coverage")
    if selection_mode != "immersion":
        return available

    if discouraged_names:
        novel_available = [
            item for item in available
            if _normalized_name(item) not in discouraged_names
        ]
        if len(novel_available) >= max(desired_count + 2, 5):
            available = novel_available

    ranked = sorted(
        available,
        key=lambda item: _activity_score(item, profile, preference_model, discouraged_names),
        reverse=True,
    )
    if not ranked:
        return []

    cluster_radius = float(profile.get("cluster_radius_km", 5.0))
    min_cluster_size = max(2, min(desired_count, 3))
    novel_ranked = [
        item for item in ranked
        if not discouraged_names or _normalized_name(item) not in discouraged_names
    ]
    seed_source = novel_ranked or ranked
    seed_candidates = seed_source[: min(len(seed_source), 16)]

    best_pool: list[dict] | None = None
    best_seed_name = ""
    best_score = float("-inf")
    for seed in seed_candidates:
        seed_name = _normalized_name(seed)
        if used_cluster_seeds and seed_name in used_cluster_seeds:
            continue
        cluster_items = [
            item
            for item in available
            if (
                item is seed
                or (_haversine_km(seed.get("lat"), seed.get("lng"), item.get("lat"), item.get("lng")) or 999)
                <= cluster_radius
            )
        ]
        if len(cluster_items) < min_cluster_size:
            continue
        cluster_ranked = sorted(
            cluster_items,
            key=lambda item: _activity_score(item, profile, preference_model, discouraged_names),
            reverse=True,
        )
        pool = cluster_ranked[: max(desired_count + 2, min_cluster_size + 1)]
        pool_centroid = _items_centroid(pool)
        fresh_count = sum(
            1 for item in pool
            if not discouraged_names or _normalized_name(item) not in discouraged_names
        )
        repeated_count = len(pool) - fresh_count
        avg_seed_distance = sum(
            (_haversine_km(seed.get("lat"), seed.get("lng"), item.get("lat"), item.get("lng")) or cluster_radius)
            for item in pool
            if item is not seed
        ) / max(1, len(pool) - 1)
        compact_count = sum(
            1
            for item in pool
            if (
                item is seed
                or (_haversine_km(seed.get("lat"), seed.get("lng"), item.get("lat"), item.get("lng")) or 999)
                <= cluster_radius * 0.55
            )
        )
        centroid_distance_bonus = 0.0
        if pool_centroid and discouraged_centroids:
            nearest_prior_cluster = min(
                (
                    _haversine_km(pool_centroid[0], pool_centroid[1], centroid[0], centroid[1]) or 0.0
                    for centroid in discouraged_centroids
                ),
                default=0.0,
            )
            centroid_distance_bonus = min(nearest_prior_cluster, 12.0) * 3.5
        cluster_score = (
            sum(
                _activity_score(item, profile, preference_model, discouraged_names)
                for item in pool[:desired_count]
            )
            + fresh_count * 10.0
            + compact_count * 6.0
            - repeated_count * 26.0
            - avg_seed_distance * 5.0
            + centroid_distance_bonus
        )
        if cluster_score > best_score:
            best_score = cluster_score
            best_pool = pool
            best_seed_name = seed_name

    if not best_pool:
        best_pool = seed_source[: max(desired_count + 2, min_cluster_size + 1)]
        if best_pool:
            best_seed_name = _normalized_name(best_pool[0])

    if best_pool and discouraged_names:
        novel_best_pool = [
            item for item in best_pool
            if _normalized_name(item) not in discouraged_names
        ]
        if len(novel_best_pool) >= min(desired_count, max(2, min_cluster_size)):
            best_pool = novel_best_pool + [
                item for item in best_pool if _normalized_name(item) in discouraged_names
            ]

    if used_cluster_seeds is not None and best_seed_name:
        used_cluster_seeds.add(best_seed_name)
    return best_pool


def _distance_to_cluster(item: dict, anchors: list[dict], centroid: tuple[float, float] | None = None) -> float | None:
    distances: list[float] = []
    for anchor in anchors:
        dist = _haversine_km(anchor.get("lat"), anchor.get("lng"), item.get("lat"), item.get("lng"))
        if dist is not None:
            distances.append(dist)
    if centroid is not None:
        dist = _haversine_km(centroid[0], centroid[1], item.get("lat"), item.get("lng"))
        if dist is not None:
            distances.append(dist)
    if not distances:
        return None
    return min(distances)


def _build_district_restaurant_pool(
    restaurants: list[dict],
    activities: list[dict],
    used_names: set[str],
    *,
    primary_radius_km: float = 2.4,
    expanded_radius_km: float = 4.2,
    desired_count: int = 4,
    discouraged_names: set[str] | None = None,
) -> list[dict]:
    available = [
        item for item in restaurants
        if item.get("name") and _normalized_name(item) not in used_names
    ]
    if not available or not activities:
        return available

    centroid = _items_centroid(activities)

    def _collect(radius_km: float) -> list[dict]:
        collected = [
            item for item in available
            if (_distance_to_cluster(item, activities, centroid) or 999.0) <= radius_km
        ]
        return sorted(
            collected,
            key=lambda item: (
                _distance_to_cluster(item, activities, centroid) or 999.0,
                -_numeric_rating(item.get("rating")),
            ),
        )

    local_pool = _collect(primary_radius_km)
    if len(local_pool) < desired_count:
        expanded_pool = _collect(expanded_radius_km)
        if len(expanded_pool) > len(local_pool):
            local_pool = expanded_pool

    if discouraged_names and local_pool:
        novel_local = [
            item for item in local_pool
            if _normalized_name(item) not in discouraged_names
        ]
        if len(novel_local) >= min(2, desired_count):
            local_pool = novel_local

    return local_pool or available


def _filter_restaurants_by_fit(
    restaurants: list[dict],
    profile: dict,
    preference_model: dict,
    *,
    quality_floor: float = -1.5,
    drop_from_best: float = 6.0,
) -> list[dict]:
    if not restaurants:
        return []
    scored = [
        (item, _restaurant_relevance_score(item, profile, preference_model))
        for item in restaurants
    ]
    best_score = max(score for _, score in scored)
    cutoff = max(quality_floor, best_score - drop_from_best)
    filtered = [item for item, score in scored if score >= cutoff]
    return filtered or restaurants


def _restaurant_pool_target_size(duration_days: int, profile: dict) -> int:
    base_meals = 1 + max(duration_days - 2, 0) * 2
    return max(6, min(base_meals, 7))


def _restaurant_role_adjustment(item: dict, profile: dict, preference_model: dict) -> float:
    blob = _text_blob(item)
    signals = _item_signal_counts(blob)
    rating = _numeric_rating(item.get("rating"))
    role = str(profile.get("meal_role") or "highlight")
    if role == "highlight":
        return (
            signals["fine_dining"] * 4.8
            + signals["traditional_cuisine"] * 3.2
            + max(0.0, rating - 4.1) * 4.5
            - (5.0 if _is_nonlocal_restaurant(item) else 0.0)
        )
    if role == "local_mix":
        return (
            signals["local_food"] * 6.8
            + signals["neighborhood"] * 3.8
            + signals["traditional_cuisine"] * 3.2
            + max(0.0, rating - 4.0) * 2.0
            - (8.5 if _is_nonlocal_restaurant(item) else 0.0)
        )
    return (
        signals["neighborhood"] * 4.0
        + signals["local_food"] * 3.6
        + signals["traditional_cuisine"] * 2.8
        + max(0.0, rating - 4.0) * 2.2
        - (5.2 if _is_nonlocal_restaurant(item) else 0.0)
    )


def _restaurant_role_compatible(item: dict, profile: dict) -> bool:
    role = str(profile.get("meal_role") or "highlight")
    if role == "highlight":
        return True
    if not _is_nonlocal_restaurant(item):
        return True
    signals = _item_signal_counts(_text_blob(item))
    rating = _numeric_rating(item.get("rating"))
    if signals["fine_dining"] > 0 and rating >= 4.4:
        return True
    return False


def _build_option_restaurant_pools(
    restaurants: list[dict],
    duration_days: int,
    preference_model: dict,
) -> dict[str, list[dict]]:
    if not restaurants:
        return {option_key: [] for option_key in _OPTION_PROFILES}

    option_pools: dict[str, list[dict]] = {}

    for option_key, profile in _OPTION_PROFILES.items():
        pool_target = _restaurant_pool_target_size(duration_days, profile)
        candidate_base = _filter_restaurants_by_fit(
            [item for item in restaurants if _restaurant_role_compatible(item, profile)],
            profile,
            preference_model,
            quality_floor=0.0 if str(profile.get("meal_role") or "") in {"local_mix", "district_coherent"} else -0.5,
            drop_from_best=7.0,
        )
        source_restaurants = candidate_base if len(candidate_base) >= pool_target else [
            item for item in restaurants if _restaurant_role_compatible(item, profile)
        ] or restaurants
        scored_candidates = sorted(
            (
                (item, _restaurant_score(item, profile, preference_model, None))
                for item in source_restaurants
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        pool: list[dict] = []
        pool_names: set[str] = set()
        for item, _score in scored_candidates:
            name = _normalized_name(item)
            if not name or name in pool_names:
                continue
            pool.append(item)
            pool_names.add(name)
            if len(pool) >= pool_target:
                break

        option_pools[option_key] = pool

    return option_pools


def _build_immersion_option_schedule(
    option_key: str,
    profile: dict,
    compact_attractions: list[dict],
    option_restaurants: list[dict],
    fallback_restaurants: list[dict],
    hotel: dict | None,
    outbound_flight: dict | None,
    return_flight: dict | None,
    duration_days: int,
    preference_model: dict,
    trip_start_date: date | None = None,
    discouraged_activities: set[str] | None = None,
    discouraged_restaurants: set[str] | None = None,
    discouraged_activity_centroids: list[tuple[float, float]] | None = None,
) -> tuple[list, dict, list[tuple[float, float]], list[dict]]:
    used_place_names: set[str] = set()
    used_cluster_seeds: set[str] = set()
    day_activity_centroids: list[tuple[float, float]] = []
    day_decisions: list[dict] = []
    allow_global_fallback = str(profile.get("meal_role") or "") == "highlight"
    days: list[dict] = []
    stats = {
        "removed_items": 0,
        "normalized_items": 0,
        "duplicate_items": 0,
        "invalid_items": 0,
        "sparse_days": [],
        "boundary_repairs": [],
        "route_warnings": [],
    }

    days.append(
        {
            "day": "Day 1",
            "items": _arrival_day_items(
                profile,
                outbound_flight,
                hotel,
                compact_attractions,
                option_restaurants,
                fallback_restaurants,
                used_place_names,
                preference_model,
                service_date=_service_date(trip_start_date, 0),
                discouraged_activities=discouraged_activities,
                discouraged_restaurants=discouraged_restaurants,
                option_key=option_key,
            ),
        }
    )
    day_decisions.append({
        "day": "Day 1",
        "day_type": "arrival",
        "theme": "arrival evening",
        "seed_name": None,
        "seed_reason": "arrival day uses a fixed light-evening template",
        "activities": [],
        "lunch": None,
        "dinner": None,
    })

    middle_days = max(duration_days - 2, 0)
    cluster_radius = float(profile.get("cluster_radius_km", 5.0))

    for idx in range(middle_days):
        service_date = _service_date(trip_start_date, idx + 1)
        remaining_middle_days = middle_days - idx
        desired_count = max(
            int(profile.get("min_activity_count", 2)),
            min(int(profile.get("target_activity_count", 2)), 2),
        )

        all_available = [
            item for item in compact_attractions
            if item.get("name") and _normalized_name(item) not in used_place_names
        ]
        novel_available = [
            item for item in all_available
            if not discouraged_activities or _normalized_name(item) not in discouraged_activities
        ]
        hard_novel_mode = len(novel_available) >= desired_count * remaining_middle_days
        source_variants: list[tuple[list[dict], set[str] | None]] = []
        if hard_novel_mode and novel_available:
            source_variants.append((novel_available, None))
        source_variants.append((all_available, discouraged_activities))

        activities: list[dict] = []
        used_source_novel_only = False
        centroid_bias = list(discouraged_activity_centroids or []) + day_activity_centroids
        probe_profile = {**profile, "selection_mode": "coverage", "fresh_activity_quota": 0, "fresh_activity_bonus": 0}

        prior_themes = [d["theme"] for d in day_decisions if d.get("theme")]
        selected_seed: dict | None = None
        day_theme = "mixed sightseeing"
        seed_reason = ""

        for source_items, source_discouraged in source_variants:
            if len(source_items) < desired_count:
                continue
            day_pool = _build_day_activity_pool(
                source_items,
                used_place_names,
                profile,
                preference_model,
                desired_count=desired_count,
                day_index=idx,
                discouraged_names=source_discouraged,
                used_cluster_seeds=used_cluster_seeds,
                discouraged_centroids=centroid_bias,
            )
            if not day_pool:
                continue

            seed_candidates = _prepare_seed_candidates(
                day_pool,
                profile,
                preference_model,
                cluster_radius=cluster_radius,
                max_candidates=5,
                discouraged_names=source_discouraged,
            )
            if seed_candidates:
                selected_seed, day_theme, seed_reason = _llm_select_seed(
                    seed_candidates,
                    day_index=idx,
                    option_key=option_key,
                    profile=profile,
                    preference_model=preference_model,
                    prior_themes=prior_themes,
                )

            probe_used_names = set(used_place_names)
            picked = _choose_cluster_activities(
                day_pool,
                probe_used_names,
                probe_profile,
                preference_model,
                count=desired_count,
                day_index=idx,
                discouraged_names=source_discouraged,
                forced_seed=selected_seed,
            )
            if len(picked) >= int(profile.get("min_activity_count", 2)):
                activities = picked
                used_source_novel_only = source_discouraged is None and hard_novel_mode
                break

        if not activities and all_available:
            fallback_pool = _build_day_activity_pool(
                all_available,
                used_place_names,
                profile,
                preference_model,
                desired_count=desired_count,
                day_index=idx,
                discouraged_names=discouraged_activities,
                used_cluster_seeds=used_cluster_seeds,
                discouraged_centroids=centroid_bias,
            )
            probe_used_names = set(used_place_names)
            activities = _choose_cluster_activities(
                fallback_pool or all_available,
                probe_used_names,
                probe_profile,
                preference_model,
                count=desired_count,
                day_index=idx,
                discouraged_names=discouraged_activities,
                forced_seed=selected_seed,
            )

        blocked_place_names = {_normalized_name(item) for item in activities if _normalized_name(item)}
        district_restaurants = _build_district_restaurant_pool(
            option_restaurants,
            activities,
            used_place_names,
            primary_radius_km=2.4,
            expanded_radius_km=4.0,
            desired_count=4,
            discouraged_names=discouraged_restaurants,
        )
        district_restaurants = _filter_restaurants_by_fit(
            district_restaurants,
            profile,
            preference_model,
            quality_floor=-0.5,
            drop_from_best=5.0,
        )

        restaurant_source = district_restaurants or option_restaurants
        novel_restaurants = [
            item for item in restaurant_source
            if not discouraged_restaurants or _normalized_name(item) not in discouraged_restaurants
        ]
        novel_restaurants = _filter_restaurants_by_fit(
            novel_restaurants,
            profile,
            preference_model,
            quality_floor=0.0,
            drop_from_best=4.0,
        )
        if len(novel_restaurants) >= min(2, 2 * remaining_middle_days):
            restaurant_source = novel_restaurants

        lunch = _pick_mandatory_meal(
            restaurant_source,
            used_place_names,
            blocked_place_names,
            profile,
            preference_model,
            activities[0] if activities else None,
            meal_kind="lunch",
            service_date=service_date,
            discouraged_names=discouraged_restaurants if not used_source_novel_only else None,
        )
        if lunch is None and allow_global_fallback and fallback_restaurants and restaurant_source is not fallback_restaurants:
            lunch = _pick_mandatory_meal(
                fallback_restaurants,
                used_place_names,
                blocked_place_names,
                profile,
                preference_model,
                activities[0] if activities else None,
                meal_kind="lunch",
                service_date=service_date,
                discouraged_names=discouraged_restaurants,
            )

        reserved_restaurants = set(used_place_names)
        if lunch and lunch.get("name"):
            reserved_restaurants.add(_normalized_name(lunch))
            blocked_place_names.add(_normalized_name(lunch))

        dinner_anchor = activities[-1] if activities else lunch
        dinner = _pick_mandatory_meal(
            restaurant_source,
            reserved_restaurants,
            blocked_place_names,
            profile,
            preference_model,
            dinner_anchor,
            meal_kind="dinner",
            service_date=service_date,
            discouraged_names=discouraged_restaurants if not used_source_novel_only else None,
        )
        if dinner is None and allow_global_fallback and fallback_restaurants and restaurant_source is not fallback_restaurants:
            dinner = _pick_mandatory_meal(
                fallback_restaurants,
                reserved_restaurants,
                blocked_place_names,
                profile,
                preference_model,
                dinner_anchor,
                meal_kind="dinner",
                service_date=service_date,
                discouraged_names=discouraged_restaurants,
            )

        items = _middle_day_items(
            profile,
            activities,
            lunch,
            dinner,
            day_index=idx + 1,
            service_date=service_date,
        )
        meal_total = sum(1 for item in items if item.get("icon") == "restaurant")
        if meal_total < 2:
            has_lunch = any(
                item.get("icon") == "restaurant"
                and (_minutes_since_midnight(item.get("time", "")) or 0) < 16 * 60
                for item in items
            )
            has_dinner = any(
                item.get("icon") == "restaurant"
                and (_minutes_since_midnight(item.get("time", "")) or 0) >= 16 * 60
                for item in items
            )
            if not has_lunch:
                added = _append_missing_meal(
                    items,
                    restaurant_source,
                    used_place_names,
                    profile,
                    preference_model,
                    meal_kind="lunch",
                    service_date=service_date,
                    discouraged_restaurants=discouraged_restaurants,
                )
                if not added and fallback_restaurants and restaurant_source is not fallback_restaurants:
                    _append_missing_meal(
                        items,
                        fallback_restaurants,
                        used_place_names,
                        profile,
                        preference_model,
                        meal_kind="lunch",
                        service_date=service_date,
                        discouraged_restaurants=discouraged_restaurants,
                    )
            if not has_dinner:
                added = _append_missing_meal(
                    items,
                    restaurant_source,
                    used_place_names,
                    profile,
                    preference_model,
                    meal_kind="dinner",
                    service_date=service_date,
                    discouraged_restaurants=discouraged_restaurants,
                )
                if not added and fallback_restaurants and restaurant_source is not fallback_restaurants:
                    _append_missing_meal(
                        items,
                        fallback_restaurants,
                        used_place_names,
                        profile,
                        preference_model,
                        meal_kind="dinner",
                        service_date=service_date,
                        discouraged_restaurants=discouraged_restaurants,
                    )
            items.sort(key=lambda item: _sort_time_value(item.get("time", "")))
        items = _enforce_restaurant_time_buffers(items, min_restaurant_block_minutes=80)

        activity_total = sum(1 for item in items if item.get("icon") == "activity")
        meal_total = sum(1 for item in items if item.get("icon") == "restaurant")
        if activity_total < int(profile.get("min_activity_count", 2)) or meal_total < 2:
            stats["sparse_days"].append(
                {"day_index": idx + 2, "activity_count": activity_total, "meal_count": meal_total}
            )

        activity_centroid = _items_centroid(activities)
        if activity_centroid is not None:
            day_activity_centroids.append(activity_centroid)

        for item in items:
            if item.get("icon") in {"activity", "restaurant"}:
                normalized = _normalized_name(item)
                if normalized:
                    used_place_names.add(normalized)
        days.append({"day": f"Day {idx + 2}", "items": items})

        day_decisions.append({
            "day": f"Day {idx + 2}",
            "day_type": "middle",
            "theme": day_theme,
            "seed_name": selected_seed.get("name") if selected_seed else None,
            "seed_reason": seed_reason,
            "activities": [
                {
                    "name": a.get("name", ""),
                    "type": str(a.get("type") or a.get("category") or ""),
                }
                for a in activities
            ],
            "lunch": _meal_trace(lunch, "lunch", preference_model),
            "dinner": _meal_trace(dinner, "dinner", preference_model),
        })

    days.append(
        {
            "day": f"Day {duration_days}",
            "items": _departure_day_items(
                profile,
                return_flight,
                hotel,
                compact_attractions,
                option_restaurants,
                fallback_restaurants,
                used_place_names,
                preference_model,
                service_date=_service_date(trip_start_date, duration_days - 1),
                discouraged_activities=discouraged_activities,
                discouraged_restaurants=discouraged_restaurants,
            ),
        }
    )
    day_decisions.append({
        "day": f"Day {duration_days}",
        "day_type": "departure",
        "theme": "departure day",
        "seed_name": None,
        "seed_reason": "departure day uses checkout + last visit + airport template",
        "activities": [],
        "lunch": None,
        "dinner": None,
    })

    seen_place_names: set[str] = set()
    for day in days:
        for item in day.get("items", []):
            name = _normalized_name(item)
            if not name:
                continue
            if item.get("icon") in {"activity", "restaurant"}:
                if name in seen_place_names:
                    stats["duplicate_items"] += 1
                else:
                    seen_place_names.add(name)
    return days, stats, day_activity_centroids, day_decisions


def _pick_restaurant_for_anchor(
    restaurants: list[dict],
    used_names: set[str],
    profile: dict,
    preference_model: dict,
    anchor: dict | None,
    discouraged_names: set[str] | None = None,
    preferred_start: int | None = None,
    service_date: date | None = None,
    meal_duration: int | None = None,
    latest_finish: int | None = None,
    variant_offset: int = 0,
) -> dict | None:
    available = [
        item for item in restaurants
        if item.get("name") and item.get("name").lower().strip() not in used_names
    ]
    if not available:
        return None
    compatible_available = [item for item in available if _restaurant_role_compatible(item, profile)]
    if compatible_available:
        available = compatible_available
    broad_available = list(available)
    strong_food_preference = preference_model["families"].get("food", 0) >= 3
    scored_available = [
        (item, _restaurant_relevance_score(item, profile, preference_model))
        for item in available
    ]
    best_relevance = max(relevance for _, relevance in scored_available)
    relevance_cutoff = max(16.0, best_relevance - (20 if strong_food_preference else 10))
    filtered = [item for item, relevance in scored_available if relevance >= relevance_cutoff]
    if filtered:
        available = filtered
    if discouraged_names:
        fresh_quota = max(0, int(profile.get("fresh_restaurant_quota", 0)))
        novel_available = [item for item in available if _normalized_name(item) not in discouraged_names]
        if str(profile.get("selection_mode") or "") == "immersion" and len(novel_available) >= 3:
            available = novel_available
        elif len(novel_available) >= max(1, fresh_quota):
            overall_best = max(
                _restaurant_score(item, profile, preference_model, discouraged_names)
                for item in available
            )
            novel_best = max(
                _restaurant_score(item, profile, preference_model, discouraged_names)
                for item in novel_available
            )
            if novel_best >= overall_best - 8:
                available = novel_available
        else:
            relaxed_novel_available = [
                item for item in broad_available if _normalized_name(item) not in discouraged_names
            ]
            if len(relaxed_novel_available) >= max(1, fresh_quota):
                available = relaxed_novel_available
    if preferred_start is not None:
        meal_span = meal_duration or 75
        time_fit = [
            item
            for item in available
            if _fit_service_start(item, preferred_start, meal_span, service_date, latest_finish) is not None
        ]
        if not time_fit:
            time_fit = [
                item
                for item in broad_available
                if item.get("name")
                and item.get("name").lower().strip() not in used_names
                and _fit_service_start(item, preferred_start, meal_span, service_date, latest_finish) is not None
            ]
        if time_fit:
            available = time_fit
        else:
            return None
    traditional_pref = (
        _preference_emphasis(preference_model, "traditional")
        + _preference_emphasis(preference_model, "traditional cultural")
        + _preference_emphasis(preference_model, "historical")
        + _preference_emphasis(preference_model, "authentic")
    )
    fine_pref = _preference_emphasis(preference_model, "fine dining") + _preference_emphasis(preference_model, "fine")
    local_pref = (
        _preference_emphasis(preference_model, "local")
        + _preference_emphasis(preference_model, "street food")
        + _preference_emphasis(preference_model, "hawker")
    )
    scored_candidates = [
        (item, _restaurant_score(item, profile, preference_model, discouraged_names))
        for item in available
    ]
    if scored_candidates:
        best_quality = max(score for _, score in scored_candidates)
        quality_floor = -6.0
        if traditional_pref or fine_pref or local_pref:
            quality_floor = 0.0
        quality_cutoff = max(quality_floor, best_quality - 8.0)
        filtered_quality = [item for item, score in scored_candidates if score >= quality_cutoff]
        if filtered_quality:
            available = filtered_quality
    anchor_lat = anchor.get("lat") if anchor else None
    anchor_lng = anchor.get("lng") if anchor else None
    ranked_candidates = sorted(
        available,
        key=lambda item: (
            _restaurant_score(item, profile, preference_model, discouraged_names)
            - profile.get("restaurant_route_penalty", 1.4)
            * ((_haversine_km(anchor_lat, anchor_lng, item.get("lat"), item.get("lng")) or 4))
        ),
        reverse=True,
    )
    if not ranked_candidates:
        return None

    if variant_offset <= 0:
        return ranked_candidates[0]

    best_candidate_score = (
        _restaurant_score(ranked_candidates[0], profile, preference_model, discouraged_names)
        - profile.get("restaurant_route_penalty", 1.4)
        * ((_haversine_km(anchor_lat, anchor_lng, ranked_candidates[0].get("lat"), ranked_candidates[0].get("lng")) or 4))
    )
    shortlist = [
        item for item in ranked_candidates
        if (
            _restaurant_score(item, profile, preference_model, discouraged_names)
            - profile.get("restaurant_route_penalty", 1.4)
            * ((_haversine_km(anchor_lat, anchor_lng, item.get("lat"), item.get("lng")) or 4))
        ) >= best_candidate_score - 4.5
    ]
    if len(shortlist) <= 1:
        return ranked_candidates[0]
    return shortlist[min(variant_offset, len(shortlist) - 1)]


def _pick_mandatory_meal(
    restaurants: list[dict],
    used_names: set[str],
    blocked_names: set[str],
    profile: dict,
    preference_model: dict,
    anchor: dict | None,
    *,
    meal_kind: str,
    service_date: date | None = None,
    discouraged_names: set[str] | None = None,
) -> dict | None:
    filtered_restaurants = _exclude_same_place(restaurants, blocked_names)
    duration = 75 if meal_kind == "lunch" else 90
    attempts = (
        list(range(11 * 60 + 15, 13 * 60 + 31, 15))
        if meal_kind == "lunch"
        else list(range(17 * 60 + 30, 19 * 60 + 31, 15))
    )
    latest_finish = 15 * 60 + 30 if meal_kind == "lunch" else 21 * 60 + 30
    traditional_pref = (
        _preference_emphasis(preference_model, "traditional")
        + _preference_emphasis(preference_model, "traditional cultural")
        + _preference_emphasis(preference_model, "historical")
        + _preference_emphasis(preference_model, "authentic")
    )
    fine_pref = _preference_emphasis(preference_model, "fine dining") + _preference_emphasis(preference_model, "fine")
    local_pref = (
        _preference_emphasis(preference_model, "local")
        + _preference_emphasis(preference_model, "street food")
        + _preference_emphasis(preference_model, "hawker")
    )
    relevance_floor = -10.0
    if traditional_pref or fine_pref or local_pref:
        relevance_floor = -3.0 if meal_kind == "dinner" else -5.0
    best_fallback: tuple[float, dict] | None = None
    best_local_fallback: tuple[float, dict] | None = None
    for preferred_start in attempts:
        picked = _pick_restaurant_for_anchor(
            filtered_restaurants,
            used_names,
            profile,
            preference_model,
            anchor,
            discouraged_names=discouraged_names,
            preferred_start=preferred_start,
            service_date=service_date,
            meal_duration=duration,
            latest_finish=latest_finish,
        )
        if picked is not None:
            picked_score = _restaurant_relevance_score(picked, profile, preference_model)
            if picked_score >= relevance_floor:
                return picked
            if best_fallback is None or picked_score > best_fallback[0]:
                best_fallback = (picked_score, picked)
            if (not _is_nonlocal_restaurant(picked)) and (
                best_local_fallback is None or picked_score > best_local_fallback[0]
            ):
                best_local_fallback = (picked_score, picked)
    for preferred_start in attempts:
        picked = _pick_restaurant_for_anchor(
            filtered_restaurants,
            used_names,
            profile,
            preference_model,
            anchor,
            discouraged_names=None,
            preferred_start=preferred_start,
            service_date=service_date,
            meal_duration=duration,
            latest_finish=latest_finish,
        )
        if picked is not None:
            picked_score = _restaurant_relevance_score(picked, profile, preference_model)
            if picked_score >= relevance_floor:
                return picked
            if best_fallback is None or picked_score > best_fallback[0]:
                best_fallback = (picked_score, picked)
            if (not _is_nonlocal_restaurant(picked)) and (
                best_local_fallback is None or picked_score > best_local_fallback[0]
            ):
                best_local_fallback = (picked_score, picked)
    if best_local_fallback is not None:
        return best_local_fallback[1]
    return best_fallback[1] if best_fallback else None


def _pick_activity_near_anchor(
    attractions: list[dict],
    used_names: set[str],
    profile: dict,
    preference_model: dict,
    anchor: dict | None,
    light_only: bool = False,
    discouraged_names: set[str] | None = None,
) -> dict | None:
    available = [
        item for item in attractions
        if item.get("name") and item.get("name").lower().strip() not in used_names
    ]
    if not available:
        return None
    broad_available = list(available)

    strong_preference = preference_model["families"].get("culture", 0) + preference_model["families"].get("food", 0) >= 3

    def _light_bonus(item: dict) -> float:
        if not light_only:
            return 0.0
        blob = _text_blob(item)
        bonus = 0.0
        if any(word in blob for word in ("tower", "sky", "temple", "garden", "park", "street")):
            bonus += 6.0
        if "museum" in blob:
            bonus -= 5.0
        return bonus

    anchor_lat = anchor.get("lat") if anchor else None
    anchor_lng = anchor.get("lng") if anchor else None
    scored_available = [
        (
            item,
            _activity_score(item, profile, preference_model, discouraged_names) + _light_bonus(item),
            _activity_relevance_score(item, profile, preference_model),
        )
        for item in available
    ]
    best_relevance = max(relevance for _, _, relevance in scored_available)
    relevance_cutoff = max(24.0, best_relevance - (34 if strong_preference else 22))
    filtered = [item for item, _, relevance in scored_available if relevance >= relevance_cutoff]
    if filtered:
        available = filtered
    if discouraged_names:
        novel_available = [item for item in available if _normalized_name(item) not in discouraged_names]
        if novel_available:
            overall_best = max(
                _activity_score(item, profile, preference_model, discouraged_names) + _light_bonus(item)
                for item in available
            )
            novel_best = max(
                _activity_score(item, profile, preference_model, discouraged_names) + _light_bonus(item)
                for item in novel_available
            )
            if novel_best >= overall_best - 10:
                available = novel_available
        else:
            relaxed_novel_available = [
                item for item in broad_available if _normalized_name(item) not in discouraged_names
            ]
            if relaxed_novel_available:
                available = relaxed_novel_available
    picked = max(
        available,
        key=lambda item: (
            _activity_score(item, profile, preference_model, discouraged_names)
            + _light_bonus(item)
            - profile.get("nearby_route_penalty", 1.6) * ((_haversine_km(anchor_lat, anchor_lng, item.get("lat"), item.get("lng")) or 4))
        ),
    )
    used_names.add(picked.get("name", "").lower().strip())
    return picked


def _preferred_activity_count(
    attractions: list[dict],
    used_names: set[str],
    profile: dict,
    preference_model: dict,
    remaining_middle_days: int,
) -> int:
    available = [
        item for item in attractions
        if item.get("name") and item.get("name").lower().strip() not in used_names
    ]
    if not available:
        return 0
    relevant_scores = sorted(
        (_activity_relevance_score(item, profile, preference_model) for item in available),
        reverse=True,
    )
    if not relevant_scores:
        return 0

    target = int(profile.get("target_activity_count", 3))
    minimum = int(profile.get("min_activity_count", 2))
    total_available = len(available)
    if total_available == 0:
        return 0
    if remaining_middle_days <= 0:
        return min(target, total_available)

    feasible_per_day = math.ceil(total_available / remaining_middle_days)
    if total_available >= minimum * remaining_middle_days:
        return min(target, max(minimum, feasible_per_day))
    return min(target, max(1, feasible_per_day))


def _middle_day_items(
    profile: dict,
    activities: list[dict],
    lunch: dict | None,
    dinner: dict | None,
    day_index: int,
    service_date: date | None = None,
) -> list[dict]:
    if not activities:
        return []
    start_minutes = profile["start_min"] + day_index * 7
    first_blob = _text_blob(activities[0])
    if any(word in first_blob for word in ("museum", "national")):
        start_minutes += 15
    if any(word in first_blob for word in ("garden", "temple", "shrine", "park")):
        start_minutes -= 10
    items: list[dict] = []
    current = start_minutes
    remaining = list(enumerate(activities))
    first_pick = _pick_next_activity_for_time(remaining, current, None, service_date)
    if not first_pick:
        return []
    remaining.remove(first_pick)
    _, act1 = first_pick
    items.append({"time": _hhmm(current), "icon": "activity", "key": act1["key"], "name": act1["name"], "cost": act1.get("price") or "TBC"})
    current += _activity_duration_minutes(act1)

    if lunch:
        current += _travel_minutes(act1, lunch, base=16)
        lunch_floor = 11 * 60 + 35 + (day_index % 3) * 10
        lunch_pref = max(lunch_floor, current)
        lunch_added = False
        lunch_start = _fit_service_start(
            lunch,
            lunch_pref,
            _meal_duration_minutes(lunch),
            service_date,
        )
        if lunch_start is not None:
            items.append({"time": _hhmm(lunch_start), "icon": "restaurant", "key": lunch["key"], "name": lunch["name"], "cost": lunch.get("price") or "TBC"})
            current = lunch_start + _meal_duration_minutes(lunch)
            lunch_added = True
    else:
        lunch_added = False

    anchor = lunch if lunch_added else act1
    slot_idx = 2
    while remaining:
        next_pick = _pick_next_activity_for_time(remaining, current, anchor, service_date)
        if not next_pick:
            break
        remaining.remove(next_pick)
        _, act = next_pick
        current += _travel_minutes(anchor, act, base=18 if str(anchor.get("key", "")).startswith("restaurant") else 15)
        latest_start = _activity_latest_start_minutes(act, service_date)
        if current > latest_start:
            continue
        if slot_idx == 2 and lunch:
            current += 5
        items.append({"time": _hhmm(current), "icon": "activity", "key": act["key"], "name": act["name"], "cost": act.get("price") or "TBC"})
        current += _activity_duration_minutes(act)
        anchor = act
        slot_idx += 1

    if dinner:
        current += _travel_minutes(anchor, dinner, base=18)
        premium_bonus = 15 if _item_signal_counts(_text_blob(dinner)).get("fine_dining", 0) > 0 else 0
        dinner_floor = 18 * 60 + (day_index % 3) * 15 + premium_bonus
        dinner_pref = max(dinner_floor, current)
        dinner_start = _fit_service_start(
            dinner,
            dinner_pref,
            _meal_duration_minutes(dinner),
            service_date,
        )
        if dinner_start is not None:
            items.append({"time": _hhmm(dinner_start), "icon": "restaurant", "key": dinner["key"], "name": dinner["name"], "cost": dinner.get("price") or "TBC"})

    return items


def _item_end_minutes(item: dict) -> int:
    start = _minutes_since_midnight(item.get("time", "")) or 0
    if item.get("icon") == "activity":
        return start + _activity_duration_minutes(item)
    if item.get("icon") == "restaurant":
        return start + _meal_duration_minutes(item)
    return start


def _append_missing_meal(
    items: list[dict],
    restaurants: list[dict],
    used_place_names: set[str],
    profile: dict,
    preference_model: dict,
    *,
    meal_kind: str,
    service_date: date | None,
    discouraged_restaurants: set[str] | None = None,
) -> bool:
    blocked_names = {_normalized_name(item) for item in items if _normalized_name(item)}
    if meal_kind == "lunch":
        anchor_item = next((item for item in items if item.get("icon") == "activity"), None)
        anchor = anchor_item or {}
        preferred_floor = 11 * 60 + 35
        latest_finish = 15 * 60 + 30
    else:
        anchor_item = next((item for item in reversed(items) if item.get("icon") in {"activity", "restaurant"}), None)
        anchor = anchor_item or {}
        preferred_floor = 18 * 60
        latest_finish = 21 * 60 + 30

    picked = _pick_mandatory_meal(
        restaurants,
        used_place_names,
        blocked_names,
        profile,
        preference_model,
        anchor if isinstance(anchor, dict) else None,
        meal_kind=meal_kind,
        service_date=service_date,
        discouraged_names=discouraged_restaurants,
    )
    if not picked:
        return False

    anchor_end = _item_end_minutes(anchor_item) if anchor_item else preferred_floor - 30
    travel = _travel_minutes(anchor if isinstance(anchor, dict) else None, picked, base=14)
    start_pref = max(preferred_floor, anchor_end + travel)
    meal_duration = _meal_duration_minutes(picked)
    start = _fit_service_start(
        picked,
        start_pref,
        meal_duration,
        service_date,
        latest_finish=latest_finish,
    )
    if start is None:
        return False

    items.append(
        {
            "time": _hhmm(start),
            "icon": "restaurant",
            "key": picked["key"],
            "name": picked["name"],
            "cost": picked.get("price") or "TBC",
        }
    )
    used_place_names.add(_normalized_name(picked))
    items.sort(key=lambda item: _sort_time_value(item.get("time", "")))
    return True


def _enforce_restaurant_time_buffers(
    items: list[dict],
    *,
    min_restaurant_block_minutes: int = 80,
) -> list[dict]:
    adjusted = sorted(deepcopy(items), key=lambda item: _sort_time_value(item.get("time", "")))
    if not adjusted:
        return adjusted

    for idx in range(1, len(adjusted)):
        previous = adjusted[idx - 1]
        current = adjusted[idx]
        previous_start = _minutes_since_midnight(previous.get("time", ""))
        current_start = _minutes_since_midnight(current.get("time", ""))
        if previous_start is None or current_start is None:
            continue

        previous_icon = str(previous.get("icon") or "").lower()
        if previous_icon == "restaurant":
            required_start = previous_start + max(min_restaurant_block_minutes, _meal_duration_minutes(previous))
        elif previous_icon == "activity":
            required_start = previous_start + _activity_duration_minutes(previous)
        else:
            continue

        if current_start < required_start:
            current["time"] = _hhmm(required_start)

    return adjusted


def _repair_restaurants_to_option_pool(
    days: list[dict],
    option_restaurants: list[dict],
    profile: dict,
    preference_model: dict,
    trip_start_date: date | None = None,
) -> list[dict]:
    if not option_restaurants:
        return days
    repaired_days = deepcopy(days)
    pool_names = {_normalized_name(item) for item in option_restaurants if _normalized_name(item)}
    used_names = {
        _normalized_name(item)
        for day in repaired_days
        for item in day.get("items", [])
        if item.get("icon") in {"activity", "restaurant"} and _normalized_name(item)
    }

    for day_idx, day in enumerate(repaired_days):
        service_date = _service_date(trip_start_date, day_idx)
        items = day.get("items", [])
        for item_idx, item in enumerate(items):
            if item.get("icon") != "restaurant":
                continue
            current_name = _normalized_name(item)
            if current_name in pool_names:
                continue

            anchor = None
            for prior_idx in range(item_idx - 1, -1, -1):
                prior_item = items[prior_idx]
                if prior_item.get("icon") in {"activity", "restaurant"}:
                    anchor = prior_item
                    break

            blocked_names = {
                _normalized_name(other)
                for other in items
                if other is not item and other.get("icon") in {"activity", "restaurant"} and _normalized_name(other)
            }
            available_pool = [
                restaurant for restaurant in option_restaurants
                if _normalized_name(restaurant) not in blocked_names
                and (_normalized_name(restaurant) == current_name or _normalized_name(restaurant) not in used_names)
            ]
            if not available_pool:
                continue

            current_minutes = _minutes_since_midnight(item.get("time", "")) or (12 * 60 if item_idx < 3 else 18 * 60)
            meal_duration = 75 if current_minutes < 16 * 60 else 90
            latest_finish = 15 * 60 + 30 if current_minutes < 16 * 60 else 21 * 60 + 30
            replacement = _pick_restaurant_for_anchor(
                available_pool,
                used_names - {current_name},
                profile,
                preference_model,
                anchor,
                preferred_start=current_minutes,
                service_date=service_date,
                meal_duration=meal_duration,
                latest_finish=latest_finish,
            )
            if not replacement:
                continue

            replacement_start = _fit_service_start(
                replacement,
                current_minutes,
                meal_duration,
                service_date,
                latest_finish=latest_finish,
            )
            if replacement_start is None:
                continue

            used_names.discard(current_name)
            used_names.add(_normalized_name(replacement))
            item["time"] = _hhmm(replacement_start)
            item["key"] = replacement["key"]
            item["name"] = replacement["name"]
            item["cost"] = replacement.get("price") or item.get("cost") or "TBC"
        day["items"] = sorted(items, key=lambda current: _sort_time_value(current.get("time", "")))
    return repaired_days


def _arrival_day_items(
    profile: dict,
    outbound_flight: dict | None,
    hotel: dict | None,
    attractions: list[dict],
    restaurants: list[dict],
    fallback_restaurants: list[dict] | None,
    used_place_names: set[str],
    preference_model: dict,
    service_date: date | None = None,
    discouraged_activities: set[str] | None = None,
    discouraged_restaurants: set[str] | None = None,
    option_key: str | None = None,
) -> list[dict]:
    items: list[dict] = []
    allow_global_fallback = str(profile.get("meal_role") or "") == "highlight"
    arrival_minutes = _minutes_since_midnight(outbound_flight.get("arrival_time", "")) if outbound_flight else None
    if outbound_flight:
        items.append(
            {
                "time": _clock_time(outbound_flight.get("arrival_time"), fallback="17:30"),
                "icon": "flight",
                "key": "flight_outbound",
                "name": _flight_display(outbound_flight),
                "cost": _usd_to_sgd_str(outbound_flight.get("price_usd")),
            }
        )

    arrival_anchor = hotel or {}
    dinner_pref = max((arrival_minutes or (17 * 60 + 30)) + 60, 18 * 60)
    if arrival_minutes is not None and arrival_minutes <= 15 * 60 + 30:
        light_activity = _pick_activity_near_anchor(
            attractions,
            used_place_names,
            profile,
            preference_model,
            arrival_anchor,
            light_only=True,
            discouraged_names=discouraged_activities,
        )
        if light_activity:
            activity_time = max(arrival_minutes + 90, 15 * 60 + 20)
            activity_end = activity_time + _activity_duration_minutes(light_activity)
            if activity_time <= _activity_latest_start_minutes(light_activity, service_date) and activity_end <= 17 * 60 + 45:
                items.append(
                    {
                        "time": _hhmm(activity_time),
                        "icon": "activity",
                        "key": light_activity["key"],
                        "name": light_activity["name"],
                        "cost": light_activity.get("price") or "TBC",
                    }
                )
                arrival_anchor = light_activity

    dinner = _pick_restaurant_for_anchor(
        restaurants,
        used_place_names,
        profile,
        preference_model,
        arrival_anchor,
        preferred_start=dinner_pref,
        service_date=service_date,
        meal_duration=90,
        discouraged_names=discouraged_restaurants,
        variant_offset=int(profile.get("arrival_restaurant_offset", 0)),
    )
    if dinner is None and allow_global_fallback and fallback_restaurants and fallback_restaurants is not restaurants:
        dinner = _pick_restaurant_for_anchor(
            fallback_restaurants,
            used_place_names,
            profile,
            preference_model,
            arrival_anchor,
            preferred_start=dinner_pref,
            service_date=service_date,
            meal_duration=90,
            discouraged_names=discouraged_restaurants,
            variant_offset=int(profile.get("arrival_restaurant_offset", 0)),
        )
    if dinner:
        premium_bonus = 15 if _item_signal_counts(_text_blob(dinner)).get("fine_dining", 0) > 0 else 0
        anchor_minutes = _minutes_since_midnight(items[-1]["time"]) if len(items) > 1 else arrival_minutes
        dinner_start_pref = max((anchor_minutes or (17 * 60 + 30)) + 60, 18 * 60 + premium_bonus)
        dinner_start = _fit_service_start(
            dinner,
            dinner_start_pref,
            _meal_duration_minutes(dinner),
            service_date,
        )
        if dinner_start is not None:
            dinner_time = _hhmm(dinner_start)
            used_place_names.add(_normalized_name(dinner))
        else:
            dinner_time = None
        if dinner_time is not None:
            items.append(
                {
                    "time": dinner_time,
                    "icon": "restaurant",
                    "key": dinner["key"],
                    "name": dinner["name"],
                    "cost": dinner.get("price") or "TBC",
                }
            )
    dinner_added = any(item.get("icon") == "restaurant" for item in items)

    if hotel:
        last_anchor_time = _minutes_since_midnight(items[-1]["time"]) if len(items) > 1 else arrival_minutes
        hotel_minute = max((arrival_minutes or (17 * 60 + 30)) + 150, (last_anchor_time or 19 * 60) + 30)
        if dinner and dinner_added:
            dinner_time = _minutes_since_midnight(items[-1]["time"]) or (18 * 60)
            hotel_minute = max(
                hotel_minute,
                dinner_time + _meal_duration_minutes(dinner) + _travel_minutes(dinner, hotel, base=14),
            )
        hotel_time = _hhmm(hotel_minute)
        items.append(
            {
                "time": hotel_time,
                "icon": "hotel",
                "key": "hotel_stay",
                "name": hotel.get("name", ""),
                "cost": _usd_to_sgd_str(hotel.get("price_per_night_usd")) + "/night" if hotel.get("price_per_night_usd") else "TBC",
            }
        )
    return items


def _departure_day_items(
    profile: dict,
    return_flight: dict | None,
    hotel: dict | None,
    attractions: list[dict],
    restaurants: list[dict],
    fallback_restaurants: list[dict] | None,
    used_place_names: set[str],
    preference_model: dict,
    service_date: date | None = None,
    discouraged_activities: set[str] | None = None,
    discouraged_restaurants: set[str] | None = None,
) -> list[dict]:
    items: list[dict] = []
    allow_global_fallback = str(profile.get("meal_role") or "") == "highlight"
    departure_minutes = _minutes_since_midnight(return_flight.get("departure_time", "")) if return_flight else None
    airport_cutoff = departure_minutes - 180 if departure_minutes is not None else None
    hotel_anchor = hotel or {}
    if airport_cutoff is None:
        checkout_time = 9 * 60
    else:
        checkout_time = max(8 * 60, min(10 * 60 + 30, airport_cutoff - 180))

    if hotel:
        items.append(
            {
                "time": _hhmm(checkout_time),
                "icon": "hotel",
                "key": "hotel_checkout",
                "name": f"Checkout from {hotel.get('name', '')}".strip(),
                "cost": _usd_to_sgd_str(hotel.get("price_per_night_usd")) + "/night" if hotel.get("price_per_night_usd") else "TBC",
            }
        )

    current = checkout_time + 20
    available_after_checkout = max(0, (airport_cutoff or current) - current)
    if available_after_checkout >= 165:
        activity = _pick_activity_near_anchor(
            attractions,
            used_place_names,
            profile,
            preference_model,
            hotel_anchor,
            light_only=True,
            discouraged_names=discouraged_activities,
        )
        if activity:
            activity_time = max(current, 9 * 60 + 5)
            activity_end = activity_time + _activity_duration_minutes(activity)
            if activity_time <= _activity_latest_start_minutes(activity, service_date) and activity_end <= (airport_cutoff or activity_end) - 80:
                items.append(
                    {
                        "time": _hhmm(activity_time),
                        "icon": "activity",
                        "key": activity["key"],
                        "name": activity["name"],
                        "cost": activity.get("price") or "TBC",
                    }
                )
                current = activity_end
                meal_anchor = activity
            else:
                meal_anchor = hotel_anchor
        else:
            meal_anchor = hotel_anchor
        brunch = _pick_restaurant_for_anchor(
            restaurants,
            used_place_names,
            profile,
            preference_model,
            meal_anchor,
            preferred_start=max(current, 10 * 60 + 20),
            service_date=service_date,
            meal_duration=75,
            latest_finish=(airport_cutoff or current) - 20,
            discouraged_names=discouraged_restaurants,
        )
        if brunch is None and allow_global_fallback and fallback_restaurants and fallback_restaurants is not restaurants:
            brunch = _pick_restaurant_for_anchor(
                fallback_restaurants,
                used_place_names,
                profile,
                preference_model,
                meal_anchor,
                preferred_start=max(current, 10 * 60 + 20),
                service_date=service_date,
                meal_duration=75,
                latest_finish=(airport_cutoff or current) - 20,
                discouraged_names=discouraged_restaurants,
            )
        if brunch:
            current += _travel_minutes(meal_anchor, brunch, base=14)
            brunch_pref = max(current, 10 * 60 + 20)
            brunch_start = _fit_service_start(
                brunch,
                brunch_pref,
                _meal_duration_minutes(brunch),
                service_date,
                latest_finish=(airport_cutoff or brunch_pref) - 20,
            )
            if brunch_start is not None:
                items.append(
                    {
                        "time": _hhmm(brunch_start),
                        "icon": "restaurant",
                        "key": brunch["key"],
                        "name": brunch["name"],
                        "cost": brunch.get("price") or "TBC",
                    }
                )
                used_place_names.add(_normalized_name(brunch))
    elif available_after_checkout >= 75:
        brunch = _pick_restaurant_for_anchor(
            restaurants,
            used_place_names,
            profile,
            preference_model,
            hotel_anchor,
            preferred_start=max(current, 9 * 60 + 40),
            service_date=service_date,
            meal_duration=75,
            latest_finish=(airport_cutoff or current) - 20,
            discouraged_names=discouraged_restaurants,
        )
        if brunch is None and allow_global_fallback and fallback_restaurants and fallback_restaurants is not restaurants:
            brunch = _pick_restaurant_for_anchor(
                fallback_restaurants,
                used_place_names,
                profile,
                preference_model,
                hotel_anchor,
                preferred_start=max(current, 9 * 60 + 40),
                service_date=service_date,
                meal_duration=75,
                latest_finish=(airport_cutoff or current) - 20,
                discouraged_names=discouraged_restaurants,
            )
        if brunch:
            brunch_pref = max(current, 9 * 60 + 40)
            brunch_start = _fit_service_start(
                brunch,
                brunch_pref,
                _meal_duration_minutes(brunch),
                service_date,
                latest_finish=(airport_cutoff or brunch_pref) - 20,
            )
            if brunch_start is not None:
                items.append(
                    {
                        "time": _hhmm(brunch_start),
                        "icon": "restaurant",
                        "key": brunch["key"],
                        "name": brunch["name"],
                        "cost": brunch.get("price") or "TBC",
                    }
                )
                used_place_names.add(_normalized_name(brunch))

    if return_flight:
        items.append(
            {
                "time": _clock_time(return_flight.get("departure_time"), fallback="12:30"),
                "icon": "flight",
                "key": "flight_return",
                "name": _flight_display(return_flight),
                "cost": _usd_to_sgd_str(return_flight.get("price_usd")),
            }
        )
    return sorted(items, key=lambda item: _sort_time_value(item.get("time", "")))


def _build_option_schedule(
    option_key: str,
    profile: dict,
    compact_attractions: list[dict],
    option_restaurants: list[dict],
    fallback_restaurants: list[dict],
    hotel: dict | None,
    outbound_flight: dict | None,
    return_flight: dict | None,
    duration_days: int,
    preference_model: dict,
    trip_start_date: date | None = None,
    discouraged_activities: set[str] | None = None,
    discouraged_restaurants: set[str] | None = None,
    discouraged_activity_centroids: list[tuple[float, float]] | None = None,
) -> tuple[list, dict, list[tuple[float, float]], list[dict]]:
    used_place_names: set[str] = set()
    used_cluster_seeds: set[str] = set()
    day_activity_centroids: list[tuple[float, float]] = []
    day_decisions: list[dict] = []
    allow_global_fallback = str(profile.get("meal_role") or "") == "highlight"
    days: list[dict] = []
    stats = {
        "removed_items": 0,
        "normalized_items": 0,
        "duplicate_items": 0,
        "invalid_items": 0,
        "sparse_days": [],
        "boundary_repairs": [],
        "route_warnings": [],
    }

    days.append(
        {
            "day": "Day 1",
            "items": _arrival_day_items(
                profile,
                outbound_flight,
                hotel,
                compact_attractions,
                option_restaurants,
                fallback_restaurants,
                used_place_names,
                preference_model,
                service_date=_service_date(trip_start_date, 0),
                discouraged_activities=discouraged_activities,
                discouraged_restaurants=discouraged_restaurants,
                option_key=option_key,
            ),
        }
    )
    day_decisions.append({
        "day": "Day 1",
        "day_type": "arrival",
        "theme": "arrival evening",
        "seed_name": None,
        "seed_reason": "arrival day uses a fixed light-evening template",
        "activities": [],
        "lunch": None,
        "dinner": None,
    })

    middle_days = max(duration_days - 2, 0)
    cluster_radius = float(profile.get("cluster_radius_km", 5.0))

    for idx in range(middle_days):
        minimum_activity_count = int(profile.get("min_activity_count", 2))
        remaining_middle_days = middle_days - idx
        activity_count = _preferred_activity_count(
            compact_attractions,
            used_place_names,
            profile,
            preference_model,
            remaining_middle_days,
        )
        if len(compact_attractions) >= minimum_activity_count:
            activity_count = max(minimum_activity_count, activity_count)

        day_activity_pool = _build_day_activity_pool(
            compact_attractions,
            used_place_names,
            profile,
            preference_model,
            desired_count=activity_count,
            day_index=idx,
            discouraged_names=discouraged_activities,
            used_cluster_seeds=used_cluster_seeds,
            discouraged_centroids=discouraged_activity_centroids,
        )

        prior_themes = [d["theme"] for d in day_decisions if d.get("theme")]
        seed_candidates = _prepare_seed_candidates(
            day_activity_pool or compact_attractions,
            profile,
            preference_model,
            cluster_radius=cluster_radius,
            max_candidates=5,
            discouraged_names=discouraged_activities,
        )
        if seed_candidates:
            selected_seed, day_theme, seed_reason = _llm_select_seed(
                seed_candidates,
                day_index=idx,
                option_key=option_key,
                profile=profile,
                preference_model=preference_model,
                prior_themes=prior_themes,
            )
        else:
            selected_seed, day_theme, seed_reason = None, "mixed sightseeing", "pool too small for seed selection"

        activity_profile = (
            profile
            if str(profile.get("selection_mode") or "coverage") != "immersion"
            else {**profile, "selection_mode": "coverage"}
        )
        activities = _choose_cluster_activities(
            day_activity_pool or compact_attractions,
            used_place_names,
            activity_profile,
            preference_model,
            count=activity_count,
            day_index=idx,
            discouraged_names=discouraged_activities,
            forced_seed=selected_seed,
        )
        blocked_place_names = {_normalized_name(item) for item in activities if _normalized_name(item)}
        lunch = _pick_mandatory_meal(
            option_restaurants,
            used_place_names,
            blocked_place_names,
            profile,
            preference_model,
            activities[0] if activities else None,
            meal_kind="lunch",
            service_date=_service_date(trip_start_date, idx + 1),
            discouraged_names=discouraged_restaurants,
        )
        if lunch is None and allow_global_fallback and fallback_restaurants and fallback_restaurants is not option_restaurants:
            lunch = _pick_mandatory_meal(
                fallback_restaurants,
                used_place_names,
                blocked_place_names,
                profile,
                preference_model,
                activities[0] if activities else None,
                meal_kind="lunch",
                service_date=_service_date(trip_start_date, idx + 1),
                discouraged_names=discouraged_restaurants,
            )
        reserved_restaurants = set(used_place_names)
        if lunch and lunch.get("name"):
            reserved_restaurants.add(_normalized_name(lunch))
            blocked_place_names.add(_normalized_name(lunch))
        dinner_anchor = activities[-1] if activities else lunch
        dinner = _pick_mandatory_meal(
            option_restaurants,
            reserved_restaurants,
            blocked_place_names,
            profile,
            preference_model,
            dinner_anchor,
            meal_kind="dinner",
            service_date=_service_date(trip_start_date, idx + 1),
            discouraged_names=discouraged_restaurants,
        )
        if dinner is None and allow_global_fallback and fallback_restaurants and fallback_restaurants is not option_restaurants:
            dinner = _pick_mandatory_meal(
                fallback_restaurants,
                reserved_restaurants,
                blocked_place_names,
                profile,
                preference_model,
                dinner_anchor,
                meal_kind="dinner",
                service_date=_service_date(trip_start_date, idx + 1),
                discouraged_names=discouraged_restaurants,
            )
        items = _middle_day_items(
            profile,
            activities,
            lunch,
            dinner,
            day_index=idx + 1,
            service_date=_service_date(trip_start_date, idx + 1),
        )
        meal_total = sum(1 for item in items if item.get("icon") == "restaurant")
        if meal_total < 2:
            has_lunch = any(
                item.get("icon") == "restaurant"
                and (_minutes_since_midnight(item.get("time", "")) or 0) < 16 * 60
                for item in items
            )
            has_dinner = any(
                item.get("icon") == "restaurant"
                and (_minutes_since_midnight(item.get("time", "")) or 0) >= 16 * 60
                for item in items
            )
            if not has_lunch:
                _append_missing_meal(
                    items, option_restaurants, used_place_names, profile, preference_model,
                    meal_kind="lunch",
                    service_date=_service_date(trip_start_date, idx + 1),
                    discouraged_restaurants=discouraged_restaurants,
                )
                if not any(
                    item.get("icon") == "restaurant"
                    and (_minutes_since_midnight(item.get("time", "")) or 0) < 16 * 60
                    for item in items
                ) and fallback_restaurants and fallback_restaurants is not option_restaurants:
                    _append_missing_meal(
                        items, fallback_restaurants, used_place_names, profile, preference_model,
                        meal_kind="lunch",
                        service_date=_service_date(trip_start_date, idx + 1),
                        discouraged_restaurants=discouraged_restaurants,
                    )
            if not has_dinner:
                _append_missing_meal(
                    items, option_restaurants, used_place_names, profile, preference_model,
                    meal_kind="dinner",
                    service_date=_service_date(trip_start_date, idx + 1),
                    discouraged_restaurants=discouraged_restaurants,
                )
                if not any(
                    item.get("icon") == "restaurant"
                    and (_minutes_since_midnight(item.get("time", "")) or 0) >= 16 * 60
                    for item in items
                ) and fallback_restaurants and fallback_restaurants is not option_restaurants:
                    _append_missing_meal(
                        items, fallback_restaurants, used_place_names, profile, preference_model,
                        meal_kind="dinner",
                        service_date=_service_date(trip_start_date, idx + 1),
                        discouraged_restaurants=discouraged_restaurants,
                    )
            items.sort(key=lambda item: _sort_time_value(item.get("time", "")))
        items = _enforce_restaurant_time_buffers(items, min_restaurant_block_minutes=80)
        activity_total = sum(1 for item in items if item.get("icon") == "activity")
        meal_total = sum(1 for item in items if item.get("icon") == "restaurant")
        if activity_total < minimum_activity_count or meal_total < 2:
            stats["sparse_days"].append(
                {"day_index": idx + 2, "activity_count": activity_total, "meal_count": meal_total}
            )
        activity_centroid = _items_centroid(activities)
        if activity_centroid is not None:
            day_activity_centroids.append(activity_centroid)
        for item in items:
            if item.get("icon") in {"activity", "restaurant"}:
                normalized = _normalized_name(item)
                if normalized:
                    used_place_names.add(normalized)
        days.append({"day": f"Day {idx + 2}", "items": items})

        day_decisions.append({
            "day": f"Day {idx + 2}",
            "day_type": "middle",
            "theme": day_theme,
            "seed_name": selected_seed.get("name") if selected_seed else None,
            "seed_reason": seed_reason,
            "activities": [
                {
                    "name": a.get("name", ""),
                    "type": str(a.get("type") or a.get("category") or ""),
                }
                for a in activities
            ],
            "lunch": _meal_trace(lunch, "lunch", preference_model),
            "dinner": _meal_trace(dinner, "dinner", preference_model),
        })

    days.append(
        {
            "day": f"Day {duration_days}",
            "items": _departure_day_items(
                profile,
                return_flight,
                hotel,
                compact_attractions,
                option_restaurants,
                fallback_restaurants,
                used_place_names,
                preference_model,
                service_date=_service_date(trip_start_date, duration_days - 1),
                discouraged_activities=discouraged_activities,
                discouraged_restaurants=discouraged_restaurants,
            ),
        }
    )
    day_decisions.append({
        "day": f"Day {duration_days}",
        "day_type": "departure",
        "theme": "departure day",
        "seed_name": None,
        "seed_reason": "departure day uses checkout + last visit + airport template",
        "activities": [],
        "lunch": None,
        "dinner": None,
    })

    seen_place_names: set[str] = set()
    for day in days:
        for item in day.get("items", []):
            name = _normalized_name(item)
            if not name:
                continue
            if item.get("icon") in {"activity", "restaurant"}:
                if name in seen_place_names:
                    stats["duplicate_items"] += 1
                else:
                    seen_place_names.add(name)
    stats["route_warnings"] = []
    return days, stats, day_activity_centroids, day_decisions


# Main route-first scheduler used by planner_from_research_1().
def _build_deterministic_plan(
    compact_attractions: list[dict],
    compact_restaurants: list[dict],
    compact_hotels: list[dict],
    compact_flights_out: list[dict],
    compact_flights_ret: list[dict],
    duration_days: int,
    preference_model: dict,
    trip_start_date: date | None = None,
) -> tuple[dict, dict, dict, dict, dict, dict]:
    compact_attractions = [item for item in compact_attractions if _is_normal_activity_candidate(item)]
    compact_restaurants = [item for item in compact_restaurants if _is_normal_restaurant_candidate(item)]
    attraction_lookup = _name_lookup(compact_attractions)
    restaurant_lookup = _name_lookup(compact_restaurants)
    outbound_flight = _pick_outbound_flight(compact_flights_out)
    return_flight = _pick_return_flight(compact_flights_ret)
    hotel_by_option = _pick_hotels_by_profile(compact_hotels)
    restaurant_pools = _build_option_restaurant_pools(
        compact_restaurants,
        duration_days,
        preference_model,
    )

    draft_options: dict = {}
    final_itineraries: dict = {}
    normalized_itineraries: dict = {}
    option_meta: dict = {}
    validation_report: dict = {}
    planner_decision_trace: dict = {}
    prior_option_activity_names: set[str] = set()
    prior_option_activity_centroids: list[tuple[float, float]] = []

    for option_key, profile in _OPTION_PROFILES.items():
        hotel = hotel_by_option.get(option_key)
        option_restaurants = restaurant_pools.get(option_key) or compact_restaurants
        if str(profile.get("selection_mode") or "") == "immersion":
            days, stats, day_centroids, day_decisions = _build_immersion_option_schedule(
                option_key,
                profile,
                compact_attractions,
                option_restaurants,
                compact_restaurants,
                hotel,
                outbound_flight,
                return_flight,
                duration_days,
                preference_model,
                trip_start_date=trip_start_date,
                discouraged_activities=prior_option_activity_names,
                discouraged_restaurants=None,
                discouraged_activity_centroids=prior_option_activity_centroids,
            )
        else:
            days, stats, day_centroids, day_decisions = _build_option_schedule(
                option_key,
                profile,
                compact_attractions,
                option_restaurants,
                compact_restaurants,
                hotel,
                outbound_flight,
                return_flight,
                duration_days,
                preference_model,
                trip_start_date=trip_start_date,
                discouraged_activities=prior_option_activity_names,
                discouraged_restaurants=None,
                discouraged_activity_centroids=prior_option_activity_centroids,
            )

        final_itineraries[option_key] = days
        normalized_itineraries[option_key] = deepcopy(days)
        draft_options[option_key] = {
            "label": profile["label"],
            "desc": profile["desc"],
            "budget": profile["budget"],
            "style": profile["style"],
            "badge": profile["badge"],
        }
        option_meta[option_key] = {
            "label": profile["label"],
            "desc": profile["desc"],
            "budget": profile["budget"],
            "style": profile["style"],
            "badge": profile["badge"],
        }
        validation_report[option_key] = stats
        planner_decision_trace[option_key] = _synchronize_planner_decision_trace(
            days,
            day_decisions,
            attraction_lookup,
            restaurant_lookup,
            preference_model,
        )

        for day in days:
            for item in day.get("items", []):
                name = _normalized_name(item)
                if not name:
                    continue
                if item.get("icon") == "activity":
                    prior_option_activity_names.add(name)
        prior_option_activity_centroids.extend(day_centroids)

    chain_of_thought = json.dumps(planner_decision_trace, ensure_ascii=False)
    raw_output = {
        "planner_mode": "deterministic_route_first_with_llm_seeds",
        "chain_of_thought": chain_of_thought,
        "options": draft_options,
    }
    return (
        raw_output,
        final_itineraries,
        normalized_itineraries,
        option_meta,
        validation_report,
        planner_decision_trace,
    )


def planner_from_research(state: dict, research_result: dict) -> dict:
    state = _normalize_trip_state(state)
    dest = state.get("destination")
    origin = state.get("origin")
    dates = state.get("dates")
    budget = state.get("budget")
    duration = state.get("duration")

    missing = [
        field
        for field, value in [
            ("destination", dest),
            ("origin", origin),
            ("dates", dates),
            ("budget", budget),
            ("duration", duration),
        ]
        if not value
    ]
    if missing:
        return {"error": f"Missing required fields: {', '.join(missing)}"}

    try:
        payload = _extract_research_payload(state, research_result)
        research = payload["research"]
        tool_log = payload["tool_log"]
        compact_attractions = payload["compact_attractions"]
        compact_restaurants = payload["compact_restaurants"]
        compact_hotels = payload["compact_hotels"]
        compact_flights_out = payload["compact_flights_out"]
        compact_flights_ret = payload["compact_flights_ret"]
        prompt_inventory = _build_prompt_inventory(
            compact_attractions,
            compact_restaurants,
            compact_hotels,
            compact_flights_out,
            compact_flights_ret,
        )
        compact_attractions = _ensure_inventory_keys(compact_attractions, "attraction")
        compact_restaurants = _ensure_inventory_keys(compact_restaurants, "restaurant")
        preference_model = _build_preference_model(state)
        trip_start_date = _trip_start_date(state)
        _cache_inventory_snapshot(
            research,
            research_result,
            tool_log,
            compact_attractions,
            compact_restaurants,
            compact_hotels,
            compact_flights_out,
            compact_flights_ret,
            prompt_inventory,
        )

        (
            result,
            itineraries,
            normalized_itineraries,
            option_meta,
            validation_report,
            planner_decision_trace,
        ) = _build_deterministic_plan(
            compact_attractions,
            compact_restaurants,
            compact_hotels,
            compact_flights_out,
            compact_flights_ret,
            _duration_days(duration),
            preference_model,
            trip_start_date=trip_start_date,
        )
        planner_chain_of_thought = result.get("chain_of_thought", "")
        tool_log.append("[planner_agent_1] Deterministic route-first scheduler with LLM seed selection used saved research inventory")
        tool_log.append(f"[planner_agent_1] Generated options: {list(itineraries.keys())}")

        return {
            "itineraries": itineraries,
            "final_itineraries": itineraries,
            "validated_itineraries": itineraries,
            "option_meta": option_meta,
            "validation_report": validation_report,
            "planner_decision_trace": planner_decision_trace,
            "planner_chain_of_thought": planner_chain_of_thought,
            "chain_of_thought": planner_chain_of_thought,
            "research": research,
            "state": state,
            "tool_log": tool_log,
            "flight_options_outbound": compact_flights_out,
            "flight_options_return": compact_flights_ret,
            "hotel_options": payload["hotel_opts"],
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[Planner1] Error: {exc}")
        return {"error": str(exc)}


# Thin wrapper kept for call sites that still expect planner to trigger research first.
def planner_agent(state: dict, tools: dict | None = None) -> dict:
    state = _normalize_trip_state(state)
    research_result = research_agent(state, tools or {})
    if "error" in research_result:
        return research_result
    return planner_from_research(state, research_result)
