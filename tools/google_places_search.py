"""
Google Places search adapters for TravelMind.

This module converts Google Places API (New) responses into the same
list[dict] shape that the planner previously consumed from
`serp_tripadvisor_structured()`.
"""

from __future__ import annotations

from typing import Any

from .google_places_tool import GooglePlacesTool, PlacesAPIError


def _contains_location(query: str, location: str) -> bool:
    q = (query or "").lower()
    loc = (location or "").lower()
    return bool(loc and loc in q)


def _final_query(query: str, location: str = "") -> str:
    query = (query or "").strip()
    location = (location or "").strip()
    if not location or _contains_location(query, location):
        return query
    return f"{query} in {location}".strip()


def _normalize_category(place: dict[str, Any]) -> str:
    raw = place.get("primary_type") or ""
    if not raw:
        raw = place.get("types", ["Attraction"])[0] if place.get("types") else "Attraction"
    return str(raw).replace("_", " ").title()


def _to_legacy_shape(place: dict[str, Any]) -> dict[str, Any]:
    weekday_descriptions = place.get("weekday_descriptions") or []
    hours_summary = " | ".join(str(part) for part in weekday_descriptions if str(part).strip())
    return {
        "place_id": place.get("id") or place.get("resource_name") or place.get("name") or "",
        "name": place.get("name") or "",
        "category": _normalize_category(place),
        "rating": str(place.get("rating") or ""),
        "reviews": str(place.get("user_rating_count") or ""),
        # Google Places price signal is noisy for this workflow, so do not feed it downstream.
        "price": "",
        "description": (place.get("editorial_summary") or "")[:120],
        "source": "google_places",
        "address": place.get("formatted_address") or "",
        "lat": place.get("latitude"),
        "lng": place.get("longitude"),
        "open_now": place.get("open_now"),
        "hours": hours_summary,
        "weekday_descriptions": weekday_descriptions,
        "google_maps_uri": place.get("google_maps_uri") or "",
    }


def google_places_search_bundle(
    query: str,
    location: str = "",
    place_kind: str = "attraction",
    max_results: int = 12,
) -> dict[str, Any]:
    final_query = _final_query(query, location)
    if not final_query:
        return {
            "original_query": query,
            "final_query": final_query,
            "location": location,
            "place_kind": place_kind,
            "raw_places": [],
            "selected_places": [],
        }

    try:
        tool = GooglePlacesTool()
        places = tool.search_text(query=final_query, max_result_count=max_results)
    except (ValueError, PlacesAPIError, Exception) as e:
        print(f"[GooglePlaces] search failed for '{final_query}': {e}")
        return {
            "original_query": query,
            "final_query": final_query,
            "location": location,
            "place_kind": place_kind,
            "error": str(e),
            "raw_places": [],
            "selected_places": [],
        }

    selected_limited = [_to_legacy_shape(place) for place in places[:max_results]]
    return {
        "original_query": query,
        "final_query": final_query,
        "location": location,
        "place_kind": place_kind,
        "raw_places": places,
        "selected_places": selected_limited,
    }


def google_places_structured(
    query: str,
    location: str = "",
    place_kind: str = "attraction",
    max_results: int = 12,
) -> list[dict[str, Any]]:
    """
    Search Google Places and return planner-compatible place dicts.

    Args:
        query: Free-text search query.
        location: Optional destination used only when the query does not already mention it.
        place_kind: "attraction" or "restaurant".
        max_results: Max returned places after filtering/deduplication.
    """
    return google_places_search_bundle(
        query=query,
        location=location,
        place_kind=place_kind,
        max_results=max_results,
    ).get("selected_places", [])
