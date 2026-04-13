"""Google Places API (New) tool for TravelMind Research Agent.

This module wraps the Places API (New) endpoints used by the planner:

- Text Search (New)
- Place Details (New)

The wrapper keeps request construction, field masks, and response
normalization in one place so service-layer code can focus on planner logic.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SEARCH_FIELDS: List[str] = [
    "places.id",
    "places.name",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.primaryType",
    "places.types",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.regularOpeningHours",
    "places.googleMapsUri",
    "places.websiteUri",
]

DEFAULT_DETAIL_FIELDS: List[str] = [
    "id",
    "name",
    "displayName",
    "formattedAddress",
    "location",
    "primaryType",
    "types",
    "rating",
    "userRatingCount",
    "priceLevel",
    "regularOpeningHours",
    "googleMapsUri",
    "websiteUri",
    "editorialSummary",
    "reviews",
]


class PlacesAPIError(RuntimeError):
    """Raised when the Places API returns an error."""


@dataclass
class NearbyCircle:
    latitude: float
    longitude: float
    radius_meters: float

    def to_location_bias(self) -> Dict[str, Any]:
        return {
            "circle": {
                "center": {
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                },
                "radius": self.radius_meters,
            }
        }

    def to_location_restriction(self) -> Dict[str, Any]:
        return self.to_location_bias()


class GooglePlacesTool:
    """Thin wrapper over Google Places API (New)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 20,
        session: Optional[requests.Session] = None,
        include_raw_payload: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing Google Maps API key. Set GOOGLE_MAPS_API_KEY or pass api_key explicitly."
            )
        self.timeout = timeout
        self.session = session or requests.Session()
        self.base_url = "https://places.googleapis.com/v1"
        self.include_raw_payload = include_raw_payload

    def search_text(
        self,
        query: str,
        *,
        max_result_count: int = 10,
        price_levels: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        remaining = max(1, int(max_result_count))
        requested_fields = list(fields or DEFAULT_SEARCH_FIELDS)
        if "nextPageToken" not in requested_fields:
            requested_fields.append("nextPageToken")

        normalized_places: List[Dict[str, Any]] = []
        page_token: Optional[str] = None

        while remaining > 0:
            body: Dict[str, Any] = {
                "textQuery": query,
                "pageSize": min(20, remaining),
            }
            if price_levels:
                body["priceLevels"] = price_levels
            if page_token:
                body["pageToken"] = page_token

            payload = self._post(
                path="/places:searchText",
                body=body,
                field_mask=requested_fields,
            )

            page_places = [
                self.normalize_place(p, include_raw=self.include_raw_payload)
                for p in payload.get("places", [])
            ]
            normalized_places.extend(page_places)
            remaining -= len(page_places)

            page_token = payload.get("nextPageToken")
            if not page_token or not page_places:
                break

        return normalized_places[:max_result_count]

    def get_place_details(
        self,
        place_resource_name: str,
        *,
        language_code: str = "en",
        region_code: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"languageCode": language_code}
        if region_code:
            params["regionCode"] = region_code

        payload = self._get(
            path=f"/{place_resource_name}",
            params=params,
            field_mask=fields or DEFAULT_DETAIL_FIELDS,
        )
        return self.normalize_place(payload, include_raw=self.include_raw_payload)

    @staticmethod
    def normalize_place(
        place: Dict[str, Any],
        *,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        display_name = place.get("displayName", {})
        editorial = place.get("editorialSummary", {})
        location = place.get("location") or {}
        opening_hours = place.get("regularOpeningHours") or {}

        normalized = {
            "id": place.get("id"),
            "resource_name": place.get("name"),
            "name": display_name.get("text") if isinstance(display_name, dict) else display_name,
            "formatted_address": place.get("formattedAddress"),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "primary_type": place.get("primaryType"),
            "types": place.get("types", []),
            "rating": place.get("rating"),
            "user_rating_count": place.get("userRatingCount"),
            "price_level": place.get("priceLevel"),
            "open_now": opening_hours.get("openNow"),
            "weekday_descriptions": opening_hours.get("weekdayDescriptions", []),
            "google_maps_uri": place.get("googleMapsUri"),
            "website_uri": place.get("websiteUri"),
            "editorial_summary": editorial.get("text") if isinstance(editorial, dict) else editorial,
            "reviews": place.get("reviews", []),
        }
        if include_raw:
            normalized["raw"] = place
        return normalized

    @staticmethod
    def deduplicate_places(places: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for place in places:
            key = place.get("resource_name") or place.get("id") or place.get("name")
            if key and key not in seen:
                seen.add(key)
                deduped.append(place)
        return deduped

    @staticmethod
    def filter_places(
        places: Iterable[Dict[str, Any]],
        *,
        min_rating: Optional[float] = None,
        allowed_primary_types: Optional[List[str]] = None,
        require_open_now: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        allowed_primary_types = allowed_primary_types or []

        for place in places:
            rating = place.get("rating")
            primary_type = place.get("primary_type")
            open_now = place.get("open_now")

            if min_rating is not None and (rating is None or rating < min_rating):
                continue
            if allowed_primary_types and primary_type not in allowed_primary_types:
                continue
            if require_open_now is True and open_now is not True:
                continue
            filtered.append(place)
        return filtered

    def _headers(self, field_mask: List[str]) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": ",".join(field_mask),
        }

    def _post(self, path: str, body: Dict[str, Any], field_mask: List[str]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.post(
            url,
            headers=self._headers(field_mask),
            data=json.dumps(body),
            timeout=self.timeout,
        )
        return self._handle_response(response)

    def _get(self, path: str, params: Dict[str, Any], field_mask: List[str]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.get(
            url,
            headers=self._headers(field_mask),
            params=params,
            timeout=self.timeout,
        )
        return self._handle_response(response)

    @staticmethod
    def _handle_response(response: requests.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            payload = {"text": response.text}

        if not response.ok:
            message = payload.get("error", {}).get("message") or payload.get("message") or response.text
            raise PlacesAPIError(f"Places API request failed ({response.status_code}): {message}")
        return payload


if __name__ == "__main__":
    tool = GooglePlacesTool()
    sample_places = tool.search_text(
        query="Best attractions in Kyoto",
        max_result_count=10,
        fields=[
            "places.id",
            "places.name",
            "places.displayName",
            "places.formattedAddress",
            "places.location",
            "places.primaryType",
            "places.types",
            "places.rating",
            "places.userRatingCount",
            "places.regularOpeningHours",
            "places.googleMapsUri",
        ],
    )
    print(json.dumps(sample_places, ensure_ascii=False, indent=2))
