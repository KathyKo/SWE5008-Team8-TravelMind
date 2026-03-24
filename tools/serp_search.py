import os
import json
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")


def _serpapi_request(engine: str, params: dict) -> dict | None:
    """Make a SerpAPI request. Returns parsed result dict or None on failure."""
    if not SERPAPI_KEY:
        print("[SerpAPI] SERPAPI_API_KEY not set.")
        return None
    try:
        from serpapi import GoogleSearch
        params = {**params, "api_key": SERPAPI_KEY, "engine": engine}
        result = GoogleSearch(params).get_dict()
        if "error" in result:
            print(f"[SerpAPI] {engine} error: {result['error']}")
            return None
        return result
    except Exception as e:
        print(f"[SerpAPI] {engine} exception: {e}")
        return None


# ---------------------------------------------------------------------------
# Google Flights
# ---------------------------------------------------------------------------

def serp_flights(origin: str, destination: str, date: str) -> str | None:
    """
    Search Google Flights via SerpAPI.
    Returns a formatted multi-line string with flight details, or None if no results.

    Args:
        origin:      City name or IATA code (e.g. "Singapore" or "SIN")
        destination: City name or IATA code (e.g. "Kuala Lumpur" or "KUL")
        date:        Travel date in YYYY-MM-DD format
    """
    print(f"[SerpAPI] Google Flights: {origin} → {destination} on {date}")
    result = _serpapi_request("google_flights", {
        "departure_id": origin,
        "arrival_id":   destination,
        "outbound_date": date,
        "currency": "USD",
        "hl": "en",
        "type": "2",  # 2 = one-way
    })
    if not result:
        return None

    flights = result.get("best_flights") or result.get("other_flights") or []
    if not flights:
        return None

    lines = []
    for f in flights[:5]:
        price          = f.get("price")
        total_duration = f.get("total_duration")
        for leg in f.get("flights", []):
            airline   = leg.get("airline", "")
            flight_no = leg.get("flight_number", "")
            dep       = leg.get("departure_airport", {})
            arr       = leg.get("arrival_airport", {})
            dep_time  = dep.get("time", "")
            arr_time  = arr.get("time", "")
            dep_name  = dep.get("name", dep.get("id", ""))
            arr_name  = arr.get("name", arr.get("id", ""))
            duration  = leg.get("duration")
            cabin     = leg.get("travel_class", "")

            line = f"{airline} {flight_no} | {dep_name} {dep_time} → {arr_name} {arr_time}"
            if duration:
                line += f" | {duration} min"
            if cabin:
                line += f" | {cabin}"
            if price:
                line += f" | USD {price}"
            lines.append(line)

    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Google Hotels
# ---------------------------------------------------------------------------

def serp_hotels(
    destination: str,
    check_in: str = "",
    check_out: str = "",
    preferences: str = "",
) -> str | None:
    """
    Search Google Hotels via SerpAPI.
    Returns a formatted multi-line string with hotel options, or None if no results.

    Args:
        destination: City or area name
        check_in:    Check-in date in YYYY-MM-DD format (optional)
        check_out:   Check-out date in YYYY-MM-DD format (optional)
        preferences: Free-text preference string (e.g. "beach resort", "budget")
    """
    print(f"[SerpAPI] Google Hotels: {destination} | {check_in}→{check_out} | {preferences}")
    params: dict = {
        "q":        f"hotels in {destination} {preferences}".strip(),
        "hl":       "en",
        "currency": "USD",
    }
    if check_in:
        params["check_in_date"] = check_in
    if check_out:
        params["check_out_date"] = check_out

    result = _serpapi_request("google_hotels", params)
    if not result:
        return None

    properties = result.get("properties", [])
    if not properties:
        return None

    lines = []
    for h in properties[:6]:
        name         = h.get("name", "")
        rating       = h.get("overall_rating", "")
        reviews      = h.get("reviews", "")
        hotel_class  = h.get("hotel_class", "")
        price_info   = h.get("rate_per_night", {})
        price        = price_info.get("lowest") or price_info.get("extracted_lowest", "")
        description  = h.get("description", "")

        line = name
        if hotel_class:
            line += f" | {hotel_class}"
        if rating:
            review_str = f" ({reviews} reviews)" if reviews else ""
            line += f" | ⭐ {rating}/5{review_str}"
        if price:
            line += f" | From USD {price}/night"
        if description:
            line += f" | {description[:120]}"
        lines.append(line)

    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Google Maps (local results)
# ---------------------------------------------------------------------------

def serp_local(destination: str, query: str) -> str | None:
    """
    Search Google Maps local results via SerpAPI.
    Returns a formatted multi-line string of places, or None if no results.

    Args:
        destination: City or area name
        query:       What to search for (e.g. "top attractions", "dive sites")
    """
    print(f"[SerpAPI] Google Maps: {query} in {destination}")
    result = _serpapi_request("google_maps", {
        "q":    f"{query} in {destination}",
        "hl":   "en",
        "type": "search",
    })
    if not result:
        return None

    places = result.get("local_results", [])
    if not places:
        return None

    lines = []
    for p in places[:8]:
        name    = p.get("title", "")
        rating  = p.get("rating", "")
        reviews = p.get("reviews", "")
        type_   = p.get("type", "")
        address = p.get("address", "")
        hours   = p.get("hours", "")

        line = name
        if type_:
            line += f" | {type_}"
        if rating:
            review_str = f" ({reviews} reviews)" if reviews else ""
            line += f" | ⭐ {rating}{review_str}"
        if address:
            line += f" | {address}"
        if hours:
            line += f" | {hours}"
        lines.append(line)

    return "\n".join(lines) if lines else None
