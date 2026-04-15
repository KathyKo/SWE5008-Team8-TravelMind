import json
from .web_search import web_search
from .google_search import google_search
from .serp_search import serp_hotels


def search_hotels(city: str, preferences: str = "best rated", dates: str = "") -> str:
    """
    Search hotel options and prices.

    Strategy:
      1. SerpAPI Google Hotels — structured data with real prices, star ratings,
         review counts, and availability.
      2. Fallback to Tavily + Google web search (Agoda, Booking.com, Trip.com)
         for niche properties not indexed by Google Hotels.

    Args:
        city:        Destination city or area
        preferences: Free-text preference (e.g. "beach resort", "budget", "luxury")
        dates:       Travel date range, e.g. "2026-05-01 to 2026-05-03"
    """
    # Parse check-in / check-out from date range string
    check_in, check_out = "", ""
    if " to " in dates:
        parts = dates.split(" to ")
        check_in  = parts[0].strip()
        check_out = parts[1].strip()

    # --- Primary: SerpAPI Google Hotels ---
    serp_result = serp_hotels(city, check_in, check_out, preferences)
    if serp_result:
        return serp_result

    # --- Fallback: Tavily + Google web search ---
    date_part = f"{dates} " if dates else ""
    queries = [
        f"{preferences} hotel {city} {date_part}price per night agoda booking.com",
        f"{city} hotel {date_part}{preferences} room rate site:trip.com OR site:agoda.com OR site:hotels.com",
        f"{preferences} hotel {city} {date_part}availability booking",
    ]

    parts = []
    for q in queries:
        print(f"[TOOL] Hotel search fallback: {q}")
        t = web_search(q)
        g = google_search(q)
        for r in [t, g]:
            if r and "error" not in r[:60].lower():
                parts.append(r)

    if not parts:
        return json.dumps({"error": f"No hotel results found for {city}."})

    return " | ".join(parts)
