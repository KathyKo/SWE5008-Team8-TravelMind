import json
from .web_search import web_search
from .google_search import google_search
from .serp_search import serp_flights


def search_flights(origin_city: str, destination_city: str, travel_date: str) -> str:
    """
    Search flight schedules and prices.

    Strategy:
      1. SerpAPI Google Flights — structured data with real departure/arrival times,
         flight numbers, airlines, and prices.
      2. Fallback to Tavily + Google web search if SerpAPI returns nothing
         (e.g. small regional airports, unusual routes).

    Args:
        origin_city:      Departure city or IATA code (e.g. "Singapore" or "SIN")
        destination_city: Arrival city or IATA code (e.g. "Kuala Lumpur" or "KUL")
        travel_date:      Date in YYYY-MM-DD format
    """
    # --- Primary: SerpAPI Google Flights ---
    serp_result = serp_flights(origin_city, destination_city, travel_date)
    if serp_result:
        return serp_result

    # --- Fallback: Tavily + Google web search ---
    queries = [
        f"{origin_city} to {destination_city} flights {travel_date} skyscanner schedule price",
        f"flights {origin_city} {destination_city} {travel_date} direct economy departure arrival time",
        f"{origin_city} {destination_city} direct flight {travel_date} flight number departure arrival",
    ]

    parts = []
    for q in queries:
        print(f"[TOOL] Flight search fallback: {q}")
        t = web_search(q)
        g = google_search(q)
        for r in [t, g]:
            if r and "error" not in r[:60].lower():
                parts.append(r)

    if not parts:
        return json.dumps({"error": f"No flight results found for {origin_city}→{destination_city} on {travel_date}."})

    return " | ".join(parts)
