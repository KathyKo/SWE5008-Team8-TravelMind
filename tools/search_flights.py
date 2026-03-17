import json
from .web_search import web_search
from .google_search import google_search


def search_flights(origin_city: str, destination_city: str, travel_date: str) -> str:
    """
    Search flight schedules and prices using both Tavily and Google,
    targeting Skyscanner, Google Flights, and airline booking pages.
    """
    queries = [
        # Skyscanner — best for price comparison and schedules
        f"{origin_city} to {destination_city} flights {travel_date} skyscanner schedule price",
        # Google Flights results via general search
        f"flights {origin_city} {destination_city} {travel_date} direct economy departure arrival time",
        # Airline-level search for schedules
        f"{origin_city} {destination_city} direct flight {travel_date} flight number departure arrival",
    ]

    parts = []
    for q in queries:
        print(f"[TOOL] Flight search: {q}")
        t = web_search(q)
        g = google_search(q)
        for r in [t, g]:
            if r and "error" not in r[:60].lower():
                parts.append(r)

    if not parts:
        return json.dumps({"error": f"No flight results found for {origin_city}→{destination_city} on {travel_date}."})

    return " | ".join(parts)
