import json
from .web_search import web_search
from .google_search import google_search


def search_hotels(city: str, preferences: str = "best rated", dates: str = "") -> str:
    """
    Search hotel options and prices using both Tavily and Google,
    targeting Agoda, Booking.com, and Trip.com for real prices.
    """
    date_part = f"{dates} " if dates else ""

    queries = [
        # OTA sites — best for actual room rates
        f"{preferences} hotel {city} {date_part}price per night agoda booking.com",
        # Trip.com / general OTA
        f"{city} hotel {date_part}{preferences} room rate site:trip.com OR site:agoda.com OR site:hotels.com",
        # Broad fallback
        f"{preferences} hotel {city} {date_part}availability booking",
    ]

    parts = []
    for q in queries:
        print(f"[TOOL] Hotel search: {q}")
        t = web_search(q)
        g = google_search(q)
        for r in [t, g]:
            if r and "error" not in r[:60].lower():
                parts.append(r)

    if not parts:
        return json.dumps({"error": f"No hotel results found for {city}."})

    return " | ".join(parts)
