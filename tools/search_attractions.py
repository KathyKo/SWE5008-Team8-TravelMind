import json
from .web_search import web_search
from .google_search import google_search


def search_attractions(city: str, interest: str = "top attractions things to do") -> str:
    """
    Search attractions, activities, and local highlights using both Tavily and Google.
    """
    queries = [
        f"{interest} in {city}",
        f"{city} best local experiences activities guide",
    ]

    parts = []
    for q in queries:
        print(f"[TOOL] Attractions search: {q}")
        t = web_search(q)
        g = google_search(q)
        for r in [t, g]:
            if r and "error" not in r[:60].lower():
                parts.append(r)

    if not parts:
        return json.dumps({"error": f"No attraction results found for {city}."})

    return " | ".join(parts)
