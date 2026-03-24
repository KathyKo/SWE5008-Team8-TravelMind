import json
from .web_search import web_search
from .google_search import google_search
from .serp_search import serp_local


def search_attractions(city: str, interest: str = "top attractions things to do") -> str:
    """
    Search attractions, activities, and local highlights.

    Strategy:
      1. SerpAPI Google Maps — real places with ratings, review counts,
         opening hours, and addresses.
      2. Fallback to Tavily + Google web search for broader editorial content
         (travel guides, blog posts) if Maps returns no results.

    Args:
        city:     Destination city or area
        interest: What to look for (e.g. "top attractions", "dive sites", "restaurants")
    """
    # --- Primary: SerpAPI Google Maps ---
    serp_result = serp_local(city, interest)
    if serp_result:
        return serp_result

    # --- Fallback: Tavily + Google web search ---
    queries = [
        f"{interest} in {city}",
        f"{city} best local experiences activities guide",
    ]

    parts = []
    for q in queries:
        print(f"[TOOL] Attractions search fallback: {q}")
        t = web_search(q)
        g = google_search(q)
        for r in [t, g]:
            if r and "error" not in r[:60].lower():
                parts.append(r)

    if not parts:
        return json.dumps({"error": f"No attraction results found for {city}."})

    return " | ".join(parts)
