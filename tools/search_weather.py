import os
import requests
import json
from dotenv import load_dotenv
from .web_search import web_search
from .google_search import google_search

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")



def _openweather_forecast(city: str, days: int = 5) -> str | None:
    """
    Fetch a rolling forecast from OpenWeather API.
    `days` controls how many days of 3-hour intervals to return (max ~5 days free tier).
    Returns JSON string on success, None on failure.
    """
    if not OPENWEATHER_API_KEY:
        return None

    # cnt = number of 3-hour intervals; 8 per day
    cnt = min(days * 8, 40)

    try:
        resp = requests.get(
            "http://api.openweathermap.org/data/2.5/forecast",
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric", "cnt": cnt},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json().get("list", [])
        if not data:
            return None

        # Sample one reading per day (every 8th interval = noon-ish)
        forecasts = []
        for i in range(0, len(data), 8):
            item = data[i]
            forecasts.append({
                "time":     item.get("dt_txt"),
                "summary":  item.get("weather", [{}])[0].get("description"),
                "temp_c":   item.get("main", {}).get("temp"),
                "humidity": item.get("main", {}).get("humidity"),
                "wind_kph": round(item.get("wind", {}).get("speed", 0) * 3.6, 1),
            })
        return json.dumps(forecasts)
    except Exception:
        return None


def _web_weather_search(city: str, travel_dates: str = "", days: int = 5) -> str:
    """
    Fallback: search seasonal / historical / forecast weather via Tavily + Google.
    Useful for far-future dates and remote locations.
    """
    date_part = f"{travel_dates} " if travel_dates else ""
    queries = [
        f"{city} weather {date_part}forecast {days} day temperature rain",
        f"{city} {date_part}travel weather what to expect climate",
    ]
    parts = []
    for q in queries:
        print(f"[TOOL] Weather web search: {q}")
        t = web_search(q)
        g = google_search(q)
        for r in [t, g]:
            if r and "error" not in r[:60].lower():
                parts.append(r)
    return " | ".join(parts)


def search_weather(city: str, travel_dates: str = "", days: int = 5) -> str:
    """
    Get weather information for any city or region.

    Args:
        city:          Destination name (any city, island, or region — no hardcoded list).
        travel_dates:  Optional travel date range (e.g. "2026-05-01 to 2026-05-03").
                       Used in web search fallback for seasonal context.
        days:          Number of forecast days requested (default 5).

    Strategy:
      1. Try OpenWeather API (accurate for near-future forecasts up to ~5 days).
      2. If the API fails or returns nothing — fall back to Tavily + Google web search,
         which works for far-future dates and seasonal planning.
    """
    print(f"[TOOL] Weather lookup: {city} | dates: {travel_dates} | days: {days}")

    # Try live API first
    api_result = _openweather_forecast(city, days)
    if api_result:
        return api_result

    # Fall back to web search
    web_result = _web_weather_search(city, travel_dates, days)
    if web_result:
        return web_result

    return json.dumps({"error": f"Could not retrieve weather data for '{city}'."})
