from importlib import import_module

__all__ = [
    "search_flights",
    "search_hotels",
    "search_weather",
    "web_search",
    "google_search",
    "serp_flights",
    "serp_flights_autocomplete",
    "serp_hotels",
    "serp_hotel_details",
    "serp_hotel_reviews",
    "serp_local",
    "serp_local_structured",
    "serp_maps_autocomplete",
    "serp_maps_reviews",
    "serp_travel_explore",
    "serp_tripadvisor",
    "serp_tripadvisor_structured",
    "get_hotel_token",
    "get_place_data_id",
]


def __getattr__(name: str):
    if name == "search_flights":
        return import_module(".search_flights", __name__).search_flights
    if name == "search_hotels":
        return import_module(".search_hotels", __name__).search_hotels
    if name == "search_weather":
        return import_module(".search_weather", __name__).search_weather
    if name == "web_search":
        return import_module(".web_search", __name__).web_search
    if name == "google_search":
        return import_module(".google_search", __name__).google_search
    if name in {
        "serp_flights",
        "serp_flights_autocomplete",
        "serp_hotels",
        "serp_hotel_details",
        "serp_hotel_reviews",
        "serp_local",
        "serp_local_structured",
        "serp_maps_autocomplete",
        "serp_maps_reviews",
        "serp_travel_explore",
        "serp_tripadvisor",
        "serp_tripadvisor_structured",
        "get_hotel_token",
        "get_place_data_id",
    }:
        serp = import_module(".serp_search", __name__)
        return getattr(serp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
