from typing import Callable, Dict

from tools import (
    search_flights,
    search_hotels,
    search_weather,
    web_search,
    search_attractions,
    google_search,
)


# Explicit per-agent tool permissions.
TOOLS_BY_AGENT: Dict[str, Dict[str, Callable]] = {
    "booking_agent": {
        "web_search": web_search,
        "search_hotels": search_hotels,
        "search_flights": search_flights,
        "google_search": google_search,
    },
    "local_guide": {
        "search_weather": search_weather,
        "search_attractions": search_attractions,
    },
}


def get_tools_for_agent(agent_name: str) -> Dict[str, Callable]:
    """
    Returns the tool dictionary for a given agent.
    This demonstrates explicit tool access control: if an agent name
    is not configured here, it has no tools by default.
    """
    return TOOLS_BY_AGENT.get(agent_name, {})
