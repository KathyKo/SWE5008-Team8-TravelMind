from typing import Callable, Dict

from tools.search_flights import search_flights
from tools.search_hotels import search_hotels
from tools.search_weather import search_weather
from tools.web_search import web_search
from tools.google_search import google_search


# Explicit per-agent tool permissions.
TOOLS_BY_AGENT: Dict[str, Dict[str, Callable]] = {
    # Agent3: Planner
    "planner_agent": {
        "search_weather":     search_weather,
        "search_flights":     search_flights,
        "search_hotels":      search_hotels,
        "web_search":         web_search,
        "google_search":      google_search,
    },
    # Research agent
    "research_agent": {
        "search_weather":     search_weather,
        "search_flights":     search_flights,
        "search_hotels":      search_hotels,
        "web_search":         web_search,
        "google_search":      google_search,
    },
}


def get_tools_for_agent(agent_name: str) -> Dict[str, Callable]:
    """
    Returns the tool dictionary for a given agent.
    This demonstrates explicit tool access control: if an agent name
    is not configured here, it has no tools by default.
    """
    return TOOLS_BY_AGENT.get(agent_name, {})
