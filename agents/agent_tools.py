from typing import Callable, Dict

from tools import (
    search_flights,
    search_hotels,
    search_weather,
    web_search,
    google_search,
)


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
    # New research-only pipeline
    "research_agent_1": {
        "search_weather":     search_weather,
        "search_flights":     search_flights,
        "search_hotels":      search_hotels,
        "web_search":         web_search,
        "google_search":      google_search,
    },
    # New planner-only pipeline (delegates search to research_agent_1)
    "planner_agent_1": {
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
