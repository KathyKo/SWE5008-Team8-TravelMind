from typing import TypedDict, Optional, Annotated, List
import operator


class State(TypedDict):
    """
    Overall state of the Travel Agency multi-agent system.
    """
    # Conversation history
    messages: Annotated[list, operator.add]

    # User preferences (filled by concierge)
    origin: Optional[str]
    destination: Optional[str]
    budget: Optional[str]
    dates: Optional[str]
    preferences: Optional[str]   # e.g. "budget", "moderate", "luxury"
    duration: Optional[str]      # e.g. "3 days"
    user_id: Optional[str]       # User identifier for logging

    # Security fields
    threat_blocked: Optional[bool]
    threat_type: Optional[str]
    threat_detail: Optional[str]
    sanitised_input: Optional[str]
    security_audit_log: Optional[List]

    # Options gathered by specialized agents
    flight_options: Optional[List]
    hotel_options: Optional[List]

    # Booking agent multi-stage state
    stage: Optional[str]           # itinerary_draft | awaiting_selection | reviewing | confirmed
    itinerary: Optional[str]       # drafted itinerary text (written once, never overwritten)
    research: Optional[dict]       # raw destination research snippets
    selections: Optional[dict]     # {"transport": "...", "accommodation": "..."}
    search_results: Optional[dict] # {"transport": {...}, "accommodation": {...}}

    # Final output
    final_itinerary: Optional[str]

    # Flow control
    next_agent: Optional[str]
    confirmed: Optional[bool]
    is_complete: bool
