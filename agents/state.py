from re import search
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

    # Intent/Profile extracted by Agent1
    user_profile: Optional[str]  # derived from preferences, e.g. "diet=vegetarian; interests=culture"
    travelers: Optional[int]
    outbound_time_pref: Optional[str]
    return_time_pref: Optional[str]
    session_id: Optional[str]
    intent_profile_output: Optional[dict]
    user_profile_structured: Optional[dict]

    # 8-agent orchestration layer artifacts
    orchestration_stage: Optional[str]
    input_guard_output: Optional[dict]
    search_output: Optional[dict]
    planner_output: Optional[dict]
    debate_output: Optional[dict]
    explain_output: Optional[dict]
    output_guard_result: Optional[dict]
    final_plan: Optional[dict]
    composite_score: Optional[float]
    replan_attempts: Optional[int]
    max_replan_attempts: Optional[int]
    approval_threshold: Optional[float]

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
