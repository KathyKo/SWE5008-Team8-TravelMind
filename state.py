from typing import TypedDict, Optional, Annotated, List
import operator


class State(TypedDict):
    """
    Overall state of the Travel Agency multi-agent system.
    """
    # Conversation history
    messages: Annotated[list, operator.add]
    
    # User Preferences (to be filled by the agents as they talk)
    origin: Optional[str]
    destination: Optional[str]
    budget: Optional[str]
    dates: Optional[str]
    
    # Options gathered by specialized agents
    flight_options: Optional[List[dict]]
    hotel_options: Optional[List[dict]]
    
    # The final output
    final_itinerary: Optional[str]
    
    # Flow control
    next_agent: Optional[str]
    confirmed: Optional[bool]
    is_complete: bool
