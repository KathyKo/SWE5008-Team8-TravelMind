from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from .nodes import (
    human_node,
    intent_profile_node,
    orchestrator_node,
    concierge_node,
    booking_node,
    local_guide_node,
    summarizer_node,
    orchestrator_routing,
    check_exit_condition,
)
from .state import State


# Load environment variables (API Keys)
load_dotenv(override=True)


def build_travel_graph():
    """
    Builds the Travel Agency multi-agent graph.
    """
    builder = StateGraph(State)

    # 1. Add all nodes
    builder.add_node("human", human_node)
    builder.add_node("intent_profile", intent_profile_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("concierge", concierge_node)
    builder.add_node("booking_agent", booking_node)
    builder.add_node("local_guide", local_guide_node)
    builder.add_node("summarizer", summarizer_node)

    # 2. Define edges (The flow)
    builder.add_edge(START, "human")

    # From human, check if user wants to quit or continue to orchestrator
    builder.add_conditional_edges(
        "human",
        check_exit_condition,
        {
            "summarizer": "summarizer",
            "intent_profile": "intent_profile",
        },
    )

    # After intent profile extraction, route to the orchestrator
    builder.add_edge("intent_profile", "orchestrator")

    # From orchestrator, route to the selected specialized agent
    builder.add_conditional_edges(
        "orchestrator",
        orchestrator_routing,
        {
            "concierge": "concierge",
            "booking_agent": "booking_agent",
            "local_guide": "local_guide",
            "summarizer": "summarizer",
        },
    )

    # Specialized agents all return to human for next user input
    builder.add_edge("concierge", "human")
    builder.add_edge("booking_agent", "human")
    builder.add_edge("local_guide", "human")

    # Summarizer is the final step
    builder.add_edge("summarizer", END)

    return builder.compile()


def run_cli():
    print("==========================================")
    print("      TRAVEL PLANNING AGENCY       ")
    print("==========================================")
    print("Welcome! Our team of experts is ready to help you plan your dream trip.")
    print("Type 'exit' or 'done' whenever you are ready to finalize your itinerary.\n")

    graph = build_travel_graph()

    # Initial state
    initial_state = State(
        messages=[],
        origin=None,
        destination=None,
        dates=None,
        budget=None,
        preferences=None,
        duration=None,
        flight_options=None,
        hotel_options=None,
        stage=None,
        itinerary=None,
        research=None,
        selections=None,
        search_results=None,
        final_itinerary=None,
        next_agent=None,
        confirmed=False,
        is_complete=False,
    )

    try:
        # Start the graph interaction with a higher recursion limit for longer conversations
        graph.invoke(initial_state, config={"recursion_limit": 100})
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Session ended by user (Ctrl+C). Happy travels!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
