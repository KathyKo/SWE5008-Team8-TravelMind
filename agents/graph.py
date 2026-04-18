from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from .nodes import (
    input_guard_node,
    intent_profile_node,
    search_node,
    orchestrator_node,
    planner_node,
    replanner_node,
    debate_node,
    explain_node,
    output_guard_node,
    orchestrator_routing,
    
)
from .state import State


# Load environment variables (API Keys)
load_dotenv(override=True)


def build_travel_graph():
    """
    Builds the Travel Agency multi-agent graph.
    """
    builder = StateGraph(State)

    # 1. Add all nodes (8 agents + orchestrator )
    builder.add_node("input_guard", input_guard_node)                # Agent 5   (input guard)
    builder.add_node("intent_profile", intent_profile_node)          # Agent 1   (intent profile)
    builder.add_node("search", search_node)                          # Agent 2   (web search)
    builder.add_node("orchestrator", orchestrator_node)              # Agent 9   (orchestrator)
    builder.add_node("planner", planner_node)                        # Agent 3   (planner)
    builder.add_node("replanner", replanner_node)                    # Agent 7   (replanner)
    builder.add_node("debate", debate_node)                          # Agent 4   (debate)
    builder.add_node("explain", explain_node)                        # Agent 6   (explain)
    builder.add_node("output_guard", output_guard_node)              # Agent 8   (output guard)

    # 2. Define edges (The flow)
    builder.add_edge(START, "input_guard")
    builder.add_edge("input_guard", "orchestrator")
    builder.add_edge("intent_profile", "orchestrator")
    builder.add_edge("search", "orchestrator")
    builder.add_edge("planner", "orchestrator")
    builder.add_edge("replanner", "orchestrator")
    builder.add_edge("debate", "orchestrator")
    builder.add_edge("explain", "orchestrator")
    builder.add_edge("output_guard", END)

    # From orchestrator, route to the selected agent
    builder.add_conditional_edges(
        "orchestrator",
        orchestrator_routing,
        {
            "input_guard": "input_guard",
            "intent_profile": "intent_profile",
            "search": "search",
            "planner": "planner",
            "replanner": "replanner",
            "debate": "debate",
            "explain": "explain",
            "output_guard": "output_guard",
            "END": END,
        },
    )

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
        outbound_time_pref=None,
        return_time_pref=None,
        flight_options=None,
        hotel_options=None,
        user_profile=None,
        travelers=None,
        session_id=None,
        intent_profile_output=None,
        user_profile_structured=None,
        orchestration_stage="input_guard",
        input_guard_output=None,
        search_output=None,
        planner_output=None,
        replanner_output=None,
        user_feedback=None,
        replan_mode=False,
        debate_output=None,
        is_valid=None,
        debate_count=0,
        explain_output=None,
        output_guard_result=None,
        composite_score=None,
        replan_attempts=0,
        max_replan_attempts=2,
        approval_threshold=75.0,
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
