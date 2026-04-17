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

    initial_state = {
        "messages": [],

        # Intent Profile — 全部由 intent_profile agent 填充
        "origin": None,
        "destination": None,
        "budget": None,
        "dates": None,
        "preferences": None,
        "duration": None,
        "user_profile": None,
        "travelers": None,
        "outbound_time_pref": None,
        "return_time_pref": None,
        "session_id": None,
        "intent_profile_output": None,
        "is_complete": False,

        # Security
        "threat_blocked": None,

        # Research / Search
        "search_results": None,
        "research": None,

        # Planner
        "itineraries": None,
        "final_itineraries": None,

        # Debate ↔ Planner 循环控制
        "is_valid": None,
        "debate_count": 0,
        "critique": None,
        "approval_threshold": 75.0,

        # Replanner（用户反馈触发）
        "user_feedback": None,
        "replan_attempts": 0,
        "max_replan_attempts": 2,

        # Explainability
        "explanation": None,
        "explain_data": None,

        # Output Guard
        "output_guard_decision": None,

        # Orchestrator
        "next_node": None,
        "error_message": None,
        "final_output": None,
    }

    try:
        # Start the graph interaction with a higher recursion limit for longer conversations
        graph.invoke(initial_state, config={"recursion_limit": 100})
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Session ended by user (Ctrl+C). Happy travels!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
