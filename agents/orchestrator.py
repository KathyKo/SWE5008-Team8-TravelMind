from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def travel_orchestrator(state):
    """
    Head Coordinator: Analyzes progress and strategically routes to the best expert.
    """
    messages = state.get("messages", [])
    destination = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget")
    
    system_prompt = f"""You are the Head Strategy Coordinator. Your role is to analyze the conversation and delegate to the most appropriate specialist.

CURRENT STATE:
- Destination: {destination if destination else 'Pending'}
- Dates: {dates if dates else 'Pending'}
- Budget: {budget if budget else 'Pending'}

ROUTING PROTOCOLS:
1. concierge: Use if ANY core detail (Destination, Dates, Budget) is missing, or if the user is asking general introductory questions.
2. booking_agent: Use once you have BOTH Destination and Dates to provide concrete flight and hotel options.
3. local_guide: Use if the Destination is known and the user wants to know about the 'vibe', weather, or things to do.
4. summarizer: Use ONLY if the user has explicitly indicated they are finished, or if you have enough info to present a complete, cohesive plan.

DECISION CRITERIA:
- Prioritize 'concierge' if the foundations aren't solid.
- Move to 'booking_agent' for logistics.
- Move to 'local_guide' for inspiration and local flavor.

RESPONSE:
Output ONLY the lowercase name of the selected agent (concierge, booking_agent, local_guide, or summarizer).
"""

    try:
        llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0)
        # Use history for context
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            *messages
        ])
        selected_agent = str(response.content).strip().lower()
        
        # Validation & fallback
        valid_agents = ["concierge", "booking_agent", "local_guide", "summarizer"]
        if selected_agent not in valid_agents:
            # Check for markdown or extra text
            for agent in valid_agents:
                if agent in selected_agent:
                    return {"next_agent": agent}
            selected_agent = "concierge"
            
        return {"next_agent": selected_agent}
    except Exception as e:
        print(f"Orchestrator error: {e}")
        return {"next_agent": "concierge"}
