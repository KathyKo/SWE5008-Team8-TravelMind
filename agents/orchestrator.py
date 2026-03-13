from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def travel_orchestrator(state):
    """
    Head Strategy Coordinator: Determines the next agent based on state.
    """
    messages = state.get("messages", [])
    destination = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget")
    
    system_prompt = f"""You are the Head Travel Coordinator. Your job is to select the best expert agent to help the customer based on their current progress.

CURRENT STATE:
- Destination: {destination if destination else 'Not set'}
- Dates: {dates if dates else 'Not set'}
- Budget: {budget if budget else 'Not set'}

ROUTING PROTOCOLS:
1. 'concierge': If we are missing basic info (Destination, Dates, or Budget).
2. 'booking_agent': If we have Destination and Dates and the user wants flight or hotel info.
3. 'local_guide': If Destination is known and the user wants to know about the weather or attractions.
4. 'summarizer': Use this when the travel plan is complete or when the user wants to finish.

Respond with ONLY the name of the agent (concierge, booking_agent, local_guide, or summarizer).
"""

    try:
        llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            *messages
        ])
        selected_agent = str(response.content).strip().lower()
        
        # Validation
        valid_agents = ["concierge", "booking_agent", "local_guide", "summarizer"]
        if selected_agent not in valid_agents:
            for agent in valid_agents:
                if agent in selected_agent:
                    return {"next_agent": agent}
            selected_agent = "concierge"
            
        return {"next_agent": selected_agent}
    except Exception as e:
        print(f"Orchestrator error: {e}")
        return {"next_agent": "concierge"}
