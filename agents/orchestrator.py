from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from llm_config import OPENAI_MODEL

def travel_orchestrator(state):
    """
    Head Strategy Coordinator: Determines the next agent based on state.
    """
    messages = state.get("messages", [])
    origin = state.get("origin")
    destination = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget")

    # Hard rule: if any core fields are missing, always go to concierge first.
    # This prevents jumping straight to booking_agent on the very first turn.
    if not origin or not destination or not dates or not budget:
        return {"next_agent": "concierge", "confirmed": False}

    system_prompt = f"""You are the Head Travel Coordinator. Your job is to select the best expert agent to help the customer based on their current progress.

CURRENT STATE:
- Origin: {origin if origin else 'Not set'}
- Destination: {destination if destination else 'Not set'}
- Dates: {dates if dates else 'Not set'}
- Budget: {budget if budget else 'Not set'}

ROUTING PROTOCOLS:
1. 'concierge': If we are missing basic info (Origin, Destination, Dates, or Budget).
2. 'booking_agent': If we have Destination and Dates and the user wants flight or hotel info.
3. 'local_guide': If Destination is known and the user wants to know about the weather or attractions.
4. 'summarizer': Use this when the travel plan is complete or when the user wants to finish.

You MUST respond with a single JSON object ONLY, no explanation, with the following shape:
{{
  "next_agent": "concierge" | "booking_agent" | "local_guide" | "summarizer",
  "confirmed": true | false
}}

"next_agent" is the next expert that should act.
"confirmed" should be true ONLY if the user clearly indicated they are satisfied with the current plan and ready to finalize the itinerary
  (for example, they say things like "yes", "ok", "looks good", "sounds good", "可以了", "就这样吧", etc.).
"""

    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            *messages
        ])

        raw = str(response.content)
        # Strip optional code fences
        if "```" in raw:
            raw = raw.split("```", 1)[1]
            if "```" in raw:
                raw = raw.split("```", 1)[0]
        raw = raw.strip()

        import json
        data = json.loads(raw)

        next_agent = str(data.get("next_agent", "")).strip().lower()
        confirmed = bool(data.get("confirmed", False))

        valid_agents = ["concierge", "booking_agent", "local_guide", "summarizer"]
        if next_agent not in valid_agents:
            for agent in valid_agents:
                if agent in next_agent:
                    next_agent = agent
                    break
            else:
                next_agent = "concierge"

        return {"next_agent": next_agent, "confirmed": confirmed}
    except Exception as e:
        print(f"Orchestrator error: {e}")
        return {"next_agent": "concierge"}
