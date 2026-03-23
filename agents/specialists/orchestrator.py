import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from ..llm_config import OPENAI_MODEL


def travel_orchestrator(state):
    """
    Head Coordinator: routes each user turn to the correct specialist agent.

    Agent responsibilities (mutually exclusive):
    - concierge      : collect missing core travel info (origin/destination/dates/budget)
    - local_guide    : weather, climate, travel season, local attractions, dive sites,
                       activities, what-to-do recommendations — NO prices
    - booking_agent  : flights, transport, hotels, itinerary planning, live prices
    - summarizer     : produce the final summary when user says done/exit
    """
    messages   = state.get("messages", [])
    origin      = state.get("origin")
    destination = state.get("destination")
    dates       = state.get("dates")
    budget      = state.get("budget")

    # ── Hard rule 1: missing core info → always concierge ──
    if not origin or not destination or not dates or not budget:
        return {"next_agent": "concierge", "confirmed": False}

    # ── Hard rule 2: booking mid-flow → lock to booking_agent ──
    booking_stage = state.get("stage")
    if booking_stage in ("awaiting_selection", "reviewing"):
        return {"next_agent": "booking_agent", "confirmed": False}

    has_data = bool(state.get("flight_options")) and bool(state.get("hotel_options"))

    system_prompt = f"""You are the Head Travel Coordinator. Route the user's latest message to exactly one specialist agent.

CURRENT TRIP STATE:
- Origin: {origin}
- Destination: {destination}
- Dates: {dates}
- Budget: {budget}
- Booking data collected: {'Yes' if has_data else 'No'}

AGENT RESPONSIBILITIES — read carefully before routing:

1. concierge
   → ONLY when origin / destination / dates / budget is still missing.
   → Do NOT use for anything else.

2. local_guide
   → Weather, climate, best travel season, monsoon info.
   → Local attractions, sightseeing, nature, dive sites, beaches.
   → What to do / see / eat at the destination.
   → Activity recommendations (diving, hiking, snorkelling, etc.) — but NOT prices.

3. booking_agent
   → Searching or comparing flights, ferries, transport options.
   → Searching or comparing hotels, resorts, accommodation.
   → Live prices, fares, availability for any provider.
   → Building or refining the day-by-day itinerary.
   → Booking-related questions (schedules, costs, operators).

4. summarizer
   → ONLY when booking data is collected ('Yes') AND the user explicitly says
     they are done / satisfied / wants the final plan.
   → Do NOT route here early.

ROUTING EXAMPLES:
- "What's the weather like in May?" → local_guide
- "Best dive sites near Tioman?" → local_guide
- "Is it rainy season in June?" → local_guide
- "Find me a hotel" → booking_agent
- "What does the ferry cost?" → booking_agent
- "Singapore Airlines flight times?" → booking_agent
- "I'm done, please summarize" → summarizer (only if has_data=Yes)

Respond with ONLY a JSON object:
{{
  "next_agent": "concierge" | "local_guide" | "booking_agent" | "summarizer",
  "confirmed": true | false
}}

Set "confirmed" to true ONLY if the user is explicitly satisfied and wants the final itinerary printed.
"""

    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        response = llm.invoke([SystemMessage(content=system_prompt), *messages])

        raw = str(response.content).strip()
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        data       = json.loads(raw)
        next_agent = str(data.get("next_agent", "")).strip().lower()
        confirmed  = bool(data.get("confirmed", False))

        # Hard rule: never confirm without booking data
        if not has_data:
            confirmed = False
            if next_agent == "summarizer":
                next_agent = "booking_agent"

        valid = ["concierge", "booking_agent", "local_guide", "summarizer"]
        if next_agent not in valid:
            next_agent = next((a for a in valid if a in next_agent), "booking_agent")

        return {"next_agent": next_agent, "confirmed": confirmed}

    except Exception as e:
        print(f"Orchestrator error: {e}")
        return {"next_agent": "concierge", "confirmed": False}
