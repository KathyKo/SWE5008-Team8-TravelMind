from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
from llm_config import OPENAI_MODEL


def booking_agent(state, tools=None):
    """
    Booking Specialist: Dynamically decides whether to search for flights, 
    ferries, or private cars based on user preference.
    """
    messages = state.get("messages", [])
    origin = state.get("origin")
    dest = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget", "standard")

    if not dest:
        return {"messages": [{"role": "assistant", "content": "I need a destination to help with your bookings."}]}

    # Try to infer origin from the latest user message if it's still missing
    if not origin and messages:
        last_msg = messages[-1]
        if last_msg.get("role") == "user":
            possible_origin = last_msg.get("content", "").strip()
            if possible_origin:
                origin = possible_origin

    # If origin is still missing after inference, explicitly ask for it
    if not origin:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "To search for flights or transport, I need to know where you will be departing from. Which city or airport will you leave from?"
                }
            ]
        }

    # Enforce tool access control
    tools = tools or {}
    if "web_search" not in tools or "search_hotels" not in tools:
        raise RuntimeError(
            "Booking agent is missing required tool permissions "
            "(web_search, search_hotels)."
        )
    web_search = tools["web_search"]
    search_hotels = tools["search_hotels"]

    # 1. Use LLM to determine the best transport search query based on history
    system_prompt = """You are a Transport & Booking Expert. 
Analyze the conversation and determine the user's preferred mode of transport.
If they mentioned 'ferry', 'car', 'bus', or 'no flights', DO NOT search for flights.
Output a JSON with two keys:
- 'transport_query': A specific search string (e.g., 'private car from Singapore to Mersing and ferry to Tioman')
- 'search_type': 'flight' or 'other'
"""

    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        decision_res = llm.invoke([
            SystemMessage(content=system_prompt),
            *messages
        ])
        
        # Clean the response to get JSON
        res_text = str(decision_res.content)
        if "```json" in res_text:
            res_text = res_text.split("```json")[1].split("```")[0].strip()
        
        decision = json.loads(res_text)
        transport_query = decision.get("transport_query", f"flights to {dest} on {dates}")
        search_type = decision.get("search_type", "flight")

        # 2. Execute Transport Search
        print(f"\n[AGENT] Booking Specialist is searching for: {transport_query}...")
        transport_info = web_search(transport_query)
        
        # 3. Execute Hotel Search
        print(f"[AGENT] Booking Specialist is searching for hotels in {dest}...")
        hotel_info = search_hotels(dest, f"{budget} rated")

        # 4. Final Response Construction
        header = "FLIGHT OPTIONS" if search_type == "flight" else "TRANSPORT OPTIONS"
        response_text = f"I've found some travel options for your trip to {dest} on {dates}:\n\n"
        response_text += f" {header}:\n{transport_info}\n\n"
        response_text += f" HOTEL OPTIONS:\n{hotel_info}\n\n"
        response_text += "Does this transport and accommodation plan work for you?"

        return {
            "messages": [{"role": "assistant", "content": response_text}],
            "flight_options": [transport_info] if search_type == "flight" else [],
            "hotel_options": [hotel_info]
        }

    except Exception as e:
        print(f"Booking Agent Error: {e}")
        # Fallback to simple search if LLM fails
        return {"messages": [{"role": "assistant", "content": "I'm having trouble searching right now. Could you try again?"}]}
