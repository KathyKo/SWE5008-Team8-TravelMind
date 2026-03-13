from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from tools import web_search, search_hotels
import json

def booking_agent(state):
    """
    Booking Specialist: Dynamically decides whether to search for flights, 
    ferries, or private cars based on user preference.
    """
    messages = state.get("messages", [])
    dest = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget", "standard")

    if not dest:
        return {"messages": [{"role": "assistant", "content": "I need a destination to help with your bookings."}]}

    # 1. Use LLM to determine the best transport search query based on history
    system_prompt = """You are a Transport & Booking Expert. 
Analyze the conversation and determine the user's preferred mode of transport.
If they mentioned 'ferry', 'car', 'bus', or 'no flights', DO NOT search for flights.
Output a JSON with two keys:
- 'transport_query': A specific search string (e.g., 'private car from Singapore to Mersing and ferry to Tioman')
- 'search_type': 'flight' or 'other'
"""

    try:
        llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0)
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
