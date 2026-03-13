from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from llm_config import OPENAI_MODEL

def travel_summarizer(state):
    """
    Master Planner: Compiles everything into a final travel itinerary.
    """
    origin = state.get("origin")
    dest = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget")
    flights = state.get("flight_options", [])
    hotels = state.get("hotel_options", [])
    
    context_data = f"""
    Origin: {origin}
    Destination: {dest}
    Dates: {dates}
    Budget: {budget}
    Flights: {flights}
    Hotels: {hotels}
    """

    system_prompt = """You are a Master Travel Planner. Your task is to compile a cohesive and final travel itinerary based on all the information gathered.

STRUCTURE:
1. Welcome & trip overview
2. Confirmed flights and timings (based on the options provided)
3. Recommended hotel choices with short rationale
4. Suggested daily plan / key activities
5. Practical tips (transport, weather, packing, misc.)

IMPORTANT:
- Assume the user has already confirmed they are happy with the options.
- Do NOT ask for any more confirmations or additional preferences.
- Do NOT mention that you "will" book or "will" generate anything later.
- Speak as if you are handing over a ready-to-use itinerary they can directly follow or book from.

Tone: Professional, concise, and organized.
"""

    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.7)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create a final itinerary with this data: {context_data}")
        ])
        
        summary = str(response.content)
        
        print("\n" + "="*50)
        print("          YOUR FINAL TRAVEL PLAN          ")
        print("="*50)
        print(summary)
        print("="*50 + "\n")

        return {
            "final_itinerary": summary,
            "is_complete": True
        }
    except Exception as e:
        print(f"Summarizer error: {e}")
        return {"is_complete": True}
