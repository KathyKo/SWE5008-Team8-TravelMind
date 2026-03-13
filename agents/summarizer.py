from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def travel_summarizer(state):
    """
    Master Planner: Compiles everything into a final travel itinerary.
    """
    dest = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget")
    flights = state.get("flight_options", [])
    hotels = state.get("hotel_options", [])
    
    context_data = f"""
    Destination: {dest}
    Dates: {dates}
    Budget: {budget}
    Flights: {flights}
    Hotels: {hotels}
    """

    system_prompt = """You are a Master Travel Planner. Your task is to compile a cohesive and final travel itinerary based on all the information gathered.

STRUCTURE:
1. Welcome & Summary
2. Flights & Hotels
3. Local Info & Attractions
4. Closing message

Tone: Professional and organized.
"""

    try:
        llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0.7)
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
