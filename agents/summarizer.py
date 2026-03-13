from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def travel_summarizer(state):
    """
    Master Planner: Compiles everything into a beautiful, ready-to-use travel itinerary.
    """
    dest = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget")
    flights = state.get("flight_options", [])
    hotels = state.get("hotel_options", [])
    
    # Simple summary of gathered data for the prompt
    context_data = f"""
    Destination: {dest}
    Dates: {dates}
    Budget: {budget}
    Flights: {flights}
    Hotels: {hotels}
    """

    system_prompt = """You are the Lead Master Planner at travel agency. Your task is to craft a truly cohesive, and inspiring final travel itinerary based on the information gathered.

Your response MUST be elegant and well-structured, using professional formatting (Markdown).

STRUCTURE:
1.  **A Signature Welcome**: A warm, sophisticated opening.
2.  **Executive Summary**: Overview of the destination and travel dates.
3.  **The Logistic Pillar**: Clear, summarized flight and hotel recommendations.
4.  **Local Flavor & Experience**: A curated guide to attractions and the expected climate.
5.  **Closing Remarks**: A gracious sign-off.

Tone: Professional, elite, encouraging, and detailed.
"""

    try:
        llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0.7)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please finalize the itinerary with the following data: {context_data}")
        ])
        
        summary = str(response.content)
        
        # Display the final itinerary beautifully
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
        return {"final_itinerary": "Your plan is ready, but I encountered a slight printing error. Happy travels!", "is_complete": True}
