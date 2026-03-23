from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from ..llm_config import OPENAI_MODEL


def local_guide(state, tools=None):
    """
    Local Guide: answers weather, attractions, activities, and local tips.
    Does NOT handle prices, bookings, or itinerary planning — those go to booking_agent.
    """
    dest   = state.get("destination")
    dates  = state.get("dates", "")
    messages = state.get("messages", [])

    if not dest:
        return {"messages": [{"role": "assistant", "content": "Please tell me your destination and I'll share local tips!"}]}

    tools = tools or {}
    if "search_weather" not in tools or "search_attractions" not in tools:
        raise RuntimeError("Local guide is missing required tools: search_weather, search_attractions.")

    search_weather     = tools["search_weather"]
    search_attractions = tools["search_attractions"]

    print(f"[AGENT] Local Guide — weather for {dest} ({dates})...")
    # Pass travel_dates so the weather tool can do seasonal web-search fallback
    try:
        weather = search_weather(dest, travel_dates=dates)
    except TypeError:
        # Fallback for tools that don't accept travel_dates kwarg
        weather = search_weather(dest)

    print(f"[AGENT] Local Guide — attractions for {dest}...")
    attractions = search_attractions(dest)

    system_prompt = f"""You are an expert Local Guide for {dest}.
Your job: answer the traveller's question about weather, attractions, activities, local tips, and what to do at the destination.

LIVE DATA RETRIEVED:
--- WEATHER ---
{weather}

--- ATTRACTIONS & ACTIVITIES ---
{attractions}

RULES:
1. Answer using ONLY facts present in the retrieved data above. Cite the source when reporting specific figures or rankings.
2. If the weather data covers a different date than the traveller's trip dates ({dates}), state that clearly and give seasonal context from the data.
3. If data for a specific attraction or activity is not in the results, say you don't have current info and suggest the official tourism site or a quick Google search.
4. Do NOT invent weather figures, attraction details, or activity prices not present in the data.
5. Do NOT handle flight/hotel prices or booking — those are handled by the booking specialist.
6. Do NOT end with offers of further help ("Would you like me to...", "If you want I can...").
7. Keep the response focused and practical.
"""

    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        response = llm.invoke([SystemMessage(content=system_prompt), *messages])
        return {"messages": [{"role": "assistant", "content": str(response.content)}]}
    except Exception as e:
        print(f"Local Guide error: {e}")
        return {"messages": [{"role": "assistant", "content": f"Weather: {weather}\n\nAttractions: {attractions}"}]}
