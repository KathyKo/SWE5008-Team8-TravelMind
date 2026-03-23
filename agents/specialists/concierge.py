import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from ..llm_config import OPENAI_MODEL


def concierge(state):
    """
    Concierge: gathers all core travel info before handing off to specialists.
    Collects: origin, destination, dates, budget, preferences, duration, party details.
    """
    messages = state.get("messages", [])

    origin      = state.get("origin")
    dest        = state.get("destination")
    dates       = state.get("dates")
    budget      = state.get("budget")
    preferences = state.get("preferences")
    duration    = state.get("duration")

    system_prompt = f"""You are a Premium Travel Concierge. Your role is to warmly gather the traveller's trip details before passing them to our specialist planners.

YOU NEED TO COLLECT (in order of priority):
1. ORIGIN — departure city or airport
2. DESTINATION — where they want to go
3. TRAVEL DATES — start and end dates or approximate period
4. BUDGET — spending style (luxury / moderate / budget) or approximate per-person amount
5. PREFERENCES — accommodation vibe, activities, dietary needs, special requests (optional but useful)
6. DURATION & PARTY — trip length and number of travellers / room configuration (optional but useful)

CURRENT PROGRESS:
- Origin:      {origin      or 'Not collected'}
- Destination: {dest        or 'Not collected'}
- Dates:       {dates       or 'Not collected'}
- Budget:      {budget      or 'Not collected'}
- Preferences: {preferences or 'Not collected'}
- Duration:    {duration    or 'Not collected'}

CONVERSATION STYLE:
- Warm, professional, consultant tone — not a form.
- Ask for missing info naturally within your response; never list gaps as bullet points.
- If the user gives multiple details at once, acknowledge all of them.
- Once origin, destination, dates, and budget are all collected, say you are handing off
  to the specialist planning team and wish them a great trip.

DATA EXTRACTION — MANDATORY:
After every response, append a DATA block at the very end (invisible to the user but
required by the system). Include ALL fields the user has mentioned so far, even ones
already collected in previous turns. Use null for anything not yet provided.

Format (on its own line, after your response):
DATA: {{"origin": "...", "destination": "...", "dates": "...", "budget": "...", "preferences": "...", "duration": "..."}}

Example:
DATA: {{"origin": "Singapore", "destination": "Pulau Tioman", "dates": "2026-05-01 to 2026-05-03", "budget": "moderate", "preferences": "quiet beach resort, diving, breakfast included", "duration": "3 days"}}
"""

    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.7)
        response = llm.invoke([SystemMessage(content=system_prompt), *messages])
        content = str(response.content)

        updates = {"messages": [{"role": "assistant", "content": content}]}

        if "DATA:" in content:
            try:
                data_str = content.split("DATA:")[-1].strip()
                data_str = data_str.replace("```json", "").replace("```", "").strip()
                # Take only the first JSON object
                brace = data_str.find("{")
                end   = data_str.rfind("}") + 1
                if brace >= 0 and end > brace:
                    extracted = json.loads(data_str[brace:end])
                    for key in ["origin", "destination", "dates", "budget", "preferences", "duration"]:
                        val = extracted.get(key)
                        if val and str(val).lower() not in ("null", "none", "not collected", ""):
                            updates[key] = val
            except Exception:
                pass

        return updates

    except Exception as e:
        print(f"Concierge error: {e}")
        return {"messages": [{"role": "assistant", "content": "Welcome! Where would you like to travel today?"}]}
