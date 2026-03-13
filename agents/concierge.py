from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
from llm_config import OPENAI_MODEL

def concierge(state):
    """
    Concierge agent: Welcomes users and extracts travel preferences.
    """
    messages = state.get("messages", [])
    
    # Extract current state for context
    origin = state.get("origin")
    dest = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget")

    system_prompt = f"""You are a Travel Concierge. Your goal is to help users plan their trip by gathering four essential pieces of information:
1. ORIGIN: Which city/airport they will depart from.
2. DESTINATION: Where they want to go.
3. TRAVEL DATES: When they plan to travel.
4. BUDGET: Their expected spending level.

Current Progress:
- Origin: {origin if origin else 'Not set'}
- Destination: {dest if dest else 'Not set'}
- Dates: {dates if dates else 'Not set'}
- Budget: {budget if budget else 'Not set'}

INSTRUCTIONS:
- Be polite and helpful.
- If information is missing, ask the user for it.
- Make sure ORIGIN is collected before handing off to booking or flight specialists.
- Once all four are gathered, let the user know that the specialists will now look into flights, hotels, and local tips.

DATA EXTRACTION (MANDATORY):
If the user provides any new details about origin, destination, dates, or budget, you MUST include a JSON block at the end of your response starting with 'DATA:'.
Example: DATA: {{"origin": "Singapore", "destination": "Tokyo", "dates": "next week", "budget": "moderate"}}
"""

    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.7)
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            *messages
        ])
        
        content = str(response.content)
        updates = {"messages": [{"role": "assistant", "content": content}]}
        
        if "DATA:" in content:
            try:
                data_part = content.split("DATA:")[1].strip()
                data_part = data_part.replace("```json", "").replace("```", "").strip()
                extracted_data = json.loads(data_part)
                
                for key in ["origin", "destination", "dates", "budget"]:
                    if key in extracted_data and extracted_data[key]:
                        updates[key] = extracted_data[key]
            except:
                pass
                
        return updates
    except Exception as e:
        print(f"Concierge error: {e}")
        return {"messages": [{"role": "assistant", "content": "Welcome! Where would you like to travel today?"}]}
