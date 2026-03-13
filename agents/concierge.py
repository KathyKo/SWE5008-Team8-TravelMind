from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json

def concierge(state):
    """
    Concierge agent: Welcomes users and extracts travel preferences.
    """
    messages = state.get("messages", [])
    
    # Extract current state for context
    dest = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget")

    system_prompt = f"""You are a Travel Concierge. Your goal is to help users plan their trip by gathering three essential pieces of information:
1. DESTINATION: Where they want to go.
2. TRAVEL DATES: When they plan to travel.
3. BUDGET: Their expected spending level.

Current Progress:
- Destination: {dest if dest else 'Not set'}
- Dates: {dates if dates else 'Not set'}
- Budget: {budget if budget else 'Not set'}

INSTRUCTIONS:
- Be polite and helpful.
- If information is missing, ask the user for it.
- Once all three are gathered, let the user know that the specialists will now look into flights, hotels, and local tips.

DATA EXTRACTION (MANDATORY):
If the user provides any new details about destination, dates, or budget, you MUST include a JSON block at the end of your response starting with 'DATA:'.
Example: DATA: {{"destination": "Tokyo", "dates": "next week", "budget": "moderate"}}
"""

    try:
        llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0.7)
        
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
                
                for key in ["destination", "dates", "budget"]:
                    if key in extracted_data and extracted_data[key]:
                        updates[key] = extracted_data[key]
            except:
                pass
                
        return updates
    except Exception as e:
        print(f"Concierge error: {e}")
        return {"messages": [{"role": "assistant", "content": "Welcome! Where would you like to travel today?"}]}
