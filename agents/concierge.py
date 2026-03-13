from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json

def concierge(state):
    """
    Concierge agent: Welcomes users and extracts travel preferences.
    """
    messages = state.get("messages", [])
    
    # Extract existing info to show current state
    dest = state.get("destination", "Not set")
    dates = state.get("dates", "Not set")
    budget = state.get("budget", "Not set")

    system_prompt = f"""You are a Travel Concierge. Your goal is to be helpful and gather information:
- Destination
- Travel Dates
- Budget

Current Information:
- Destination: {dest}
- Dates: {dates}
- Budget: {budget}

If information is missing, ask the user politely. If all info is present, let them know our specialists are looking into it.
Always respond in a professional and welcoming manner.

CRITICAL: At the end of your response, if you identified any NEW information from the user's last message, output it in JSON format starting with 'DATA:'
Example: DATA: {{"destination": "Tokyo", "dates": "Dec 20-25", "budget": "5000 USD"}}
"""

    try:
        llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0.7)
        user_input = messages[-1].get("content") if messages else "Hello"
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ])
        
        content = str(response.content)
        updates = {"messages": [{"role": "assistant", "content": content}]}
        
        # Simple extraction logic from 'DATA:' tag
        if "DATA:" in content:
            data_part = content.split("DATA:")[1].strip()
            try:
                extracted_data = json.loads(data_part)
                for key in ["destination", "dates", "budget"]:
                    if key in extracted_data:
                        updates[key] = extracted_data[key]
            except:
                pass
                
        return updates
    except Exception as e:
        print(f"Concierge error: {e}")
        return {"messages": [{"role": "assistant", "content": "Welcome! Where would you like to travel today?"}]}
