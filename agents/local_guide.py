from tools import search_weather, search_attractions

def local_guide(state):
    """
    Local Guide: Provides elite insider tips, weather, and must-see landmarks.
    """
    dest = state.get("destination")
    if not dest:
        return {"messages": [{"role": "assistant", "content": "Our Local Insider team is ready to curate your experience, but first, we must confirm your destination!"}]}

    print(f"\n[INSIDER] Local Guide is assessing the climate for {dest}...")
    weather = search_weather(dest)
    
    print(f"[INSIDER] Local Guide is discovering the best of {dest} for you...")
    attractions = search_attractions(dest)

    response_text = f"Bonjour from 'Luxe Voyage' Insider Services! I have personally vetted the latest for your journey to {dest}:\n\n"
    response_text += f" **CLIMATE OUTLOOK**:\n{weather}\n\n"
    response_text += f" **CURATED ATTRACTIONS**:\n{attractions}\n\n"
    response_text += "These gems are chosen to give you a truly authentic and high-end experience of the city. Shall I delve deeper into any of these for you?"

    return {
        "messages": [{"role": "assistant", "content": response_text}]
    }
