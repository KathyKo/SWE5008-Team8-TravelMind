def local_guide(state, tools=None):
    """
    Local Guide: Provides weather and attractions info.
    """
    dest = state.get("destination")
    if not dest:
        return {"messages": [{"role": "assistant", "content": "I need to know your destination before I can give you local tips!"}]}

    # Enforce tool access control
    tools = tools or {}
    if "search_weather" not in tools or "search_attractions" not in tools:
        raise RuntimeError(
            "Local guide is missing required tool permissions "
            "(search_weather, search_attractions)."
        )
    search_weather = tools["search_weather"]
    search_attractions = tools["search_attractions"]

    print(f"\n[AGENT] Local Guide is checking weather for {dest}...")
    weather = search_weather(dest)
    
    print(f"[AGENT] Local Guide is finding attractions in {dest}...")
    attractions = search_attractions(dest)

    response_text = f"Here is some local information for your trip to {dest}:\n\n"
    response_text += f"WEATHER:\n{weather}\n\n"
    response_text += f"ATTRACTIONS:\n{attractions}\n\n"
    response_text += "Does this help you plan your activities?"

    return {
        "messages": [{"role": "assistant", "content": response_text}]
    }
