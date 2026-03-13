from tools import search_flights, search_hotels

def booking_agent(state):
    """
    Booking Specialist: Finds premier flights and hotels, presenting them with professional flair.
    """
    dest = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget", "standard")

    if not dest or not dates:
        return {"messages": [{"role": "assistant", "content": "Our Logistics team requires a destination and travel dates to secure your bookings. Please let the Concierge know your plans!"}]}

    print(f"\n[LOGISTICS] Booking Specialist is curating flight options to {dest}...")
    flight_info = search_flights("Home", dest, dates)
    
    print(f"[LOGISTICS] Booking Specialist is securing premier hotel options in {dest}...")
    hotel_info = search_hotels(dest, f"{budget} rated")

    response_text = f"Greetings from 'Luxe Voyage' Logistics! I have curated the most suitable travel options for your journey to {dest} for {dates}:\n\n"
    response_text += f"✈️ **ELITE FLIGHT OPTIONS**:\n{flight_info}\n\n"
    response_text += f"🏨 **PREMIER ACCOMMODATIONS**:\n{hotel_info}\n\n"
    response_text += "These options have been selected to match your preferred travel style. Do any of these pique your interest, or shall I refine the search?"

    return {
        "messages": [{"role": "assistant", "content": response_text}],
        "flight_options": [flight_info],
        "hotel_options": [hotel_info]
    }
