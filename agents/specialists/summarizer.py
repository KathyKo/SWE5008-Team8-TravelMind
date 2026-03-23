from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..llm_config import OPENAI_MODEL
import json


def travel_summarizer(state):
    """
    Summarizer: compiles the final confirmed travel plan from all state data.
    """
    messages       = state.get("messages", [])
    origin         = state.get("origin")
    dest           = state.get("destination")
    dates          = state.get("dates")
    budget         = state.get("budget")
    preferences    = state.get("preferences", "")
    duration       = state.get("duration", "")
    selections     = state.get("selections", {})
    search_results = state.get("search_results", {})
    itinerary      = state.get("itinerary", "")

    # Build confirmed pricing block from booking agent's search results
    price_lines = []
    for key, val in search_results.items():
        provider = selections.get(key, key)
        data     = val.get("data", "") if isinstance(val, dict) else str(val)
        price_lines.append(f"- {key.capitalize()} ({provider}): {data}")

    # Full conversation for context
    conversation_text = "\n".join(
        f"{'Traveler' if m.get('role') == 'user' else 'Agent'}: {m.get('content', '')}"
        for m in messages
    )

    context_data = f"""
=== CONFIRMED TRIP DETAILS ===
Origin:      {origin}
Destination: {dest}
Dates:       {dates}
Budget:      {budget}
Preferences: {preferences}
Duration:    {duration}

=== CONFIRMED SELECTIONS & VERIFIED PRICING ===
{chr(10).join(price_lines) if price_lines else 'See conversation history.'}

=== DRAFTED ITINERARY (from booking specialist) ===
{itinerary if itinerary else 'See conversation history.'}

=== FULL CONVERSATION HISTORY ===
{conversation_text}
"""

    system_prompt = """You are a Senior Travel Product Manager compiling the final confirmed travel plan.

RULES:
1. PRIORITISE confirmed selections and verified pricing from the state data above.
   Use the conversation history to fill any gaps — do not invent details.
2. NO META-TALK: Do not say "I am an AI", "This is a proposal", or "subject to availability".
3. NATURAL LANGUAGE: No placeholders like "[Price]", "TBC", or "HH:MM".
   Use natural descriptions ("around noon", "approximately SGD 220").
4. CURRENCY: Use the destination's local currency. Do not show conversions unless requested.
5. NO INVENTIONS: If a detail is unknown, omit it gracefully — do not guess.
6. HARD STOP — NO OFFERS: Do not end with any sentence offering further help.
   ("If you'd like...", "Would you like me to...", "Let me know if...", etc.)
   The plan ends at the last practical tip. Full stop.

STRUCTURE (in this order):
1. Trip Overview — one short paragraph: who, where, when, vibe.
2. Getting There — confirmed transport with timings and operator names.
3. Accommodation — confirmed hotel, room type, board basis.
4. Day-by-Day — flowing narrative (not a checklist). Honour the exact dive/activity plan.
5. Confirmed Pricing — one line per item with source citation.
6. Essential Tips — 4–6 highly relevant practical tips only.

Tone: premium, polished, definitive.
"""

    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.5)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Compile the final travel plan:\n{context_data}")
        ])

        summary = str(response.content)

        print("\n" + "=" * 50)
        print("          YOUR FINAL TRAVEL PLAN          ")
        print("=" * 50)
        print(summary)
        print("=" * 50 + "\n")

        return {"final_itinerary": summary, "is_complete": True}

    except Exception as e:
        print(f"Summarizer error: {e}")
        return {"is_complete": True}
