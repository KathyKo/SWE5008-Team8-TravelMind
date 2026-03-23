from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
import json
from ..llm_config import OPENAI_MODEL

# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------
STAGE_ITINERARY = "itinerary_draft"    # Step 1: research + draft itinerary + option lists
STAGE_SELECTING = "awaiting_selection" # Step 2: user picks, agent fetches live prices
STAGE_REVIEWING = "reviewing"          # Step 3: user reviews; can change one item or confirm
STAGE_CONFIRMED = "confirmed"          # Done


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


def _json_invoke(llm, system_prompt: str, messages: list) -> dict:
    """Call the LLM expecting a JSON object. Returns {} on parse failure."""
    res = llm.invoke(
        [SystemMessage(content=system_prompt), *messages],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(res.content)
    except (json.JSONDecodeError, AttributeError):
        return {}


def _text_invoke(llm, system_prompt: str, messages: list) -> str:
    """Call the LLM expecting plain text."""
    res = llm.invoke([SystemMessage(content=system_prompt), *messages])
    return str(res.content)


# ---------------------------------------------------------------------------
# Step 1a — destination research (4 targeted searches)
# ---------------------------------------------------------------------------

def _research_destination(dest: str, origin: str, dates: str, web_search, google_search) -> dict:
    """
    Run 4 targeted searches to ground the itinerary in verified facts.
    Returns raw search snippets keyed by topic.
    """
    queries = {
        "attractions":   f"{dest} top attractions things to do",
        "logistics":     f"{origin} to {dest} transport options {dates}",
        "accommodation": f"{dest} beach resort hotel accommodation options",
        "season":        f"{dest} weather travel conditions {dates}",
        "advisory":      f"{dest} travel advisory entry requirements",
    }
    results = {}
    for topic, q in queries.items():
        print(f"[AGENT] Researching {topic}: {q}")
        tavily = web_search(q)    if web_search    else ""
        google = google_search(q) if google_search else ""
        results[topic] = f"{tavily} | {google}"
    return results


# ---------------------------------------------------------------------------
# Step 1b — draft day-by-day itinerary + numbered option lists (no prices)
# ---------------------------------------------------------------------------

def _plan_itinerary(
    llm, messages: list,
    dest: str, origin: str, dates: str, duration: str,
    preferences: str, research: dict,
) -> str:
    prompt = f"""You are a professional travel planner creating a grounded trip proposal.

Trip details:
- Route: {origin} → {dest}
- Dates: {dates}
- Duration: {duration}
- Traveller preference: {preferences}

Verified destination data (use ONLY what appears below — do not add facts not present here):
- Attractions & things to do: {research["attractions"]}
- Getting there & local transport: {research["logistics"]}
- Accommodation options found: {research["accommodation"]}
- Weather & conditions: {research["season"]}
- Entry requirements & advisories: {research["advisory"]}

Write your response in exactly this structure:

---
### Itinerary
A day-by-day schedule using specific activity names from the research data.
Include rough timing (morning / afternoon / evening). Keep it concise and inspiring.

### Transport Options
2–3 concrete ways to travel from {origin} to {dest} found in the research data.
Format each as a numbered line:
  1. [Provider / mode] — [brief description]
No prices — these will be fetched live once you select one.

### Accommodation Options
2–3 choices that match the traveller's preference ({preferences}) found in the research data.
Format each as a numbered line:
  1. [Hotel / area / type] — [brief description]
No prices.
---

Close with this exact sentence:
"Please reply with your preferred transport and accommodation (by number or by name), or describe your own choice — I'll fetch live prices for you."

Hard rules:
- Only mention attractions, providers, or hotels that appear in the research data above.
- If the research data is sparse, list only what is confirmed; do not pad with invented options.
- Never invent prices, schedules, or provider names.
- Do not mention prices anywhere in this response.
- Each day must be logistically coherent: if the user has dive sessions on a day, do NOT also schedule strenuous activities (e.g., full-day treks) on the same day.
- If the user specified a dive plan (e.g., 1 night dive Day 1, 3 dives Day 2), honour that exactly — do not rearrange dives or add activities that conflict.
"""
    return _text_invoke(llm, prompt, messages)


# ---------------------------------------------------------------------------
# Step 2 — live price search for one specific provider
# ---------------------------------------------------------------------------

def _clean_provider(provider: str) -> str:
    """Take the first named option when user writes 'A / B' or 'A or B'."""
    for sep in (" / ", " or ", " | ", "/"):
        if sep in provider:
            provider = provider.split(sep)[0]
    return provider.strip()


def _search_provider_price(
    provider: str, category: str,
    dest: str, dates: str,
    web_search, google_search,
) -> dict:
    """
    Search live pricing for one named provider.
    Runs three queries: OTA price, broad info, direct booking.
    Returns {"found": bool, "data": str}.
    """
    name = _clean_provider(provider)

    if category == "accommodation":
        q_price = f"{name} {dest} room rate agoda booking.com site:agoda.com OR site:booking.com OR site:trip.com"
        q_info  = f"{name} {dest} hotel resort price"
    elif category == "custom_service":
        q_price = f"{name} {dest} price package rate"
        q_info  = f"{name} {dest} dive shop tour operator"
    else:  # transport
        q_price = f"{name} {dest} price fare {dates}"
        q_info  = f"{name} {dest} transport booking"

    q_direct = f"{name} {dest} {dates}"

    print(f"[AGENT] Searching ({category}): {q_price}")
    print(f"[AGENT] Searching ({category}): {q_info}")

    tavily_price  = web_search(q_price)    if web_search    else ""
    google_price  = google_search(q_price) if google_search else ""
    tavily_info   = web_search(q_info)     if web_search    else ""
    google_direct = google_search(q_direct) if google_search else ""

    combined = " | ".join(
        s for s in [tavily_price, google_price, tavily_info, google_direct] if s and s.strip()
    )
    return {
        "found": bool(combined.strip()),
        "data":  combined,
    }


def _build_price_reply(
    llm, messages: list, selections: dict, search_results: dict
) -> str:
    """
    Present live search results as a concise price summary.
    No estimation — if data is missing, say so explicitly.
    """
    results_block = "\n".join(
        f"- {key.capitalize()} ({selections.get(key, '')}): {val['data']}"
        for key, val in search_results.items()
    )
    prompt = f"""You are a travel booking assistant presenting live search results to the client.

User's selections:
{json.dumps(selections, indent=2)}

Live search data retrieved:
{results_block}

Write a concise price summary — one section per selected item.

Strict rules:
1. Report ONLY figures and facts that explicitly appear in the search data above.
   Do NOT estimate, interpolate, or invent any number or availability status.
2. Cite the source for every price or schedule you mention
   (e.g., "per [website]" or "per [operator name]").
3. Distinguish two "no price" cases:
   a. Search data mentions the provider (website, listing, contact) but contains no price →
      Write: "Found [name] at [source] but no prices are listed online.
      Contact them directly for a quote: [website/contact if available]."
   b. Search data returned nothing at all →
      Write: "I was unable to find any information for [name]. Please provide
      the details directly (e.g., a booking link or quoted price), or choose another option."
   Never substitute a placeholder figure or an estimated range in either case.
4. Do NOT repeat the full itinerary — only cover the selected items.
5. End with: "Does this look right, or would you like to change anything?"
"""
    return _text_invoke(llm, prompt, messages)


# ---------------------------------------------------------------------------
# Step 3 — detect change request or final confirmation
# ---------------------------------------------------------------------------

def _live_search(query: str, web_search, google_search) -> str:
    """Run a single ad-hoc query through both tools. Returns combined raw snippets."""
    print(f"[AGENT] Live search: {query}")
    t = web_search(query)    if web_search    else ""
    g = google_search(query) if google_search else ""
    return " | ".join(s for s in [t, g] if s and s.strip())


def _answer_with_live_data(llm, messages: list, query: str, live_data: str) -> str:
    prompt = f"""The user asked a question. Answer it directly using ONLY the live search data below.

Search query used: {query}
Live search data: {live_data}

Rules:
1. Lead with the answer — do not open with "I found..." or "Based on the data...".
2. Report only facts and figures present in the search data. Cite source site names inline (e.g., "per Skyscanner").
3. If the data contains no relevant answer, say in one sentence which official site to check. Nothing more.
4. Never invent schedules, prices, or availability.
5. Keep the answer concise and direct.
6. Do NOT end with any offer of further help ("If you want I can...", "Would you like me to...", etc.).
"""
    return _text_invoke(llm, prompt, messages)


def _detect_change(llm, messages: list, current_selections: dict) -> dict:
    """
    Read the latest user message.
    Returns {"changed_key": str|None, "new_value": str|None}.
    """
    prompt = f"""The user is reviewing their booking selections:
{json.dumps(current_selections, indent=2)}

Read the latest user message and decide:
- Are they requesting a change to transport or accommodation?
- Or are they satisfied and confirming the plan?

Return ONLY a JSON object:
{{
  "changed_key": "transport" | "accommodation" | "custom_service" | null,
  "new_value":   "the new provider or choice they named, or null"
}}

Return null for changed_key if the user is happy and not requesting any change.
"""
    return _json_invoke(llm, prompt, messages)


def _build_confirmation_summary(llm, messages: list, state: dict) -> str:
    """Final summary shown once when the user confirms."""
    search_results = state.get("search_results", {})
    selections     = state.get("selections", {})

    price_lines = [
        f"- {key.capitalize()} ({selections.get(key, key)}): {val.get('data', 'N/A')}"
        for key, val in search_results.items()
    ]

    prompt = f"""The user has confirmed their trip. Write a clean, final booking summary.

Confirmed itinerary:
{state.get("itinerary", "")}

Confirmed selections and verified pricing:
{chr(10).join(price_lines)}

Format:
- One-sentence intro confirming the booking.
- The day-by-day itinerary (copy from above, unchanged).
- A pricing section listing confirmed transport and accommodation with source citations.

Keep it concise and professional. End with: "Have a wonderful trip!"
"""
    return _text_invoke(llm, prompt, messages)


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

def booking_agent(state: dict, tools: dict = None) -> dict:
    """
    Stage-aware booking agent.

    Flow:
      STAGE_ITINERARY  → research destination, draft day-by-day itinerary,
                         present numbered transport + accommodation options (no prices yet)
      STAGE_SELECTING  → parse user's choice (from list OR custom), fetch live prices,
                         present price card with source citations; say "not found" if missing
      STAGE_REVIEWING  → detect change request or confirmation;
                         if change → re-search that item only (itinerary NOT repeated);
                         if confirmed → show final summary once
      STAGE_CONFIRMED  → trip locked, nothing more to do
    """
    messages    = state.get("messages", [])
    stage       = state.get("stage") or STAGE_ITINERARY   # None → treat as fresh start
    origin      = state.get("origin", "Singapore")
    dest        = state.get("destination")
    dates       = state.get("dates", "")
    duration    = state.get("duration", "3 days")
    # fall back to budget field if preferences not explicitly set
    preferences = state.get("preferences") or state.get("budget", "moderate")

    if not dest:
        return {"messages": [{"role": "assistant", "content": "I need a destination to help plan your trip."}]}

    tools         = tools or {}
    web_search    = tools.get("web_search")
    google_search = tools.get("google_search")
    llm           = _llm()

    # -----------------------------------------------------------------------
    # PRE-STAGE: if the user is asking a question that needs live data,
    # search immediately and return — the booking stage is preserved
    # -----------------------------------------------------------------------
    pre_intent_prompt = """Read ONLY the latest user message.
Classify it as ONE of:
- "question": user is requesting information (e.g. flight times, prices, schedules, availability, comparisons)
- "action": user is planning, selecting, confirming, or changing something in the booking flow

Return ONLY a JSON object:
{
  "intent":       "question" | "action",
  "search_query": "a concise search query to answer the question, or null if intent is action"
}
"""
    pre_intent = _json_invoke(llm, pre_intent_prompt, messages)
    if pre_intent.get("intent") == "question":
        query = pre_intent.get("search_query") or messages[-1].get("content", "") if messages else ""
        if query:
            live_data = _live_search(query, web_search, google_search)
            reply     = _answer_with_live_data(llm, messages, query, live_data)
            return {
                "messages": [{"role": "assistant", "content": reply}],
                "stage":    stage,   # preserve — booking flow continues next turn
            }

    # -----------------------------------------------------------------------
    # STAGE 1 — Research destination, draft itinerary + numbered option lists
    # -----------------------------------------------------------------------
    if stage == STAGE_ITINERARY:
        try:
            research       = _research_destination(dest, origin, dates, web_search, google_search)
            itinerary_text = _plan_itinerary(
                llm, messages, dest, origin, dates, duration, preferences, research
            )
            return {
                "messages":      [{"role": "assistant", "content": itinerary_text}],
                "stage":         STAGE_SELECTING,
                "itinerary":     itinerary_text,
                "research":      research,
                # Non-None so orchestrator knows booking is in progress
                "flight_options": [],
                "hotel_options":  [],
            }
        except Exception as e:
            print(f"[Stage 1 error] {e}")
            return {"messages": [{"role": "assistant", "content": "Something went wrong while researching your destination. Please try again."}]}

    # -----------------------------------------------------------------------
    # STAGE 2 — Parse user's selection, fetch live prices, show price card
    #           OR answer an ad-hoc question and stay in this stage
    # -----------------------------------------------------------------------
    if stage == STAGE_SELECTING:
        try:
            intent_prompt = """Read the user's latest message.
Decide: are they making specific transport/accommodation/service selections, or asking a question?

Return ONLY a JSON object:
{
  "intent":       "selection" | "question",
  "search_query": "a concise web search query to answer their question, or null if intent is selection"
}
"""
            intent_data   = _json_invoke(llm, intent_prompt, messages)
            intent        = intent_data.get("intent", "selection")
            search_query  = intent_data.get("search_query")

            # ── User is asking a question — search live and answer, stay in stage ──
            if intent == "question" and search_query:
                live_data = _live_search(search_query, web_search, google_search)
                reply     = _answer_with_live_data(llm, messages, search_query, live_data)
                return {
                    "messages": [{"role": "assistant", "content": reply}],
                    "stage":    STAGE_SELECTING,   # stay here until they select
                }

            # ── User is making selections — extract, price-search, advance stage ──
            selection_prompt = """Read the user's latest message and extract their selections.

Return ONLY a JSON object:
{
  "transport":      "the exact transport provider or mode the user chose, or null if not mentioned",
  "accommodation":  "the exact hotel or accommodation the user chose, or null if not mentioned",
  "custom_service": "any dive operator, tour company, activity provider, or other specific service the user mentioned, or null"
}

Preserve the user's exact words — do not paraphrase or normalise provider names.
If the user provided their own choice (not from the numbered list), capture it as-is.
"""
            selections     = _json_invoke(llm, selection_prompt, messages)
            search_results = {}

            for category in ("transport", "accommodation", "custom_service"):
                provider = selections.get(category)
                if provider:
                    search_results[category] = _search_provider_price(
                        provider, category, dest, dates, web_search, google_search
                    )

            reply = _build_price_reply(llm, messages, selections, search_results)

            return {
                "messages":       [{"role": "assistant", "content": reply}],
                "stage":          STAGE_REVIEWING,
                "selections":     selections,
                "search_results": search_results,
                "flight_options": [selections.get("transport")],
                "hotel_options":  [selections.get("accommodation")],
            }
        except Exception as e:
            print(f"[Stage 2 error] {e}")
            return {"messages": [{"role": "assistant", "content": "Something went wrong during the price search. Please try again."}]}

    # -----------------------------------------------------------------------
    # STAGE 3 — User reviewing: detect change or final confirmation
    # -----------------------------------------------------------------------
    if stage == STAGE_REVIEWING:
        try:
            current_selections   = state.get("selections", {})
            current_search       = state.get("search_results", {})
            change               = _detect_change(llm, messages, current_selections)
            changed_key          = change.get("changed_key")
            new_value            = change.get("new_value")

            # User is satisfied — show final confirmation summary (itinerary shown once here)
            if not changed_key:
                summary = _build_confirmation_summary(llm, messages, state)
                return {
                    "messages":  [{"role": "assistant", "content": summary}],
                    "stage":     STAGE_CONFIRMED,
                    "confirmed": True,
                }

            # User changed one item — re-search that item only, preserve everything else
            new_result             = _search_provider_price(
                new_value, changed_key, dest, dates, web_search, google_search
            )
            updated_selections     = {**current_selections, changed_key: new_value}
            updated_search_results = {**current_search,     changed_key: new_result}

            # Price reply only — itinerary is NOT repeated
            reply = _build_price_reply(llm, messages, updated_selections, updated_search_results)

            return {
                "messages":       [{"role": "assistant", "content": reply}],
                "stage":          STAGE_REVIEWING,
                "selections":     updated_selections,
                "search_results": updated_search_results,
                "flight_options": [updated_selections.get("transport")],
                "hotel_options":  [updated_selections.get("accommodation")],
            }
        except Exception as e:
            print(f"[Stage 3 error] {e}")
            return {"messages": [{"role": "assistant", "content": "Something went wrong while updating your selection. Please try again."}]}

    # -----------------------------------------------------------------------
    # STAGE_CONFIRMED — trip locked; answer follow-up questions with live search
    # -----------------------------------------------------------------------
    if stage == STAGE_CONFIRMED:
        # Extract a search query from the user's question
        query_prompt = """The user's trip is confirmed. They have a follow-up question.
Extract a concise web search query to answer it.
Return ONLY a JSON object: {"query": "the search query, or null if no search is needed"}
"""
        query_data   = _json_invoke(llm, query_prompt, messages)
        search_query = query_data.get("query")

        live_data = ""
        if search_query:
            live_data = _live_search(search_query, web_search, google_search)

        search_results = state.get("search_results", {})
        selections     = state.get("selections", {})
        price_lines    = "\n".join(
            f"- {k.capitalize()} ({selections.get(k, k)}): {v.get('data', 'N/A')}"
            for k, v in search_results.items()
        )
        followup_prompt = f"""The user's trip is confirmed. Answer their follow-up question.

Confirmed trip data (already searched):
{price_lines}

Live search data for their question:
{live_data if live_data else "(no additional search performed)"}

Rules:
- Prioritise the live search data for new questions (e.g., flight schedules, timings).
- Cite source URLs or site names for every figure or schedule you report.
- If neither dataset contains the answer, say clearly you couldn't find it and name the official site to check.
- Do not invent schedules, prices, or availability.
- Keep the answer concise.
"""
        return {
            "messages": [{"role": "assistant", "content": _text_invoke(llm, followup_prompt, messages)}]
        }

    # Unknown stage fallback
    return {"messages": [{"role": "assistant", "content": "I'm not sure where we are in the planning. What would you like to do?"}]}
