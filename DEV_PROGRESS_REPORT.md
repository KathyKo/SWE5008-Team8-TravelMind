# TravelAgents — Development Progress Report
*Generated: 2026-03-27*

---

## (A) Agents Built — Status

| Agent | File | Status | Notes |
|---|---|---|---|
| **Orchestrator** | `agents/specialists/orchestrator.py` | Complete | Routes messages to concierge / local_guide / booking_agent / summarizer via LLM JSON decision + hard rules |
| **Concierge** | `agents/specialists/concierge.py` | Complete | Gathers origin, destination, dates, budget, preferences, duration. Embeds structured `DATA:` block in response for state extraction |
| **Local Guide** | `agents/specialists/local_guide.py` | Complete | Calls `search_weather` + `search_attractions` tools; answers local tips strictly from retrieved data |
| **Booking Agent** | `agents/specialists/booking_agent.py` | Complete | 4-stage FSM: `itinerary_draft → awaiting_selection → reviewing → confirmed`. Runs multi-query destination research, live price search, change detection, and final confirmation summary |
| **Summarizer** | `agents/specialists/summarizer.py` | Complete | Compiles final polished travel plan from all state fields (itinerary, selections, search_results, preferences) |
| **Planner (Agent3)** | `agents/specialists/planner_agent.py` | Complete | See Section B |
| **Explainability (Agent6)** | `agents/specialists/explainability_agent.py` | Complete | See Section B |

### Tools Status

| Tool | File | Description |
|---|---|---|
| `search_flights` | `tools/search_flights.py` | SerpAPI Google Flights primary; Tavily + Google web search fallback |
| `search_hotels` | `tools/search_hotels.py` | SerpAPI Google Hotels with property token caching |
| `search_attractions` | `tools/search_attractions.py` | Local attractions search |
| `search_weather` | `tools/search_weather.py` | Weather data retrieval |
| `web_search` | `tools/web_search.py` | Tavily web search |
| `google_search` | `tools/google_search.py` | Google web search |
| `serp_search` | `tools/serp_search.py` | SerpAPI unified wrapper (Flights, Hotels, Maps, TripAdvisor, Travel Explore) |

---

## (B) Integration Logic

### Agent3: Planner

**Input:** `destination, origin, dates, budget, preferences, duration, user_profile, outbound_time_pref, return_time_pref`

**Output:** `itineraries, option_meta, chain_of_thought, research, tool_log, flight_options_outbound, flight_options_return, hotel_options`

**Data Flow:**

```
Input State
  destination, origin, dates, budget, preferences, duration,
  user_profile, outbound_time_pref, return_time_pref
        │
        ▼
Phase 1 — Parallel Data Collection (ThreadPoolExecutor, 10 workers)
  ┌─────────────────────────────────────────────────────────────┐
  │  search_flights(origin→dest, outbound_date)                 │  → flight options outbound
  │  search_flights(dest→origin, return_date)                   │  → flight options return
  │  search_hotels(dest, budget, dates)                         │  → hotel list + tokens
  │  serp_local_structured(dest, "things to do attractions")    │  → Maps attractions (with GPS)
  │  serp_local_structured(dest, "restaurants")                 │  → Maps restaurants
  │  serp_tripadvisor_structured("things to do", dest)          │  → TripAdvisor attractions
  │  serp_tripadvisor_structured("restaurants", dest)           │  → TripAdvisor restaurants
  │  search_weather(dest, dates)                                │  → weather context
  │  web_search(dest + travel guide)                            │  → general travel context
  │  serp_local/tripadvisor per user preference keyword (x2)    │  → preference-specific places
  └─────────────────────────────────────────────────────────────┘
        │
        ▼
Phase 2 — LLM Scheduling (single LLM call)
  Clean inventory (real names, ratings, prices from APIs)
  LLM role: only schedule items into days — no data lookup
  Rules: no repeats, geographic grouping, preference matching
  Output: 3 itinerary variants (Options A / B / C)
        │
        ▼
Output State
  itineraries, option_meta, chain_of_thought,
  research, tool_log,
  flight_options_outbound, flight_options_return, hotel_options
```

**Key design decision:** Two-phase separation prevents hallucination — the LLM only schedules items that already exist in the structured inventory; it never invents hotel names, attractions, or prices.

---

### Agent6: Explainability

**Input:** `itineraries, option_meta, research, chain_of_thought, tool_log, user_profile, preferences, budget, destination` (from Agent3 output, cached by `plan_id` in backend)

**Output:** `explain_data, chain_of_thought, agent_steps`

**Data Flow:**

```
Input State (cached by plan_id in backend)
  itineraries, research, chain_of_thought, tool_log,
  user_profile, preferences, budget, destination
        │
        ▼
Step 1 — Deduplicate items across all 3 options
  Walk all itinerary days → collect unique items by name
  (same hotel/attraction in Options A, B, C counted once)
        │
        ▼
Step 2 — Single LLM call for unique items only
  System prompt: user prefs + budget + research snippets
  Input: unique_items JSON dict
  Output JSON:
    explain_data: { item_key: { matches[], rating, review_highlights[], chain_of_thought } }
    overall_chain_of_thought: theme summary across Options A/B/C
        │
        ▼
Step 3 — Map explanations back to ALL itinerary occurrences
  name_to_expl lookup → final_explain_data covers every key in every option
        │
        ▼
Step 4 — Build agent_steps trace for frontend Agent Activity panel
  [ Research Agent (N tool calls), Planner Agent, Explainability ]
        │
        ▼
Output State
  explain_data     → "Why?" popups in Plan + My Trip pages
  chain_of_thought → side panel in My Trip (Chain of Thought from Agent3 + Agent6)
  agent_steps      → agent activity trace shown in frontend
```

**Frontend rendering — lazy loading design:**

| Endpoint | Trigger | Agent called | Purpose |
|---|---|---|---|
| `POST /planner/generate` | User submits trip form | Agent3 only | Returns 3 itinerary options quickly |
| `POST /planner/explain` | My Trip page first load | Agent6 only | Loads explanations lazily using cached `plan_id` |

This decoupling keeps the Plan page fast while Agent6 runs asynchronously in the background.

---

## (C) Problems Encountered

### 1. API retrieves data but itinerary arrangement is wrong

SerpAPI returns correct flight and hotel data, but the LLM scheduling phase sometimes produces logistically incoherent day plans — e.g., scheduling a full-day trek on the same day as multiple dive sessions, or misaligning activity timing with geographic clusters. The structured inventory is sound but the LLM does not always respect activity-day limits or geographic proximity when composing the schedule.

### 2. Hallucination in responses

When SerpAPI returns sparse or partial results (e.g., small island destinations with few Google Maps listings), the LLM occasionally fills gaps with invented attraction names, hotel brands, or approximate prices not present in the retrieved data. The two-phase design reduces this significantly but does not eliminate it when the inventory is thin.

### 3. Explainability Agent slow execution

Agent6 processes all unique items in a single LLM call. When Agent3 generates itineraries with many unique items across 3 options (15–25 items), the single LLM call with a large JSON payload and research context can take 20–40 seconds. Deduplication mitigates this but does not fully solve the latency issue, particularly for popular multi-day destinations.

### 4. Frontend and backend not yet integrated

The Streamlit Plan page (`plan.py`) and My Trip page (`my_trip.py`) call `/planner/generate` and `/planner/explain`, and the backend router (`routers/planner.py`) plus `agent_client.py` are written. However, the main chat flow (concierge → orchestrator → booking agent) running via `frontend/app.py` and `backend/main.py` remains on a separate path. The two flows have not been merged into a single unified user journey — trip details collected in the chat (origin, destination, dates, budget) do not yet flow automatically into the Planner page.

---

## (D) Dev Plan — To Do

### High Priority

| Task | File(s) |
|---|---|
| **Fix itinerary scheduling logic** — add explicit constraints to Phase 2 LLM prompt: enforce activity-per-day limits, geographic zone grouping, dive/activity mutual exclusion | `agents/specialists/planner_agent.py` |
| **Reduce hallucination** — add a post-scheduling validation pass: compare every scheduled item name against the Phase 1 inventory list; flag or remove items not found in structured data | `agents/specialists/planner_agent.py` |
| **Integrate Plan/My Trip flow with main chat flow** — unify `frontend/app.py` (chat) and `frontend/pages/plan.py` / `my_trip.py` into a single session so concierge-collected data flows directly into the Planner page without re-entry | `frontend/`, `backend/main.py` |

### Medium Priority

| Task | File(s) |
|---|---|
| **Explainability latency** — batch items into chunks of 5–8 per LLM call, or move Agent6 to a background async task and push results to the frontend via polling or SSE | `agents/specialists/explainability_agent.py` |
| **Error recovery UX** — sparse-data failures currently return empty itinerary sections silently; surface a clear fallback message and suggest broadening search criteria | `frontend/pages/plan.py`, Agent3 |

### Low Priority

| Task | File(s) |
|---|---|
| **Chain-of-thought display** — CoT text is generated and stored in state but the My Trip side panel rendering is partially implemented; complete the frontend rendering of the step-by-step reasoning | `frontend/pages/my_trip.py` |
| **End-to-end integration test** — run a full session (concierge → orchestrator → planner → explainability → summarizer) with a real destination and validate output quality | QA / Integration |
