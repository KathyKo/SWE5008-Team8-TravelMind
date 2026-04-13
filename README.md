## Kathy's Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Browser                        │
└──────────────────────────┬──────────────────────────────┘
                           │ http://localhost:8501
┌──────────────────────────▼──────────────────────────────┐
│               Frontend  (Streamlit)                     │
│        app.py (chat) · plan.py · my_trip.py             │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP REST  :8000
┌──────────────────────────▼──────────────────────────────┐
│                Backend  (FastAPI)                       │
│   POST /planner/generate · POST /planner/explain        │
└──────────────────────────┬──────────────────────────────┘
                           │ Python imports
┌──────────────────────────▼──────────────────────────────┐
│              Agents                                     │
│  Agent3: Planner · Agent6: Explainability               │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   Tools                                 │
│  SerpAPI (Flights · Hotels · Maps · TripAdvisor)        │
│  Tavily · Google Custom Search · OpenWeatherMap         │
└─────────────────────────────────────────────────────────┘
```

---

## User Flows TO BE DISCUSS

| Flow | Entry Point | Agents Called | Purpose |
|---|---|---|---|
| **Plan page** | `POST /planner/generate` | Agent3 (Planner) | User submits a trip form; returns 3 itinerary options |
| **My Trip page** | `POST /planner/explain` | Agent6 (Explainability) | Loads "Why?" explanations lazily using a cached `plan_id` |

The Plan page returns quickly because only Agent3 runs synchronously. Agent6 runs on first load of the My Trip page, using the inventory already cached by Agent3 — no extra API calls.

---

## Repository Structure

```
travelagents/
│
├── frontend/                    # Streamlit UI
│   ├── app.py                   # Chat interface (legacy flow)
│   ├── utils.py                 # Shared frontend helpers
│   └── pages/
│       ├── plan.py              # Trip form → calls /planner/generate
│       └── my_trip.py           # Itinerary display + /planner/explain
│
├── backend/                     # FastAPI
│   ├── main.py
│   ├── agent_client.py          # Thin wrapper calling planner/explainability agents
│   └── routers/
│       └── planner.py           # /planner/generate + /planner/explain endpoints
│
├── agents/                      # Agent implementations
│   ├── __init__.py
│   ├── llm_config.py            # OpenAI model config
│   └── specialists/
│       ├── planner_agent.py     # Agent3 — 3-phase itinerary planner
│       └── explainability_agent.py  # Agent6 — "Why?" explanation generator
│
├── agent4_critic/               # Agent4 — Debate/critique loop (experimental)
│
├── tools/                       # External API integrations
│   ├── __init__.py
│   ├── serp_search.py           # SerpAPI unified wrapper (Flights/Hotels/Maps/TripAdvisor)
│   ├── search_flights.py        # SerpAPI → Tavily+Google fallback
│   ├── search_hotels.py         # SerpAPI → Tavily+Google fallback
│   ├── search_weather.py        # OpenWeather → web search fallback
│   ├── web_search.py            # Tavily API
│   └── google_search.py        # Google Custom Search API
│
├── .env.example                 # API key template
├── .gitignore
├── DEV_PROGRESS_REPORT.md       # Full development status + known issues
└── README.md                    # This file
```

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd travelagents

cp .env.example .env
# Edit .env and fill in all API keys (see Environment Variables below)
```

### 2. Run services

```bash
# Backend
cd backend && uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && streamlit run app.py
```

| Service | URL |
|---|---|
| Frontend | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in all values:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | Model name (default: `gpt-5.4-nano-2026-03-17`) |
| `SERPAPI_API_KEY` | SerpAPI key — Flights, Hotels, Maps, TripAdvisor |
| `TAVILY_API_KEY` | Tavily web search (primary web fallback) |
| `GOOGLE_API_KEY` | Google Custom Search API key |
| `GOOGLE_CSE_ID` | Google Custom Search Engine ID |
| `OPENWEATHER_API_KEY` | OpenWeatherMap API (5-day forecast) |

---

## Agents

### Agent3: Planner (`agents/specialists/planner_agent.py`)

The planner is the core of the system. It runs a **3-phase pipeline** to produce three differentiated itinerary options (A / B / C) without hallucinating place names or prices.

#### Input

```
destination, origin, dates, budget, preferences, duration,
user_profile, outbound_time_pref, return_time_pref
```

#### Phase 1 — Parallel Data Collection

All API calls run concurrently via `ThreadPoolExecutor` (up to 10 workers). Each worker runs independently so total wall time ≈ the slowest single call, not the sum.

| Task | API called | What it returns |
|---|---|---|
| `search_flights(origin→dest, outbound_date)` | SerpAPI Google Flights → Tavily+Google fallback | Outbound flight options with airline, flight number, times, price |
| `search_flights(dest→origin, return_date)` | SerpAPI Google Flights → Tavily+Google fallback | Return flight options |
| `search_hotels(dest, budget, dates)` | SerpAPI Google Hotels → Tavily+Google fallback | Hotels with star class, rating, price/night, `property_token` |
| `serp_local_structured(dest, "things to do attractions")` | SerpAPI Google Maps | Attractions with GPS coords, rating, price |
| `serp_local_structured(dest, "restaurants")` | SerpAPI Google Maps | Restaurants with address and price level |
| `serp_tripadvisor_structured("things to do attractions", dest)` | SerpAPI TripAdvisor | Attractions with ratings and descriptions |
| `serp_tripadvisor_structured("restaurants", dest)` | SerpAPI TripAdvisor | Restaurants with ratings and prices |
| `search_weather(dest, dates)` | OpenWeatherMap → Tavily+Google fallback | 5-day forecast or seasonal climate text |
| `web_search(dest + "travel guide")` | Tavily | General travel context |
| `serp_local_structured + serp_tripadvisor_structured` (×2) | SerpAPI Maps + TripAdvisor | Per-preference searches (e.g. "diving", "family friendly") |

After the parallel fetch, the agent runs **two sequential sub-steps**:

1. **Hotel reviews** — for the top 2 hotels (using their cached `property_token`), calls `serp_hotel_details()` + `serp_hotel_reviews()` to enrich the hotel description.
2. **Price enrichment** — attractions without a known price trigger a second parallel batch of web searches (Tavily + Google + SerpAPI organic) followed by a LLM call to extract prices from the search corpus. This runs in batches of 15 attractions at a time to stay within context limits.

The result is a **clean, validated inventory**: deduplicated attractions (TripAdvisor merged with Maps results, corporate names and tour-operator products filtered out), deduplicated restaurants (hotel names and non-dining experiences excluded), compact hotel list, and compact flight lists.

#### Phase 2 — LLM Scheduling

A single LLM call (gpt-5.4-nano-2026-03-17, temperature=0.5) receives the complete inventory as structured text and is instructed to schedule items into days. The LLM's **only job is scheduling** — it cannot invent any new names, prices, or flights. Hard rules encoded in the prompt:

- Day 1 (arrival): flight_outbound first → max 2 light activities → hotel check-in.
- Middle days: geographic clustering using GPS coordinates; at least 3 activities per day; 1 restaurant per day (geographically near the cluster); hotel overnight at end.
- Last day: calculate airport cutoff = return flight departure − 3 h; number of allowed morning activities depends on the cutoff time.
- Options A, B, C must use **different attraction subsets** (genuinely different themes/clusters).
- All costs converted to SGD in the output.

The LLM is called with `response_format={"type": "json_object"}` and retried up to 3 times if it returns an empty options dict.

#### Phase 3 — Post-Processing & Validation

`_post_process_options()` walks every item in every option and:

- **Flights**: validates flight number against the inventory list; drops any flight the LLM invented; converts cost to SGD.
- **Hotels**: always kept; normalises cost to SGD/night.
- **Attractions**: checks the item name against the inventory (fuzzy substring match); drops any name not found; strips pipe-appended metadata the LLM may copy from the inventory line.
- **Restaurants**: additionally validates that the name is actually from the restaurant inventory (not an attraction placed in a restaurant slot).
- **Deduplication**: within each option, each attraction/restaurant can appear at most once.
- **Chronological sort**: items within each day sorted by `time` field.
- **Sparse-day fill**: if any middle day ends up with fewer than 3 activity items after validation, unused attractions from the inventory are injected automatically.

#### Output

```
itineraries          — { "A": [days], "B": [days], "C": [days] }
option_meta          — { "A": { label, desc, budget, style, badge }, ... }
chain_of_thought     — LLM's geographic cluster reasoning + last-day cutoff logic
research             — raw API results (maps_attractions, ta_restaurants, weather, ...)
tool_log             — ordered list of tool calls made
flight_options_outbound / flight_options_return  — structured flight alternatives
hotel_options        — structured hotel alternatives (for "Change hotel" UI)
```

The inventory is also saved to `_inventory_cache` so that `revise_itinerary()` can re-use it without making any new API calls. This is used by the Agent4 debate loop: Agent4 critiques the itinerary and Agent3 revises it in a single additional LLM call.

---

### Agent6: Explainability (`agents/specialists/explainability_agent.py`)

Produces the "Why was this recommended?" popup text for every item across all three options.

**Flow:**
1. Deduplicate items across Options A, B, C — same hotel or attraction in multiple options is explained once.
2. Single LLM call with the unique items + user preferences + research snippets as context.
3. Map explanations back to every occurrence in all three options.
4. Build the `agent_steps` activity trace shown in the frontend Agent Activity panel.

**Triggered lazily** on first load of the My Trip page via `POST /planner/explain`, using the `plan_id` cached in the backend from the earlier `/planner/generate` call.

---

## Tools

### `serp_search.py` — SerpAPI Unified Wrapper

Central module for all SerpAPI calls. Maintains module-level caches that persist within a request:

| Cache | Key | Value | Used by |
|---|---|---|---|
| `_hotel_tokens` | lowercase hotel name | `property_token` | `serp_hotel_details`, `serp_hotel_reviews`, explainability |
| `_place_data_ids` | lowercase place name | `data_id` | `serp_maps_reviews` |
| `_flight_options_outbound/return` | — | list of structured flight dicts | "Change flight" UI |
| `_hotel_options` | — | list of structured hotel dicts | "Change hotel" UI |

**Functions:**

| Function | SerpAPI engine | When called | Returns |
|---|---|---|---|
| `serp_flights()` | `google_flights` | Phase 1, for both outbound and return | Text block of up to 5 flights; populates flight cache |
| `serp_flights_autocomplete()` | `google_flights_autocomplete` | City → IATA resolution (inside `_to_iata()`) | 3-letter IATA code |
| `serp_hotels()` | `google_hotels` | Phase 1 | Hotel list with prices; populates `_hotel_tokens` |
| `serp_hotel_details()` | `google_hotels` (property token) | After hotel search, top 2 hotels | Amenities, typical price range, nearby places |
| `serp_hotel_reviews()` | `google_hotels_reviews` | After hotel search, top 2 hotels | Top-rated guest review snippets |
| `serp_local()` | `google_maps` | (text output variant — used in older agents) | Formatted place list |
| `serp_local_structured()` | `google_maps` | Phase 1 (attractions + restaurants + per-pref) | `list[dict]` with GPS coordinates |
| `serp_maps_reviews()` | `google_maps_reviews` | Explainability agent | Review snippets for a specific place |
| `serp_tripadvisor()` | `tripadvisor` | (text output variant) | Formatted results |
| `serp_tripadvisor_structured()` | `tripadvisor` | Phase 1 (attractions + restaurants + per-pref) | `list[dict]` with category and ratings |
| `serp_travel_explore()` | `google_travel_explore` | Destination discovery feature (not in main flow) | Affordable destination list from an origin |

**API used:** SerpAPI (`SERPAPI_API_KEY`). All calls go through `_serpapi_request()` which handles the `GoogleSearch` client, key injection, and error logging.

---

### `search_flights.py` — Flight Search with Fallback

**When called:** Phase 1, twice — once for the outbound leg and once for the return leg.

**Logic:**
1. Call `serp_flights()` (SerpAPI Google Flights). Returns structured flight data with airline, flight number, departure/arrival times, and price.
2. If SerpAPI returns nothing (e.g., unusual route, small regional airport): fall back to three web queries combining Tavily (`web_search`) and Google Custom Search (`google_search`) — searching Skyscanner-style queries for schedule and price data.

**APIs used:** SerpAPI, Tavily (`TAVILY_API_KEY`), Google Custom Search (`GOOGLE_API_KEY` + `GOOGLE_CSE_ID`).

---

### `search_hotels.py` — Hotel Search with Fallback

**When called:** Phase 1.

**Logic:**
1. Call `serp_hotels()` (SerpAPI Google Hotels) with check-in/check-out dates parsed from the `dates` string and the user's budget preference passed as a search qualifier.
2. If SerpAPI returns nothing: fall back to three Agoda/Booking.com/Trip.com–targeted queries via Tavily + Google Custom Search.

**APIs used:** SerpAPI, Tavily, Google Custom Search.

---

### `search_weather.py` — Weather with Fallback

**When called:** Phase 1.

**Logic:**
1. Try `OpenWeatherMap Forecast API` (`/data/2.5/forecast`) — returns up to 5 days of 3-hour interval forecasts, sampled once per day. Works well for near-future dates.
2. If the API key is missing or the city is not found: fall back to Tavily + Google Custom Search using seasonal / climate queries. This handles far-future dates and remote destinations where live forecast data is unavailable.

**APIs used:** OpenWeatherMap (`OPENWEATHER_API_KEY`), Tavily, Google Custom Search.

---

### `web_search.py` — Tavily Web Search

**When called:**
- Phase 1 general travel context fetch.
- Price enrichment corpus fetch.
- As fallback in `search_flights`, `search_hotels`, `search_weather`.

**Logic:** Single POST to the Tavily `/search` endpoint (`search_depth=advanced`, `max_results=5`, `include_answer=True`). Returns a JSON list of `{title, snippet, source}` dicts. Tavily's `include_answer` flag allows it to return a direct synthesised answer for simple factual queries (e.g., admission prices).

**API used:** Tavily (`TAVILY_API_KEY`).

---

### `google_search.py` — Google Custom Search

**When called:** Same occasions as `web_search` — they are typically called in parallel for the same query to maximise recall.

**Logic:** Uses `langchain_google_community.GoogleSearchAPIWrapper` (wraps Google Custom Search JSON API), returning up to 5 results with title, link, and snippet.

**API used:** Google Custom Search (`GOOGLE_API_KEY` + `GOOGLE_CSE_ID`).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| LLM (Agent3 / Agent6) | OpenAI `gpt-5.4-nano-2026-03-17` (via LangChain `ChatOpenAI`) |
| LLM (Agent4) | DeepSeek `deepseek-chat` (OpenAI-compatible API) |
| Flight / Hotel / Maps Search | SerpAPI (Google Flights, Hotels, Maps, TripAdvisor) |
| General Web Search | Tavily + Google Custom Search |
| Weather | OpenWeatherMap API |
| Concurrency | Python `concurrent.futures.ThreadPoolExecutor` |

---

## Known Issues

See [DEV_PROGRESS_REPORT.md](DEV_PROGRESS_REPORT.md) for the full issue list. Summary:

1. **Itinerary scheduling** — LLM occasionally violates activity-per-day limits or geographic grouping despite explicit prompt rules.
2. **Hallucination on sparse destinations** — when SerpAPI returns few results for small islands or niche locations, the post-processor validation reduces but does not eliminate invented names.
3. **Explainability latency** — a single LLM call with 15–25 unique items can take 20–40 s. 

---

## Database Design (Agent3 / Agent4 / Agent6)

Currently the backend stores all state in a process-level Python dict (`_plan_cache`) which is lost on restart and capped at 50 entries. Below is the proposed relational schema to replace it with a persistent database (SQLite for development, PostgreSQL for production).

The schema is organised around a single **trip session** (`trips`) that everything else hangs off of via `plan_id`.

---

### Entity Relationship Overview

```
trips
 ├── itinerary_options (A / B / C)
 │    └── itinerary_days
 │         └── itinerary_items
 ├── inventory_flights   (outbound + return alternatives)
 ├── inventory_hotels    (hotel alternatives)
 ├── inventory_attractions
 ├── inventory_restaurants
 ├── tool_log
 ├── critique_issues     (Agent4 — one row per issue, per round)
 ├── debate_rounds       (Agent4 — one row per message exchanged)
 ├── debate_verdict      (Agent4 — final accept/reject decision)
 └── explain_items       (Agent6 — one row per unique item explained)
```

---

### Table Definitions

#### `trips` — one row per `/planner/generate` call

| Column | Type | Notes |
|---|---|---|
| `plan_id` | VARCHAR(8) PK | Short UUID generated by the backend |
| `origin` | TEXT | Departure city |
| `destination` | TEXT | |
| `dates` | TEXT | e.g. `"2026-05-10 to 2026-05-14"` |
| `duration` | TEXT | e.g. `"4"` (days) |
| `budget` | TEXT | e.g. `"moderate"` or `"SGD 3000"` |
| `preferences` | TEXT | Comma-separated tags from user |
| `outbound_time_pref` | TEXT | e.g. `"Morning (09:00–12:00)"` |
| `return_time_pref` | TEXT | |
| `user_profile` | JSON | Full user profile dict |
| `chain_of_thought` | TEXT | Agent3's cluster reasoning + last-day logic |
| `status` | TEXT | `generated` / `debated` / `explained` |
| `created_at` | TIMESTAMP | |

---

#### `itinerary_options` — Options A, B, C per trip

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `option_key` | CHAR(1) | `"A"`, `"B"`, or `"C"` |
| `label` | TEXT | e.g. `"Option A — Cultural Explorer"` |
| `desc` | TEXT | One-sentence description |
| `budget_estimate` | TEXT | e.g. `"approx SGD 2800"` |
| `style` | TEXT | e.g. `"relaxed"` |
| `badge` | TEXT | 2–3 word badge shown in UI |
| `is_revised` | BOOLEAN | `TRUE` after Agent4 debate revision |
| `revision_round` | INTEGER | Which debate round produced this version |

---

#### `itinerary_days` — days within each option

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `option_id` | INTEGER FK → itinerary_options |
| `day_index` | INTEGER | 0-based position |
| `day_title` | TEXT | e.g. `"Day 1 — Arrival · Marina Bay (2026-05-10)"` |

---

#### `itinerary_items` — individual schedule entries

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `day_id` | INTEGER FK → itinerary_days |
| `item_index` | INTEGER | Sort order within the day |
| `time` | VARCHAR(5) | `"HH:MM"` |
| `icon` | VARCHAR(4) | Emoji: `"✈️"`, `"🏛️"`, `"🍽️"`, `"🏨"` |
| `name` | TEXT | Exact place / flight / hotel name |
| `cost` | TEXT | e.g. `"SGD 39"`, `"Free"`, `"TBC"`, `"SGD 244/night"` |
| `item_key` | TEXT | e.g. `"flight_outbound"`, `"a_d2_0"`, `"a_d2_r0"` |

---

#### `inventory_flights` — raw flight alternatives from SerpAPI

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `direction` | TEXT | `"outbound"` or `"return"` |
| `airline` | TEXT | |
| `flight_number` | TEXT | May contain `/` for multi-leg (e.g. `"TR 890 / TR 2410"`) |
| `departure_airport` | TEXT | Full name from SerpAPI |
| `arrival_airport` | TEXT | |
| `departure_time` | VARCHAR(5) | `"HH:MM"` |
| `arrival_time` | VARCHAR(5) | |
| `duration_min` | INTEGER | Total journey minutes |
| `travel_class` | TEXT | e.g. `"Economy"` |
| `price_usd` | NUMERIC(10,2) | Raw USD price from SerpAPI |
| `display` | TEXT | Pre-formatted one-line display string |

---

#### `inventory_hotels` — hotel alternatives from SerpAPI

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `name` | TEXT | |
| `hotel_class` | TEXT | e.g. `"4-star hotel"` |
| `rating` | NUMERIC(3,1) | 0.0–5.0 |
| `reviews` | INTEGER | Review count |
| `price_per_night_usd` | NUMERIC(10,2) | Lowest nightly rate |
| `property_token` | TEXT | SerpAPI token — used for review lookups |
| `description` | TEXT | Truncated hotel description |
| `display` | TEXT | Pre-formatted one-line string |

---

#### `inventory_attractions` — merged Maps + TripAdvisor attraction list

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `name` | TEXT | |
| `type` | TEXT | e.g. `"Tourist attraction"` (from Maps) |
| `category` | TEXT | e.g. `"Attractions"` (from TripAdvisor) |
| `rating` | TEXT | |
| `reviews` | TEXT | Review count string |
| `price` | TEXT | Normalised to SGD or price level symbol |
| `address` | TEXT | |
| `lat` | NUMERIC(9,6) | GPS latitude (from Maps) |
| `lng` | NUMERIC(9,6) | GPS longitude (from Maps) |
| `data_id` | TEXT | Google Maps `data_id` for review lookups |
| `source` | TEXT | `"maps"` or `"tripadvisor"` |
| `matches_preference` | TEXT | Which user preference keyword matched |

---

#### `inventory_restaurants` — merged Maps + TripAdvisor restaurant list

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `name` | TEXT | |
| `category` | TEXT | |
| `rating` | TEXT | |
| `price` | TEXT | Price level (e.g. `"$$"`) or SGD amount |
| `address` | TEXT | |
| `lat` | NUMERIC(9,6) | |
| `lng` | NUMERIC(9,6) | |
| `source` | TEXT | `"maps"` or `"tripadvisor"` |

---

#### `tool_log` — ordered log of tool calls for Agent Activity panel

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `log_index` | INTEGER | Preserves insertion order |
| `entry` | TEXT | e.g. `"[search_flights(SIN→NRT)]"` |

---

#### `critique_issues` — Agent4 issues (one row per issue per debate round)

Agent4 runs 4 parallel critic nodes (bias, logistics, preference, diversity). Each produces `CritiqueReport` objects containing a list of `IssueItem`s.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `debate_round` | INTEGER | Round number (starts at 1) |
| `issue_id` | TEXT | Agent4's own UUID for the issue |
| `itinerary_id` | CHAR(1) | `"A"`, `"B"`, or `"C"` |
| `critic_name` | TEXT | `"bias_critic"` / `"logistics_critic"` / `"preference_critic"` / `"diversity_critic"` |
| `issue_type` | TEXT | `"bias_fairness"` / `"logistics"` / `"preference_mismatch"` / `"diversity_overlap"` / `"accessibility"` / `"budget_violation"` / `"time_unrealistic"` / `"other"` |
| `severity` | TEXT | `"critical"` / `"major"` / `"minor"` |
| `day` | INTEGER | Affected day number (nullable) |
| `slot_name` | TEXT | Affected item name (nullable) |
| `description` | TEXT | What is wrong |
| `suggestion` | TEXT | How to fix it |
| `overall_score` | NUMERIC(4,1) | Per-report score 0–10 from that critic |

---

#### `debate_rounds` — full message exchange between Agent4 and Agent3

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `round_number` | INTEGER | |
| `sender` | TEXT | `"agent4_critic"` or `"agent3_planner"` |
| `content` | TEXT | Feedback letter (Agent4) or revision acknowledgement (Agent3) |
| `debate_status` | TEXT | `"ongoing"` / `"accepted"` / `"max_rounds_reached"` |
| `sent_at` | TIMESTAMP | |

---

#### `debate_verdict` — final outcome of the Agent4 debate loop

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `accepted` | BOOLEAN | `TRUE` if Agent4 accepted Agent3's final revision |
| `final_round` | INTEGER | Which round ended the debate |
| `reason` | TEXT | Agent4's acceptance / rejection rationale |
| `remaining_issues` | JSON | List of unresolved `IssueItem` dicts |

---

#### `explain_items` — Agent6 explanations per unique inventory item

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `item_key` | TEXT | Normalised item name (used as lookup key) |
| `item_name` | TEXT | Display name |
| `matches_prefs` | JSON | List of user preference tags this item satisfies |
| `rating` | TEXT | Rating string from inventory |
| `review_highlights` | JSON | List of short review snippets |
| `chain_of_thought` | TEXT | Per-item "why" reasoning from Agent6 |

---

#### `explain_sessions` — Agent6 session-level output

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTO |
| `plan_id` | VARCHAR(8) FK → trips |
| `overall_chain_of_thought` | TEXT | Theme summary across Options A / B / C |
| `created_at` | TIMESTAMP | |

---

### Which Agent Writes / Reads Each Table

| Table | Written by | Read by |
|---|---|---|
| `trips` | Backend `/generate` | Backend `/explain`, `/revise`, `/debate` |
| `itinerary_options/days/items` | Agent3 (Planner) | Agent4 (critic input), Agent6 (explain input), Frontend |
| `inventory_*` | Agent3 Phase 1 | Agent3 Phase 2 prompt builder, Agent3 `revise_itinerary()`, Agent6 |
| `critique_issues` | Agent4 critic nodes | Agent4 aggregator, Frontend debate view |
| `debate_rounds` | Agent4 aggregator + Agent3 revise | Frontend debate history panel |
| `debate_verdict` | Agent4 debate manager | Backend `/debate` response, Frontend |
| `explain_items` | Agent6 | Frontend "Why?" popup |
| `explain_sessions` | Agent6 | Frontend My Trip chain-of-thought panel |
