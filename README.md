# TravelMind 🗺️

An agentic AI multi-agent travel planning system.
NUS-ISS Graduate Certificate in Architecting AI Systems — SWE5008, Team 8.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Browser                        │
└──────────────────────────┬──────────────────────────────┘
                           │ http://localhost:8501
┌──────────────────────────▼──────────────────────────────┐
│               Frontend  (Streamlit)                     │
│   Plan · My Trip · Re-plan · Security Demo              │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP REST  :8000
┌──────────────────────────▼──────────────────────────────┐
│                Backend  (FastAPI)                       │
│         POST /travel/plan · POST /travel/summarize      │
└──────────────────────────┬──────────────────────────────┘
                           │ Python imports
┌──────────────────────────▼──────────────────────────────┐
│              Agents  (LangGraph)                        │
│  Orchestrator · Concierge · Booking · Local Guide       │
│  Summarizer                                             │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   Tools                                 │
│  SerpAPI (Flights · Hotels · Maps)                      │
│  Tavily · Google Search · OpenWeather                   │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
SWE5008-TEAM8-TRAVELMIND/
│
├── frontend/                    # Streamlit UI
│   ├── app.py
│   ├── pages/
│   │   ├── plan.py
│   │   ├── my_trip.py
│   │   ├── replan.py
│   │   └── security.py
│   ├── data/
│   │   └── store.py
│   ├── .streamlit/
│   │   └── config.toml
│   ├── Dockerfile
│   ├── docker-compose.yml       # Standalone frontend only
│   ├── requirements.txt
│   └── README.md
│
├── backend/                     # FastAPI
│   ├── main.py
│   ├── routers/
│   │   └── travel.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── agents/                      # LangGraph agent logic
│   ├── README.md
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent_tools.py           # Per-agent tool access control
│   ├── graph.py                 # build_travel_graph()
│   ├── state.py                 # TypedDict State schema
│   ├── nodes.py                 # Node functions
│   ├── llm_config.py            # Model config
│   ├── visualize.py             # Graph diagram generation
│   └── specialists/             # Specialist agent implementations
│       ├── orchestrator.py
│       ├── concierge.py
│       ├── booking_agent.py
│       ├── local_guide.py
│       └── summarizer.py
│
├── tools/                       # External API integrations
│   ├── __init__.py
│   ├── serp_search.py           # SerpAPI helpers (flights / hotels / maps)
│   ├── web_search.py            # Tavily
│   ├── google_search.py         # Google Custom Search
│   ├── search_flights.py        # SerpAPI → Tavily+Google fallback
│   ├── search_hotels.py         # SerpAPI → Tavily+Google fallback
│   ├── search_weather.py        # OpenWeather → web search fallback
│   └── search_attractions.py   # SerpAPI Maps → Tavily+Google fallback
│
├── docker-compose.yml           # Full stack — all services
├── .env.example                 # API key template
├── .gitignore
└── README.md                    # This file
```

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd SWE5008-TEAM8-TRAVELMIND

# Set up environment variables
cp .env.example .env
# Edit .env and fill in all API keys
```

### 2. Run with Docker Compose (recommended)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

### 3. Run services individually

```bash
# Frontend only
cd frontend && streamlit run app.py

# Backend only
cd backend && uvicorn main:app --reload --port 8000
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in all values:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | Model name (default: `gpt-4o-mini`) |
| `TAVILY_API_KEY` | Tavily web search API key |
| `GOOGLE_API_KEY` | Google Custom Search API key |
| `GOOGLE_CSE_ID` | Google Custom Search Engine ID |
| `OPENWEATHER_API_KEY` | OpenWeatherMap API key |
| `SERPAPI_API_KEY` | SerpAPI key — powers Google Flights, Hotels, and Maps search |

---

## Demo Accounts

| User | Email | Password | Profile |
|---|---|---|---|
| Alice | alice@example.com | demo123 | Culture lover · Vegetarian · Low intensity |
| Bob | bob@example.com | demo123 | Foodie · Moderate |
| Carol | carol@example.com | demo123 | Adventure · Outdoor |

---

## Agents

| Agent | Responsibility |
|---|---|
| Orchestrator | Routes user requests to the appropriate specialist |
| Concierge | Gathers user preferences — destination, dates, budget, interests |
| Booking Agent | Searches and presents flight and hotel options with live prices |
| Local Guide | Recommends attractions, activities, and local experiences |
| Summarizer | Produces the final consolidated travel itinerary |

### Booking Agent — 4-Stage Flow

```
Stage 1 (itinerary_draft)   → Research destination, draft day-by-day plan + numbered options
Stage 2 (awaiting_selection) → User picks transport/hotel, agent fetches live prices via SerpAPI
Stage 3 (reviewing)          → User confirms or changes one item; only changed item is re-searched
Stage 4 (confirmed)          → Trip locked; follow-up questions answered via live search
```

---

## Tools

| Tool | Primary Source | Fallback |
|---|---|---|
| `search_flights` | SerpAPI Google Flights (times, flight no., prices) | Tavily + Google web search |
| `search_hotels` | SerpAPI Google Hotels (ratings, prices, availability) | Tavily + Google web search |
| `search_attractions` | SerpAPI Google Maps (places, ratings, hours) | Tavily + Google web search |
| `search_weather` | OpenWeatherMap API (5-day forecast) | Tavily + Google web search |
| `web_search` | Tavily API | — |
| `google_search` | Google Custom Search API | — |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Agent Orchestration | LangGraph |
| LLM | OpenAI GPT-4o-mini |
| Flight / Hotel / Maps Search | SerpAPI (Google Flights, Hotels, Maps) |
| General Web Search | Tavily + Google Custom Search |
| Weather | OpenWeatherMap API |
| Containerisation | Docker + Docker Compose |
