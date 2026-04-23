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
│  Tavily · Google Search · OpenWeather                   │
│  Flights · Hotels · Attractions                         │
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
│   │   └── research.py
│   │   └── planner.py
│   │   └── explainability.py
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
│       ├── intent_profile.py
│       ├── input_guard_agent.py
│       ├── output_guard_agent.py
│       ├── planner_agent_1.py
│       ├── research_agent_1.py
│       ├── explainability_agent.py
│       ├── replanner_agent.py
│       ├── debate_agent.py
│       ├── orchestrator.py
│
├── tools/                       # External API integrations
│   ├── __init__.py
│   ├── web_search.py            # Tavily
│   ├── google_search.py         # Google Custom Search
│   ├── search_flights.py
│   ├── search_hotels.py
│   ├── search_weather.py
│   └── search_attractions.py
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
| `OPENAI_MODEL` | Model name (default: `gpt-5-mini-2025-08-07`) |
| `TAVILY_API_KEY` | Tavily web search API key |
| `GOOGLE_API_KEY` | Google Custom Search API key |
| `GOOGLE_CSE_ID` | Google Custom Search Engine ID |
| `OPENWEATHER_API_KEY` | OpenWeatherMap API key |

---

## Demo Accounts

| User | Email | Password | Profile |
|---|---|---|---|
| Alice | alice@example.com | demo123 | Culture lover · Vegetarian · Low intensity |
| Bob | bob@example.com | demo123 | Foodie · Moderate |
| Carol | carol@example.com | demo123 | Adventure · Outdoor |

---

## Agents

| Agent | Port | Responsibility |
|---|---|---|
| Orchestrator | 8001 | Entry point — routes requests through the multi-agent pipeline |
| Input Guard | 8100 | Screens user input for prompt injection and PII before processing |
| Intent Profile | 8101 | Extracts travel intent, preferences, hard constraints, and soft preferences |
| Research | 8102 | Fetches and ranks flights, hotels, attractions, and restaurants via external APIs |
| Planner | 8103 | Generates 3 itinerary options (A/B/C) via ReAct loop using research data |
| Debate & Critique | 8104 | Reviews plans, raises critiques, and triggers revision loops (up to 3 rounds) |
| Explainability | 8105 | Generates reasoning traces and per-item explanations for each itinerary option |
| Output Guard | 8106 | Validates agent output for hallucinations and policy compliance before returning to user |
| Replanner | 8107 | Handles real-time re-planning requests (fatigue, bad weather, venue closures) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Agent Orchestration | LangGraph 0.2 |
| LLM | OpenAI GPT-5-mini |
| Search | Tavily + Google Custom Search |
| Weather | OpenWeatherMap API |
| Containerisation | Docker + Docker Compose |

---
