# TravelMind — Backend ⚙️

FastAPI backend for the TravelMind multi-agent travel planning system.
Receives HTTP requests from the frontend, invokes the LangGraph agent workflow, and returns structured responses.
Part of NUS-ISS SWE5008 Group Project — Team 8.

---

## Overview

The backend acts as the bridge between the Streamlit frontend and the LangGraph agent graph. It exposes a simple REST API so the frontend never needs to know about LangGraph internals — it just sends HTTP requests and receives JSON responses.

```
Frontend (Streamlit)
      │
      │  HTTP POST /travel/plan
      ▼
Backend (FastAPI)           ← You are here
      │
      │  Invokes LangGraph graph
      ▼
Agents (LangGraph)
      │
      ├── Orchestrator Agent
      ├── Concierge Agent
      ├── Booking Agent
      ├── Local Guide Agent
      └── Summarizer Agent
            │
            └── Tools (Tavily, Google, OpenWeather, etc.)
```

---

## Project Structure

```
backend/
├── main.py                 # FastAPI app — CORS, routers, health check
├── routers/
│   ├── __init__.py
│   └── travel.py           # POST /travel/plan, POST /travel/summarize
├── .env.example            # API key template — copy to .env
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Auto-generated Swagger UI |
| `POST` | `/travel/plan` | Send user message → agent responds |
| `POST` | `/travel/summarize` | Confirm plan → generate final itinerary |

### POST `/travel/plan`

**Request**
```json
{
  "message": "I want to visit Kyoto for 5 days, budget SGD 3000, vegetarian",
  "origin": "Singapore",
  "destination": "Kyoto",
  "dates": "2026-03-10 to 2026-03-14",
  "budget": "SGD 3000",
  "preferences": "vegetarian, culture",
  "duration": "5 days"
}
```

**Response**
```json
{
  "reply": "Great choice! Here are some options for your Kyoto trip...",
  "next_agent": "booking_agent",
  "destination": "Kyoto",
  "flight_options": [...],
  "hotel_options": [...],
  "itinerary": null,
  "final_itinerary": null,
  "stage": "awaiting_selection",
  "is_complete": false
}
```

### POST `/travel/summarize`

**Request**
```json
{
  "origin": "Singapore",
  "destination": "Kyoto",
  "dates": "2026-03-10 to 2026-03-14",
  "budget": "SGD 3000",
  "preferences": "vegetarian, culture",
  "duration": "5 days",
  "selections": {
    "transport": "SQ flights, depart 08:00",
    "accommodation": "Kyomachiya Guesthouse"
  },
  "itinerary": "Day 1: Fushimi Inari..."
}
```

**Response**
```json
{
  "reply": "Here is your finalised itinerary...",
  "final_itinerary": "Your 5-day Kyoto itinerary:\n\nDay 1...",
  "is_complete": true
}
```

---

## Getting Started

### Prerequisites

- Python 3.12
- API keys for OpenAI, Tavily, Google Search, and OpenWeatherMap

### Option 1 — Run locally

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys

# 5. Run the server
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/docs** to explore the API via Swagger UI.

---

### Option 2 — Run with Docker

```bash
cd backend

# Build image
docker build -t travelmind-backend .

# Run container (pass .env file)
docker run -p 8000:8000 --env-file .env travelmind-backend
```

---

### Option 3 — Run full stack with Docker Compose

From the **project root**:

```bash
# Copy and fill in your API keys first
cp backend/.env.example .env

docker compose up --build
```

- Frontend: http://localhost:8501
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for LLM calls |
| `OPENAI_MODEL` | Optional | Model name (default: `gpt-4o-mini`) |
| `TAVILY_API_KEY` | ✅ | Tavily search API key |
| `GOOGLE_API_KEY` | ✅ | Google Custom Search API key |
| `GOOGLE_CSE_ID` | ✅ | Google Custom Search Engine ID |
| `OPENWEATHER_API_KEY` | ✅ | OpenWeatherMap API key |

Copy `.env.example` to `.env` and fill in all values. Never commit `.env` to version control.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| fastapi | 0.115.0 | Web framework |
| uvicorn | 0.30.6 | ASGI server |
| pydantic | 2.8.2 | Request/response validation |
| langgraph | 0.2.28 | Agent orchestration |
| langchain | 0.3.0 | LLM tooling |
| langchain-openai | 0.2.0 | OpenAI integration |
| langchain-google-community | 2.0.0 | Google Search integration |
| requests | 2.32.3 | HTTP client for tool calls |
| python-dotenv | 1.0.1 | Environment variable loading |

---

## Team

NUS-ISS Graduate Certificate in Architecting AI Systems — SWE5008
**Team 8:** Ko Hung Chi · Lin Mingjie · Yan Binghao · Zhao Hongming · Zhao Ziyang
**Instructor:** Chen Junhua
