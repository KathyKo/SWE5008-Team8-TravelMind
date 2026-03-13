#  Travel Planning Agent (Multi-Agent System)

## Project Overview
This project is an AI-powered multi-agent travel agency designed to help users plan a complete trip. Whether it's finding flights, hotels, or discovering local attractions, our expert agents collaborate to generate a personalized itinerary. 

This system demonstrates a robust **Orchestrator-Worker architecture** using **LangGraph**, where a specialized coordinator manages the flow between different domain experts.

## Key Features
- **Multi-Agent Coordination**: An intelligent Orchestrator routes tasks to specialized agents (Concierge, Booking Specialist, Local Guide).
- **Real-time Tool Integration**: Agents use real APIs (Tavily web search, OpenWeather) to provide up-to-date travel information.
- **Dynamic State Management**: Shared state maintains user preferences and gathered travel data throughout the session.
- **Smart Itinerary Generation**: A final summary agent compiles all gathered data into a beautiful, ready-to-use travel plan.

## Agent Roles & Personas
1. **Travel Orchestrator (The Coordinator)**: Analyzes the conversation and assigns the most suitable expert to the task.
2. **Travel Concierge (The Facilitator)**: Interacts with the user to gather essential trip details (destination, dates, budget).
3. **Booking Specialist (The Logistician)**: Specialized in finding real-time flight and hotel options based on user constraints.
4. **Local Guide (The Insider)**: Provides weather forecasts and suggests must-see tourist attractions.
5. **Master Planner (The Summarizer)**: Reviews all expert input and creates the final, cohesive itinerary.

## Tools Integrated
- **`search_flights`**: Searches for current flight options using the shared `web_search` helper.
- **`search_hotels`**: Finds top-rated accommodations based on destination and budget (via Tavily-powered web search).
- **`search_weather`**: Fetches 2-day weather forecasts via OpenWeather API.
- **`search_attractions`**: Suggests popular landmarks and activities in the target city (via Tavily-powered web search).
- **`web_search`**: A generic real-time information retrieval tool backed by the Tavily API.

## Agent Tool Access Control (Demo)

This project also demonstrates **per-agent tool permissions**:

- **Central tool registry**:  
  - `tools/__init__.py` exposes reusable tools (`search_flights`, `search_hotels`, `search_weather`, `search_attractions`, `web_search`).  
  - `agent_tools.py` defines which agent is allowed to use which tools:
    - `booking_agent`: `web_search`, `search_hotels`, `search_flights`
    - `local_guide`: `search_weather`, `search_attractions`
- **Injection instead of global imports**:  
  - In `nodes.py`, each node calls agents with an injected `tools` dict:
    - `booking_node` → `booking_agent(state, tools=...)`
    - `local_guide_node` → `local_guide(state, tools=...)`
- **Runtime enforcement inside agents**:  
  - `booking_agent` / `local_guide` check for required tools in the injected dict and raise an error if missing.  
  - Other agents (`concierge`, `orchestrator`, `summarizer`) do not receive tools and therefore cannot accidentally call them.

This pattern keeps **capabilities explicit per agent** and makes it easy to reason about and modify who can access which external tools.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python installed. We recommend using **`uv`** or **`pip`**.

### 2. Environment Configuration
Copy the `.env.example` file and rename it to `.env`:
```sh
cp .env.example .env
```
Fill in the following keys:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `TAVILY_API_KEY`: Tavily search API Key (used for flights, hotels, attractions, and general web search).
- `OPENWEATHER_API_KEY`: OpenWeatherMap API Key.

### 3. Install Dependencies

We recommend using a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

pip install -r requirements.txt
```

### 4. Run the Agents
```sh
python main.py
```

## How to use
- Simply start chatting! (e.g., "I want to plan a trip to Tokyo in December.")
- The agents will ask for more details if needed.
- Type **'exit'** or **'done'** to generate your final itinerary summary.
