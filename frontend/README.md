# TravelMind 🗺️

An agentic AI multi-agent travel planning system — NUS-ISS SWE5008 Group Project (Team 8).

---

## Overview

TravelMind coordinates seven specialised AI agents to support the full travel planning lifecycle: from understanding user intent and retrieving real-time travel data, to generating multiple itinerary options, reviewing them through a debate mechanism, and dynamically re-planning mid-trip.

This repository contains the **frontend prototype** built with Streamlit, demonstrating the full user-facing workflow with simulated agent behaviour.

---

## Pages

| Page | Description |
|---|---|
| 🗺️ **Plan** | Enter a trip request → agents generate 3 itinerary options → select preferred option |
| 📅 **My Trip** | View full itinerary · check off visited places (implicit feedback) · trigger re-planning |
| 🔄 **Re-plan** | Select situation + time available → Dynamic Re-planning Agent finds nearby alternatives |
| 🛡️ **Security** | Live demo of the Risk & Safety Agent intercepting prompt injection, PII probes, and hallucination attempts |

---

## Project Structure

```
travelmind/
├── app.py                  # Entry point — login screen, topbar, tab routing
├── pages/
│   ├── plan.py             # Plan Your Trip page
│   ├── my_trip.py          # My Trip page
│   ├── replan.py           # Dynamic Re-planning page
│   └── security.py         # Security Demo page
├── data/
│   └── store.py            # Shared data — users, itineraries, agent steps, security patterns
├── .streamlit/
│   └── config.toml         # Streamlit theme and server config
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Getting Started

### Option 1 — Run locally

**Prerequisites:** Python 3.11+

```bash
# 1. Clone the repository
git clone <repo-url>
cd travelmind

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

### Option 2 — Run with Docker

**Prerequisites:** Docker and Docker Compose installed

```bash
# 1. Clone the repository
git clone <repo-url>
cd travelmind

# 2. Build and start the container
docker compose up --build

# 3. Open your browser at
http://localhost:8501

# Stop the container
docker compose down
```

---

### Option 3 — Run with Docker (without Compose)

```bash
# Build the image
docker build -t travelmind .

# Run the container
docker run -p 8501:8501 travelmind
```

---

## Demo Accounts

| User | Email | Password | Profile |
|---|---|---|---|
| Alice | alice@example.com | demo123 | Culture lover · Vegetarian · Low intensity |
| Bob | bob@example.com | demo123 | Foodie · Moderate |
| Carol | carol@example.com | demo123 | Adventure · Outdoor |

---

## Demo Scenarios

**1. Normal planning flow**
- Sign in as Alice
- Go to 🗺️ Plan → click **Generate Options →**
- Watch all 6 agents run, then the Debate & Critique exchange
- Switch between Option A / B / C and select a preferred itinerary

**2. Visit check-in feedback loop**
- Go to 📅 My Trip
- Check off visited locations — notice the AI profile confidence update in the side panel

**3. Dynamic re-planning**
- Go to 🔄 Re-plan
- Select **I'm tired** → **2–3 hours**
- Watch the Re-planning Agent log and choose from 3 nearby alternatives

**4. Security attack demo**
- Go to 🛡️ Security
- Try the **Prompt Injection** or **PII Probe** presets
- Watch the Risk & Safety Agent intercept and log the attack in real time
- Compare with the **Normal Query** preset which passes through cleanly

---

## Tech Stack (Prototype)

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Agent Orchestration | LangGraph (planned) |
| LLM | OpenAI GPT-4o / Anthropic Claude (planned) |
| Vector Store | ChromaDB (planned) |
| Containerisation | Docker |
| CI/CD | GitHub Actions (planned) |
| Monitoring | LangSmith (planned) |

---
