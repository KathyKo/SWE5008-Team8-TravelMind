"""
backend/main.py — TravelMind Backend (Kathy: Agent2 / Agent3 / Agent6)

Routers:
  /research       → Agent2 (Research)       research_agent_1
  /planner        → Agent3 (Planner)        planner_agent_1 + revise_itinerary_1
  /explainability → Agent6 (Explainability) explainability_agent

Run (from project root — travelagents/):
  python -m uvicorn backend.main:app --reload --port 8000

Interactive docs:
  http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import research, planner, explainability

app = FastAPI(
    title="TravelMind API — Kathy (Agent2 / Agent3 / Agent6)",
    description=(
        "Agent2 (Research): POST /research/run\n"
        "Agent3 (Planner):  POST /planner/run  |  POST /planner/revise\n"
        "Agent6 (Explain):  POST /explainability/run"
    ),
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(research.router,       prefix="/research",       tags=["Agent2 Research"])
app.include_router(planner.router,        prefix="/planner",        tags=["Agent3 Planner"])
app.include_router(explainability.router, prefix="/explainability", tags=["Agent6 Explainability"])


# ── Health ────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "service": "travelmind-backend", "agents": ["Agent2", "Agent3", "Agent6"]}


@app.get("/", tags=["system"])
def root():
    return {
        "message": "TravelMind API — Kathy scope",
        "agents": {
            "Agent2 Research":       "POST /research/run",
            "Agent3 Planner":        "POST /planner/run",
            "Agent3 Revise":         "POST /planner/revise",
            "Agent6 Explainability": "POST /explainability/run",
        },
        "docs": "/docs",
    }
