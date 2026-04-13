"""
backend/main.py — TravelMind Backend (Kathy's scope)

Routers:
  /planner  → Agent3 (Planner) + Agent6 (Explainability)

Run:
  cd backend
  python -m uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import planner

app = FastAPI(
    title="TravelMind API",
    description="Agent3 (Planner) + Agent6 (Explainability)",
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
app.include_router(planner.router, prefix="/planner", tags=["planner"])


# ── Health ────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "service": "travelmind-backend-planner"}


@app.get("/", tags=["system"])
def root():
    return {"message": "TravelMind Planner API. Visit /docs for reference."}
