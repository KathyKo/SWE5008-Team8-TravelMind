"""
backend/main.py — FastAPI entry point for TravelMind backend
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import travel

app = FastAPI(
    title="TravelMind API",
    description="Multi-agent travel planning backend powered by LangGraph",
    version="0.1.0",
)

# ── CORS ─────────────────────────────────────────────────────
# Allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",       # local Streamlit dev
        "http://frontend:8501",        # Docker internal
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(travel.router, prefix="/travel", tags=["travel"])


# ── Health check ──────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "service": "travelmind-backend"}


@app.get("/", tags=["system"])
def root():
    return {"message": "TravelMind API is running. Visit /docs for the API reference."}
