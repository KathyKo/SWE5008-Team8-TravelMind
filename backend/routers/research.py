"""
backend/routers/research.py — Agent2 (Research) endpoints

POST /research/run  → research_agent(state)
                      Returns raw flights, hotels, attractions, restaurants
GET  /research/health
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.agent_client import call_research

router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────

class ResearchRequest(BaseModel):
    """
    Input to Agent2 (Research).
    Matches the output format from Agent1 (Intent Profile).
    """
    origin:             str
    destination:        str
    dates:              str             # e.g. "2026-05-01 to 2026-05-07"
    duration:           str             # e.g. "7 days"
    budget:             str
    preferences:        str
    outbound_time_pref: Optional[str]        = None
    return_time_pref:   Optional[str]        = None
    user_profile:       Optional[dict]       = None
    search_queries:     Optional[list[dict]] = None  # RAG queries from Agent1
    hard_constraints:   Optional[dict]       = None
    soft_preferences:   Optional[dict]       = None


class ResearchResponse(BaseModel):
    """
    Output from Agent2 (Research).
    Feed this into Agent3 (Planner) via POST /planner/run.
    """
    flight_options_outbound: list   # ranked outbound flight options
    flight_options_return:   list   # ranked return flight options
    hotel_options:           list   # ranked hotel options
    attractions:             list   # scored attraction candidates
    restaurants:             list   # scored restaurant candidates
    tool_log:                list   # step-by-step execution log


# ── Endpoints ──────────────────────────────────────────────────

@router.post("/run", response_model=ResearchResponse)
def run_research(request: ResearchRequest):
    """
    Agent2 (research_agent) — collects local info and transportation data.

    Input  : Travel info + user profile + preferences  (from Agent1)
    Output : Flights, hotels, attractions, restaurants (for Agent3)
    """
    state = {
        "origin":             request.origin,
        "destination":        request.destination,
        "dates":              request.dates,
        "duration":           request.duration,
        "budget":             request.budget,
        "preferences":        request.preferences,
        "outbound_time_pref": request.outbound_time_pref or "",
        "return_time_pref":   request.return_time_pref   or "",
        "user_profile":       request.user_profile       or {},
        "search_queries":     request.search_queries     or [],
        "hard_constraints":   request.hard_constraints   or {},
        "soft_preferences":   request.soft_preferences   or {},
    }

    print(f"[Agent2] Research: {request.origin} → {request.destination} | {request.dates}")
    result = call_research(state)

    if "error" in result:
        raise HTTPException(status_code=500, detail=f"Agent2 error: {result['error']}")

    return ResearchResponse(
        flight_options_outbound = result.get("compact_flights_out", []),
        flight_options_return   = result.get("compact_flights_ret",  []),
        hotel_options           = result.get("hotel_opts",           []),
        attractions             = result.get("compact_attractions",  []),
        restaurants             = result.get("compact_restaurants",  []),
        tool_log                = result.get("tool_log",             []),
    )


@router.get("/health")
def health():
    return {"status": "ok", "agent": "Agent2-Research"}
