"""
backend/routers/planner.py — Agent3 (Planner) endpoints

POST /planner/run     → planner_agent(state)
                        Runs Agent2 internally, then generates 3 itinerary options
POST /planner/revise  → revise_itinerary(state, critique, current_result)
                        Accepts Agent4 critique and returns a revised plan (1 LLM call)
GET  /planner/health
"""

import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from backend.agent_client import call_planner, call_planner_revise
from backend.db.database import get_db
from backend.db import crud

router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────

class PlannerRequest(BaseModel):
    """
    Input to Agent3 (Planner).
    Matches the output format from Agent1 (Intent Profile) / Agent2 (Research).
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


class PlannerResponse(BaseModel):
    """
    Output from Agent3 (Planner).
    Feed plan_id into Agent6 (Explainability) via POST /explainability/run.
    Feed itineraries into Agent4 (Debate & Critique) directly.
    """
    plan_id:                 str    # store this — pass to /explainability/run
    itineraries:             dict   # {"A": [...], "B": [...], "C": [...]}
    option_meta:             dict   # label, style, budget per option
    tool_log:                list
    flight_options_outbound: list
    flight_options_return:   list
    hotel_options:           list


class ReviseRequest(BaseModel):
    """
    Input to Agent3 revision mode.
    Agent4 (Debate & Critique) passes its critique here after reviewing the plan.
    """
    plan_id:  str   # from PlannerResponse — identifies the plan to revise
    critique: str   # critique_summary string from Agent4


# ── Endpoints ──────────────────────────────────────────────────

@router.post("/run", response_model=PlannerResponse)
def run_planner(request: PlannerRequest, db: Session = Depends(get_db)):
    """
    Agent3 (planner_agent) — generates 3 itinerary options.

    Input  : Travel info + user profile + preferences  (from Agent1 / Agent2)
    Output : 3 itinerary options A/B/C + flights + hotels  (for Agent4 / Agent6)

    Internally calls Agent2 (research_agent) to fetch live data,
    then schedules the itinerary deterministically with LLM seed selection.
    The plan is saved to DB under plan_id — pass this to /explainability/run.
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

    print(f"[Agent3] Planner: {request.origin} → {request.destination} | {request.dates}")
    result = call_planner(state)

    if "error" in result:
        raise HTTPException(status_code=500, detail=f"Agent3 error: {result['error']}")

    plan_id = str(uuid.uuid4())[:8]
    crud.save_plan(db, plan_id, state, result, via_debate=False)

    return PlannerResponse(
        plan_id                 = plan_id,
        itineraries             = result.get("itineraries",             {}),
        option_meta             = result.get("option_meta",             {}),
        tool_log                = result.get("tool_log",                []),
        flight_options_outbound = result.get("flight_options_outbound", []),
        flight_options_return   = result.get("flight_options_return",   []),
        hotel_options           = result.get("hotel_options",           []),
    )


@router.post("/revise", response_model=PlannerResponse)
def revise_planner(request: ReviseRequest, db: Session = Depends(get_db)):
    """
    Agent3 revision mode (revise_itinerary) — revises itineraries based on Agent4 critique.

    Input  : plan_id (from /planner/run) + critique string from Agent4
    Output : revised itineraries A/B/C (same format as /planner/run)

    Reuses the cached research inventory — only 1 LLM call.
    Can be called up to 3 times per plan (Agent4 debate loop).
    """
    cached = crud.load_plan(db, request.plan_id)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail=f"Plan '{request.plan_id}' not found. Call /planner/run first."
        )

    state_keys = (
        "origin", "destination", "dates", "duration", "budget", "preferences",
        "outbound_time_pref", "return_time_pref", "user_profile",
        "search_queries", "hard_constraints", "soft_preferences",
    )
    state = {k: cached[k] for k in state_keys if k in cached}
    current_result = {
        k: cached[k]
        for k in ("itineraries", "option_meta", "tool_log",
                  "flight_options_outbound", "flight_options_return", "hotel_options")
        if k in cached
    }

    print(f"[Agent3] Revise plan {request.plan_id} — critique: {len(request.critique)} chars")
    revised = call_planner_revise(state, request.critique, current_result)

    if "error" in revised:
        raise HTTPException(status_code=500, detail=f"Agent3 revision error: {revised['error']}")

    crud.update_plan_result(db, request.plan_id, revised)

    return PlannerResponse(
        plan_id                 = request.plan_id,
        itineraries             = revised.get("itineraries",             {}),
        option_meta             = revised.get("option_meta",             {}),
        tool_log                = revised.get("tool_log",                []),
        flight_options_outbound = revised.get("flight_options_outbound", []),
        flight_options_return   = revised.get("flight_options_return",   []),
        hotel_options           = revised.get("hotel_options",           []),
    )


@router.get("/health")
def health():
    return {"status": "ok", "agent": "Agent3-Planner"}
