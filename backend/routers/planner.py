"""
backend/routers/planner.py — Kathy's planner endpoints

POST /planner/generate   → Agent3 (Planner) only — fast, returns itineraries immediately
POST /planner/explain    → Agent6 (Explainability) — call lazily from My Trip page
GET  /planner/health     → health check
"""

import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from backend.agent_client import call_planner, call_planner_revise, call_explainability, run_debate_loop
from backend.db.database import engine, get_db
from backend.db.models import Base
from backend.db import crud

router = APIRouter()

# Create tables on startup (idempotent — skipped if tables already exist)
Base.metadata.create_all(bind=engine)


# ── Request / Response schemas ────────────────────────────────

class GenerateRequest(BaseModel):
    origin:              str
    destination:         str
    dates:               str
    duration:            str
    budget:              str
    preferences:         str
    outbound_time_pref:  Optional[str] = None
    return_time_pref:    Optional[str] = None
    user_profile:        Optional[dict] = None
    search_queries:      Optional[list[dict]] = None
    hard_constraints:    Optional[dict] = None
    soft_preferences:    Optional[dict] = None


class GenerateResponse(BaseModel):
    plan_id:                  str    # pass back to /explain
    itineraries:              dict
    option_meta:              dict
    agent_steps:              list
    tool_log:                 list
    flight_options_outbound:  list
    flight_options_return:    list
    hotel_options:            list


class ReviseRequest(BaseModel):
    plan_id:   str
    critique:  str     # critique_summary from Agent4


class DebateRequest(BaseModel):
    origin:              str
    destination:         str
    dates:               str
    duration:            str
    budget:              str
    preferences:         str
    outbound_time_pref:  Optional[str] = None
    return_time_pref:    Optional[str] = None
    user_profile:        Optional[dict] = None
    search_queries:      Optional[list[dict]] = None
    hard_constraints:    Optional[dict] = None
    soft_preferences:    Optional[dict] = None


class DebateResponse(BaseModel):
    plan_id:                  str
    itineraries:              dict
    option_meta:              dict
    tool_log:                 list
    flight_options_outbound:  list
    flight_options_return:    list
    hotel_options:            list
    debate_verdict:           Optional[dict] = None
    debate_history:           Optional[list] = None
    # Agent6 explainability
    explain_data:             Optional[dict] = None
    chain_of_thought:         Optional[str]  = None


class ExplainRequest(BaseModel):
    plan_id: str


class ExplainResponse(BaseModel):
    explain_data:     dict
    chain_of_thought: str
    agent_steps:      list


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/generate", response_model=GenerateResponse)
def generate_plan(request: GenerateRequest, db: Session = Depends(get_db)):
    """Agent3 only — returns 3 itinerary options quickly (no Agent6)."""
    state = {
        "origin":             request.origin,
        "destination":        request.destination,
        "dates":              request.dates,
        "duration":           request.duration,
        "budget":             request.budget,
        "preferences":        request.preferences,
        "outbound_time_pref": request.outbound_time_pref or "",
        "return_time_pref":   request.return_time_pref or "",
        "user_profile":       request.user_profile or {},
        "search_queries":     request.search_queries or [],
        "hard_constraints":   request.hard_constraints or {},
        "soft_preferences":   request.soft_preferences or {},
    }

    print(f"[Backend] Calling Agent3 (Planner): {request.origin} → {request.destination} | {request.dates}")
    planner_result = call_planner(state)

    if "error" in planner_result:
        raise HTTPException(status_code=500, detail=f"Planner error: {planner_result['error']}")

    plan_id = str(uuid.uuid4())[:8]
    crud.save_plan(db, plan_id, state, planner_result, via_debate=False)

    return GenerateResponse(
        plan_id                 = plan_id,
        itineraries             = planner_result.get("itineraries",             {}),
        option_meta             = planner_result.get("option_meta",             {}),
        agent_steps             = planner_result.get("tool_log",                [])[:3],
        tool_log                = planner_result.get("tool_log",                []),
        flight_options_outbound = planner_result.get("flight_options_outbound", []),
        flight_options_return   = planner_result.get("flight_options_return",   []),
        hotel_options           = planner_result.get("hotel_options",           []),
    )


@router.post("/revise", response_model=GenerateResponse)
def revise_plan(request: ReviseRequest, db: Session = Depends(get_db)):
    """Agent3 revision mode — receives Agent4 critique, revises using cached inventory (1 LLM call)."""
    cached = crud.load_plan(db, request.plan_id)
    if not cached:
        raise HTTPException(status_code=404, detail=f"Plan {request.plan_id} not found. Generate a plan first.")

    state_keys = ("origin", "destination", "dates", "duration", "budget",
                  "preferences", "outbound_time_pref", "return_time_pref", "user_profile",
                  "search_queries", "hard_constraints", "soft_preferences")
    state        = {k: cached[k] for k in state_keys if k in cached}
    current_result = {k: cached[k] for k in ("itineraries", "option_meta", "tool_log",
                      "flight_options_outbound", "flight_options_return", "hotel_options") if k in cached}

    print(f"[Backend] Calling Agent3 revision for plan {request.plan_id} ({len(request.critique)} chars critique)")
    revised = call_planner_revise(state, request.critique, current_result)

    if "error" in revised:
        raise HTTPException(status_code=500, detail=f"Revision error: {revised['error']}")

    crud.update_plan_result(db, request.plan_id, revised)

    return GenerateResponse(
        plan_id                 = request.plan_id,
        itineraries             = revised.get("itineraries",             {}),
        option_meta             = revised.get("option_meta",             {}),
        agent_steps             = revised.get("tool_log",                [])[:3],
        tool_log                = revised.get("tool_log",                []),
        flight_options_outbound = revised.get("flight_options_outbound", []),
        flight_options_return   = revised.get("flight_options_return",   []),
        hotel_options           = revised.get("hotel_options",           []),
    )


@router.post("/debate", response_model=DebateResponse)
def debate_plan(request: DebateRequest, db: Session = Depends(get_db)):
    """Agent3 → Agent4 debate loop (up to 3 rounds) → Agent6 explain final output."""
    state = {
        "origin":             request.origin,
        "destination":        request.destination,
        "dates":              request.dates,
        "duration":           request.duration,
        "budget":             request.budget,
        "preferences":        request.preferences,
        "outbound_time_pref": request.outbound_time_pref or "",
        "return_time_pref":   request.return_time_pref or "",
        "user_profile":       request.user_profile or {},
        "search_queries":     request.search_queries or [],
        "hard_constraints":   request.hard_constraints or {},
        "soft_preferences":   request.soft_preferences or {},
    }

    print(f"[Backend] Starting debate: {request.origin} → {request.destination} | {request.dates}")
    result = run_debate_loop(state)

    if "error" in result:
        raise HTTPException(status_code=500, detail=f"Debate error: {result['error']}")

    plan_id = str(uuid.uuid4())[:8]
    crud.save_plan(db, plan_id, state, result, via_debate=True)

    # Agent6: explain the final itineraries
    print(f"[Backend] Calling Agent6 (Explainability) on debate result...")
    full_cached = crud.load_plan(db, plan_id)
    explain_result = call_explainability(full_cached)
    crud.save_explain(db, plan_id, explain_result)

    return DebateResponse(
        plan_id                 = plan_id,
        itineraries             = result.get("itineraries",             {}),
        option_meta             = result.get("option_meta",             {}),
        tool_log                = result.get("tool_log",                []),
        flight_options_outbound = result.get("flight_options_outbound", []),
        flight_options_return   = result.get("flight_options_return",   []),
        hotel_options           = result.get("hotel_options",           []),
        debate_verdict          = result.get("debate_verdict"),
        debate_history          = result.get("debate_history"),
        explain_data            = explain_result.get("explain_data",     {}),
        chain_of_thought        = explain_result.get("chain_of_thought", ""),
    )


@router.post("/explain", response_model=ExplainResponse)
def explain_plan(request: ExplainRequest, db: Session = Depends(get_db)):
    """Agent6 — runs explainability on a previously generated plan."""
    cached = crud.load_plan(db, request.plan_id)
    if not cached:
        raise HTTPException(status_code=404, detail=f"Plan {request.plan_id} not found. Generate a plan first.")

    print(f"[Backend] Calling Agent6 (Explainability) for plan {request.plan_id}...")
    explain_result = call_explainability(cached)
    crud.save_explain(db, request.plan_id, explain_result)

    return ExplainResponse(
        explain_data     = explain_result.get("explain_data",     {}),
        chain_of_thought = explain_result.get("chain_of_thought", ""),
        agent_steps      = explain_result.get("agent_steps",      []),
    )


@router.get("/health")
def health():
    return {"status": "ok", "router": "planner"}
