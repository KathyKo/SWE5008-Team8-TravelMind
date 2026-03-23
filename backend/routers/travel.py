"""
backend/routers/travel.py — Travel planning API endpoints

All endpoints invoke the LangGraph workflow defined in agents/graph.py.
The graph and agents live outside this folder; backend only handles
HTTP request/response and delegates logic to the agents layer.
"""

import sys
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

# ── Path setup ────────────────────────────────────────────────
# Allow importing from agents/ and tools/ at the project root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.graph import build_travel_graph
from agents.state import State

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────

class PlanRequest(BaseModel):
    message: str
    origin: Optional[str] = None
    destination: Optional[str] = None
    dates: Optional[str] = None
    budget: Optional[str] = None
    preferences: Optional[str] = None
    duration: Optional[str] = None


class PlanResponse(BaseModel):
    reply: str
    next_agent: Optional[str] = None
    destination: Optional[str] = None
    flight_options: Optional[list] = None
    hotel_options: Optional[list] = None
    itinerary: Optional[str] = None
    final_itinerary: Optional[str] = None
    stage: Optional[str] = None
    is_complete: bool = False


class SummarizeRequest(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    dates: Optional[str] = None
    budget: Optional[str] = None
    preferences: Optional[str] = None
    duration: Optional[str] = None
    selections: Optional[dict] = None
    itinerary: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/plan", response_model=PlanResponse)
def plan(request: PlanRequest):
    """
    Send a user message to the travel planning agent graph.
    The graph routes to the appropriate agent (concierge, booking, local guide)
    and returns the agent's response along with updated state.
    """
    try:
        graph = build_travel_graph()

        initial_state = State(
            messages=[{"role": "user", "content": request.message}],
            origin=request.origin,
            destination=request.destination,
            dates=request.dates,
            budget=request.budget,
            preferences=request.preferences,
            duration=request.duration,
            flight_options=None,
            hotel_options=None,
            stage=None,
            itinerary=None,
            research=None,
            selections=None,
            search_results=None,
            final_itinerary=None,
            next_agent=None,
            confirmed=False,
            is_complete=False,
        )

        result = graph.invoke(initial_state, config={"recursion_limit": 50})

        # Extract last assistant message
        messages = result.get("messages", [])
        last_reply = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_reply = msg.get("content", "")
                break

        return PlanResponse(
            reply=last_reply,
            next_agent=result.get("next_agent"),
            destination=result.get("destination"),
            flight_options=result.get("flight_options"),
            hotel_options=result.get("hotel_options"),
            itinerary=result.get("itinerary"),
            final_itinerary=result.get("final_itinerary"),
            stage=result.get("stage"),
            is_complete=result.get("is_complete", False),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize", response_model=PlanResponse)
def summarize(request: SummarizeRequest):
    """
    Trigger the summarizer agent directly to produce a final itinerary.
    Call this when the user confirms their plan.
    """
    try:
        graph = build_travel_graph()

        state = State(
            messages=[{"role": "user", "content": "done"}],
            origin=request.origin,
            destination=request.destination,
            dates=request.dates,
            budget=request.budget,
            preferences=request.preferences,
            duration=request.duration,
            flight_options=None,
            hotel_options=None,
            stage=request.itinerary and "confirmed",
            itinerary=request.itinerary,
            research=None,
            selections=request.selections,
            search_results=None,
            final_itinerary=None,
            next_agent=None,
            confirmed=True,
            is_complete=False,
        )

        result = graph.invoke(state, config={"recursion_limit": 20})

        messages = result.get("messages", [])
        last_reply = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_reply = msg.get("content", "")
                break

        return PlanResponse(
            reply=last_reply,
            final_itinerary=result.get("final_itinerary"),
            is_complete=True,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", tags=["system"])
def router_health():
    return {"status": "ok", "router": "travel"}
