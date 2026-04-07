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
# In Docker: agents and tools are at /app/agents and /app/tools
# Locally: they're at project root/../agents and project root/../tools
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.exists(os.path.join(project_root, "agents")):
    sys.path.insert(0, project_root)
elif os.path.exists("/app/agents"):
    sys.path.insert(0, "/app")

from agents.graph import build_travel_graph
from agents.state import State
from agents import input_guard_agent
from agents.specialists.output_guard_agent import output_guard_agent

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────

class SecurityCheckRequest(BaseModel):
    text: str
    user_id: Optional[str] = None


class SecurityCheckResponse(BaseModel):
    threat_blocked: bool
    threat_type: Optional[str] = None
    threat_detail: Optional[str] = None
    sanitised_input: str
    security_audit_log: Optional[list] = None


class OutputCheckRequest(BaseModel):
    text: str
    user_id: Optional[str] = None


class OutputCheckResponse(BaseModel):
    threat_blocked: bool
    threat_type: Optional[str] = None
    threat_detail: Optional[str] = None
    security_audit_log: Optional[list] = None


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

@router.post("/security/check", response_model=SecurityCheckResponse)
def security_check(request: SecurityCheckRequest):
    """
    Security check endpoint — validates user input against the input guard agent.
    Tests the input through the two-layer security pipeline:
      Layer 1: Regex-based injection detection
      Layer 2: LLM Guard semantic scanner
    
    Returns whether the input was blocked and any sanitised version of it.
    """
    print(f"[Backend] 🔐 Received security check request")
    print(f"[Backend] Text: {request.text[:100]}...")
    print(f"[Backend] User ID: {request.user_id}")
    
    try:
        # Build initial state with user message
        initial_state = State(
            messages=[{"role": "user", "content": request.text}],
            user_id=request.user_id or "test_user",
            # Fill other required fields with defaults
            origin=None,
            destination=None,
            dates=None,
            budget=None,
            preferences=None,
            duration=None,
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
            threat_blocked=False,
            threat_type=None,
            threat_detail=None,
            sanitised_input="",
            security_audit_log=[],
        )
        
        # Run the input guard agent
        print(f"[Backend] 🔐 Running input_guard_agent...")
        result = input_guard_agent(initial_state)
        print(f"[Backend] ✅ input_guard_agent completed")
        print(f"[Backend] Result: threat_blocked={result.get('threat_blocked')}, threat_type={result.get('threat_type')}")
        
        response = SecurityCheckResponse(
            threat_blocked=result.get("threat_blocked", False),
            threat_type=result.get("threat_type"),
            threat_detail=result.get("threat_detail"),
            sanitised_input=result.get("sanitised_input", request.text),
            security_audit_log=result.get("security_audit_log", []),
        )
        print(f"[Backend] 📤 Returning response")
        return response
    
    except Exception as e:
        print(f"[Backend] ❌ Error in security_check: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Security check failed: {str(e)}")


@router.post("/security/check-output", response_model=OutputCheckResponse)
def security_check_output(request: OutputCheckRequest):
    """
    Output security check endpoint — validates model output against the output guard agent.
    Tests the output through the multi-layer security pipeline:
      Layer 1: Hallucination detection
      Layer 2: Regex-based PII detection
      Layer 3: Rule-based unsafe topics check
      Layer 4: LLM semantic unsafe-content check
      Layer 5: LLM Guard output scanner
    
    Returns whether the output was flagged for security issues.
    """
    print(f"[Backend] 🔐 Received output security check request")
    print(f"[Backend] Text: {request.text[:100]}...")
    print(f"[Backend] User ID: {request.user_id}")
    
    try:
        # Build initial state with output to validate
        initial_state = State(
            messages=[{"role": "assistant", "content": request.text}],
            user_id=request.user_id or "test_user",
            # Fill other required fields with defaults
            origin=None,
            destination=None,
            dates=None,
            budget=None,
            preferences=None,
            duration=None,
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
            threat_blocked=False,
            threat_type=None,
            threat_detail=None,
            sanitised_input="",
            security_audit_log=[],
        )
        
        # Run the output guard agent
        print(f"[Backend] 🔐 Running output_guard_agent...")
        result = output_guard_agent(initial_state)
        print(f"[Backend] ✅ output_guard_agent completed")
        print(f"[Backend] Result: output_flagged={result.get('output_flagged')}, output_flag_reason={result.get('output_flag_reason')}")
        
        # Map output_guard_agent result keys to response keys
        threat_blocked = result.get("output_flagged", False)
        threat_type = result.get("output_flag_reason")
        threat_detail = result.get("output_guard_decision", {}).get("reason", "Output passed validation")
        
        response = OutputCheckResponse(
            threat_blocked=threat_blocked,
            threat_type=threat_type,
            threat_detail=threat_detail,
            security_audit_log=result.get("security_audit_log", []),
        )
        print(f"[Backend] 📤 Returning response: threat_blocked={threat_blocked}")
        return response
    
    except Exception as e:
        print(f"[Backend] ❌ Error in security_check_output: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Output security check failed: {str(e)}")


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
