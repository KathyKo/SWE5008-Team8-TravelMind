"""
backend/routers/travel.py — Travel planning API endpoints

Backend acts as an API gateway and delegates agent execution
to the standalone agents service over HTTPS.
"""

import os
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

AGENTS_BASE_URL = os.getenv("AGENTS_BASE_URL", "http://agents:8001").rstrip("/")
AGENTS_TIMEOUT_SECONDS = float(os.getenv("AGENTS_TIMEOUT_SECONDS", "180"))


def _agents_post(path: str, payload: dict) -> dict:
    url = f"{AGENTS_BASE_URL}{path}"
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=AGENTS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = f"Failed to reach agents service at {url}: {exc}"
        raise HTTPException(status_code=502, detail=detail) from exc

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
    outbound_time_pref: Optional[str] = None
    return_time_pref: Optional[str] = None
    user_id: Optional[str] = None

class PlanResponse(BaseModel):
    reply: str
    next_agent: Optional[str] = None
    destination: Optional[str] = None
    flight_options: Optional[list] = None
    hotel_options: Optional[list] = None
    itinerary: Optional[str] = None
    final_itinerary: Optional[str] = None
    intent_profile_output: Optional[dict] = None
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
    payload = {"text": request.text, "user_id": request.user_id}
    result = _agents_post("/security/check", payload)
    return SecurityCheckResponse(
        threat_blocked=result.get("threat_blocked", False),
        threat_type=result.get("threat_type"),
        threat_detail=result.get("threat_detail"),
        sanitised_input=result.get("sanitised_input", request.text),
        security_audit_log=result.get("security_audit_log", []),
    )


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
    payload = {"text": request.text, "user_id": request.user_id}
    result = _agents_post("/security/check-output", payload)
    return OutputCheckResponse(
        threat_blocked=result.get("threat_blocked", False),
        threat_type=result.get("threat_type"),
        threat_detail=result.get("threat_detail"),
        security_audit_log=result.get("security_audit_log", []),
    )


@router.post("/plan", response_model=PlanResponse)
def plan(request: PlanRequest):
    """
    Send a user message to the travel planning agent graph.
    The graph routes to the appropriate agent (concierge, booking, local guide)
    and returns the agent's response along with updated state.
    """
    payload = request.model_dump()
    result = _agents_post("/plan", payload)
    return PlanResponse(
        reply=result.get("reply", ""),
        next_agent=result.get("next_agent"),
        destination=result.get("destination"),
        flight_options=result.get("flight_options"),
        hotel_options=result.get("hotel_options"),
        itinerary=result.get("itinerary"),
        final_itinerary=result.get("final_itinerary"),
        intent_profile_output=result.get("intent_profile_output"),
        stage=result.get("stage"),
        is_complete=result.get("is_complete", False),
    )


@router.post("/summarize", response_model=PlanResponse)
def summarize(request: SummarizeRequest):
    """
    Trigger the summarizer agent directly to produce a final itinerary.
    Call this when the user confirms their plan.
    """
    payload = request.model_dump()
    result = _agents_post("/summarize", payload)
    return PlanResponse(
        reply=result.get("reply", ""),
        final_itinerary=result.get("final_itinerary"),
        is_complete=result.get("is_complete", True),
    )


@router.get("/health", tags=["system"])
def router_health():
    return {"status": "ok", "router": "travel"}
