"""
agents/routers/security.py — security check endpoints

Provides:
  POST /security/check        — input guard (threat detection on user text)
  POST /security/check-output — output guard (validation of assistant responses)
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.state import State
from agents.specialists.input_guard_agent import input_guard_agent
from agents.specialists.output_guard_agent import output_guard_agent

router = APIRouter()


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


def _empty_state(role: str, text: str, user_id: Optional[str]) -> State:
    return State(
        messages=[{"role": role, "content": text}],
        user_id=user_id or "test_user",
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


@router.post("/security/check", response_model=SecurityCheckResponse)
def security_check(request: SecurityCheckRequest):
    try:
        result = input_guard_agent(_empty_state("user", request.text, request.user_id))
        return SecurityCheckResponse(
            threat_blocked=result.get("threat_blocked", False),
            threat_type=result.get("threat_type"),
            threat_detail=result.get("threat_detail"),
            sanitised_input=result.get("sanitised_input", request.text),
            security_audit_log=result.get("security_audit_log", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Security check failed: {exc}") from exc


@router.post("/security/check-output", response_model=OutputCheckResponse)
def security_check_output(request: OutputCheckRequest):
    try:
        result = output_guard_agent(_empty_state("assistant", request.text, request.user_id))
        return OutputCheckResponse(
            threat_blocked=result.get("output_flagged", False),
            threat_type=result.get("output_flag_reason"),
            threat_detail=result.get("output_guard_decision", {}).get("reason", "Output passed validation"),
            security_audit_log=result.get("security_audit_log", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Output security check failed: {exc}") from exc
