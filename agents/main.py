"""
agents/main.py — standalone Agents API service

This service hosts the heavy security and planning logic.
Backend and peer agents call this service over HTTP.
"""

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from agents.graph import build_travel_graph
from agents.state import State
from agents.specialists.input_guard_agent import input_guard_agent
from agents.specialists.output_guard_agent import output_guard_agent
from agents.specialists.debate_agent import debate_agent
from agents.specialists.dynamic_replan_agent import dynamic_replan_agent

app = FastAPI(
    title="TravelMind Agents API",
    description="Standalone service for TravelMind agent orchestration and security guards",
    version="0.1.0",
)


def _enforce_agent_port(request: Request, expected_port: int, agent_name: str) -> None:
    actual_port = request.url.port
    if actual_port != expected_port:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Port mismatch for {agent_name}: expected {expected_port}, got {actual_port}. "
                f"Please call {agent_name} on port {expected_port}."
            ),
        )


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


class ReplanRequest(BaseModel):
    user_feedback: dict[str, Any]
    plan_id: Optional[str] = None
    user_id: Optional[str] = None


class AgentInvokeRequest(BaseModel):
    state: dict[str, Any]


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "service": "travelmind-agents"}


@app.post("/api/invoke/input_guard")
def invoke_input_guard(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8100, "input_guard")
    try:
        result = input_guard_agent(payload.state)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"input_guard failed: {exc}") from exc


@app.post("/api/invoke/output_guard")
def invoke_output_guard(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8106, "output_guard")
    try:
        result = output_guard_agent(payload.state)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"output_guard failed: {exc}") from exc


@app.post("/api/invoke/intent_profile")
def invoke_intent_profile(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8101, "intent_profile")
    state = payload.state
    return {
        "intent_profile_output": state.get("intent_profile_output", {}),
        "next_node": state.get("next_node", "orchestrator"),
    }


@app.post("/api/invoke/search")
def invoke_search(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8102, "search")
    state = payload.state
    return {
        "search_results": state.get("search_results", []),
        "next_node": state.get("next_node", "orchestrator"),
    }


@app.post("/api/invoke/planner")
def invoke_planner(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8103, "planner")
    state = payload.state
    return {
        "itinerary": state.get("itinerary"),
        "next_node": state.get("next_node", "orchestrator"),
    }


@app.post("/api/invoke/debate")
def invoke_debate(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8104, "debate")
    try:
        result = debate_agent(payload.state)
        return {
            "plan_id": result.get("plan_id"),
            "is_valid": result.get("is_valid"),
            "debate_count": result.get("debate_count"),
            "debate_output": result.get("debate_output", {}),
            "next_node": result.get("next_node", "orchestrator"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"debate failed: {exc}") from exc


@app.post("/api/invoke/explain")
def invoke_explain(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8105, "explain")
    state = payload.state
    return {
        "explain_output": state.get("explain_output", {}),
        "next_node": state.get("next_node", "orchestrator"),
    }


@app.post("/api/invoke/replanner")
def invoke_replanner(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8107, "replanner")
    try:
        result = dynamic_replan_agent(payload.state)
        return {
            "plan_id": result.get("plan_id"),
            "replanner_output": result.get("replanner_output", {}),
            "user_feedback": result.get("user_feedback"),
            "next_node": result.get("next_node", "orchestrator"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"replanner failed: {exc}") from exc


@app.post("/security/check", response_model=SecurityCheckResponse)
def security_check(request: SecurityCheckRequest):
    try:
        initial_state = State(
            messages=[{"role": "user", "content": request.text}],
            user_id=request.user_id or "test_user",
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

        result = input_guard_agent(initial_state)
        return SecurityCheckResponse(
            threat_blocked=result.get("threat_blocked", False),
            threat_type=result.get("threat_type"),
            threat_detail=result.get("threat_detail"),
            sanitised_input=result.get("sanitised_input", request.text),
            security_audit_log=result.get("security_audit_log", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Security check failed: {exc}") from exc


@app.post("/security/check-output", response_model=OutputCheckResponse)
def security_check_output(request: OutputCheckRequest):
    try:
        initial_state = State(
            messages=[{"role": "assistant", "content": request.text}],
            user_id=request.user_id or "test_user",
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

        result = output_guard_agent(initial_state)
        return OutputCheckResponse(
            threat_blocked=result.get("output_flagged", False),
            threat_type=result.get("output_flag_reason"),
            threat_detail=result.get("output_guard_decision", {}).get("reason", "Output passed validation"),
            security_audit_log=result.get("security_audit_log", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Output security check failed: {exc}") from exc


@app.post("/plan", response_model=PlanResponse)
def plan(request: PlanRequest):
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
            intent_profile_output=result.get("intent_profile_output"),
            stage=result.get("stage"),
            is_complete=result.get("is_complete", False),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/summarize", response_model=PlanResponse)
def summarize(request: SummarizeRequest):
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/replan")
def replan(request: ReplanRequest):
    try:
        graph = build_travel_graph()
        initial_state = State(
            messages=[{"role": "user", "content": str(request.user_feedback)}],
            user_id=request.user_id or "replan_user",
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
            user_feedback={**request.user_feedback, "plan_id": request.plan_id} if request.plan_id else request.user_feedback,
            replanner_output=None,
            replan_mode=True,
            is_valid=None,
            debate_count=0,
        )
        result = graph.invoke(initial_state, config={"recursion_limit": 30})
        return {
            "plan_id": result.get("plan_id"),
            "replanner_output": result.get("replanner_output", {}),
            "next_node": result.get("next_node", "END"),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"replan failed: {exc}") from exc
