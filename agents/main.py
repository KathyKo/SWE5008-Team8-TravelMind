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
from agents.specialists.intent_profile import intent_profile
from agents.specialists.planner_agent import planner_agent
from agents.specialists.research_agent import research_agent
from agents.specialists.explainability_agent import explainability_agent
from agents.specialists.replanner_agent import replanner_agent
from agents.specialists.debate_agent import debate_agent
from agents.routers.security import router as security_router

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
    try:
        result = intent_profile(state)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"intent_profile failed: {exc}") from exc



@app.post("/api/invoke/search")
def invoke_search(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8102, "search")
    state = payload.state
    try:
        result = research_agent(state)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"search failed: {exc}") from exc


@app.post("/api/invoke/planner")
def invoke_planner(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8103, "planner")
    state = payload.state
    try:
        result = planner_agent(state)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"planner failed: {exc}") from exc


@app.post("/api/invoke/debate")
def invoke_debate(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8104, "debate")
    state = payload.state
    try:
        result = debate_agent(state)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"debate failed: {exc}") from exc


@app.post("/api/invoke/explain")
def invoke_explain(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8105, "explain")
    state = payload.state
    try:
        result = explainability_agent(state)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"explain failed: {exc}") from exc


@app.post("/api/invoke/replanner")
def invoke_replanner(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8107, "replanner")
    state = payload.state
    try:
        result = replanner_agent(state)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"replanner failed: {exc}") from exc

app.include_router(security_router)