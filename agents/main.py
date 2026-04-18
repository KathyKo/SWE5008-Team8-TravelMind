"""
agents/main.py — standalone Agents API service

This service hosts the heavy security and planning logic.
Backend and peer agents call this service over HTTP.
"""

import json
import logging
import traceback
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.graph import build_travel_graph, default_graph_initial_state
from agents.state import State
from agents.specialists.input_guard_agent import input_guard_agent
from agents.specialists.output_guard_agent import output_guard_agent
from agents.specialists.intent_profile import intent_profile
from agents.specialists.planner_agent import planner_agent
from agents.specialists.research_agent import research_agent
from agents.specialists.explainability_agent import explainability_agent
from agents.specialists.debate_agent import debate_agent
from agents.specialists.dynamic_replan_agent import dynamic_replan_agent

log = logging.getLogger(__name__)

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


def _merge_graph_initial(incoming: dict[str, Any]) -> dict[str, Any]:
    base = default_graph_initial_state()
    base.update(incoming)
    return base


def _state_for_client(state: dict[str, Any]) -> dict[str, Any]:
    """Strip large / non-JSON-friendly pieces before sending to the browser."""
    keys = (
        "origin",
        "destination",
        "dates",
        "duration",
        "budget",
        "preferences",
        "session_id",
        "intent_profile_output",
        "search_results",
        "research",
        "itineraries",
        "final_itineraries",
        "option_meta",
        "tool_log",
        "flight_options_outbound",
        "flight_options_return",
        "hotel_options",
        "is_valid",
        "debate_count",
        "critique",
        "composite_score",
        "debate_output",
        "explanation",
        "explain_data",
        "output_guard_decision",
        "output_flagged",
        "output_flag_reason",
        "threat_blocked",
        "threat_detail",
        "threat_type",
        "error_message",
        "input_guard_decision",
        "sanitised_input",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k not in state:
            continue
        try:
            json.dumps(state[k], default=str)
            out[k] = state[k]
        except TypeError:
            out[k] = str(state[k])
    return out


def _ndjson_line(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, default=str) + "\n").encode("utf-8")


@app.post("/api/invoke/graph/stream")
def invoke_graph_stream(payload: AgentInvokeRequest):
    """
    Run the full LangGraph orchestrator with stream_mode='values'.
    Emits NDJSON: {"type":"progress","step":N} per super-step, then
    {"type":"done","state":{...}} or {"type":"error","message":"..."}.
    """

    def event_stream() -> Iterator[bytes]:
        graph = build_travel_graph()
        merged = _merge_graph_initial(payload.state)
        step = 0
        last_snapshot: dict[str, Any] | None = None
        try:
            for item in graph.stream(
                merged,
                config={"recursion_limit": 120},
                stream_mode="values",
            ):
                step += 1
                if isinstance(item, tuple) and len(item) >= 2:
                    snapshot = item[-1]
                else:
                    snapshot = item
                if isinstance(snapshot, dict):
                    last_snapshot = snapshot
                yield _ndjson_line({"type": "progress", "step": step})
            if not isinstance(last_snapshot, dict):
                yield _ndjson_line({"type": "error", "message": "graph produced no state"})
                return
            yield _ndjson_line({"type": "done", "state": _state_for_client(last_snapshot)})
        except Exception as exc:
            log.exception("graph/stream failed")
            yield _ndjson_line({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
        log.exception("input_guard failed")
        raise HTTPException(status_code=500, detail=f"input_guard failed: {exc}") from exc


@app.post("/api/invoke/output_guard")
def invoke_output_guard(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8106, "output_guard")
    try:
        result = output_guard_agent(payload.state)
        return result
    except Exception as exc:
        log.exception("output_guard failed")
        raise HTTPException(status_code=500, detail=f"output_guard failed: {exc}") from exc


@app.post("/api/invoke/intent_profile")
def invoke_intent_profile(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8101, "intent_profile")
    state = payload.state
    try:
        result = intent_profile(state)
        return result
    except Exception as exc:
        log.exception("intent_profile failed")
        raise HTTPException(status_code=500, detail=f"intent_profile failed: {exc}") from exc



@app.post("/api/invoke/search")
def invoke_search(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8102, "search")
    state = payload.state
    try:
        result = research_agent(state)
        return result
    except Exception as exc:
        traceback.print_exc()
        log.exception("search (research_agent) failed")
        raise HTTPException(status_code=500, detail=f"search failed: {exc}") from exc


@app.post("/api/invoke/planner")
def invoke_planner(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8103, "planner")
    state = payload.state
    try:
        result = planner_agent(state)
        return result
    except Exception as exc:
        log.exception("planner failed")
        raise HTTPException(status_code=500, detail=f"planner failed: {exc}") from exc


@app.post("/api/invoke/debate")
def invoke_debate(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8104, "debate")
    state = payload.state
    try:
        result = debate_agent(state)
        return result
    except Exception as exc:
        log.exception("debate failed")
        raise HTTPException(status_code=500, detail=f"debate failed: {exc}") from exc


@app.post("/api/invoke/explain")
def invoke_explain(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8105, "explain")
    state = payload.state
    try:
        result = explainability_agent(state)
        return result
    except Exception as exc:
        log.exception("explain failed")
        raise HTTPException(status_code=500, detail=f"explain failed: {exc}") from exc


@app.post("/api/invoke/replanner")
def invoke_replanner(request: Request, payload: AgentInvokeRequest):
    _enforce_agent_port(request, 8107, "replanner")
    state = payload.state
    try:
        result = dynamic_replan_agent(state)
        return result
    except Exception as exc:
        log.exception("replanner failed")
        raise HTTPException(status_code=500, detail=f"replanner failed: {exc}") from exc


