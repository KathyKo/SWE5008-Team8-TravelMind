"""
backend/routers/agent_proxy.py — frontend-safe proxy endpoints

Forces frontend traffic through backend:
  frontend -> backend (/agent/*) -> agents service
"""

from __future__ import annotations

import os
from typing import Any

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

AGENTS_BASE_URL = os.getenv("AGENTS_BASE_URL", "http://agents:8001").rstrip("/")
AGENTS_GRAPH_STREAM_URL = os.getenv(
    "AGENTS_GRAPH_STREAM_URL",
    f"{AGENTS_BASE_URL}/api/invoke/graph/stream",
).rstrip("/")
AGENTS_FAIRNESS_CHECK_URL = os.getenv(
    "AGENTS_FAIRNESS_CHECK_URL",
    f"{AGENTS_BASE_URL}/api/invoke/fairness-check",
).rstrip("/")
AGENTS_REPLAN_URL = os.getenv(
    "AGENTS_REPLAN_URL",
    "http://agents:8107/api/invoke/replanner",
).rstrip("/")


class GraphStreamRequest(BaseModel):
    state: dict[str, Any]


class FairnessCheckRequest(BaseModel):
    state: dict[str, Any]
    selected_option: str | None = None


class ReplanRequest(BaseModel):
    state: dict[str, Any]


@router.post("/graph/stream")
def proxy_graph_stream(request: GraphStreamRequest):
    try:
        upstream = requests.post(
            AGENTS_GRAPH_STREAM_URL,
            json={"state": request.state},
            stream=True,
            timeout=600,
        )
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Agents graph stream unavailable: {exc}") from exc

    if upstream.status_code != 200:
        detail = upstream.text
        upstream.close()
        raise HTTPException(status_code=502, detail=f"Agents graph stream error: {detail}")

    def _iter_bytes():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return StreamingResponse(_iter_bytes(), media_type="application/x-ndjson")


@router.post("/fairness-check")
def proxy_fairness_check(request: FairnessCheckRequest):
    payload = {"state": request.state}
    if request.selected_option:
        payload["selected_option"] = request.selected_option
    try:
        resp = requests.post(
            AGENTS_FAIRNESS_CHECK_URL,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Agents fairness-check error: {detail}") from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Agents fairness-check unavailable: {exc}") from exc


@router.post("/replan")
def proxy_replan(request: ReplanRequest):
    try:
        resp = requests.post(
            AGENTS_REPLAN_URL,
            json={"state": request.state},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Agents replan error: {detail}") from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Agents replan unavailable: {exc}") from exc

