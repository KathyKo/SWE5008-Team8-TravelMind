"""
backend/routers/graph_proxy.py — /api/invoke/* proxy endpoints

Exposes the agents service's LangGraph streaming endpoint under the backend,
so the frontend can reach it via BACKEND_URL (port 8000) instead of needing
direct network access to the agents service (port 8001).

Routes (mounted under prefix="/api/invoke"):
  POST /api/invoke/graph/stream  -> proxies to agents /api/invoke/graph/stream
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

GRAPH_STREAM_TIMEOUT = int(os.getenv("AGENTS_GRAPH_STREAM_TIMEOUT", "600"))


class GraphStreamRequest(BaseModel):
    state: dict[str, Any]


@router.post("/graph/stream")
def proxy_graph_stream(request: GraphStreamRequest):
    """
    Stream-forward the LangGraph orchestrator output from the agents service.

    The agents endpoint emits NDJSON lines
    ({"type":"progress",...} / {"type":"done",...} / {"type":"error",...});
    we stream those bytes back to the caller unchanged.
    """
    try:
        upstream = requests.post(
            AGENTS_GRAPH_STREAM_URL,
            json={"state": request.state},
            stream=True,
            timeout=GRAPH_STREAM_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Agents graph stream unavailable: {exc}",
        ) from exc

    if upstream.status_code != 200:
        detail = upstream.text
        upstream.close()
        raise HTTPException(
            status_code=502,
            detail=f"Agents graph stream error: {detail}",
        )

    def _iter_bytes():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return StreamingResponse(
        _iter_bytes(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
