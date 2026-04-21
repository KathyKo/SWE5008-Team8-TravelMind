"""
backend/routers/security.py — legacy compatibility endpoints for frontend

Provides:
  POST /travel/security/check
  POST /travel/security/check-output

These endpoints proxy requests to the standalone agents service so existing
frontend paths remain unchanged after backend router refactor.
"""

import os

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

AGENTS_BASE_URL = os.getenv("AGENTS_BASE_URL", "http://agents:8001").rstrip("/")


class SecurityRequest(BaseModel):
    text: str
    user_id: str | None = None


@router.post("/security/check")
def security_check(request: SecurityRequest):
    try:
        resp = requests.post(
            f"{AGENTS_BASE_URL}/security/check",
            json={"text": request.text, "user_id": request.user_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Agents security check error: {detail}") from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Agents service unavailable: {exc}") from exc


@router.post("/security/check-output")
def security_check_output(request: SecurityRequest):
    try:
        resp = requests.post(
            f"{AGENTS_BASE_URL}/security/check-output",
            json={"text": request.text, "user_id": request.user_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Agents output security check error: {detail}") from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Agents service unavailable: {exc}") from exc

@router.get("/health")
def health():
    return {"status": "ok", "router": "security-compat"}
