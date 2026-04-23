"""
backend/routers/fairness.py — frontend-facing fairness check endpoint

Provides:
  POST /fairness/check-selected-option

This endpoint proxies the selected-option fairness check to the agents service.
"""

import os

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

AGENTS_BASE_URL = os.getenv("AGENTS_BASE_URL", "http://agents:8001").rstrip("/")


class SelectedOptionFairnessRequest(BaseModel):
    state: dict
    selected_option: str


@router.post("/check-selected-option")
def check_selected_option_fairness(request: SelectedOptionFairnessRequest):
    try:
        resp = requests.post(
            f"{AGENTS_BASE_URL}/api/invoke/fairness-check",
            json={"state": request.state, "selected_option": request.selected_option},
            timeout=120,
        )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                payload = resp.json()
                if isinstance(payload, dict):
                    detail = payload.get("detail", detail)
            except Exception:
                pass
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Agents fairness-check error: {detail}",
            )
        return resp.json()
    except HTTPException:
        raise
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Agents service unavailable: {exc}") from exc


@router.get("/health")
def health():
    return {"status": "ok", "router": "fairness"}
