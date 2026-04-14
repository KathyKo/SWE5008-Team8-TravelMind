"""
backend/routers/explainability.py — Agent6 (Explainability) endpoints

POST /explainability/run  → explainability_agent(state)
                            Generates natural-language summaries + item explanations
                            for a previously generated plan
GET  /explainability/health
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from backend.agent_client import call_explainability
from backend.db.database import get_db
from backend.db import crud

router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────

class ExplainRequest(BaseModel):
    """
    Input to Agent6 (Explainability).
    Requires a plan_id returned by POST /planner/run.
    Optionally specify which itinerary option to explain (default: A).
    """
    plan_id:        str
    explain_option: Optional[str] = None   # "A", "B", or "C" — defaults to "A"


class ExplainResponse(BaseModel):
    """
    Output from Agent6 (Explainability).

    summary        : overall_summary + per-day day_summaries (natural language)
    item_explanations : per-item metadata keyed by item_key and occurrence_id
    chain_of_thought  : full planning + explainability reasoning text
    agent_steps       : tool execution log for display in frontend
    """
    plan_id:           str
    explain_option:    str
    summary:           dict   # {"overall_summary": "...", "day_summaries": {"Day 1": "..."}}
    item_explanations: dict   # {"by_key": {...}, "by_occurrence": {...}}
    chain_of_thought:  str
    agent_steps:       list


# ── Endpoints ──────────────────────────────────────────────────

@router.post("/run", response_model=ExplainResponse)
def run_explainability(request: ExplainRequest, db: Session = Depends(get_db)):
    """
    Agent6 (explainability_agent) — explains the selected itinerary option.

    Input  : plan_id (from /planner/run) + optional explain_option ("A"/"B"/"C")
    Output : natural-language summaries + per-item explanations + chain of thought

    Reads the planner's decision trace from DB to generate trace-based explanations.
    Results are saved back to DB (idempotent — re-running updates the stored explain).
    """
    cached = crud.load_plan(db, request.plan_id)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail=f"Plan '{request.plan_id}' not found. Call /planner/run first."
        )

    # Inject the requested option so Agent6 knows which to explain
    if request.explain_option:
        cached["explain_option"] = request.explain_option.upper()

    print(f"[Agent6] Explainability for plan {request.plan_id} "
          f"option {request.explain_option or 'A'}")
    result = call_explainability(cached)

    if "error" in result:
        raise HTTPException(status_code=500, detail=f"Agent6 error: {result['error']}")

    crud.save_explain(db, request.plan_id, result)

    return ExplainResponse(
        plan_id           = request.plan_id,
        explain_option    = result.get("explain_option",   request.explain_option or "A"),
        summary           = result.get("summary",          {}),
        item_explanations = result.get("item_explanations",{}),
        chain_of_thought  = result.get("chain_of_thought", ""),
        agent_steps       = result.get("agent_steps",      []),
    )


@router.get("/health")
def health():
    return {"status": "ok", "agent": "Agent6-Explainability"}
