"""Debate specialist: DB-driven rounds with planner revision and judge fallback."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agents.db.crud import load_plan, update_plan_result
from agents.db.database import SessionLocal
from agents.db.models import Plan
from agents.llm_config import DEBATE_MODEL, DMX_API_KEY, DMX_BASE_URL, JUDGE_MODEL
from agents.specialists.planner_agent import revise_itinerary

MAX_DEBATE_ROUNDS = 3
DIMENSIONS = ("bias_fairness", "logistics", "preference_alignment", "option_diversity")

ROUND_CRITIQUE_SYSTEM_PROMPT = """
You are Agent4, a strict but constructive travel debate critic.
Evaluate itinerary options with exactly these 4 dimensions:
1) bias_fairness
2) logistics
3) preference_alignment
4) option_diversity

Return strict JSON:
{
  "round_decision": "continue" | "decide",
  "winner_option": "A" | "B" | "C" | null,
  "winner_reason": "<short reason>",
  "critique_summary": "<actionable critique to planner>",
  "dimension_scores": {
    "A": {"bias_fairness":0-100,"logistics":0-100,"preference_alignment":0-100,"option_diversity":0-100},
    "B": {"bias_fairness":0-100,"logistics":0-100,"preference_alignment":0-100,"option_diversity":0-100},
    "C": {"bias_fairness":0-100,"logistics":0-100,"preference_alignment":0-100,"option_diversity":0-100}
  }
}
If one option clearly wins, set round_decision=decide. Otherwise continue.
"""

ROUND_CRITIQUE_HUMAN_PROMPT = """
Round: {round_num}/{max_rounds}

Itineraries:
{itineraries_json}

User profile:
{user_profile_json}

Flights outbound:
{flight_out_json}
Flights return:
{flight_ret_json}

Hotels:
{hotel_json}

Debate history:
{history_json}
"""

JUDGE_SYSTEM_PROMPT = """
You are the final judge when max debate rounds are reached.
Use all context and score A/B/C on 4 dimensions:
- bias_fairness
- logistics
- preference_alignment
- option_diversity

Return strict JSON:
{
  "winner_option": "A" | "B" | "C",
  "winner_reason": "<short reason>",
  "dimension_scores": {
    "A": {"bias_fairness":0-100,"logistics":0-100,"preference_alignment":0-100,"option_diversity":0-100},
    "B": {"bias_fairness":0-100,"logistics":0-100,"preference_alignment":0-100,"option_diversity":0-100},
    "C": {"bias_fairness":0-100,"logistics":0-100,"preference_alignment":0-100,"option_diversity":0-100}
  }
}
"""


def _debate_llm(*, temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEBATE_MODEL,
        temperature=temperature,
        base_url=DMX_BASE_URL,
        api_key=DMX_API_KEY,
    )


def _judge_llm(*, temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=temperature,
        base_url=DMX_BASE_URL,
        api_key=DMX_API_KEY,
    )


def _extract_duration_days(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _safe_round_dim_scores(payload: dict) -> dict:
    scores = payload.get("dimension_scores") or {}
    sanitized: dict[str, dict[str, float]] = {}
    for option_key, option_scores in scores.items():
        if not isinstance(option_scores, dict):
            continue
        dims: dict[str, float] = {}
        for dim in DIMENSIONS:
            raw = option_scores.get(dim, 70)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = 70.0
            dims[dim] = max(0.0, min(100.0, val))
        dims["composite"] = round(sum(dims[d] for d in DIMENSIONS) / len(DIMENSIONS), 2)
        sanitized[str(option_key)] = dims
    return sanitized


def _winner_by_scores(scores: dict) -> Optional[str]:
    best_key: Optional[str] = None
    best_score = float("-inf")
    for key, dims in scores.items():
        composite = float((dims or {}).get("composite", 0.0))
        if composite > best_score:
            best_score = composite
            best_key = key
    return best_key


def _resolve_active_plan_id() -> Optional[str]:
    """
    Resolve debate target directly from DB to avoid state-input coupling.
    Priority:
    1) latest plan with no debate_verdict yet
    2) latest plan overall
    """
    db = SessionLocal()
    try:
        row = (
            db.query(Plan.plan_id)
            .filter(Plan.debate_verdict.is_(None))
            .order_by(Plan.created_at.desc())
            .first()
        )
        if row:
            return row[0]
        latest = db.query(Plan.plan_id).order_by(Plan.created_at.desc()).first()
        return latest[0] if latest else None
    finally:
        db.close()


def _build_round_critique_payload(plan_payload: dict, history: list[dict], round_num: int) -> dict:
    chain = (
        ChatPromptTemplate.from_messages(
            [("system", ROUND_CRITIQUE_SYSTEM_PROMPT), ("human", ROUND_CRITIQUE_HUMAN_PROMPT)]
        )
        | _debate_llm(temperature=0.2)
        | JsonOutputParser()
    )
    raw = chain.invoke(
        {
            "round_num": round_num,
            "max_rounds": MAX_DEBATE_ROUNDS,
            "itineraries_json": json.dumps(plan_payload.get("itineraries", {}), ensure_ascii=False, indent=2),
            "user_profile_json": json.dumps(
                {
                    "origin": plan_payload.get("origin"),
                    "destination": plan_payload.get("destination"),
                    "dates": plan_payload.get("dates"),
                    "duration_days": _extract_duration_days(plan_payload.get("duration")),
                    "budget": plan_payload.get("budget"),
                    "preferences": plan_payload.get("preferences"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "flight_out_json": json.dumps(plan_payload.get("flight_options_outbound", []), ensure_ascii=False),
            "flight_ret_json": json.dumps(plan_payload.get("flight_options_return", []), ensure_ascii=False),
            "hotel_json": json.dumps(plan_payload.get("hotel_options", []), ensure_ascii=False),
            "history_json": json.dumps(history, ensure_ascii=False, indent=2),
        }
    )
    raw["dimension_scores"] = _safe_round_dim_scores(raw)
    return raw


def _build_judge_payload(plan_payload: dict, history: list[dict]) -> dict:
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", JUDGE_SYSTEM_PROMPT),
                ("human", "Itineraries: {itineraries_json}\nUser profile: {user_profile_json}\nDebate history: {history_json}"),
            ]
        )
        | _judge_llm(temperature=0.1)
        | JsonOutputParser()
    )
    raw = chain.invoke(
        {
            "itineraries_json": json.dumps(plan_payload.get("itineraries", {}), ensure_ascii=False, indent=2),
            "user_profile_json": json.dumps(
                {
                    "origin": plan_payload.get("origin"),
                    "destination": plan_payload.get("destination"),
                    "dates": plan_payload.get("dates"),
                    "duration": plan_payload.get("duration"),
                    "budget": plan_payload.get("budget"),
                    "preferences": plan_payload.get("preferences"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "history_json": json.dumps(history, ensure_ascii=False, indent=2),
        }
    )
    raw["dimension_scores"] = _safe_round_dim_scores(raw)
    return raw


def _round_from_history(history: list[dict]) -> int:
    critic_rounds = [msg for msg in history if isinstance(msg, dict) and msg.get("sender") == "agent4_critic"]
    return len(critic_rounds) + 1


def _planner_state_from_plan_payload(plan_payload: dict) -> dict:
    return {
        "origin": plan_payload.get("origin"),
        "destination": plan_payload.get("destination"),
        "dates": plan_payload.get("dates"),
        "duration": plan_payload.get("duration"),
        "budget": plan_payload.get("budget"),
        "preferences": plan_payload.get("preferences"),
        "outbound_time_pref": "",
        "return_time_pref": "",
    }


def _merge_revised_plan(plan_payload: dict, revised: dict) -> dict:
    merged = deepcopy(plan_payload)
    for key in (
        "itineraries",
        "option_meta",
        "flight_options_outbound",
        "flight_options_return",
        "hotel_options",
        "tool_log",
        "planner_decision_trace",
        "chain_of_thought",
    ):
        if key in revised:
            merged[key] = revised.get(key)
    return merged


def _persist_plan_debate(
    *,
    plan_id: str,
    plan_payload: dict,
    debate_history: list[dict],
    debate_verdict: Optional[dict],
) -> None:
    db = SessionLocal()
    try:
        revised = {
            "itineraries": plan_payload.get("itineraries", {}),
            "option_meta": plan_payload.get("option_meta", {}),
            "flight_options_outbound": plan_payload.get("flight_options_outbound", []),
            "flight_options_return": plan_payload.get("flight_options_return", []),
            "hotel_options": plan_payload.get("hotel_options", []),
            "tool_log": plan_payload.get("tool_log", []),
            "planner_decision_trace": plan_payload.get("planner_decision_trace", {}),
            "chain_of_thought": plan_payload.get("chain_of_thought", ""),
            "debate_history": debate_history,
            "debate_verdict": debate_verdict,
        }
        update_plan_result(db, plan_id, revised)
    finally:
        db.close()


def debate_agent(_state: dict) -> dict:
    """
    Debate Agent entrypoint.
    Input source is DB only (to minimize state-coupling and cross-turn mismatch).
    """
    plan_id = _resolve_active_plan_id()
    if not plan_id:
        return {
            "is_valid": False,
            "debate_count": 0,
            "debate_output": {"error": "No plan found in DB for debate."},
            "next_node": "orchestrator",
        }

    db = SessionLocal()
    try:
        plan_payload = load_plan(db, plan_id)
    finally:
        db.close()

    if not plan_payload:
        return {
            "plan_id": plan_id,
            "is_valid": False,
            "debate_count": 0,
            "debate_output": {"error": f"Plan '{plan_id}' not found in agents/db."},
            "next_node": "orchestrator",
        }

    debate_history = deepcopy(plan_payload.get("debate_history") or [])
    round_num = _round_from_history(debate_history)

    if round_num > MAX_DEBATE_ROUNDS:
        existing_verdict = plan_payload.get("debate_verdict")
        return {
            "plan_id": plan_id,
            "is_valid": bool(existing_verdict),
            "debate_count": MAX_DEBATE_ROUNDS,
            "debate_output": {
                "debate_history": debate_history,
                "debate_verdict": existing_verdict,
                "current_round_summary": "Debate rounds already completed.",
            },
            "next_node": "orchestrator",
        }

    try:
        critique_payload = _build_round_critique_payload(plan_payload, debate_history, round_num)
    except Exception as e:
        critique_payload = {
            "round_decision": "continue",
            "winner_option": None,
            "winner_reason": f"Round critique failed: {e}",
            "critique_summary": f"Round critique failed: {e}",
            "dimension_scores": {},
        }

    critique_summary = str(critique_payload.get("critique_summary", "")).strip()
    debate_history.append(
        {
            "round": round_num,
            "sender": "agent4_critic",
            "content": {
                "critique_summary": critique_summary,
                "winner_hint": critique_payload.get("winner_option"),
                "round_decision": critique_payload.get("round_decision"),
                "dimension_scores": critique_payload.get("dimension_scores", {}),
            },
        }
    )

    round_decision = str(critique_payload.get("round_decision", "continue")).lower()
    winner_option = critique_payload.get("winner_option")
    scores = critique_payload.get("dimension_scores", {})
    immediate_win = round_decision == "decide" and winner_option in plan_payload.get("itineraries", {})

    if immediate_win:
        final_winner = str(winner_option)
        debate_verdict = {
            "accepted": True,
            "final_round": round_num,
            "winner_option": final_winner,
            "winner_reason": critique_payload.get("winner_reason", "Debate converged before max rounds."),
            "dimension_scores": scores,
            "selected_plan": (plan_payload.get("itineraries") or {}).get(final_winner),
            "selected_option_meta": (plan_payload.get("option_meta") or {}).get(final_winner),
            "via_judge": False,
        }
        _persist_plan_debate(
            plan_id=plan_id,
            plan_payload=plan_payload,
            debate_history=debate_history,
            debate_verdict=debate_verdict,
        )
        return {
            "plan_id": plan_id,
            "is_valid": True,
            "debate_count": round_num,
            "debate_output": {
                "debate_history": debate_history,
                "debate_verdict": debate_verdict,
                "current_round_summary": critique_summary,
            },
            "next_node": "orchestrator",
        }

    # Continue debate if rounds remain: trigger real Planner (Agent3) revision via LLM.
    if round_num < MAX_DEBATE_ROUNDS:
        planner_state = _planner_state_from_plan_payload(plan_payload)
        revised = revise_itinerary(planner_state, critique_summary, plan_payload)
        planner_reply = {
            "stance": "accept_most" if "error" not in revised else "mixed",
            "action": "revise" if "error" not in revised else "keep_due_to_error",
            "message": (
                revised.get("planner_chain_of_thought")
                if isinstance(revised, dict)
                else ""
            )
            or ("Planner revision generated." if "error" not in revised else str(revised.get("error"))),
        }
        debate_history.append(
            {
                "round": round_num,
                "sender": "agent3_response",
                "content": planner_reply,
            }
        )
        if isinstance(revised, dict) and "error" not in revised:
            plan_payload = _merge_revised_plan(plan_payload, revised)
        _persist_plan_debate(
            plan_id=plan_id,
            plan_payload=plan_payload,
            debate_history=debate_history,
            debate_verdict=None,
        )
        return {
            "plan_id": plan_id,
            "is_valid": False,
            "debate_count": round_num,
            "debate_output": {
                "debate_history": debate_history,
                "debate_verdict": None,
                "current_round_summary": critique_summary,
            },
            "next_node": "orchestrator",
        }

    # Max rounds reached: judge decides.
    judge_payload: Optional[dict] = None
    try:
        judge_payload = _build_judge_payload(plan_payload, debate_history)
    except Exception:
        fallback_winner = _winner_by_scores(scores) or "A"
        judge_payload = {
            "winner_option": fallback_winner,
            "winner_reason": "Judge fallback selected highest composite score from latest round.",
            "dimension_scores": scores,
        }

    final_winner = str(judge_payload.get("winner_option", "A"))
    debate_verdict = {
        "accepted": True,
        "final_round": round_num,
        "winner_option": final_winner,
        "winner_reason": judge_payload.get("winner_reason", "Judge selected final option after max rounds."),
        "dimension_scores": judge_payload.get("dimension_scores", {}),
        "selected_plan": (plan_payload.get("itineraries") or {}).get(final_winner),
        "selected_option_meta": (plan_payload.get("option_meta") or {}).get(final_winner),
        "via_judge": True,
    }
    _persist_plan_debate(
        plan_id=plan_id,
        plan_payload=plan_payload,
        debate_history=debate_history,
        debate_verdict=debate_verdict,
    )
    return {
        "plan_id": plan_id,
        "is_valid": True,
        "debate_count": round_num,
        "debate_output": {
            "debate_history": debate_history,
            "debate_verdict": debate_verdict,
            "current_round_summary": critique_summary,
        },
        "next_node": "orchestrator",
    }

