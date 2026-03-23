"""
agents/output_guard_agent.py — OutputGuardAgent (Agent 5b)

Runs after all planning agents, before the response reaches the user.

Two-layer output validation:
  Layer 1 — Regex + rule-based checks (fast, deterministic)
      - Unsafe keyword detection (war zones, illegal activities)
      - Fabricated flight number detection (hallucination guard)

  Layer 2 — LLM Guard output scanners (ML-based)
      - Sensitive: detect PII leaking in agent responses
      - BanTopics: semantic check for unsafe travel content

Full pipeline:
  1. Extract last assistant message
  2. Layer 1: Regex hallucination + unsafe content check
  3. Layer 1: Regex PII scan (SG-specific patterns)
  4. Layer 2: LLM Guard Sensitive + BanTopics scan
  5. Return validated (and possibly redacted) output

LangGraph routing:
  [specialists] → output_guard → output_flagged  (issue found)
                               → output_safe      (output clean)
"""

from agents.state import State
from tools.hallucination_guard import check_hallucination
from tools.pii_scanner import scan_pii, has_high_risk_pii
from tools.llm_guard_scanner import scan_output_llm_guard
from tools.security_logger import log_event, EventType


AGENT_NAME = "agent_5b_output_guard"

UNSAFE_KEYWORDS = [
    "active war zone", "level 4 travel advisory", "do not travel",
    "illegal activity", "smuggling", "drug trafficking",
]

FLAGGED_RESPONSE = (
    "I encountered an issue preparing your travel information. "
    "Please try rephrasing your request."
)

def output_guard_agent(state: State) -> dict:
    """Compatibility alias for callers expecting `output_guard_agent` as node function."""
    return output_guard_node(state)

def _check_unsafe_content(text: str) -> tuple[bool, str]:
    lowered = text.lower()
    for kw in UNSAFE_KEYWORDS:
        if kw in lowered:
            return True, f"Unsafe content keyword detected: '{kw}'"
    return False, ""


def output_guard_node(state: State) -> dict:
    """
    LangGraph node for OutputGuardAgent (Agent 5b).

    Validates the last assistant message before it reaches the user.
    """
    messages = state.get("messages", [])
    user_id = state.get("user_id")

    # Find last assistant message
    last_assistant_msg = None
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if (isinstance(msg, dict) and msg.get("role") == "assistant"):
            last_assistant_msg = msg
            last_assistant_idx = i
            break

    if last_assistant_msg is None:
        return {"output_flagged": False}

    output_text = last_assistant_msg.get("content", "")
    # Get original user prompt for LLM Guard context
    user_prompt = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_prompt = msg.get("content", "")

    print(f"\n[AGENT 5b] OutputGuardAgent validating response ({len(output_text)} chars)...")

    # ── Layer 1: Hallucination check ──────────────────────────
    halluc_result = check_hallucination(output_text)
    if halluc_result.has_hallucination:
        log_event(
            EventType.OUTPUT_FLAGGED, AGENT_NAME,
            {
                "layer": "regex",
                "reason": "hallucination_warning",
                "flagged_entities": halluc_result.flagged_entities,
            },
            user_id,
        )
        # Advisory warning — log but don't block (Research Agent is primary owner)
        print(f"[AGENT 5b] ⚠️  Hallucination warning: {halluc_result.reason}")

    # ── Layer 1: Regex PII scan ───────────────────────────────
    pii_result = scan_pii(output_text)

    if has_high_risk_pii(output_text):
        log_event(
            EventType.OUTPUT_PII_REDACTED, AGENT_NAME,
            {"layer": "regex", "reason": "high_risk_pii_in_output", "findings": pii_result.findings},
            user_id,
        )
        print(f"[AGENT 5b] ⛔ FLAGGED (Layer 1 — Regex PII): High-risk PII in output")
        return {
            "output_flagged": True,
            "output_flag_reason": "pii_leakage",
            "messages": messages[:last_assistant_idx] + [
                {"role": "assistant", "content": FLAGGED_RESPONSE}
            ],
        }

    if pii_result.has_pii:
        output_text = pii_result.redacted_text
        log_event(
            EventType.OUTPUT_PII_REDACTED, AGENT_NAME,
            {"layer": "regex", "reason": "medium_pii_redacted", "findings": pii_result.findings},
            user_id,
        )
        print(f"[AGENT 5b] ℹ️  Medium-risk PII redacted (regex layer)")

    # ── Layer 1: Unsafe content check ────────────────────────
    is_unsafe, unsafe_reason = _check_unsafe_content(output_text)
    if is_unsafe:
        log_event(
            EventType.OUTPUT_FLAGGED, AGENT_NAME,
            {"layer": "regex", "reason": "unsafe_content", "detail": unsafe_reason},
            user_id,
        )
        print(f"[AGENT 5b] ⛔ FLAGGED (Layer 1 — Regex): {unsafe_reason}")
        return {
            "output_flagged": True,
            "output_flag_reason": "unsafe_content",
            "messages": messages[:last_assistant_idx] + [
                {"role": "assistant", "content": FLAGGED_RESPONSE}
            ],
        }

    # ── Layer 2: LLM Guard output scan ────────────────────────
    llm_guard_result = scan_output_llm_guard(user_prompt, output_text)

    if not llm_guard_result.is_safe:
        log_event(
            EventType.OUTPUT_FLAGGED, AGENT_NAME,
            {
                "layer": "llm_guard",
                "reason": llm_guard_result.reason,
                "flags": llm_guard_result.flags,
                "risk_score": llm_guard_result.risk_score,
            },
            user_id,
        )
        print(f"[AGENT 5b] ⛔ FLAGGED (Layer 2 — LLM Guard): {llm_guard_result.reason}")
        return {
            "output_flagged": True,
            "output_flag_reason": f"llm_guard:{','.join(llm_guard_result.flags)}",
            "messages": messages[:last_assistant_idx] + [
                {"role": "assistant", "content": FLAGGED_RESPONSE}
            ],
        }

    # Use LLM Guard's sanitised output (may have redacted additional PII)
    output_text = llm_guard_result.sanitised_text

    # ── All checks passed ─────────────────────────────────────
    log_event(
        EventType.OUTPUT_PASSED, AGENT_NAME,
        {"message_length": len(output_text)},
        user_id,
    )
    print("[AGENT 5b] ✅ Output passed both layers")

    updated_messages = list(messages)
    updated_messages[last_assistant_idx] = {
        **last_assistant_msg,
        "content": output_text,
    }

    return {
        "output_flagged": False,
        "output_flag_reason": None,
        "messages": updated_messages,
    }


def output_guard_routing(state: State) -> str:
    """
    Conditional edge for LangGraph.
    Returns 'output_flagged' or 'output_safe'.
    """
    if state.get("output_flagged", False):
        return "output_flagged"
    return "output_safe"


