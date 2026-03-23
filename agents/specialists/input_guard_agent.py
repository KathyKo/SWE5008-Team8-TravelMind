"""
agents/input_guard_agent.py — InputGuardAgent (Agent 5a)

First line of defence. Runs immediately after the human node,
before any planning agent sees the user's input.

Two-layer injection detection:
  Layer 1 — Regex (tools/injection_detector.py)
      Fast pattern matching against known injection templates.
      Zero latency, no external calls. Catches obvious attacks immediately.

  Layer 2 — LLM Guard (tools/llm_guard_scanner.py)
      ML-based semantic scanner. Only runs if Layer 1 passes.
      Catches obfuscated or novel phrasings that regex misses.

Full pipeline:
  1. Sanitise input (normalise unicode, strip control chars, truncate)
  2. Layer 1: Regex injection detection   → block on match
  3. Layer 2: LLM Guard injection scan    → block if flagged
  4. PII scan (regex + LLM Guard Sensitive) → redact or block
  5. Pass sanitised input to Agent 1 (Intent & Profile)

LangGraph routing:
  human → input_guard → threat_blocked  (threat detected)
                      → intent_profile  (input clean)
"""

from agents.state import State
from tools.sanitiser import sanitise
from tools.injection_detector import detect_injection_regex
from tools.llm_guard_scanner import scan_input_llm_guard
from tools.pii_scanner import scan_pii, has_high_risk_pii
from tools.security_logger import log_event, EventType


AGENT_NAME = "input_guard_agent"
MAX_INPUT_LENGTH = 2000 

BLOCKED_RESPONSE = (
    "I'm sorry, I'm unable to process that request. "
    "Please describe your travel plans and I'll be happy to help."
)

def input_guard_agent(state: State) -> dict:
    """Compatibility alias for callers expecting `input_guard_agent` as node function."""
    return input_guard_node(state)

def input_guard_node(state: State) -> dict:
    """
    LangGraph node for InputGuardAgent (Agent 5a).

    Reads the last user message, runs the two-layer security pipeline,
    and returns an updated state dict.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"threat_blocked": False}

    last_msg = messages[-1]
    raw_input = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
    user_id = state.get("user_id")

    print(f"\n[AGENT 5a] InputGuardAgent scanning input ({len(raw_input)} chars)...")

    # ── Step 1: Length check ──────────────────────────────────
    # Oversized inputs are rejected outright — not truncated.
    # No legitimate travel query requires more than 2000 characters.
    # Silent truncation could allow malicious content hidden past the
    # cutoff point to survive; explicit rejection is safer and clearer.
    if len(raw_input) > MAX_INPUT_LENGTH:
        log_event(
            EventType.INPUT_BLOCKED, AGENT_NAME,
            {"reason": "oversized_input", "length": len(raw_input), "limit": MAX_INPUT_LENGTH},
            user_id,
        )
        print(f"[AGENT 5a] ⛔ BLOCKED — Input length {len(raw_input)} exceeds {MAX_INPUT_LENGTH} characters")
        return {
            "threat_blocked": True,
            "threat_type": "oversized_input",
            "messages": [{"role": "assistant", "content": BLOCKED_RESPONSE}],
        }

    # ── Step 2: Sanitise ──────────────────────────────────────
    clean_text = sanitise(raw_input)

    # ── Step 3: Layer 1 — Regex injection detection ───────────
    regex_result = detect_injection_regex(clean_text)

    if regex_result.is_injection:
        log_event(
            EventType.INPUT_BLOCKED, AGENT_NAME,
            {
                "layer": "regex",
                "reason": regex_result.reason,
                "threat_type": regex_result.threat_type,
                "matched_pattern": regex_result.matched_pattern,
                "confidence": regex_result.confidence,
            },
            user_id,
        )
        print(f"[AGENT 5a] ⛔ BLOCKED (Layer 1 — Regex): {regex_result.reason}")
        return {
            "threat_blocked": True,
            "threat_type": regex_result.threat_type,
            "messages": [{"role": "assistant", "content": BLOCKED_RESPONSE}],
        }

    # ── Step 4: Layer 2 — LLM Guard injection scan ───────────
    llm_guard_result = scan_input_llm_guard(clean_text)

    if not llm_guard_result.is_safe:
        log_event(
            EventType.INPUT_BLOCKED, AGENT_NAME,
            {
                "layer": "llm_guard",
                "reason": llm_guard_result.reason,
                "threat_type": llm_guard_result.threat_type,
                "risk_score": llm_guard_result.risk_score,
            },
            user_id,
        )
        print(f"[AGENT 5a] ⛔ BLOCKED (Layer 2 — LLM Guard): {llm_guard_result.reason}")
        return {
            "threat_blocked": True,
            "threat_type": llm_guard_result.threat_type,
            "messages": [{"role": "assistant", "content": BLOCKED_RESPONSE}],
        }

    # Use LLM Guard's sanitised text (may have PII redacted by LLM Guard PII scanner)
    clean_text = llm_guard_result.sanitised_text

    # ── Step 5: PII scan ─────────────────────────────────────
    # LLM Guard's PII scanner already ran in Step 3 (redact=True).
    # Regex PII scan here as an additional layer for SG-specific patterns
    # (NRIC, SG phone) that general NER models may miss.
    pii_result = scan_pii(clean_text)

    if has_high_risk_pii(clean_text):
        log_event(
            EventType.INPUT_PII_DETECTED, AGENT_NAME,
            {"reason": "high_risk_pii_in_input", "findings": pii_result.findings},
            user_id,
        )
        print(f"[AGENT 5a] ⛔ BLOCKED — High-risk PII detected in input")
        return {
            "threat_blocked": True,
            "threat_type": "pii_probe",
            "messages": [{"role": "assistant", "content": BLOCKED_RESPONSE}],
        }

    if pii_result.has_pii:
        log_event(
            EventType.INPUT_PII_DETECTED, AGENT_NAME,
            {"reason": "medium_risk_pii_redacted", "findings": pii_result.findings},
            user_id,
        )
        clean_text = pii_result.redacted_text
        print(f"[AGENT 5a] ℹ️  Medium-risk PII redacted from input")

    # ── Step 6: Pass through ──────────────────────────────────
    log_event(
        EventType.INPUT_PASSED, AGENT_NAME,
        {"message_length": len(clean_text)},
        user_id,
    )
    print("[AGENT 5a] ✅ Input passed both layers — routing to Intent & Profile Agent")

    updated_messages = messages[:-1] + [
        {**last_msg, "content": clean_text}
        if isinstance(last_msg, dict)
        else {"role": "user", "content": clean_text}
    ]

    return {
        "threat_blocked": False,
        "threat_type": None,
        "messages": updated_messages,
    }


def input_guard_routing(state: State) -> str:
    """
    Conditional edge for LangGraph.
    Returns 'threat_blocked' or 'intent_profile'.
    """
    if state.get("threat_blocked", False):
        return "threat_blocked"
    return "intent_profile"
