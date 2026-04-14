"""
agents/input_guard_agent.py — input_guard_agent

Purpose:
    Input-side security gate. Runs immediately after human input and before
    any planning/specialist agent consumes user text.

Validation flow:
    Step 1: Length policy check (hard block if oversized)
    Step 2: Input sanitisation
    Step 3: LLM planner (LangChain tool-calling) selects adaptive checks
    Step 4: Regex injection detection (mandatory)
    Step 5: PII scan/redaction (adaptive, SG-specific + generic)
    Step 6: LLM Guard injection scan (mandatory, runs on cleaned text)
    Step 7: OpenAI moderation (adaptive)
    Step 8: Pass through with decision trace

Configuration sources:
    - Prompts: agents/prompts/security_prompts.yaml
    - Runtime controls: ENABLE_SECURITY_LLM_PLANNER, INPUT_GUARD_PROMPT_VERSION

Routing:
    human → input_guard → threat_blocked
                        → intent_profile

Design decision (v2):
    PII detection moved before LLM Guard to correctly identify PII probe attacks
    (e.g., "What is X of user Y?") as PII threats rather than injection threats.
    This improves threat classification accuracy and reduces false positives.
"""

import os
import importlib
from pathlib import Path
from typing import Any

from agents.llm_config import OPENAI_MODEL
from agents.state import State
from tools.security.sanitiser import sanitise
from tools.security.injection_detector import detect_injection_regex
from tools.security.llm_guard_scanner import scan_input_llm_guard
from tools.security.openai_moderation import check_moderation
from tools.security.pii_scanner import scan_pii, has_high_risk_pii
from tools.security.security_logger import log_event, EventType


AGENT_NAME = "input_guard_agent"
MAX_INPUT_LENGTH = 2000 
ENABLE_LLM_PLANNER = os.getenv("ENABLE_SECURITY_LLM_PLANNER", "1") == "1"
INPUT_GUARD_PROMPT_VERSION = os.getenv("INPUT_GUARD_PROMPT_VERSION", "v1")
PROMPTS_FILE = Path(__file__).resolve().parents[1] / "prompts" / "security_prompts.yaml"
DECISION_SCHEMA_VERSION = "v1"

BLOCKED_RESPONSE = (
    "I'm sorry, I'm unable to process that request. "
    "Please describe your travel plans and I'll be happy to help."
)


def _get_prompt(name: str, version: str, fallback: str) -> str:
    try:
        yaml = importlib.import_module("yaml")
        with PROMPTS_FILE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("input_guard", {}).get(name, {}).get(version, fallback)
    except Exception:
        return fallback


def _make_decision(action: str, reason: str, confidence: float, evidence: dict[str, Any]) -> dict[str, Any]:
    """Structured decision trace to make this node auditable as a security agent."""
    return {
        "agent": AGENT_NAME,
        "goal": "maximize input safety while minimizing false positives",
        "action": action,  # allow | block | redact_then_allow
        "reason": reason,
        "confidence": confidence,
        "schema_version": DECISION_SCHEMA_VERSION,
        "evidence": evidence,
    }


def _plan_input_tools(text: str) -> dict[str, Any]:
    """Use LangChain tool-calling to select an adaptive input-scan plan."""
    default_plan = {
        "risk_level": "high",
        "run_pii_scan": True,
        "run_moderation": True,
        "reason": "fallback_full_scan",
    }
    if not ENABLE_LLM_PLANNER:
        return default_plan

    system_prompt = _get_prompt(
        "input_tool_planner_system",
        INPUT_GUARD_PROMPT_VERSION,
        (
            "You are a security triage planner for a travel assistant. "
            "Given user input, decide which deep safety tools must run. "
            "Return strict JSON with keys: risk_level(low|medium|high), run_pii_scan(boolean), "
            "run_moderation(boolean), reason(string). "
            "Select exactly one tool call based on risk: "
            "plan_full_scan for uncertain/risky/adversarial input; "
            "plan_low_risk_fast_path for clearly benign input; "
            "plan_low_risk_with_moderation for benign input that still needs policy screening. "
            "When unsure, always choose plan_full_scan."
        ),
    )

    try:
        tool = getattr(importlib.import_module("langchain_core.tools"), "tool")
        HumanMessage = getattr(importlib.import_module("langchain_core.messages"), "HumanMessage")
        SystemMessage = getattr(importlib.import_module("langchain_core.messages"), "SystemMessage")
        ChatOpenAI = getattr(importlib.import_module("langchain_openai"), "ChatOpenAI")

        @tool
        def plan_full_scan(reason: str = "default_conservative") -> dict:
            """Use full security checks: run PII scan and OpenAI moderation."""
            return {
                "risk_level": "high",
                "run_pii_scan": True,
                "run_moderation": True,
                "reason": reason,
            }

        @tool
        def plan_low_risk_fast_path(reason: str = "low_risk_fast_path") -> dict:
            """Use fast path: skip extra PII regex scan and moderation check."""
            return {
                "risk_level": "low",
                "run_pii_scan": False,
                "run_moderation": False,
                "reason": reason,
            }

        @tool
        def plan_low_risk_with_moderation(reason: str = "low_risk_moderation_only") -> dict:
            """Low-risk path that still runs moderation; skip extra regex PII scan."""
            return {
                "risk_level": "low",
                "run_pii_scan": False,
                "run_moderation": True,
                "reason": reason,
            }

        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        planner = llm.bind_tools(
            [plan_full_scan, plan_low_risk_fast_path, plan_low_risk_with_moderation],
            tool_choice="required",
        )
        ai_msg = planner.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User input:\n{text}"),
        ])

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            return default_plan

        chosen = tool_calls[0]
        if isinstance(chosen, dict):
            tool_name = chosen.get("name", "")
            args = chosen.get("args", {}) or {}
        else:
            tool_name = getattr(chosen, "name", "")
            args = getattr(chosen, "args", {}) or {}
        reason = str(args.get("reason", "llm_tool_choice")) if isinstance(args, dict) else "llm_tool_choice"

        if tool_name == "plan_low_risk_fast_path":
            return {
                "risk_level": "low",
                "run_pii_scan": False,
                "run_moderation": False,
                "reason": reason,
            }
        if tool_name == "plan_low_risk_with_moderation":
            return {
                "risk_level": "low",
                "run_pii_scan": False,
                "run_moderation": True,
                "reason": reason,
            }
        return {
            "risk_level": "high",
            "run_pii_scan": True,
            "run_moderation": True,
            "reason": reason,
        }
    except Exception:
        return default_plan

def input_guard_agent(state: State) -> dict:
    """Compatibility alias for callers expecting `input_guard_agent` as node function."""
    return input_guard_node(state)

def input_guard_node(state: State) -> dict:
    """
    LangGraph node for `input_guard_agent`.

    Reads the latest user message, executes security checks,
    and returns updated state with decision trace.
    """
    messages = state.get("messages", [])
    if not messages:
        return {
            "threat_blocked": False,
            "input_guard_decision": _make_decision(
                action="allow",
                reason="no_messages",
                confidence=1.0,
                evidence={"message_count": 0},
            ),
        }

    last_msg = messages[-1]
    raw_input = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
    user_id = state.get("user_id")

    print(f"\n[input_guard_agent] scanning input ({len(raw_input)} chars)...")

    # ── Step 1: Length check ──────────────────────────────────
    # Oversized inputs are rejected outright — not truncated.
    # No legitimate travel query requires more than 2000 characters.
    # Silent truncation could allow malicious content hidden past the
    # cutoff point to survive; explicit rejection is safer and clearer.
    if len(raw_input) > MAX_INPUT_LENGTH:
        decision = _make_decision(
            action="block",
            reason="oversized_input",
            confidence=1.0,
            evidence={"length": len(raw_input), "limit": MAX_INPUT_LENGTH},
        )
        log_event(
            EventType.INPUT_BLOCKED, AGENT_NAME,
            {"reason": "oversized_input", "length": len(raw_input), "limit": MAX_INPUT_LENGTH},
            user_id,
        )
        print(f"[input_guard_agent] ⛔ BLOCKED — Input length {len(raw_input)} exceeds {MAX_INPUT_LENGTH} characters")
        return {
            "threat_blocked": True,
            "threat_type": "oversized_input",
            "threat_detail": "Input exceeds maximum length policy",
            "input_guard_decision": decision,
            "messages": [{"role": "assistant", "content": BLOCKED_RESPONSE}],
        }

    # ── Step 2: Sanitise ──────────────────────────────────────
    clean_text = sanitise(raw_input)

    # ── Step 3: LLM planning for tool invocation ────────────
    tool_plan = _plan_input_tools(clean_text)
    run_pii_scan = tool_plan["run_pii_scan"]
    run_moderation = tool_plan["run_moderation"]

    # Safety floor: never disable deep checks on medium/high risk.
    if tool_plan["risk_level"] in {"medium", "high"}:
        run_pii_scan = True
        run_moderation = True
    print(
        f"[input_guard_agent] 🧠 (Step 3) Planner risk={tool_plan['risk_level']} "
        f"PII={run_pii_scan} Moderation={run_moderation}"
    )

    # ── Step 4: Regex injection detection ──────────────────────
    regex_result = detect_injection_regex(clean_text)

    if regex_result.is_injection:
        decision = _make_decision(
            action="block",
            reason="regex_injection_detected",
            confidence=float(regex_result.confidence),
            evidence={
                "layer": "regex",
                "threat_type": regex_result.threat_type,
                "matched_pattern": regex_result.matched_pattern,
            },
        )
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
        print(f"[input_guard_agent] ⛔ BLOCKED (Step 4 — Regex): {regex_result.reason}")
        return {
            "threat_blocked": True,
            "threat_type": regex_result.threat_type,
            "threat_detail": regex_result.reason,
            "input_guard_decision": decision,
            "messages": [{"role": "assistant", "content": BLOCKED_RESPONSE}],
        }

    # ── Step 5: PII scan/redaction ────
    # Detection of PII probes should happen before ML-based injection detection
    # to correctly classify threats as "pii_probe" rather than misclassifying them
    # as "prompt_injection". This also gives us cleaner text for LLM Guard.
    pii_redacted = False
    pii_findings: list[Any] = []
    
    if run_pii_scan:
        pii_result = scan_pii(clean_text)
        pii_findings = pii_result.findings

        if has_high_risk_pii(clean_text):
            decision = _make_decision(
                action="block",
                reason="high_risk_pii_detected",
                confidence=0.95,
                evidence={"findings": pii_result.findings, "tool_plan": tool_plan},
            )
            log_event(
                EventType.INPUT_PII_DETECTED, AGENT_NAME,
                {"reason": "high_risk_pii_in_input", "findings": pii_result.findings},
                user_id,
            )
            print(f"[input_guard_agent] ⛔ BLOCKED (Step 5 — PII Probe): High-risk PII detected in input")
            return {
                "threat_blocked": True,
                "threat_type": "pii_probe",
                "threat_detail": "Attempted PII extraction detected in user input",
                "input_guard_decision": decision,
                "messages": [{"role": "assistant", "content": BLOCKED_RESPONSE}],
            }

        if pii_result.has_pii:
            log_event(
                EventType.INPUT_PII_DETECTED, AGENT_NAME,
                {"reason": "medium_risk_pii_redacted", "findings": pii_result.findings},
                user_id,
            )
            clean_text = pii_result.redacted_text
            pii_redacted = True
            print(f"[input_guard_agent] ℹ️  (Step 5) Medium-risk PII redacted from input")

    # ── Step 6: LLM Guard injection scan ──────────────────────
    # Now runs on PII-cleaned text for more accurate injection detection
    llm_guard_result = scan_input_llm_guard(clean_text)

    if not llm_guard_result.is_safe:
        decision = _make_decision(
            action="block",
            reason="llm_guard_injection_detected",
            confidence=float(llm_guard_result.risk_score),
            evidence={
                "layer": "llm_guard",
                "threat_type": llm_guard_result.threat_type,
                "risk_score": llm_guard_result.risk_score,
            },
        )
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
        print(f"[input_guard_agent] ⛔ BLOCKED (Step 6 — LLM Guard): {llm_guard_result.reason}")
        return {
            "threat_blocked": True,
            "threat_type": llm_guard_result.threat_type,
            "threat_detail": llm_guard_result.reason,
            "input_guard_decision": decision,
            "messages": [{"role": "assistant", "content": BLOCKED_RESPONSE}],
        }

    # Use LLM Guard's sanitised text
    clean_text = llm_guard_result.sanitised_text

    # ── Step 7: OpenAI Moderation (harmful content check) ─────
    moderation_categories: list[str] = []
    if run_moderation:
        moderation_result = check_moderation(clean_text)
        moderation_categories = moderation_result.blocked_categories

        if moderation_result.is_flagged:
            decision = _make_decision(
                action="block",
                reason="moderation_flagged",
                confidence=0.9,
                evidence={
                    "blocked_categories": moderation_result.blocked_categories,
                    "category_scores": moderation_result.category_scores,
                    "tool_plan": tool_plan,
                },
            )
            log_event(
                EventType.INPUT_BLOCKED, AGENT_NAME,
                {
                    "layer": "openai_moderation",
                    "reason": moderation_result.reason,
                    "blocked_categories": moderation_result.blocked_categories,
                    "category_scores": moderation_result.category_scores,
                },
                user_id,
            )
            print(f"[input_guard_agent] ⛔ BLOCKED (Step 7 — OpenAI Moderation): {moderation_result.reason}")
            return {
                "threat_blocked": True,
                "threat_type": "harmful_content",
                "threat_detail": f"Content violates policy: {', '.join(moderation_result.blocked_categories)}",
                "input_guard_decision": decision,
                "messages": [{"role": "assistant", "content": BLOCKED_RESPONSE}],
            }

    # ── Step 8: Pass through ──────────────────────────────────
    log_event(
        EventType.INPUT_PASSED, AGENT_NAME,
        {"message_length": len(clean_text)},
        user_id,
    )
    print("[input_guard_agent] ✅ Input passed all checks — routing to Intent & Profile Agent")

    updated_messages = messages[:-1] + [
        {**last_msg, "content": clean_text}
        if isinstance(last_msg, dict)
        else {"role": "user", "content": clean_text}
    ]

    final_decision = _make_decision(
        action="redact_then_allow" if pii_redacted else "allow",
        reason="input_safe_after_checks" if not pii_redacted else "input_safe_after_redaction",
        confidence=0.92 if not pii_redacted else 0.85,
        evidence={
            "pii_redacted": pii_redacted,
            "final_length": len(clean_text),
            "tool_plan": tool_plan,
            "pii_findings": pii_findings,
            "moderation_categories": moderation_categories,
        },
    )

    return {
        "threat_blocked": False,
        "threat_type": None,
        "threat_detail": None,
        "input_guard_decision": final_decision,
        "sanitised_input": clean_text,
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
