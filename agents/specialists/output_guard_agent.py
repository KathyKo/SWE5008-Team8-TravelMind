"""
agents/output_guard_agent.py — output_guard_agent

Purpose:
    Output-side safety gate. Runs after specialist responses are generated and
    before content is returned to the user.

Validation steps:
    Step 1: Hallucination pattern check (advisory)
    Step 2: Regex PII detection/redaction (includes SG-specific checks)
    Step 3: Rule-based unsafe topic keyword check
    Step 4: LLM semantic unsafe-content check 
    Step 5: LLM Guard output scan (Sensitive + BanTopics)

Configuration sources:
    - Prompts: agents/prompts/security_prompts.yaml
    - Policy/rules: agents/prompts/security_policy.yaml
    - Runtime controls: ENABLE_LLM_UNSAFE_CHECK, OUTPUT_GUARD_PROMPT_VERSION

Routing:
    [specialists] → output_guard → output_flagged
                               → output_safe
"""

import json
import os
import re
import logging
import sys
import yaml
from pathlib import Path
from typing import Any

from openai import OpenAI
from agents.llm_config import OPENAI_MODEL
from agents.state import State
from tools.security.hallucination_guard import check_hallucination
from tools.security.pii_scanner import scan_pii, has_high_risk_pii
from tools.security.llm_guard_scanner import scan_output_llm_guard
from tools.security.security_logger import log_event, EventType


AGENT_NAME = "output_guard_agent"
ENABLE_LLM_UNSAFE_CHECK = os.getenv("ENABLE_LLM_UNSAFE_CHECK", "1") == "1"
OUTPUT_GUARD_PROMPT_VERSION = os.getenv("OUTPUT_GUARD_PROMPT_VERSION", "v1")
PROMPTS_FILE = Path(__file__).resolve().parents[1] / "prompts" / "security_prompts.yaml"
POLICY_FILE = Path(__file__).resolve().parents[1] / "prompts" / "security_policy.yaml"
DECISION_SCHEMA_VERSION = "v1"

FLAGGED_RESPONSE = (
    "I encountered an issue preparing your travel information. "
    "Please try rephrasing your request."
)


logger = logging.getLogger("travelmind.agents.output_guard")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s — %(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


def _get_prompt(name: str, version: str, fallback: str) -> str:
    try:
        with PROMPTS_FILE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("output_guard", {}).get(name, {}).get(version, fallback)
    except Exception:
        return fallback


def _get_output_guard_config(version: str) -> dict[str, Any]:
    fallback = {"llm_unsafe_conf_threshold": 0.70}
    try:
        with POLICY_FILE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("output_guard", {}).get("config", {}).get(version, {})
        if not isinstance(cfg, dict):
            return fallback
        return {**fallback, **cfg}
    except Exception:
        return fallback


def _get_unsafe_regex_rules(version: str) -> list[tuple[str, str]]:
    fallback = [
        (
            r"\b(how\s+to|guide\s+to|steps\s+to|tutorial\s+for|best\s+way\s+to)\b.{0,60}"
            r"\b(smuggl(?:e|ing)|drug\s+trafficking|human\s+trafficking|weapon\s+trafficking)\b",
            "Actionable guidance for trafficking/smuggling detected",
        )
    ]
    try:
        with POLICY_FILE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_rules = data.get("output_guard", {}).get("unsafe_regex_rules", {}).get(version, [])
        rules: list[tuple[str, str]] = []
        for item in raw_rules:
            if isinstance(item, dict):
                pattern = str(item.get("pattern", "")).strip()
                reason = str(item.get("reason", "Unsafe content pattern detected")).strip()
                if pattern:
                    rules.append((pattern, reason))
        return rules or fallback
    except Exception:
        return fallback


def _make_decision(action: str, reason: str, confidence: float, evidence: dict[str, Any]) -> dict[str, Any]:
    """Structured decision trace to show agent-like reasoning and actions."""
    return {
        "agent": AGENT_NAME,
        "goal": "maximize output safety while preserving helpful travel guidance",
        "action": action,  # allow | block | redact_then_allow | warn_then_allow
        "reason": reason,
        "confidence": confidence,
        "schema_version": DECISION_SCHEMA_VERSION,
        "evidence": evidence,
    }


def output_guard_agent(state: State) -> dict:
    """Compatibility alias for callers expecting `output_guard_agent` as node function."""
    return output_guard_node(state)

def _check_unsafe_content(text: str) -> tuple[bool, str]:
    for pattern, reason in _get_unsafe_regex_rules(OUTPUT_GUARD_PROMPT_VERSION):
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return True, reason
    return False, ""


def _llm_check_unsafe_content(user_prompt: str, output_text: str) -> tuple[bool, str, float, dict[str, Any]]:
    """
    Semantic unsafe-content check using a structured JSON response.
    Returns: (is_unsafe, reason, confidence, evidence)
    """
    if not ENABLE_LLM_UNSAFE_CHECK:
        return False, "llm_unsafe_check_disabled", 0.0, {"skipped": True}

    default_safe = (False, "llm_unsafe_check_fallback_safe", 0.0, {"fallback": True})
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return default_safe

        prompt = _get_prompt(
            "unsafe_semantic_classifier_system",
            OUTPUT_GUARD_PROMPT_VERSION,
            (
                "You are a travel response safety classifier. "
                "Decide if the assistant output should be blocked for unsafe-content policy. "
                "Block if content gives actionable guidance for illegal acts, violence, smuggling, drug trafficking, "
                "or other harmful wrongdoing. "
                "Do NOT block benign safety context (e.g., warning users to avoid dangerous areas). "
                "Return strict JSON with keys: unsafe(boolean), reason(string), confidence(0..1), category(string)."
            ),
        )

        client = OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"User request:\n{user_prompt}\n\nAssistant output:\n{output_text}",
                },
            ],
        )
        content = (res.choices[0].message.content or "{}").strip()
        data = json.loads(content)
        is_unsafe = bool(data.get("unsafe", False))
        reason = str(data.get("reason", "semantic_unsafe_check"))
        confidence = float(data.get("confidence", 0.5))
        category = str(data.get("category", "unspecified"))

        conf_threshold = float(_get_output_guard_config(OUTPUT_GUARD_PROMPT_VERSION).get("llm_unsafe_conf_threshold", 0.70))
        # Conservative thresholding
        if not is_unsafe:
            return False, reason, confidence, {"category": category}
        if confidence < conf_threshold:
            return False, f"low_confidence_unsafe:{reason}", confidence, {"category": category}

        return True, reason, confidence, {"category": category}
    except Exception:
        return default_safe


def output_guard_node(state: State) -> dict:
    """
    LangGraph node for `output_guard_agent`.

    Validates the latest assistant message before it reaches the user.
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
        return {
            "output_flagged": False,
            "output_guard_decision": _make_decision(
                action="allow",
                reason="no_assistant_message",
                confidence=1.0,
                evidence={"message_count": len(messages)},
            ),
        }

    output_text = last_assistant_msg.get("content", "")
    # Get original user prompt for LLM Guard context
    user_prompt = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_prompt = msg.get("content", "")

    logger.info("validating_response chars=%s", len(output_text))

    logger.info("planner_disabled running_all_safety_tools=true")

    # ── Step 1: Hallucination check (advisory) ────────────────
    hallucination_warning = False
    hallucination_evidence: dict[str, Any] = {}
    halluc_result = check_hallucination(output_text)
    if halluc_result.has_hallucination:
        hallucination_warning = True
        hallucination_evidence = {
            "reason": halluc_result.reason,
            "flagged_entities": halluc_result.flagged_entities,
        }
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
        logger.warning("hallucination_warning reason=%s", halluc_result.reason)

    # ── Step 2: Regex PII scan ────────────────────────────────
    pii_redacted = False
    pii_findings: list[Any] = []
    pii_result = scan_pii(output_text)
    pii_findings = pii_result.findings

    if has_high_risk_pii(output_text):
        decision = _make_decision(
            action="block",
            reason="high_risk_pii_output",
            confidence=0.95,
            evidence={"findings": pii_result.findings},
        )
        log_event(
            EventType.OUTPUT_PII_REDACTED, AGENT_NAME,
            {"layer": "regex", "reason": "high_risk_pii_in_output", "findings": pii_result.findings},
            user_id,
        )
        logger.warning("flagged step=regex_pii reason=high_risk_pii_output")
        return {
            "output_flagged": True,
            "output_flag_reason": "pii_leakage",
            "output_guard_decision": decision,
            "messages": messages[:last_assistant_idx] + [
                {"role": "assistant", "content": FLAGGED_RESPONSE}
            ],
        }

    if pii_result.has_pii:
        output_text = pii_result.redacted_text
        pii_redacted = True
        log_event(
            EventType.OUTPUT_PII_REDACTED, AGENT_NAME,
            {"layer": "regex", "reason": "medium_pii_redacted", "findings": pii_result.findings},
            user_id,
        )
        logger.info("pii_redacted step=regex_pii severity=medium")

    # ── Step 3: Rule-based unsafe content check ───────────────
    is_unsafe, unsafe_reason = _check_unsafe_content(output_text)
    if is_unsafe:
        decision = _make_decision(
            action="block",
            reason="unsafe_topic_detected",
            confidence=0.9,
            evidence={"detail": unsafe_reason},
        )
        log_event(
            EventType.OUTPUT_FLAGGED, AGENT_NAME,
            {"layer": "regex", "reason": "unsafe_content", "detail": unsafe_reason},
            user_id,
        )
        logger.warning("flagged step=rule_based reason=%s", unsafe_reason)
        return {
            "output_flagged": True,
            "output_flag_reason": "unsafe_content",
            "output_guard_decision": decision,
            "messages": messages[:last_assistant_idx] + [
                {"role": "assistant", "content": FLAGGED_RESPONSE}
            ],
        }

    # ── Step 4: LLM semantic unsafe-content check ─────────────
    llm_unsafe, llm_unsafe_reason, llm_unsafe_conf, llm_unsafe_evidence = _llm_check_unsafe_content(
        user_prompt=user_prompt,
        output_text=output_text,
    )
    if llm_unsafe:
        decision = _make_decision(
            action="block",
            reason="unsafe_topic_detected_semantic",
            confidence=llm_unsafe_conf,
            evidence={
                "detail": llm_unsafe_reason,
                "semantic": llm_unsafe_evidence,
            },
        )
        log_event(
            EventType.OUTPUT_FLAGGED, AGENT_NAME,
            {
                "layer": "llm_semantic",
                "reason": "unsafe_content",
                "detail": llm_unsafe_reason,
                "confidence": llm_unsafe_conf,
                "semantic": llm_unsafe_evidence,
            },
            user_id,
        )
        logger.warning("flagged step=llm_semantic reason=%s", llm_unsafe_reason)
        return {
            "output_flagged": True,
            "output_flag_reason": "unsafe_content_semantic",
            "output_guard_decision": decision,
            "messages": messages[:last_assistant_idx] + [
                {"role": "assistant", "content": FLAGGED_RESPONSE}
            ],
        }

    # ── Step 5: LLM Guard output scan ─────────────────────────
    llm_guard_flags: list[str] = []
    llm_guard_result = scan_output_llm_guard(user_prompt, output_text)
    llm_guard_flags = llm_guard_result.flags

    if not llm_guard_result.is_safe:
        decision = _make_decision(
            action="block",
            reason="llm_guard_output_flagged",
            confidence=float(llm_guard_result.risk_score),
            evidence={
                "flags": llm_guard_result.flags,
                "risk_score": llm_guard_result.risk_score,
            },
        )
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
        logger.warning("flagged step=llm_guard reason=%s", llm_guard_result.reason)
        return {
            "output_flagged": True,
            "output_flag_reason": f"llm_guard:{','.join(llm_guard_result.flags)}",
            "output_guard_decision": decision,
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
    logger.info("output_passed all_checks=true")

    updated_messages = list(messages)
    updated_messages[last_assistant_idx] = {
        **last_assistant_msg,
        "content": output_text,
    }

    if pii_redacted:
        final_action = "redact_then_allow"
        final_reason = "output_safe_after_redaction"
        confidence = 0.86
    elif hallucination_warning:
        final_action = "warn_then_allow"
        final_reason = "minor_hallucination_warning_only"
        confidence = 0.8
    else:
        final_action = "allow"
        final_reason = "output_safe_after_checks"
        confidence = 0.92

    final_decision = _make_decision(
        action=final_action,
        reason=final_reason,
        confidence=confidence,
        evidence={
            "pii_redacted": pii_redacted,
            "hallucination_warning": hallucination_warning,
            "hallucination": hallucination_evidence,
            "pii_findings": pii_findings,
            "llm_guard_flags": llm_guard_flags,
        },
    )

    return {
        "output_flagged": False,
        "output_flag_reason": None,
        "output_guard_decision": final_decision,
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


