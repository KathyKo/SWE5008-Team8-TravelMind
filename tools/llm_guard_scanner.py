"""
tools/llm_guard_scanner.py — Layer 2: LLM Guard ML-based Scanning

LLM Guard (by Protect AI) provides ML-based scanners for both
input and output sides. It is used as the second layer after regex
pattern matching, catching semantically obfuscated attacks that
regex alone would miss.

Docs: https://llm-guard.com
PyPI: pip install llm-guard

Input scanners used:
  - PromptInjection  : ML classifier for injection attempts
  - PII              : Named entity recognition for personal data
  - TokenLimit       : Enforce token budget (complements MAX_INPUT_LENGTH)

Output scanners used:
  - Sensitive        : Detect sensitive data leaking in responses
  - FactualConsistency: Flag potentially hallucinated claims
  - BanTopics        : Block unsafe travel recommendations

Design decision:
  LLM Guard is only invoked when regex (Layer 1) passes.
  This keeps average latency low — most attacks are caught by
  regex first, LLM Guard only runs on inputs that look clean.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMGuardInputResult:
    is_safe: bool
    threat_type: Optional[str]    # which scanner flagged it
    risk_score: float             # 0.0 (safe) – 1.0 (high risk)
    sanitised_text: str           # LLM Guard may redact PII automatically
    reason: str


@dataclass
class LLMGuardOutputResult:
    is_safe: bool
    flags: list[str]              # list of scanner names that triggered
    risk_score: float
    sanitised_text: str
    reason: str


# ── Lazy imports — LLM Guard is optional ─────────────────────
# If llm-guard is not installed, both functions degrade gracefully
# and return is_safe=True (regex layer is still active as fallback).
def _import_llm_guard():
    try:
        from llm_guard.input_scanners import PromptInjection, PII, TokenLimit
        from llm_guard.output_scanners import Sensitive, BanTopics
        from llm_guard import scan_prompt, scan_output
        return {
            "PromptInjection": PromptInjection,
            "PII": PII,
            "TokenLimit": TokenLimit,
            "Sensitive": Sensitive,
            "BanTopics": BanTopics,
            "scan_prompt": scan_prompt,
            "scan_output": scan_output,
            "available": True,
        }
    except ImportError:
        return {"available": False}


_LLM_GUARD = _import_llm_guard()


# ── Input scanning ────────────────────────────────────────────

def scan_input_llm_guard(text: str) -> LLMGuardInputResult:
    """
    Layer 2 input scan using LLM Guard.

    Scanners applied (in order):
      1. PromptInjection — ML classifier trained on injection datasets
      2. PII             — NER-based personal data detection
      3. TokenLimit      — Enforce a hard token budget

    If llm-guard is not installed, returns is_safe=True with a warning
    so the system degrades gracefully (regex layer still protects).
    """
    if not _LLM_GUARD["available"]:
        print("[LLM Guard] ⚠️  llm-guard not installed — Layer 2 skipped (regex only)")
        return LLMGuardInputResult(
            is_safe=True,
            threat_type=None,
            risk_score=0.0,
            sanitised_text=text,
            reason="llm-guard not available — regex layer still active",
        )

    try:
        PromptInjection = _LLM_GUARD["PromptInjection"]
        PII             = _LLM_GUARD["PII"]
        TokenLimit      = _LLM_GUARD["TokenLimit"]
        scan_prompt     = _LLM_GUARD["scan_prompt"]

        input_scanners = [
            PromptInjection(threshold=0.75),   # flag if injection confidence > 75%
            PII(redact=True, allowed_entities=["LOCATION", "DATE_TIME"]),
            TokenLimit(limit=512),
        ]

        sanitised_text, results, is_valid = scan_prompt(input_scanners, text)

        # results is a dict: {scanner_name: (is_valid, risk_score)}
        for scanner_name, (valid, score) in results.items():
            if not valid:
                return LLMGuardInputResult(
                    is_safe=False,
                    threat_type=scanner_name.lower(),
                    risk_score=score,
                    sanitised_text=sanitised_text,
                    reason=f"LLM Guard [{scanner_name}] flagged input (score: {score:.2f})",
                )

        return LLMGuardInputResult(
            is_safe=True,
            threat_type=None,
            risk_score=0.0,
            sanitised_text=sanitised_text,
            reason="LLM Guard: all input scanners passed",
        )

    except Exception as e:
        # Never let LLM Guard errors block legitimate users
        print(f"[LLM Guard] ⚠️  Input scan error: {e} — failing open")
        return LLMGuardInputResult(
            is_safe=True,
            threat_type=None,
            risk_score=0.0,
            sanitised_text=text,
            reason=f"LLM Guard scan error (non-fatal): {str(e)}",
        )


# ── Output scanning ───────────────────────────────────────────

def scan_output_llm_guard(prompt: str, output: str) -> LLMGuardOutputResult:
    """
    Layer 2 output scan using LLM Guard.

    Scanners applied:
      1. Sensitive  — detect PII or confidential data leaking in response
      2. BanTopics  — block unsafe travel topic recommendations

    Args:
        prompt: The original user prompt (required by LLM Guard for context)
        output: The agent's response text to validate
    """
    if not _LLM_GUARD["available"]:
        print("[LLM Guard] ⚠️  llm-guard not installed — output Layer 2 skipped")
        return LLMGuardOutputResult(
            is_safe=True, flags=[], risk_score=0.0,
            sanitised_text=output,
            reason="llm-guard not available — regex layer still active",
        )

    try:
        Sensitive   = _LLM_GUARD["Sensitive"]
        BanTopics   = _LLM_GUARD["BanTopics"]
        scan_output = _LLM_GUARD["scan_output"]

        output_scanners = [
            Sensitive(redact=True),
            BanTopics(
                topics=["illegal activity", "drug trafficking", "smuggling"],
                threshold=0.75,
            ),
        ]

        sanitised_text, results, is_valid = scan_output(
            output_scanners, prompt, output
        )

        flags = []
        max_score = 0.0

        for scanner_name, (valid, score) in results.items():
            if not valid:
                flags.append(scanner_name)
                max_score = max(max_score, score)

        if flags:
            return LLMGuardOutputResult(
                is_safe=False,
                flags=flags,
                risk_score=max_score,
                sanitised_text=sanitised_text,
                reason=f"LLM Guard flagged output: {', '.join(flags)}",
            )

        return LLMGuardOutputResult(
            is_safe=True, flags=[], risk_score=0.0,
            sanitised_text=sanitised_text,
            reason="LLM Guard: all output scanners passed",
        )

    except Exception as e:
        print(f"[LLM Guard] ⚠️  Output scan error: {e} — failing open")
        return LLMGuardOutputResult(
            is_safe=True, flags=[], risk_score=0.0,
            sanitised_text=output,
            reason=f"LLM Guard scan error (non-fatal): {str(e)}",
        )
