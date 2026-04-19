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
  - BanTopics        : Block unsafe topics/keywords
  - TokenLimit       : Enforce token budget (complements MAX_INPUT_LENGTH)

Output scanners used:
  - Sensitive        : Detect sensitive data leaking in responses
  - BanTopics        : Block unsafe travel topic recommendations

Design decision:
  LLM Guard is only invoked when regex (Layer 1) passes.
  This keeps average latency low — most attacks are caught by
  regex first, LLM Guard only runs on inputs that look clean.
"""

from dataclasses import dataclass
from typing import Optional
from llm_guard.input_scanners import PromptInjection, TokenLimit, BanTopics as InputBanTopics
from llm_guard.output_scanners import Sensitive, BanTopics
from llm_guard import scan_prompt, scan_output

# Module-level singletons — models load once at import time (app startup),
# not on the first request.
_INPUT_SCANNERS = [
    PromptInjection(threshold=0.75),
    InputBanTopics(
        topics=["illegal activity", "drug trafficking", "smuggling"],
        threshold=0.75,
    ),
    TokenLimit(limit=512),
]

_OUTPUT_SCANNERS = [
    Sensitive(redact=True),
    BanTopics(
        topics=["illegal activity", "drug trafficking", "smuggling"],
        threshold=0.75,
    ),
]


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


# ── Input scanning ────────────────────────────────────────────

def scan_input_llm_guard(text: str) -> LLMGuardInputResult:
    """
    Layer 2 input scan using LLM Guard.

    Scanners applied (in order):
      1. PromptInjection — ML classifier trained on injection datasets
      2. BanTopics       — Block unsafe topics/keywords
      3. TokenLimit      — Enforce a hard token budget

    Note: PII detection is handled separately via pii_scanner module
          (regex-based NER, independent of LLM Guard)
    """
    try:
        sanitised_text, results, is_valid = scan_prompt(_INPUT_SCANNERS, text)

        # results may be numeric scores (0..1) or booleans depending on llm-guard version.
        # - numeric: score >= 0.75 means threat
        # - boolean: True means pass / valid, False means threat
        for scanner_name, score in results.items():
            if isinstance(score, bool):
                is_threat = not score
                risk_score = 1.0 if is_threat else 0.0
            else:
                risk_score = float(score)
                is_threat = risk_score >= 0.75

            if is_threat:
                return LLMGuardInputResult(
                    is_safe=False,
                    threat_type=scanner_name.lower(),
                    risk_score=risk_score,
                    sanitised_text=sanitised_text,
                    reason=f"LLM Guard [{scanner_name}] flagged input (score: {risk_score:.2f})",
                )

        return LLMGuardInputResult(
            is_safe=True,
            threat_type=None,
            risk_score=0.0,
            sanitised_text=sanitised_text,
            reason="LLM Guard: all input scanners passed",
        )
    except Exception as e:
        # Fail-safe behavior: if security scanner is unavailable, block the request
        # instead of returning 500 from API endpoints.
        return LLMGuardInputResult(
            is_safe=False,
            threat_type="llm_guard_runtime_error",
            risk_score=1.0,
            sanitised_text=text,
            reason=f"LLM Guard runtime error: {str(e)}",
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
    try:
        sanitised_text, results, is_valid = scan_output(
            _OUTPUT_SCANNERS, prompt, output
        )

        flags = []
        max_score = 0.0

        # results may be numeric scores (0..1) or booleans depending on llm-guard version.
        for scanner_name, score in results.items():
            if isinstance(score, bool):
                is_threat = not score
                numeric_score = 1.0 if is_threat else 0.0
            else:
                numeric_score = float(score)
                is_threat = numeric_score >= 0.75

            if is_threat:
                flags.append(scanner_name)
                max_score = max(max_score, numeric_score)

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
        # Fail-safe behavior: mark output unsafe when scanner runtime fails.
        return LLMGuardOutputResult(
            is_safe=False,
            flags=["llm_guard_runtime_error"],
            risk_score=1.0,
            sanitised_text=output,
            reason=f"LLM Guard runtime error: {str(e)}",
        )
