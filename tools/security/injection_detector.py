"""
tools/injection_detector.py — Layer 1: Regex-based Prompt Injection Detection

This is the FIRST layer of a two-layer injection defence:
  Layer 1 (this file) — Regex pattern matching
      Fast, zero latency, no external dependencies.
      Catches known, high-confidence injection patterns immediately.

  Layer 2 (llm_guard_scanner.py) — LLM Guard PromptInjection scanner
      ML-based, semantic understanding.
      Catches novel phrasings and obfuscated variants that regex misses.

Both layers are called sequentially by InputGuardAgent (Agent 5a).
A positive result from EITHER layer triggers a block.
"""

import re
from dataclasses import dataclass


@dataclass
class RegexDetectionResult:
    is_injection: bool
    threat_type: str | None
    matched_pattern: str | None
    confidence: float           # 0.0 – 1.0
    reason: str


# ── Pattern library ───────────────────────────────────────────
# Each entry: (regex, threat_type, confidence, description)
# Only HIGH confidence patterns here — low confidence deferred to LLM Guard
INJECTION_PATTERNS: list[tuple[str, str, float, str]] = [

    # Instruction override
    (r"\bignore\b.{0,30}\b(previous|above|prior|all)\b.{0,30}\b(instructions?|prompt|rules?|constraints?)\b",
     "instruction_override", 0.95,
     "Attempting to override system instructions"),

    (r"\bforget\b.{0,30}\b(everything|all|previous|instructions?|prompt)\b",
     "instruction_override", 0.90,
     "Attempting to clear system instructions"),

    (r"\bdisregard\b.{0,30}\b(previous|above|all|the)\b.{0,30}\b(instructions?|rules?|prompt)\b",
     "instruction_override", 0.90,
     "Attempting to disregard system instructions"),

    # Role hijacking
    (r"\byou\s+are\s+now\b.{0,50}\b(a|an|the)\b",
     "role_hijacking", 0.85,
     "Attempting to reassign agent role"),

    (r"\bact\s+as\b.{0,30}\b(a|an|the)\b.{0,30}\b(different|new|another|unrestricted)\b",
     "role_hijacking", 0.85,
     "Attempting to impersonate a different agent"),

    (r"\bpretend\b.{0,30}\b(you\s+are|to\s+be)\b",
     "role_hijacking", 0.80,
     "Role impersonation attempt"),

    # Jailbreak keywords
    (r"\b(jailbreak|dan\s+mode|developer\s+mode|unrestricted\s+mode|no\s+restrictions?)\b",
     "jailbreak", 0.95,
     "Known jailbreak keyword detected"),

    (r"\bbypass\b.{0,30}\b(safety|filter|restriction|rule|guideline|policy)\b",
     "jailbreak", 0.90,
     "Attempting to bypass safety measures"),

    # System prompt extraction
    (r"\b(reveal|show|print|output|repeat|tell\s+me)\b.{0,30}\b(your\s+)?(system\s+prompt|instructions?|initial\s+prompt)\b",
     "prompt_extraction", 0.92,
     "Attempting to extract system prompt"),

    (r"\bwhat\s+(are|were)\s+your\s+(original\s+)?(instructions?|rules?|prompt|guidelines?)\b",
     "prompt_extraction", 0.85,
     "Attempting to read system instructions"),

    # Delimiter injection
    (r"(```|\-{2,}|\*{2,}|#{2,}|<{2,}|>{2,}).{0,20}(system|assistant|human|user|instruction)",
     "delimiter_injection", 0.88,
     "Delimiter-based injection attempt"),

    (r"\[SYSTEM\]|\[INST\]|\[ADMIN\]|\<\|system\|\>|\<\|im_start\|\>",
     "delimiter_injection", 0.95,
     "Known prompt delimiter injection"),

    # Overflow / malformed
    (r"(.)\1{40,}",
     "malformed_input", 0.80,
     "Excessive character repetition — potential buffer overflow attempt"),
]

MAX_INPUT_LENGTH = 2000


def detect_injection_regex(user_input: str) -> RegexDetectionResult:
    """
    Layer 1: Scan user input for known injection patterns using regex.

    Returns RegexDetectionResult.
    If is_injection=True, InputGuardAgent blocks immediately without
    proceeding to Layer 2 (LLM Guard) — fast-fail on known threats.
    """
    if not user_input or not user_input.strip():
        return RegexDetectionResult(
            is_injection=False, threat_type=None,
            matched_pattern=None, confidence=0.0, reason="Empty input",
        )

    # Length check (see MAX_INPUT_LENGTH for rationale)
    if len(user_input) > MAX_INPUT_LENGTH:
        return RegexDetectionResult(
            is_injection=True,
            threat_type="oversized_input",
            matched_pattern=f"length={len(user_input)}",
            confidence=0.75,
            reason=f"Input length {len(user_input)} exceeds maximum {MAX_INPUT_LENGTH} characters",
        )

    lowered = user_input.lower()

    for pattern, threat_type, confidence, description in INJECTION_PATTERNS:
        match = re.search(pattern, lowered, re.IGNORECASE | re.DOTALL)
        if match:
            return RegexDetectionResult(
                is_injection=True,
                threat_type=threat_type,
                matched_pattern=match.group(0)[:80],
                confidence=confidence,
                reason=description,
            )

    return RegexDetectionResult(
        is_injection=False, threat_type=None,
        matched_pattern=None, confidence=0.0,
        reason="No regex injection patterns detected — proceeding to LLM Guard",
    )
