"""
tools/pii_scanner.py — PII (Personally Identifiable Information) Scanner

Scans text for PII patterns on both the input side (user probe attempts)
and output side (prevent data leakage in agent responses).

Used by both InputGuardAgent (Agent 5a) (InputGuardAgent) and OutputGuardAgent (Agent 5b) (OutputGuardAgent).
"""

import re
from dataclasses import dataclass


@dataclass
class PIIResult:
    has_pii: bool
    findings: list[dict]       # list of {pii_type, matched_value, position}
    redacted_text: str         # original text with PII replaced by [REDACTED]
    reason: str


# ── PII pattern library ──────────────────────────────────────
# Each entry: (label, regex, risk_level)
# risk_level: "high" = block, "medium" = redact and log, "low" = log only
PII_PATTERNS: list[tuple[str, str, str]] = [

    # Email addresses
    ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "high"),

    # Singapore NRIC / FIN  (S/T/F/G + 7 digits + letter)
    ("sg_nric", r"\b[STFG]\d{7}[A-Z]\b", "high"),

    # Passport numbers (generic: letter(s) + 6-9 digits)
    ("passport", r"\b[A-Z]{1,2}\d{6,9}\b", "high"),

    # Credit / debit card numbers (13-19 digits, with optional spaces/dashes)
    ("credit_card", r"\b(?:\d[ \-]?){13,19}\b", "high"),

    # Singapore phone numbers (+65 or local 8-digit starting with 6/8/9)
    ("sg_phone", r"(\+65[\s\-]?)?\b[689]\d{7}\b", "medium"),

    # Generic international phone (+ country code + 7-12 digits)
    ("phone_intl", r"\+\d{1,3}[\s\-]?\d{7,12}", "medium"),

    # Physical addresses (basic heuristic: block/unit numbers + street keywords)
    ("address",
     r"\b\d{1,4}\b.{0,20}\b(street|st|avenue|ave|road|rd|drive|dr|lane|ln|blvd|boulevard|place|pl)\b",
     "medium"),

    # Date of birth patterns
    ("date_of_birth",
     r"\b(DOB|date\s+of\s+birth|born\s+on)\b.{0,20}\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}",
     "medium"),

    # Social security / national ID generic
    ("national_id", r"\b\d{3}[\-\s]\d{2}[\-\s]\d{4}\b", "high"),
]

# Mask value: show first 2 chars + asterisks
def _mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 2)


def scan_pii(text: str) -> PIIResult:
    """
    Scan text for PII patterns.

    Returns PIIResult with:
    - has_pii: whether any PII was found
    - findings: list of detected PII items with type and masked value
    - redacted_text: text with all PII replaced by [REDACTED:<type>]
    - reason: human-readable summary
    """
    if not text or not text.strip():
        return PIIResult(
            has_pii=False,
            findings=[],
            redacted_text=text or "",
            reason="Empty input",
        )

    findings: list[dict] = []
    redacted = text

    for pii_type, pattern, risk_level in PII_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matched_value = match.group(0)
            findings.append({
                "pii_type": pii_type,
                "matched_value": _mask(matched_value),
                "position": match.start(),
                "risk_level": risk_level,
            })
            # Replace in redacted copy
            redacted = redacted.replace(
                matched_value, f"[REDACTED:{pii_type.upper()}]", 1
            )

    if findings:
        types_found = list({f["pii_type"] for f in findings})
        high_risk = [f for f in findings if f["risk_level"] == "high"]
        reason = (
            f"PII detected: {', '.join(types_found)}. "
            f"{len(high_risk)} high-risk item(s) found."
        )
    else:
        reason = "No PII detected"

    return PIIResult(
        has_pii=bool(findings),
        findings=findings,
        redacted_text=redacted,
        reason=reason,
    )


def has_high_risk_pii(text: str) -> bool:
    """Convenience function — returns True if any high-risk PII is present."""
    result = scan_pii(text)
    return any(f["risk_level"] == "high" for f in result.findings)
