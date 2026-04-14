"""
tools/hallucination_guard.py — Hallucination Guard

Verifies that entities mentioned in agent outputs (flight numbers,
hotel names, cities, etc.) are plausible and not fabricated.

Strategy:
1. Extract entity claims from the output text
2. Check against a set of structural rules (format validation)
3. Flag entities that match known hallucination patterns (e.g. fake flight numbers)

Used by OutputGuardAgent (Agent 5b) (OutputGuardAgent).

Note: Full real-time entity verification (e.g. checking a flight number
against live airline APIs) is handled by the Research Agent (Agent 2).
This guard handles structural and pattern-level checks.
"""

import re
from dataclasses import dataclass, field


@dataclass
class HallucinationResult:
    has_hallucination: bool
    flagged_entities: list[dict]   # list of {entity_type, value, reason}
    confidence: float
    reason: str


# ── Known fake / test flight patterns ───────────────────────
# Flight numbers that are commonly hallucinated or used in tests
FAKE_FLIGHT_PATTERNS = [
    r"\b[A-Z]{2}\d{4}\b",          # Generic: check format is valid airline code
]

# IATA airline codes that are known to be valid (non-exhaustive sample)
VALID_AIRLINE_CODES = {
    "SQ", "MH", "TG", "CX", "EK", "QF", "JL", "NH", "KE", "OZ",
    "AI", "BA", "LH", "AF", "KL", "UA", "AA", "DL", "QR", "EY",
    "TR", "D7", "AK", "FD", "SL", "VZ", "TW", "VJ", "BX", "OD",
}

# Flight numbers with 5+ digits are not standard IATA patterns and should be flagged.
INVALID_FLIGHT_NUMBER_LENGTH_PATTERN = r"\b([A-Z]{2})(\d{5,})\b"

# ── Suspicious output patterns ───────────────────────────────
HALLUCINATION_PATTERNS: list[tuple[str, str, str]] = [

    # Fabricated URLs (model often generates plausible-looking but fake URLs)
    (r"https?://[a-z0-9\-]+\.(com|net|org)/booking/[a-z0-9\-/]+",
     "fabricated_url",
     "URL appears to be a generated booking link — verify before presenting to user"),

    # Suspiciously round prices (often hallucinated)
    (r"\b(SGD|USD|EUR|GBP)\s*(100|200|300|400|500|1000|2000|3000|5000)\.00\b",
     "suspiciously_round_price",
     "Price is suspiciously round — verify against live data"),

    # Fake confirmation codes (alphanumeric 6-char codes)
    (r"\b(confirmation|booking|reference)\s*(code|number|#|no\.?)?\s*:?\s*[A-Z0-9]{6}\b",
     "fabricated_confirmation_code",
     "Confirmation code may be fabricated — do not present to user as real"),

    # Specific flight number format with invalid airline code
    # Checked separately in check_flight_numbers()
]


def _check_flight_numbers(text: str) -> list[dict]:
    """
    Extract flight numbers and validate the airline code portion.
    Flags numbers with unrecognised airline codes or invalid digit length as potentially hallucinated.
    """
    flagged = []

    # Non-standard flight numbers with 5+ digits should be flagged as likely hallucinations
    for match in re.finditer(INVALID_FLIGHT_NUMBER_LENGTH_PATTERN, text, re.IGNORECASE):
        full = match.group(0)
        flagged.append({
            "entity_type": "flight_number",
            "value": full,
            "reason": f"Flight number '{full}' has too many digits to be a standard IATA flight number",
        })

    # Match pattern: 2-letter code + 1-4 digits (standard IATA flight number)
    for match in re.finditer(r"\b([A-Z]{2})(\d{1,4})\b", text, re.IGNORECASE):
        airline_code = match.group(1).upper()
        flight_num = match.group(2)
        full = match.group(0)
        if airline_code not in VALID_AIRLINE_CODES:
            flagged.append({
                "entity_type": "flight_number",
                "value": full,
                "reason": f"Airline code '{airline_code}' not in known IATA list — may be hallucinated",
            })
    return flagged


def check_hallucination(output_text: str) -> HallucinationResult:
    """
    Check an agent's output text for potential hallucinated entities.

    Returns HallucinationResult with flagged entities and overall assessment.
    """
    if not output_text or not output_text.strip():
        return HallucinationResult(
            has_hallucination=False,
            flagged_entities=[],
            confidence=0.0,
            reason="Empty output",
        )

    flagged: list[dict] = []

    # Pattern-based checks
    for pattern, entity_type, reason in HALLUCINATION_PATTERNS:
        for match in re.finditer(pattern, output_text, re.IGNORECASE):
            flagged.append({
                "entity_type": entity_type,
                "value": match.group(0)[:60],
                "reason": reason,
            })

    # Flight number validation
    flagged.extend(_check_flight_numbers(output_text))

    if flagged:
        types = list({f["entity_type"] for f in flagged})
        confidence = min(0.95, 0.6 + 0.1 * len(flagged))
        reason = f"Potential hallucination detected in: {', '.join(types)}"
    else:
        confidence = 0.0
        reason = "No hallucination patterns detected"

    return HallucinationResult(
        has_hallucination=bool(flagged),
        flagged_entities=flagged,
        confidence=confidence,
        reason=reason,
    )
