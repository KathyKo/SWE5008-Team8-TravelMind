"""
tools/sanitiser.py — Input Sanitisation

Cleans and normalises raw user input before it enters the agent pipeline.
Runs before injection detection — removes noise that could mask patterns.

Responsibility: cleaning only.
Length enforcement is handled by InputGuardAgent (Agent 5a) as an
explicit block condition — oversized inputs are rejected, not silently
truncated.

Used by InputGuardAgent (Agent 5a).
"""

import re
import unicodedata


def sanitise(text: str) -> str:
    """
    Clean and normalise a raw input string.

    Steps:
    1. Normalise unicode to NFC form — prevents homoglyph attacks
    2. Strip null bytes and control characters (except newline/tab)
    3. Collapse excessive whitespace
    4. Trim leading/trailing whitespace
    """
    if not text:
        return ""

    # Normalise unicode (NFC) — prevents homoglyph attacks
    text = unicodedata.normalize("NFC", text)

    # Remove null bytes and most control characters
    # Keep: \t (tab), \n (newline), \r (carriage return)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse 3+ consecutive spaces to 1
    text = re.sub(r" {3,}", " ", text)

    return text.strip()
