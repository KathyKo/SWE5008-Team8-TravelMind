"""
tools/sanitiser.py — Input Sanitisation

Cleans and normalises raw user input before it enters the agent pipeline.
Runs before injection detection — removes noise that could mask patterns.

Used by Agent 5a (Input Gatekeeper).
"""

import re
import unicodedata


def sanitise(text: str) -> str:
    """
    Clean and normalise a raw input string.

    Steps:
    1. Decode unicode escapes and normalise to NFC form
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


def truncate(text: str, max_length: int = 2000) -> tuple[str, bool]:
    """
    Truncate text to max_length characters.
    Returns (truncated_text, was_truncated).
    """
    if len(text) <= max_length:
        return text, False
    return text[:max_length], True


def sanitise_and_truncate(text: str, max_length: int = 2000) -> dict:
    """
    Combined sanitise + truncate. Returns a dict with:
    - clean_text: the processed text
    - was_truncated: bool
    - original_length: int
    """
    cleaned = sanitise(text)
    truncated, was_truncated = truncate(cleaned, max_length)
    return {
        "clean_text": truncated,
        "was_truncated": was_truncated,
        "original_length": len(text),
    }
