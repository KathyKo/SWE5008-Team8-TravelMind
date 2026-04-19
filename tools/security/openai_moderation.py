"""
tools/openai_moderation.py — OpenAI Moderation API Integration

Uses OpenAI's Moderation API to detect harmful content:
  - Sexual content
  - Violence
  - Harassment
  - Hate speech
  - Self-harm
  - Illegal activities

This is Layer 3 in the security pipeline, running AFTER injection detection
but BEFORE the input reaches planning agents.

API: https://platform.openai.com/docs/guides/moderation
"""

import os
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class ModerationResult:
    """Result of OpenAI moderation check."""
    is_flagged: bool
    categories: dict[str, bool]  # category name → is_flagged
    category_scores: dict[str, float]  # category name → confidence score
    reason: str  # Human-readable explanation
    blocked_categories: list[str]  # List of flagged categories


def check_moderation(text: str) -> ModerationResult:
    """
    Check text against OpenAI Moderation API.
    
    Returns:
        ModerationResult with flagged status and category details.
        
    If API unavailable or text is empty:
        Returns safe result (not flagged)
    """
    
    # Empty input is safe
    if not text or not text.strip():
        return ModerationResult(
            is_flagged=False,
            categories={},
            category_scores={},
            reason="Empty input",
            blocked_categories=[]
        )
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ModerationResult(
                is_flagged=False,
                categories={},
                category_scores={},
                reason="OPENAI_API_KEY not configured",
                blocked_categories=[]
            )
        
        client = OpenAI(api_key=api_key)
        
        # Call moderation API
        response = client.moderations.create(input=text)
        result = response.results[0]
        
        # Extract flagged categories
        flagged_categories = []
        for category, is_flagged in result.category_scores.items():
            if result.categories.__dict__.get(category, False):
                flagged_categories.append(category)
        
        is_flagged = result.flagged
        
        return ModerationResult(
            is_flagged=is_flagged,
            categories=result.categories.model_dump(),
            category_scores=result.category_scores,
            reason=f"Flagged categories: {', '.join(flagged_categories)}" if flagged_categories else "Content passed moderation",
            blocked_categories=flagged_categories
        )
        
    except Exception as e:
        # If API call fails, don't block (fail-safe: pass through)
        # Log the error but continue
        return ModerationResult(
            is_flagged=False,
            categories={},
            category_scores={},
            reason=f"Moderation check failed: {str(e)}",
            blocked_categories=[]
        )


def is_safe_for_travel_context(text: str) -> tuple[bool, str]:
    """
    Quick check: is the text safe in a travel planning context?
    
    Returns:
        (is_safe, reason)
    """
    result = check_moderation(text)
    
    if not result.is_flagged:
        return True, "Content passed moderation"
    
    # In travel context, we block content flagged for:
    # - Sexual content
    # - Violence (but not context like "avoid violent areas")
    # - Self-harm
    # - Illegal activities (but not "how to avoid illegal activity")
    
    blocked = result.blocked_categories
    reason = f"Content violates policy: {', '.join(blocked)}"
    
    return False, reason
