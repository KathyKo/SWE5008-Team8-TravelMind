"""
tools/security_logger.py — Security Event Logger

Logs all security events (blocked inputs, flagged outputs, PII detections)
with timestamps for audit trail. Aligns with PDPA accountability requirements.

Used by both Agent 5a and Agent 5b.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from enum import Enum


class EventType(str, Enum):
    INPUT_BLOCKED       = "INPUT_BLOCKED"
    INPUT_PII_DETECTED  = "INPUT_PII_DETECTED"
    INPUT_SANITISED     = "INPUT_SANITISED"
    INPUT_PASSED        = "INPUT_PASSED"
    OUTPUT_FLAGGED      = "OUTPUT_FLAGGED"
    OUTPUT_PII_REDACTED = "OUTPUT_PII_REDACTED"
    OUTPUT_PASSED       = "OUTPUT_PASSED"
    SYSTEM_ERROR        = "SYSTEM_ERROR"


# ── Logger setup ─────────────────────────────────────────────
logger = logging.getLogger("travelmind.security")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    ))
    logger.addHandler(handler)
logger.propagate = False


def _build_event(
    event_type: EventType,
    agent: str,
    details: dict,
    user_id: str | None = None,
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type.value,
        "agent": agent,
        "user_id": user_id or "anonymous",
        "details": details,
    }


def log_event(
    event_type: EventType,
    agent: str,
    details: dict,
    user_id: str | None = None,
) -> dict:
    """
    Log a security event and return the event dict.
    The event dict can be stored in state for audit purposes.
    """
    event = _build_event(event_type, agent, details, user_id)

    if event_type in (EventType.INPUT_BLOCKED, EventType.OUTPUT_FLAGGED):
        logger.warning(json.dumps(event))
    elif event_type in (EventType.INPUT_PII_DETECTED, EventType.OUTPUT_PII_REDACTED):
        logger.warning(json.dumps(event))
    elif event_type == EventType.SYSTEM_ERROR:
        logger.error(json.dumps(event))
    else:
        logger.info(json.dumps(event))

    return event
