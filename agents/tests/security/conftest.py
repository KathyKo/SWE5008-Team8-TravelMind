from types import SimpleNamespace
from importlib import import_module

import pytest

ig = import_module("agents.specialists.input_guard_agent")
og = import_module("agents.specialists.output_guard_agent")


@pytest.fixture
def input_state_builder():
    def _build(text: str) -> dict:
        return {
            "messages": [{"role": "user", "content": text}],
            "user_id": "test_user",
        }

    return _build


@pytest.fixture
def output_state_builder():
    def _build(text: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": "Plan my trip"},
                {"role": "assistant", "content": text},
            ],
            "user_id": "test_user",
        }

    return _build


@pytest.fixture
def patch_input_defaults(monkeypatch):
    def _patch(risk_level: str = "high"):
        monkeypatch.setattr(ig, "log_event", lambda *args, **kwargs: None)
        monkeypatch.setattr(ig, "sanitise", lambda text: text)
        monkeypatch.setattr(
            ig,
            "_plan_input_tools",
            lambda text: {
                "risk_level": risk_level,
                "run_pii_scan": risk_level != "low",
                "run_moderation": risk_level != "low",
                "reason": "test_plan",
            },
        )
        monkeypatch.setattr(
            ig,
            "detect_injection_regex",
            lambda text: SimpleNamespace(
                is_injection=False,
                confidence=0.0,
                threat_type=None,
                matched_pattern=None,
                reason="clean",
            ),
        )
        monkeypatch.setattr(
            ig,
            "scan_pii",
            lambda text: SimpleNamespace(has_pii=False, findings=[], redacted_text=text),
        )
        monkeypatch.setattr(ig, "has_high_risk_pii", lambda text: False)
        monkeypatch.setattr(
            ig,
            "scan_input_llm_guard",
            lambda text: SimpleNamespace(
                is_safe=True,
                sanitised_text=text,
                threat_type=None,
                risk_score=0.0,
                reason="safe",
                flags=[],
            ),
        )
        monkeypatch.setattr(
            ig,
            "check_moderation",
            lambda text: SimpleNamespace(
                is_flagged=False,
                blocked_categories=[],
                category_scores={},
                reason="safe",
            ),
        )

    return _patch


@pytest.fixture
def patch_output_defaults(monkeypatch):
    def _patch():
        monkeypatch.setattr(og, "log_event", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            og,
            "check_hallucination",
            lambda text: SimpleNamespace(has_hallucination=False, reason="", flagged_entities=[]),
        )
        monkeypatch.setattr(
            og,
            "scan_pii",
            lambda text: SimpleNamespace(has_pii=False, findings=[], redacted_text=text),
        )
        monkeypatch.setattr(og, "has_high_risk_pii", lambda text: False)
        monkeypatch.setattr(og, "_check_unsafe_content", lambda text: (False, ""))
        monkeypatch.setattr(
            og,
            "_llm_check_unsafe_content",
            lambda user_prompt, output_text: (False, "safe", 0.0, {}),
        )
        monkeypatch.setattr(
            og,
            "scan_output_llm_guard",
            lambda prompt, output: SimpleNamespace(
                is_safe=True,
                flags=[],
                risk_score=0.0,
                reason="safe",
                sanitised_text=output,
            ),
        )

    return _patch
