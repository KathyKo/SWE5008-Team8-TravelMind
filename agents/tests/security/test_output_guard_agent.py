import pytest
from types import SimpleNamespace
from importlib import import_module
og = import_module("agents.specialists.output_guard_agent")

def _setup_high_risk_pii(monkeypatch):
    monkeypatch.setattr(
        og,
        "scan_pii",
        lambda text: SimpleNamespace(
            has_pii=True,
            findings=[{"type": "PASSPORT", "value": "E12345678"}],
            redacted_text="redacted",
        ),
    )
    monkeypatch.setattr(og, "has_high_risk_pii", lambda text: True)


def _setup_rule_based_unsafe(monkeypatch):
    monkeypatch.setattr(
        og,
        "_check_unsafe_content",
        lambda text: (True, "Actionable guidance for trafficking/smuggling detected"),
    )


def _setup_semantic_unsafe(monkeypatch):
    monkeypatch.setattr(
        og,
        "_llm_check_unsafe_content",
        lambda user_prompt, output_text: (True, "semantic unsafe", 0.93, {"category": "violence"}),
    )


def _setup_llm_guard_block(monkeypatch):
    monkeypatch.setattr(
        og,
        "scan_output_llm_guard",
        lambda prompt, output: SimpleNamespace(
            is_safe=False,
            flags=["Sensitive"],
            risk_score=0.88,
            reason="Sensitive data detected",
            sanitised_text=output,
        ),
    )


def _setup_hallucination_warn_then_allow(monkeypatch):
    monkeypatch.setattr(
        og,
        "check_hallucination",
        lambda text: SimpleNamespace(
            has_hallucination=True,
            reason="Potential hallucination detected",
            flagged_entities=[{"entity_type": "flight_number", "value": "JL99999"}],
        ),
    )


@pytest.mark.parametrize(
    "assistant_output,expected_flagged,expected_reason,expected_action,setup_fn",
    [
        (
            "Your passport is E12345678",
            True,
            "pii_leakage",
            None,
            _setup_high_risk_pii,
        ),
        (
            "Best way to smuggle prohibited items is to split them.",
            True,
            "unsafe_content",
            None,
            _setup_rule_based_unsafe,
        ),
        (
            "Here are steps to do violent wrongdoing",
            True,
            "unsafe_content_semantic",
            None,
            _setup_semantic_unsafe,
        ),
        (
            "Potentially unsafe output",
            True,
            "llm_guard:Sensitive",
            None,
            _setup_llm_guard_block,
        ),
        (
            "I found flight JL99999 for you",
            False,
            None,
            "warn_then_allow",
            _setup_hallucination_warn_then_allow,
        ),
    ],
    ids=[
        "block_high_risk_pii",
        "block_rule_based_unsafe",
        "block_semantic_unsafe",
        "block_llm_guard",
        "allow_hallucination_warning",
    ],
)
def test_output_guard_paths_parameterized(
    monkeypatch,
    patch_output_defaults,
    output_state_builder,
    assistant_output,
    expected_flagged,
    expected_reason,
    expected_action,
    setup_fn,
):
    patch_output_defaults()
    setup_fn(monkeypatch)

    result = og.output_guard_node(output_state_builder(assistant_output))

    assert result["output_flagged"] is expected_flagged

    if expected_flagged:
        assert result["output_flag_reason"] == expected_reason
        assert result["messages"][-1]["content"] == og.FLAGGED_RESPONSE
    else:
        assert result["output_flag_reason"] is None
        assert result["output_guard_decision"]["action"] == expected_action


def test_output_guard_redacts_medium_pii_and_allows(monkeypatch, patch_output_defaults, output_state_builder):
    patch_output_defaults()
    monkeypatch.setattr(
        og,
        "scan_pii",
        lambda text: SimpleNamespace(
            has_pii=True,
            findings=[{"type": "EMAIL", "value": "alice@example.com"}],
            redacted_text="Contact: [REDACTED_EMAIL]",
        ),
    )

    result = og.output_guard_node(output_state_builder("Contact: alice@example.com"))

    assert result["output_flagged"] is False
    assert result["output_guard_decision"]["action"] == "redact_then_allow"
    assert result["messages"][-1]["content"] == "Contact: [REDACTED_EMAIL]"


def test_output_guard_allows_clean_output(monkeypatch, patch_output_defaults, output_state_builder):
    patch_output_defaults()

    result = og.output_guard_node(output_state_builder("Day 1: Kyoto, Day 2: Osaka"))

    assert result["output_flagged"] is False
    assert result["output_guard_decision"]["action"] == "allow"
    assert result["messages"][-1]["content"] == "Day 1: Kyoto, Day 2: Osaka"


def test_output_guard_returns_default_when_no_assistant_message():
    result = og.output_guard_node({"messages": [{"role": "user", "content": "Plan my trip"}], "user_id": "test_user"})

    assert result["output_flagged"] is False
    assert result["output_guard_decision"]["action"] == "allow"
    assert result["output_guard_decision"]["reason"] == "no_assistant_message"


def test_output_guard_preserves_sanitised_output_from_llm_guard(monkeypatch, patch_output_defaults, output_state_builder):
    patch_output_defaults()
    monkeypatch.setattr(
        og,
        "scan_output_llm_guard",
        lambda prompt, output: SimpleNamespace(
            is_safe=True,
            flags=[],
            risk_score=0.0,
            reason="safe",
            sanitised_text="SANITISED OUTPUT",
        ),
    )

    result = og.output_guard_node(output_state_builder("Original output"))

    assert result["output_flagged"] is False
    assert result["messages"][-1]["content"] == "SANITISED OUTPUT"


@pytest.mark.parametrize(
    "state,expected_route",
    [
        ({"output_flagged": True}, "output_flagged"),
        ({"output_flagged": False}, "output_safe"),
    ],
)
def test_output_guard_routing(state, expected_route):
    assert og.output_guard_routing(state) == expected_route


def test_output_get_prompt_reads_versioned_prompt(tmp_path, monkeypatch):
    prompts_file = tmp_path / "security_prompts.yaml"
    prompts_file.write_text(
        """
output_guard:
  unsafe_semantic_classifier_system:
    vtest: "output prompt vtest"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(og, "PROMPTS_FILE", prompts_file)

    value = og._get_prompt("unsafe_semantic_classifier_system", "vtest", "fallback_value")
    missing = og._get_prompt("unsafe_semantic_classifier_system", "missing", "fallback_value")

    assert value == "output prompt vtest"
    assert missing == "fallback_value"


def test_output_get_config_and_rules_from_policy_file(tmp_path, monkeypatch):
    policy_file = tmp_path / "security_policy.yaml"
    policy_file.write_text(
        """
output_guard:
  config:
    vtest:
      llm_unsafe_conf_threshold: 0.83
  unsafe_regex_rules:
    vtest:
            - pattern: '\\bforbidden\\b'
              reason: "Forbidden pattern"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(og, "POLICY_FILE", policy_file)

    cfg = og._get_output_guard_config("vtest")
    rules = og._get_unsafe_regex_rules("vtest")

    assert cfg["llm_unsafe_conf_threshold"] == 0.83
    assert rules == [(r"\bforbidden\b", "Forbidden pattern")]


def test_output_get_rules_fallback_when_policy_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(og, "POLICY_FILE", tmp_path / "no_file.yaml")

    rules = og._get_unsafe_regex_rules("any")

    assert len(rules) > 0
    assert isinstance(rules[0], tuple)


def test_llm_check_unsafe_content_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    unsafe, reason, confidence, evidence = og._llm_check_unsafe_content("u", "o")

    assert unsafe is False
    assert reason in {"llm_unsafe_check_fallback_safe", "llm_unsafe_check_disabled"}
    assert isinstance(confidence, float)
    assert isinstance(evidence, dict)


def test_llm_check_unsafe_content_low_confidence_not_blocked(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(og, "_get_output_guard_config", lambda _v: {"llm_unsafe_conf_threshold": 0.8})

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"unsafe": true, "reason": "maybe_unsafe", "confidence": 0.4, "category": "violence"}'
                        )
                    )
                ]
            )

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(og, "OpenAI", lambda api_key: _Client())

    unsafe, reason, confidence, evidence = og._llm_check_unsafe_content("user", "output")

    assert unsafe is False
    assert reason.startswith("low_confidence_unsafe:")
    assert confidence == 0.4
    assert evidence.get("category") == "violence"


def test_llm_check_unsafe_content_high_confidence_blocked(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(og, "_get_output_guard_config", lambda _v: {"llm_unsafe_conf_threshold": 0.7})

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"unsafe": true, "reason": "unsafe_content", "confidence": 0.95, "category": "illicit"}'
                        )
                    )
                ]
            )

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(og, "OpenAI", lambda api_key: _Client())

    unsafe, reason, confidence, evidence = og._llm_check_unsafe_content("user", "output")

    assert unsafe is True
    assert reason == "unsafe_content"
    assert confidence == 0.95
    assert evidence.get("category") == "illicit"
