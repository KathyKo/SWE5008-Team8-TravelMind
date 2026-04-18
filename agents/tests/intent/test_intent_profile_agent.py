import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
from importlib import import_module

import pytest

_STUB_MODULES = [
    "llm_guard", "llm_guard.input_scanners", "llm_guard.output_scanners",
    "torch", "transformers",
]
for _m in _STUB_MODULES:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

ip = import_module("agents.specialists.intent_profile")


# ---------------------------------------------------------------------------
# _normalize_preferences
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("", None),
        ("vegetarian, historical, zen_gardens", "vegetarian, historical, zen_gardens"),
        ("vegetarian and historical and zen_gardens", "vegetarian, historical, zen_gardens"),
        ("vegetarian; historical; zen_gardens", "vegetarian, historical, zen_gardens"),
        ("  food ,  culture  ", "food, culture"),
    ],
    ids=[
        "none_input",
        "empty_string",
        "comma_separated",
        "and_separated",
        "semicolon_separated",
        "whitespace_trimmed",
    ],
)
def test_normalize_preferences(raw, expected):
    assert ip._normalize_preferences(raw) == expected


# ---------------------------------------------------------------------------
# _parse_date_range
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dates_str,expected_start,expected_end",
    [
        (None, None, None),
        ("", None, None),
        ("2026-05-10 to 2026-05-15", "2026-05-10", "2026-05-15"),
        ("2026-06-01 ~ 2026-06-04", "2026-06-01", "2026-06-04"),
        ("2026-07-20", "2026-07-20", "2026-07-20"),
    ],
    ids=[
        "none_input",
        "empty_string",
        "standard_to_separator",
        "tilde_separator",
        "single_date",
    ],
)
def test_parse_date_range(dates_str, expected_start, expected_end):
    start, end = ip._parse_date_range(dates_str)
    assert start == expected_start
    assert end == expected_end


# ---------------------------------------------------------------------------
# _parse_budget
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "budget_str,expected_amount,expected_currency,expected_flexibility",
    [
        (None, None, "SGD", "flexible"),
        ("SGD 5000 strict", 5000, "SGD", "strict"),
        ("USD 3000 flexible", 3000, "USD", "flexible"),
        ("EUR 2000 no limit", 2000, "EUR", "no_limit"),
        ("$1500", 1500, "SGD", "flexible"),
    ],
    ids=[
        "none_input",
        "sgd_strict",
        "usd_flexible",
        "eur_no_limit",
        "dollar_sign_default",
    ],
)
def test_parse_budget(budget_str, expected_amount, expected_currency, expected_flexibility):
    result = ip._parse_budget(budget_str)
    assert result["amount"] == expected_amount
    assert result["currency"] == expected_currency
    assert result["flexibility"] == expected_flexibility


# ---------------------------------------------------------------------------
# _infer_travelers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Plan a solo trip", 1),
        ("Trip for 2 travelers", 2),
        ("5 people going to Bali", 5),
        ("My couple trip to Paris", 2),
        ("three persons going to Tokyo", 3),
    ],
    ids=[
        "solo_default",
        "numeric_travelers",
        "numeric_people",
        "couple_keyword",
        "word_persons",
    ],
)
def test_infer_travelers(text, expected):
    assert ip._infer_travelers(text) == expected


# ---------------------------------------------------------------------------
# _extract_requirements
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw_text,preferences,expected",
    [
        ("I am vegetarian, no red-eye flights", None, ["vegetarian", "no_red_eye_flights"]),
        ("Normal trip", "halal, pet-friendly", ["halal", "pet_friendly"]),
        ("Just a vacation", None, []),
        ("vegan traveler", "vegan food", ["vegan"]),
    ],
    ids=[
        "vegetarian_no_redeye",
        "halal_pet_friendly_in_prefs",
        "no_requirements",
        "vegan_dedup",
    ],
)
def test_extract_requirements(raw_text, preferences, expected):
    assert ip._extract_requirements(raw_text, preferences) == expected


# ---------------------------------------------------------------------------
# _infer_priority
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prefs,text,expected",
    [
        ("budget trip", "", "cost_effective"),
        ("", "time efficient schedule", "time_saving"),
        ("relaxed cultural", "enjoy the view", "comfort"),
    ],
    ids=["cost_effective", "time_saving", "comfort_default"],
)
def test_infer_priority(prefs, text, expected):
    assert ip._infer_priority(prefs, text) == expected


# ---------------------------------------------------------------------------
# _build_soft_preferences
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "preferences,raw_text,expected_style,expected_pace",
    [
        ("relaxed, historical", "slow trip", "relaxed", 2),
        ("intense, adventure", "fast-paced trip", "intense", 4),
        ("modern, food", "city break", "balanced", 3),
    ],
    ids=["relaxed", "intense", "balanced"],
)
def test_build_soft_preferences_style_and_pace(preferences, raw_text, expected_style, expected_pace):
    result = ip._build_soft_preferences(preferences, raw_text)
    assert result["travel_style"] == expected_style
    assert result["pace"] == expected_pace


def test_build_soft_preferences_interest_tags():
    result = ip._build_soft_preferences("culture, zen, food", "historical temple tour")
    assert "historical" in result["interest_tags"]
    assert "zen_gardens" in result["interest_tags"]
    assert "fine_dining" in result["interest_tags"]


def test_build_soft_preferences_default_tags():
    result = ip._build_soft_preferences("", "some random trip")
    assert result["interest_tags"] == ["local_experience"]


def test_build_soft_preferences_vibe_traditional():
    result = ip._build_soft_preferences("traditional", "")
    assert result["vibe"] == "traditional_cultural"


def test_build_soft_preferences_vibe_modern():
    result = ip._build_soft_preferences("modern", "modern city")
    assert result["vibe"] == "modern_urban"


# ---------------------------------------------------------------------------
# _normalize_time_pref
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("morning", "morning"),
        ("early_morning", "early morning"),
        ("  Evening  ", "evening"),
        ("invalid_value", None),
    ],
    ids=["none", "canonical", "underscore_to_space", "whitespace_trimmed", "invalid"],
)
def test_normalize_time_pref(value, expected):
    assert ip._normalize_time_pref(value) == expected


# ---------------------------------------------------------------------------
# _build_search_queries
# ---------------------------------------------------------------------------
def test_build_search_queries_structure():
    queries = ip._build_search_queries(
        origin="Singapore",
        destination="Kyoto",
        start_date="2026-05-10",
        soft_preferences={"travel_style": "relaxed", "interest_tags": ["historical", "zen_gardens"]},
        requirements=["vegetarian"],
    )
    assert len(queries) == 4

    types = [q["type"] for q in queries]
    assert types == ["api_flight", "api_hotel", "rag_attraction", "rag_local_info"]

    assert all(q.get("query") for q in queries)


def test_build_search_queries_vegetarian_clause():
    queries = ip._build_search_queries(
        origin="Singapore",
        destination="Kyoto",
        start_date="2026-05-10",
        soft_preferences={"travel_style": "relaxed", "interest_tags": ["historical"]},
        requirements=["vegetarian"],
    )
    hotel_query = next(q for q in queries if q["type"] == "api_hotel")
    assert "vegetarian breakfast" in hotel_query["query"]


def test_build_search_queries_no_vegetarian_clause():
    queries = ip._build_search_queries(
        origin="Singapore",
        destination="Kyoto",
        start_date="2026-05-10",
        soft_preferences={"travel_style": "balanced", "interest_tags": ["modern"]},
        requirements=[],
    )
    hotel_query = next(q for q in queries if q["type"] == "api_hotel")
    assert "vegetarian" not in hotel_query["query"]


# ---------------------------------------------------------------------------
# _build_user_profile
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "preferences,substring",
    [
        ("vegetarian, culture, relaxed", "diet=vegetarian"),
        ("modern, food", "interests=modern+food"),
        (None, None),
        ("", None),
    ],
    ids=["diet_present", "interests_present", "none_input", "empty_input"],
)
def test_build_user_profile(preferences, substring):
    result = ip._build_user_profile(preferences)
    if substring is None:
        assert result is None
    else:
        assert substring in result


# ---------------------------------------------------------------------------
# _build_user_profile_structured
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "preferences,expected_dietary,expected_accommodation",
    [
        ("vegetarian, historical", "vegetarian", "budget"),
        ("vegan, luxury", "vegan", "luxury"),
        ("halal, hostel", "halal", "hostel"),
        ("modern, boutique", "none", "boutique"),
        (None, "none", "budget"),
    ],
    ids=[
        "vegetarian_budget",
        "vegan_luxury",
        "halal_hostel",
        "no_diet_boutique",
        "none_defaults",
    ],
)
def test_build_user_profile_structured_base_preferences(
    preferences, expected_dietary, expected_accommodation
):
    state = {"user_id": "u-test"}
    budget_obj = {"amount": 3000, "currency": "SGD", "flexibility": "flexible"}
    soft_prefs = {"interest_tags": ["historical"]}

    result = ip._build_user_profile_structured(
        state=state,
        preferences=preferences,
        soft_preferences=soft_prefs,
        budget_obj=budget_obj,
        requirements=[],
    )
    assert result["base_preferences"]["dietary"] == expected_dietary
    assert result["base_preferences"]["accommodation"] == expected_accommodation
    assert result["user_id"] == "u-test"


@pytest.mark.parametrize(
    "amount,expected_level",
    [
        (None, "medium"),
        (1000, "low"),
        (3000, "medium"),
        (7000, "high"),
        (15000, "ultra_luxury"),
    ],
    ids=["none_medium", "low", "medium", "high", "ultra_luxury"],
)
def test_build_user_profile_structured_budget_level(amount, expected_level):
    state = {"user_id": "u-test"}
    budget_obj = {"amount": amount, "currency": "SGD", "flexibility": "flexible"}
    soft_prefs = {"interest_tags": []}

    result = ip._build_user_profile_structured(
        state=state,
        preferences=None,
        soft_preferences=soft_prefs,
        budget_obj=budget_obj,
        requirements=[],
    )
    assert result["travel_history"]["avg_budget_level"] == expected_level


# ---------------------------------------------------------------------------
# intent_profile (main node) — empty messages
# ---------------------------------------------------------------------------
def test_intent_profile_returns_incomplete_when_no_messages():
    result = ip.intent_profile({"messages": []})
    assert result == {"is_complete": False}


# ---------------------------------------------------------------------------
# intent_profile — full state (LLM branch skipped)
# ---------------------------------------------------------------------------
def test_intent_profile_full_state_skips_llm(full_state_builder):
    state = full_state_builder()
    result = ip.intent_profile(state)

    payload = result.get("intent_profile_output", {})
    assert "session_id" in payload
    assert "hard_constraints" in payload
    assert "soft_preferences" in payload
    assert "search_queries" in payload

    hc = payload["hard_constraints"]
    assert hc["origin"] == "Singapore"
    assert hc["destination"] == "Kyoto, Japan"
    assert hc["start_date"] == "2026-05-10"
    assert hc["end_date"] == "2026-05-15"
    assert hc["budget"]["amount"] == 5000
    assert hc["budget"]["currency"] == "SGD"
    assert hc["budget"]["flexibility"] == "strict"


# ---------------------------------------------------------------------------
# intent_profile — schema validation (parameterized)
# ---------------------------------------------------------------------------
ALLOWED_FLEXIBILITY = {"strict", "flexible", "no_limit"}
ALLOWED_TRAVEL_STYLE = {"relaxed", "balanced", "intense"}
ALLOWED_PRIORITY = {"cost_effective", "time_saving", "comfort"}
ALLOWED_QUERY_TYPES = {"api_flight", "api_hotel", "rag_attraction", "rag_local_info"}
ALLOWED_TIME_PREFS = {"midnight", "early morning", "morning", "afternoon", "evening", "night"}


def _validate_output(payload: dict) -> list[str]:
    issues = []
    required_top = {"session_id", "outbound_time_pref", "return_time_pref", "hard_constraints", "soft_preferences", "search_queries"}
    missing_top = [k for k in required_top if k not in payload]
    if missing_top:
        issues.append(f"missing top-level keys: {missing_top}")
    if payload.get("outbound_time_pref") not in ALLOWED_TIME_PREFS:
        issues.append(f"outbound_time_pref invalid: {payload.get('outbound_time_pref')}")
    if payload.get("return_time_pref") not in ALLOWED_TIME_PREFS:
        issues.append(f"return_time_pref invalid: {payload.get('return_time_pref')}")

    hc = payload.get("hard_constraints", {})
    for key in ["origin", "destination", "start_date", "end_date", "budget", "travelers", "requirements"]:
        if key not in hc:
            issues.append(f"hard_constraints missing '{key}'")

    budget = hc.get("budget", {})
    if not isinstance(budget, dict):
        issues.append("hard_constraints.budget must be object")
    else:
        for key in ["amount", "currency", "flexibility"]:
            if key not in budget:
                issues.append(f"budget missing '{key}'")
        if budget.get("flexibility") not in ALLOWED_FLEXIBILITY:
            issues.append("budget.flexibility invalid")

    sp = payload.get("soft_preferences", {})
    for key in ["travel_style", "interest_tags", "pace", "vibe", "priority"]:
        if key not in sp:
            issues.append(f"soft_preferences missing '{key}'")
    if sp.get("travel_style") not in ALLOWED_TRAVEL_STYLE:
        issues.append("soft_preferences.travel_style invalid")
    if sp.get("priority") not in ALLOWED_PRIORITY:
        issues.append("soft_preferences.priority invalid")

    sq = payload.get("search_queries", [])
    if not isinstance(sq, list):
        issues.append("search_queries must be list")
    else:
        for i, item in enumerate(sq):
            if not isinstance(item, dict):
                issues.append(f"search_queries[{i}] must be object")
                continue
            if item.get("type") not in ALLOWED_QUERY_TYPES:
                issues.append(f"search_queries[{i}].type invalid")
            if not item.get("query"):
                issues.append(f"search_queries[{i}].query empty")

    return issues


@pytest.mark.parametrize(
    "text,origin,destination,dates,budget,preferences,duration,outbound,ret",
    [
        (
            "I want to visit Kyoto for 5 days, vegetarian, no red-eye flights, relaxed traditional cultural trip.",
            "Singapore",
            "Kyoto, Japan",
            "2026-05-10 to 2026-05-15",
            "SGD 5000 strict",
            "vegetarian, historical, zen_gardens, fine dining, relaxed, traditional",
            "5 days",
            "morning",
            "afternoon",
        ),
        (
            "Plan a balanced 4-day Tokyo trip for one person. Budget is flexible around USD 3000, prefer modern and food.",
            "Singapore",
            "Tokyo, Japan",
            "2026-06-01 to 2026-06-04",
            "USD 3000 flexible",
            "modern, food, balanced",
            "4 days",
            "early morning",
            "evening",
        ),
    ],
    ids=["kyoto_relaxed", "tokyo_balanced"],
)
def test_intent_profile_schema_validation(
    text, origin, destination, dates, budget, preferences, duration, outbound, ret,
    full_state_builder,
):
    state = full_state_builder(
        text=text,
        origin=origin,
        destination=destination,
        dates=dates,
        budget=budget,
        preferences=preferences,
        duration=duration,
        outbound_time_pref=outbound,
        return_time_pref=ret,
    )
    result = ip.intent_profile(state)
    payload = result.get("intent_profile_output", {})
    issues = _validate_output(payload)
    assert issues == [], f"schema validation failed: {issues}"


# ---------------------------------------------------------------------------
# intent_profile — LLM extraction branch (monkeypatched)
# ---------------------------------------------------------------------------
def _setup_llm_extraction(monkeypatch, extracted: dict):
    """Patch ChatOpenAI + _json_invoke so no real LLM call is made."""

    monkeypatch.setattr(
        ip,
        "_json_invoke",
        lambda llm, system_prompt, messages: extracted,
    )
    monkeypatch.setattr(
        ip,
        "ChatOpenAI",
        lambda *args, **kwargs: SimpleNamespace(),
    )


def test_intent_profile_llm_extraction_fills_missing_fields(monkeypatch, intent_state_builder):
    extracted = {
        "origin": "Singapore",
        "destination": "Osaka, Japan",
        "dates": "2026-08-01 to 2026-08-05",
        "budget": "SGD 2000 flexible",
        "preferences": "food, modern",
        "duration": "5 days",
        "outbound_time_pref": "morning",
        "return_time_pref": "evening",
    }
    _setup_llm_extraction(monkeypatch, extracted)

    state = intent_state_builder("Plan a food trip to Osaka for 5 days")
    result = ip.intent_profile(state)

    assert result["origin"] == "Singapore"
    assert result["destination"] == "Osaka, Japan"
    assert result["dates"] == "2026-08-01 to 2026-08-05"
    assert result["budget"] == "SGD 2000 flexible"

    payload = result.get("intent_profile_output", {})
    assert payload["hard_constraints"]["destination"] == "Osaka, Japan"
    issues = _validate_output(payload)
    assert issues == [], f"schema validation failed: {issues}"


def test_intent_profile_llm_extraction_partial_result(monkeypatch, intent_state_builder):
    extracted = {
        "origin": "Singapore",
        "destination": "Seoul, Korea",
        "dates": None,
        "budget": None,
        "preferences": "culture",
        "duration": None,
        "outbound_time_pref": None,
        "return_time_pref": None,
    }
    _setup_llm_extraction(monkeypatch, extracted)

    state = intent_state_builder("I want to visit Seoul")
    result = ip.intent_profile(state)

    assert result["origin"] == "Singapore"
    assert result["destination"] == "Seoul, Korea"
    assert result["preferences"] == "culture"


def test_intent_profile_llm_extraction_returns_empty(monkeypatch, intent_state_builder):
    _setup_llm_extraction(monkeypatch, {})

    state = intent_state_builder("Hello there")
    result = ip.intent_profile(state)

    assert result.get("is_complete") is False
    assert "intent_profile_output" in result


# ---------------------------------------------------------------------------
# intent_profile — state field passthrough
# ---------------------------------------------------------------------------
def test_intent_profile_preserves_user_id(full_state_builder):
    state = full_state_builder(user_id="u-alice-001")
    result = ip.intent_profile(state)

    profile = result.get("user_profile_structured", {})
    assert profile["user_id"] == "u-alice-001"


def test_intent_profile_uses_existing_session_id(full_state_builder):
    state = full_state_builder()
    state["session_id"] = "tm-custom-session"
    result = ip.intent_profile(state)

    assert result["session_id"] == "tm-custom-session"
    assert result["intent_profile_output"]["session_id"] == "tm-custom-session"


# ---------------------------------------------------------------------------
# intent_profile — time preferences forwarded
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "outbound,ret,expected_outbound,expected_ret",
    [
        ("morning", "afternoon", "morning", "afternoon"),
        ("early_morning", "evening", "early morning", "evening"),
        ("midnight", "night", "midnight", "night"),
    ],
    ids=["standard", "underscore_normalized", "edge_times"],
)
def test_intent_profile_time_preferences(
    full_state_builder, outbound, ret, expected_outbound, expected_ret
):
    state = full_state_builder(outbound_time_pref=outbound, return_time_pref=ret)
    result = ip.intent_profile(state)

    payload = result.get("intent_profile_output", {})
    assert payload["outbound_time_pref"] == expected_outbound
    assert payload["return_time_pref"] == expected_ret


# ---------------------------------------------------------------------------
# intent_profile — requirements propagation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,preferences,expected_reqs",
    [
        ("vegetarian traveler, no red-eye flights", "vegetarian", ["vegetarian", "no_red_eye_flights"]),
        ("normal trip", "halal", ["halal"]),
        ("just a vacation", None, []),
    ],
    ids=["vegetarian_redeye", "halal_only", "no_reqs"],
)
def test_intent_profile_requirements_in_output(
    full_state_builder, text, preferences, expected_reqs
):
    state = full_state_builder(text=text, preferences=preferences or "general")
    result = ip.intent_profile(state)

    reqs = result["intent_profile_output"]["hard_constraints"]["requirements"]
    assert reqs == expected_reqs


# ---------------------------------------------------------------------------
# _build_session_id
# ---------------------------------------------------------------------------
def test_build_session_id_format():
    sid = ip._build_session_id("any text")
    assert sid.startswith("tm-")
    parts = sid.split("-")
    assert len(parts) >= 4


def test_build_session_id_deterministic():
    a = ip._build_session_id("same input")
    b = ip._build_session_id("same input")
    assert a == b


def test_build_session_id_differs_for_different_input():
    a = ip._build_session_id("input one")
    b = ip._build_session_id("input two")
    assert a != b
