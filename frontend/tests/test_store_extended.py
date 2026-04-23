"""
frontend/tests/test_store_extended.py — extended coverage for data/store.py
"""
from pathlib import Path
import sys

from data.store import (
    USERS, OPTION_META, ITINERARIES,
    EXPLAIN_DATA, AGENT_STEPS, DEBATE_MESSAGES,
    SITUATIONS, TIME_OPTIONS, ALT_OPTIONS, REPLAN_LOG,
    ATTACK_PATTERNS, PRESETS, PIPELINE_STAGES,
)

frontend_dir = Path(__file__).resolve().parents[1]
if str(frontend_dir) not in sys.path:
    sys.path.insert(0, str(frontend_dir))




# ── USERS ─────────────────────────────────────────────────────────────────────

def test_all_users_have_required_fields():
    for email, profile in USERS.items():
        assert "@" in email
        assert "name" in profile
        assert "avatar" in profile
        assert "prefs" in profile
        assert "password" in profile
        assert isinstance(profile["prefs"], list)


def test_user_prefs_are_non_empty():
    for email, profile in USERS.items():
        assert len(profile["prefs"]) >= 1, f"{email} has empty prefs"


def test_carol_has_adventure_pref():
    assert "adventure" in USERS["carol@example.com"]["prefs"]


def test_bob_profile_values():
    bob = USERS["bob@example.com"]
    assert bob["name"] == "Bob"
    assert "food" in bob["prefs"]


# ── OPTION_META ───────────────────────────────────────────────────────────────

def test_option_meta_required_keys():
    required = {"label", "desc", "budget", "style", "badge"}
    for key, meta in OPTION_META.items():
        missing = required - set(meta.keys())
        assert not missing, f"Option {key} missing keys: {missing}"


def test_option_c_is_cheapest():
    budgets = {k: int(v["budget"].replace("SGD ", "").replace(",", ""))
               for k, v in OPTION_META.items()}
    assert budgets["C"] < budgets["B"] < budgets["A"]


def test_option_meta_labels_contain_option_letter():
    for letter in ("A", "B", "C"):
        assert f"Option {letter}" in OPTION_META[letter]["label"]


# ── ITINERARIES ───────────────────────────────────────────────────────────────

def test_all_options_have_itineraries():
    assert set(ITINERARIES.keys()) == {"A", "B", "C"}


def test_each_day_has_required_keys():
    for option, days in ITINERARIES.items():
        for day in days:
            assert "day" in day, f"Option {option} missing 'day' key"
            assert "items" in day, f"Option {option} missing 'items' key"
            assert "budget" in day


def test_items_have_required_fields():
    for option, days in ITINERARIES.items():
        for day in days:
            for item in day["items"]:
                assert "time" in item
                assert "name" in item
                assert "cost" in item


def test_option_a_has_three_days():
    assert len(ITINERARIES["A"]) == 3


def test_fushimi_is_first_item_day1_option_a():
    first_item = ITINERARIES["A"][0]["items"][0]
    assert first_item["key"] == "fushimi"
    assert first_item["cost"] == "Free"


# ── EXPLAIN_DATA ──────────────────────────────────────────────────────────────

def test_explain_data_keys_match_itinerary_keys():
    itinerary_keys = {
        item["key"]
        for days in ITINERARIES.values()
        for day in days
        for item in day["items"]
        if item.get("key")
    }
    for key in EXPLAIN_DATA:
        assert key in itinerary_keys, f"Explain key '{key}' not found in itineraries"


def test_explain_data_structure():
    for key, data in EXPLAIN_DATA.items():
        assert "name" in data
        assert "matches" in data
        assert isinstance(data["matches"], list)
        assert len(data["matches"]) >= 1
        assert "scores" in data
        for score_name, score_val in data["scores"]:
            assert isinstance(score_val, int)
            assert 0 <= score_val <= 100


# ── AGENT_STEPS ───────────────────────────────────────────────────────────────

def test_agent_steps_count():
    assert len(AGENT_STEPS) == 6


def test_agent_steps_structure():
    for step in AGENT_STEPS:
        assert "icon" in step
        assert "name" in step
        assert "detail" in step


# ── DEBATE_MESSAGES ───────────────────────────────────────────────────────────

def test_debate_messages_have_valid_roles():
    valid_roles = {"critique", "reply", "verdict"}
    for msg in DEBATE_MESSAGES:
        assert len(msg) == 3
        role, agent, text = msg
        assert role in valid_roles
        assert isinstance(text, str) and len(text) > 0


def test_debate_ends_with_verdict():
    last = DEBATE_MESSAGES[-1]
    assert last[0] == "verdict"


# ── REPLAN DATA ───────────────────────────────────────────────────────────────

def test_situations_have_required_fields():
    for s in SITUATIONS:
        assert "key" in s
        assert "emoji" in s
        assert "label" in s
        assert "desc" in s


def test_time_options_count():
    assert len(TIME_OPTIONS) == 3


def test_alt_options_have_price_and_rating():
    for opt in ALT_OPTIONS:
        assert "price" in opt
        assert "rating" in opt
        assert "name" in opt
        assert "dist" in opt


def test_replan_log_entries_have_level_and_message():
    for entry in REPLAN_LOG:
        assert len(entry) == 2
        level, msg = entry
        assert level in {"INFO", "OK", "WARN", "ERROR"}
        assert isinstance(msg, str)


# ── SECURITY DATA ─────────────────────────────────────────────────────────────

def test_attack_patterns_have_required_fields():
    for pattern in ATTACK_PATTERNS:
        assert "keywords" in pattern
        assert "type" in pattern
        assert "stage" in pattern
        assert "reason" in pattern
        assert isinstance(pattern["keywords"], list)
        assert isinstance(pattern["stage"], int)


def test_presets_are_tuples_with_label_and_text():
    for preset in PRESETS:
        assert len(preset) == 2
        label, text = preset
        assert isinstance(label, str)
        assert isinstance(text, str)


def test_pipeline_stages_count_and_structure():
    assert len(PIPELINE_STAGES) == 5
    for stage in PIPELINE_STAGES:
        name, desc = stage
        assert isinstance(name, str)
        assert isinstance(desc, str)


def test_normal_query_preset_exists():
    labels = [p[0] for p in PRESETS]
    assert any("Normal" in label for label in labels)
