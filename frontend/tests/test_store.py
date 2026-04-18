from pathlib import Path
import sys
from data.store import USERS, OPTION_META, ITINERARIES

frontend_dir = Path(__file__).resolve().parents[1]
if str(frontend_dir) not in sys.path:
    sys.path.insert(0, str(frontend_dir))



def test_demo_users_are_seeded():
    assert set(USERS) == {"alice@example.com", "bob@example.com", "carol@example.com"}
    assert USERS["alice@example.com"]["password"] == "demo123"
    assert "culture" in USERS["alice@example.com"]["prefs"]


def test_itinerary_option_metadata_has_three_variants():
    assert set(OPTION_META) == {"A", "B", "C"}
    assert OPTION_META["A"]["label"].startswith("Option A")
    assert OPTION_META["B"]["budget"].startswith("SGD")


def test_itineraries_have_entries_and_item_keys():
    assert len(ITINERARIES["A"]) >= 1
    first_day = ITINERARIES["A"][0]
    assert "day" in first_day
    assert any(item.get("key") for item in first_day["items"])
