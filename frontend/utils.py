"""
utils.py — Shared frontend utilities
"""

import re

def sanitize_cost(cost: str) -> str:
    """
    Normalises cost strings for display.
    - Collapses preference tags (★pref:...) to TBC
    - Normalises S$X / $X  →  SGD X
    - Strips verbose phrases like 'from $X per adult'
    """
    if not cost:
        return "TBC"
    c = cost.strip()
    # Preference tag leaked into cost field
    if c.startswith("★") or c.lower().startswith("pref:"):
        return "TBC"
    # Already clean
    if c.upper().startswith("SGD") or c.upper().startswith("USD") or c.lower() in ("free", "tbc"):
        return c
    # Strip "from … per adult/person/pax"
    c = re.sub(r'(?i)^from\s+', '', c)
    c = re.sub(r'(?i)\s+per\s+(adult|person|pax).*$', '', c).strip()
    # S$X → SGD X
    if c.upper().startswith("S$"):
        num = c[2:].strip()
        return f"SGD {num}"
    # Bare $X or $X–$Y → SGD X (or SGD X – Y)
    if c.startswith("$") and not c.startswith("$$"):
        num = c[1:].strip()
        parts = re.split(r'\s*[–\-]\s*\$?', num)
        if len(parts) == 2:
            return f"SGD {parts[0].strip()} – {parts[1].strip()}"
        return f"SGD {num}"
    return c

def sanitize_name(name: str) -> str:
    """Strip any pipe-appended metadata from non-flight item names."""
    if not name or "→" in name or "Airport" in name:  # flight names — leave intact
        return name
    return name.split(" | ")[0].strip() if " | " in name else name

def get_item_id(option: str, day_idx: int, item_idx: int, item_name: str) -> str:
    """Generate a consistent unique ID for an itinerary item."""
    # We include option to keep state (like visited) isolated per plan option
    # We include indices to handle multiple items with same name
    safe_name = re.sub(r'[^a-zA-Z0-9]', '', item_name).lower()[:20]
    return f"itinerary_{option}_d{day_idx}_i{item_idx}_{safe_name}"

def is_flight_item(item: dict) -> bool:
    return item.get("icon", "") == "✈️" or "flight" in item.get("name", "").lower()

def is_hotel_item(item: dict) -> bool:
    k = item.get("key", "") or ""
    return "hotel" in k or item.get("icon", "") == "🏨"
