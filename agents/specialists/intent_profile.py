import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..llm_config import OPENAI_MODEL

TIME_PREF_CHOICES = {
    "midnight",
    "early morning",
    "morning",
    "afternoon",
    "evening",
    "night",
}


def _normalize_preferences(prefs: Optional[str]) -> Optional[str]:
    if not prefs:
        return None
    # Normalize separators to comma for consistent downstream parsing.
    s = prefs.replace(" and ", ", ").replace(";", ",")
    parts = [p.strip() for p in re.split(r",|\s{2,}", s) if p.strip()]
    # Keep original casing for user-facing, but strip whitespace.
    return ", ".join(parts) if parts else None


def _build_user_profile(preferences: Optional[str]) -> Optional[str]:
    """
    Simple deterministic profile builder from preferences keywords.
    Example output: "diet=vegetarian; interests=culture; intensity=low-intensity"
    """
    prefs = _normalize_preferences(preferences)
    if not prefs:
        return None

    tokens = [t.strip().lower() for t in prefs.split(",") if t.strip()]
    dietary = [t for t in tokens if any(x in t for x in ["vegetarian", "vegan", "pesc", "halal"])]
    intensity = [t for t in tokens if any(x in t for x in ["low", "moderate", "high", "relaxed", "intensity"])]
    interests = [t for t in tokens if t not in dietary and t not in intensity]

    parts = []
    if dietary:
        parts.append(f"diet={'+'.join(dietary)}")
    if interests:
        parts.append(f"interests={'+'.join(interests)}")
    if intensity:
        parts.append(f"intensity={'+'.join(intensity)}")
    if not parts:
        return prefs
    return "; ".join(parts)


def _json_invoke(llm: ChatOpenAI, system_prompt: str, messages: list) -> Dict[str, Any]:
    """
    Call the LLM expecting a JSON object. Returns {} on parse failure.
    """
    res = llm.invoke(
        [SystemMessage(content=system_prompt), *messages],
        response_format={"type": "json_object"},
    )
    try:
        # LangChain message content should be a JSON string.
        import json

        return json.loads(res.content)
    except Exception:
        return {}


def _parse_date_range(dates: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not dates:
        return None, None
    text = dates.strip()
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|~|–|-)\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1), m.group(2)
    m2 = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m2:
        return m2.group(1), m2.group(1)
    return None, None


def _parse_budget(budget: Optional[str]) -> Dict[str, Any]:
    if not budget:
        return {"amount": None, "currency": "SGD", "flexibility": "flexible"}
    text = str(budget).strip()
    amount = None
    currency = "SGD"
    flexibility = "flexible"
    m = re.search(r"(\d[\d,]*)", text.replace(" ", ""))
    if m:
        try:
            amount = int(m.group(1).replace(",", ""))
        except ValueError:
            amount = None
    if "usd" in text.lower():
        currency = "USD"
    elif "eur" in text.lower():
        currency = "EUR"
    elif "sgd" in text.lower() or "$" in text:
        currency = "SGD"
    if any(k in text.lower() for k in ["strict", "hard", "fixed"]):
        flexibility = "strict"
    elif any(k in text.lower() for k in ["no limit", "unlimited", "no_limit"]):
        flexibility = "no_limit"
    elif any(k in text.lower() for k in ["flex", "flexible", "can exceed"]):
        flexibility = "flexible"
    return {"amount": amount, "currency": currency, "flexibility": flexibility}


def _infer_travelers(raw_text: str) -> int:
    m = re.search(r"\b(\d+)\s*(?:travellers|travelers|people|persons|pax)\b", raw_text, flags=re.I)
    if m:
        return max(1, int(m.group(1)))
    m_word = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:travellers|travelers|people|persons|pax)\b",
        raw_text,
        flags=re.I,
    )
    if m_word:
        words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        return words[m_word.group(1).lower()]
    if re.search(r"\b(couple|we|us|two of us)\b", raw_text, flags=re.I):
        return 2
    return 1


def _extract_requirements(raw_text: str, preferences: Optional[str]) -> List[str]:
    reqs: List[str] = []
    source = f"{raw_text} {preferences or ''}".lower()
    if "vegetarian" in source:
        reqs.append("vegetarian")
    if "vegan" in source:
        reqs.append("vegan")
    if "halal" in source:
        reqs.append("halal")
    if re.search(r"no red[- ]?eye", source):
        reqs.append("no_red_eye_flights")
    if "pet friendly" in source or "pet-friendly" in source:
        reqs.append("pet_friendly")
    return reqs


def _build_soft_preferences(preferences: Optional[str], raw_text: str) -> Dict[str, Any]:
    prefs = (preferences or "").lower()
    text = raw_text.lower()
    if any(k in prefs or k in text for k in ["relaxed", "low-intensity", "slow", "laid-back"]):
        travel_style = "relaxed"
        pace = 2
    elif any(k in prefs or k in text for k in ["intense", "fast-paced", "adventure"]):
        travel_style = "intense"
        pace = 4
    else:
        travel_style = "balanced"
        pace = 3

    tags: List[str] = []
    if any(k in prefs or k in text for k in ["culture", "historical", "history"]):
        tags.append("historical")
    if any(k in prefs or k in text for k in ["modern", "city", "urban"]):
        tags.append("modern")
    if any(k in prefs or k in text for k in ["sport", "running", "cycling", "ski"]):
        tags.append("sport")
    if any(k in prefs or k in text for k in ["nature", "outdoor", "hiking"]):
        tags.append("nature")
    if any(k in prefs or k in text for k in ["zen", "temple", "garden"]):
        tags.append("zen_gardens")
    if any(k in prefs or k in text for k in ["food", "fine dining", "dining"]):
        tags.append("fine_dining")
    if not tags:
        tags = ["local_experience"]
    tags = list(dict.fromkeys(tags))

    if "traditional" in prefs or "traditional" in text:
        vibe = "traditional_cultural"
    elif "modern" in prefs or "modern" in text:
        vibe = "modern_urban"
    else:
        vibe = "traditional_cultural"

    return {
        "travel_style": travel_style,
        "interest_tags": tags,
        "pace": pace,
        "vibe": vibe,
        "priority": _infer_priority(prefs, text),
    }


def _infer_priority(prefs: str, text: str) -> str:
    if any(k in prefs or k in text for k in ["budget", "cheap", "cost", "value"]):
        return "cost_effective"
    if any(k in prefs or k in text for k in ["fast", "time", "efficient", "tight schedule"]):
        return "time_saving"
    return "comfort"


def _build_search_queries(origin: Optional[str], destination: Optional[str], start_date: Optional[str], soft_preferences: Dict[str, Any], requirements: List[str]) -> List[Dict[str, str]]:
    o = origin or "origin"
    d = destination or "destination"
    s = start_date or "travel_date"
    veg_clause = " with vegetarian breakfast" if ("vegetarian" in requirements or "vegan" in requirements) else ""
    style = soft_preferences.get("travel_style", "balanced")
    interest_text = " and ".join(soft_preferences.get("interest_tags", ["local highlights"]))

    return [
        {"type": "api_flight", "query": f"flights from {o} to {d} on {s}"},
        {"type": "api_hotel", "query": f"{style} traditional ryokans in {d}{veg_clause}"},
        {"type": "rag_attraction", "query": f"best {interest_text} in {d} for photography"},
        {"type": "rag_local_info", "query": f"{d} transportation guide for {style} pace travel"},
    ]


def _build_session_id(raw_text: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = str(abs(hash(raw_text)) % 1000).zfill(3)
    return f"tm-{day}-{suffix}"


def _normalize_time_pref(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower().replace("_", " ")
    # accept "early_morning" style input, output canonical value.
    if v in TIME_PREF_CHOICES:
        return v
    return None


def _build_user_profile_structured(
    state: dict,
    preferences: Optional[str],
    soft_preferences: Dict[str, Any],
    budget_obj: Dict[str, Any],
    requirements: List[str],
) -> Dict[str, Any]:
    pref_text = (preferences or "").lower()
    if "vegan" in pref_text:
        dietary = "vegan"
    elif "vegetarian" in pref_text:
        dietary = "vegetarian"
    elif "halal" in pref_text:
        dietary = "halal"
    elif "gluten" in pref_text:
        dietary = "gluten_free"
    else:
        dietary = "none"

    if any(k in pref_text for k in ["luxury", "5-star", "premium"]):
        accommodation = "luxury"
    elif any(k in pref_text for k in ["boutique", "ryokan"]):
        accommodation = "boutique"
    elif any(k in pref_text for k in ["hostel"]):
        accommodation = "hostel"
    elif any(k in pref_text for k in ["apartment", "airbnb"]):
        accommodation = "apartment"
    else:
        accommodation = "budget"

    if any(k in pref_text for k in ["wheelchair"]):
        accessibility = "wheelchair"
    elif any(k in pref_text for k in ["low mobility", "limited mobility"]):
        accessibility = "low_mobility"
    else:
        accessibility = "none"

    amount = budget_obj.get("amount")
    if amount is None:
        avg_budget_level = "medium"
    elif amount < 1500:
        avg_budget_level = "low"
    elif amount < 5000:
        avg_budget_level = "medium"
    elif amount < 10000:
        avg_budget_level = "high"
    else:
        avg_budget_level = "ultra_luxury"

    tags = soft_preferences.get("interest_tags", [])
    bias_score = {
        "culture": 0.8 if any(t in tags for t in ["historical", "zen_gardens"]) else 0.3,
        "adventure": 0.7 if "sport" in tags or "nature" in tags else 0.2,
        "food": 0.7 if "fine_dining" in tags else 0.3,
    }

    return {
        "user_id": state.get("user_id") or "anonymous_user",
        "base_preferences": {
            "dietary": dietary,
            "accommodation": accommodation,
            "accessibility": accessibility,
        },
        "travel_history": {
            "top_interests": tags[:3] if tags else ["historical"],
            "avg_budget_level": avg_budget_level,
            "bias_score": bias_score,
        },
        "derived_requirements": requirements,
    }


def intent_profile(state: dict) -> dict:
    """
    Agent1: Intent Profile

    Input:
      - user message (state.messages)
      - possibly already provided fields (origin/destination/dates/budget/preferences/duration)

    Output (state patch):
      - standardized travel info: origin, destination, dates, budget, duration
      - user profile summary: user_profile (derived from preferences)
      - standardized preferences: preferences
    """
    messages = state.get("messages", [])
    if not messages:
        # Nothing to extract; return a no-op patch.
        return {"is_complete": False}

    last_user = messages[-1]
    raw_text = last_user.get("content", "") if isinstance(last_user, dict) else str(last_user)

    # Prefer fields already provided by the API/UI.
    origin = state.get("origin")
    destination = state.get("destination")
    dates = state.get("dates")
    budget = state.get("budget")
    preferences = state.get("preferences")
    duration = state.get("duration")
    outbound_time_pref = _normalize_time_pref(state.get("outbound_time_pref"))
    return_time_pref = _normalize_time_pref(state.get("return_time_pref"))

    has_core = all([origin, destination, dates, budget])
    if not has_core or not preferences:
        system_prompt = """You are the Agent1 Intent Profile extractor for a travel planning system.

Your task: extract the traveller's intent and profile from the latest user message.

Return ONLY a single JSON object with exactly these keys:
{{
  "origin": string | null,
  "destination": string | null,
  "dates": string | null,
  "budget": string | null,
  "preferences": string | null,
  "duration": string | null,
  "user_profile": string | null,
  "travelers": integer | null,
  "outbound_time_pref": string | null,
  "return_time_pref": string | null
}}

Rules:
1) Use ONLY explicit information found in the latest user message.
2) If an item is not present, set it to null.
3) For dates, keep the user's format (e.g., "2026-03-10 to 2026-03-14" or "next May").
4) For preferences, return a comma-separated keyword string (e.g., "culture, vegetarian, low-intensity").
5) user_profile should be a short human-readable summary derived from preferences (e.g., "diet=vegetarian; interests=culture; intensity=low-intensity").
6) travelers should be an integer when mentioned (e.g. "2 travelers"), otherwise null.
7) outbound_time_pref and return_time_pref must be one of:
   midnight, early morning, morning, afternoon, evening, night.
"""

        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        extracted = _json_invoke(llm, system_prompt, messages)
        origin = extracted.get("origin") or origin
        destination = extracted.get("destination") or destination
        dates = extracted.get("dates") or dates
        budget = extracted.get("budget") or budget
        preferences = extracted.get("preferences") or preferences
        duration = extracted.get("duration") or duration
        outbound_time_pref = _normalize_time_pref(extracted.get("outbound_time_pref")) or outbound_time_pref
        return_time_pref = _normalize_time_pref(extracted.get("return_time_pref")) or return_time_pref

    preferences = _normalize_preferences(preferences)
    user_profile = _build_user_profile(preferences) or state.get("user_profile")
    travelers = state.get("travelers") or _infer_travelers(raw_text)
    start_date, end_date = _parse_date_range(dates)
    budget_obj = _parse_budget(budget)
    requirements = _extract_requirements(raw_text, preferences)
    soft_preferences = _build_soft_preferences(preferences, raw_text)
    session_id = state.get("session_id") or _build_session_id(raw_text)
    user_profile_structured = _build_user_profile_structured(
        state=state,
        preferences=preferences,
        soft_preferences=soft_preferences,
        budget_obj=budget_obj,
        requirements=requirements,
    )

    intent_profile_output = {
        "session_id": session_id,
        "outbound_time_pref": outbound_time_pref,
        "return_time_pref": return_time_pref,
        "hard_constraints": {
            "origin": origin,
            "destination": destination,
            "start_date": start_date,
            "end_date": end_date,
            "budget": budget_obj,
            "travelers": travelers,
            "requirements": requirements,
        },
        "soft_preferences": soft_preferences,
        "search_queries": _build_search_queries(
            origin=origin,
            destination=destination,
            start_date=start_date,
            soft_preferences=soft_preferences,
            requirements=requirements,
        ),
    }

    patch: dict = {
        "origin": origin,
        "destination": destination,
        "dates": dates,
        "budget": budget,
        "preferences": preferences,
        "duration": duration,
        "user_profile": user_profile,
        "travelers": travelers,
        "outbound_time_pref": outbound_time_pref,
        "return_time_pref": return_time_pref,
        "session_id": session_id,
        "intent_profile_output": intent_profile_output,
        "user_profile_structured": user_profile_structured,
        "is_complete": False,
    }

    # Keep existing behavior compact for nullable scalar fields, but always keep
    # the contract payload for Agent2 chaining.
    return {
        k: v
        for k, v in patch.items()
        if v is not None or k in ("is_complete", "intent_profile_output")
    }

