"""
tools/serp_search.py — SerpAPI wrappers for TravelMind

Engines covered:
  google_flights              : Flight search + price insights
  google_flights_autocomplete : City/airport name → IATA code
  google_hotels               : Hotel list search
  google_hotels_reviews       : Hotel reviews (requires property_token)
  google_maps                 : Local place/attraction search
  google_maps_autocomplete    : Place name → data_id
  google_maps_reviews         : Place reviews (requires data_id)
  google_travel_explore       : Destination discovery from an origin

Module-level token caches:
  _hotel_tokens    : hotel name → property_token  (populated by serp_hotels)
  _place_data_ids  : place name → data_id         (populated by serp_local)
  These caches allow explainability_agent to fetch reviews without re-searching.
"""

import os
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", "").strip()

# Caches populated during search calls — used by explainability follow-ups
_hotel_tokens: dict[str, str] = {}   # lowercase hotel name → property_token
_place_data_ids: dict[str, str] = {} # lowercase place name → data_id

# Structured alternatives — used by frontend "Change flight/hotel" feature
_flight_options_outbound: list[dict] = []   # list of structured flight dicts (origin→dest)
_flight_options_return:   list[dict] = []   # list of structured flight dicts (dest→origin)
_hotel_options:           list[dict] = []   # list of structured hotel dicts


# ─────────────────────────────────────────────────────────────────────────────
# Core request
# ─────────────────────────────────────────────────────────────────────────────

def _serpapi_request(engine: str, params: dict) -> dict | None:
    if not SERPAPI_KEY:
        print("[SerpAPI] SERPAPI_API_KEY not set.")
        return None
    try:
        from serpapi import GoogleSearch
        result = GoogleSearch({**params, "api_key": SERPAPI_KEY, "engine": engine}).get_dict()
        if "error" in result:
            print(f"[SerpAPI] {engine} error: {result['error']}")
            return None
        return result
    except Exception as e:
        print(f"[SerpAPI] {engine} exception: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Google Flights Autocomplete  →  city/airport name → IATA code
# ─────────────────────────────────────────────────────────────────────────────

def serp_flights_autocomplete(query: str) -> str | None:
    """Resolve airport/city name to IATA code. Returns 3-letter code or None."""
    result = _serpapi_request("google_flights_autocomplete", {"q": query, "hl": "en"})
    if not result:
        return None
    for c in result.get("completions", []):
        code = c.get("value") or c.get("id", "")
        if code and len(code) == 3 and code.isupper():
            return code
    return None


# ─────────────────────────────────────────────────────────────────────────────
# City → IATA helper
# ─────────────────────────────────────────────────────────────────────────────

_CITY_TO_IATA = {
    # Southeast Asia
    "singapore": "SIN",
    "kuala lumpur": "KUL", "kl": "KUL",
    "bangkok": "BKK", "suvarnabhumi": "BKK",
    "phuket": "HKT",
    "chiang mai": "CNX",
    "bali": "DPS", "denpasar": "DPS",
    "jakarta": "CGK",
    "surabaya": "SUB",
    "yogyakarta": "JOG",
    "manila": "MNL",
    "cebu": "CEB",
    "ho chi minh": "SGN", "ho chi minh city": "SGN", "saigon": "SGN",
    "hanoi": "HAN",
    "da nang": "DAD",
    "yangon": "RGN",
    "phnom penh": "PNH",
    "siem reap": "REP",
    "vientiane": "VTE",
    "brunei": "BWN",
    # East Asia
    "tokyo": "NRT", "narita": "NRT",
    "tokyo haneda": "HND", "haneda": "HND",
    "osaka": "KIX", "kansai": "KIX",
    "kyoto": "KIX",
    "nagoya": "NGO",
    "sapporo": "CTS",
    "fukuoka": "FUK",
    "okinawa": "OKA", "naha": "OKA",
    "seoul": "ICN", "incheon": "ICN",
    "busan": "PUS",
    "jeju": "CJU",
    "taipei": "TPE",
    "tainan": "TNN",
    "taichung": "RMQ",
    "kaohsiung": "KHH",
    "hong kong": "HKG",
    "macau": "MFM",
    "beijing": "PEK",
    "shanghai": "PVG", "shanghai pudong": "PVG",
    "shanghai hongqiao": "SHA",
    "guangzhou": "CAN",
    "shenzhen": "SZX",
    "chengdu": "CTU",
    "chongqing": "CKG",
    "xi'an": "XIY", "xian": "XIY",
    "wuhan": "WUH",
    "hangzhou": "HGH",
    "kunming": "KMG",
    "guilin": "KWL",
    "sanya": "SYX",
    "urumqi": "URC",
    # South Asia
    "mumbai": "BOM", "bombay": "BOM",
    "delhi": "DEL", "new delhi": "DEL",
    "bangalore": "BLR", "bengaluru": "BLR",
    "chennai": "MAA",
    "kolkata": "CCU",
    "hyderabad": "HYD",
    "goa": "GOI",
    "ahmedabad": "AMD",
    "kathmandu": "KTM",
    "colombo": "CMB",
    "dhaka": "DAC",
    "karachi": "KHI",
    "islamabad": "ISB",
    "lahore": "LHE",
    # Middle East
    "dubai": "DXB",
    "abu dhabi": "AUH",
    "doha": "DOH",
    "riyadh": "RUH",
    "jeddah": "JED",
    "muscat": "MCT",
    "kuwait city": "KWI",
    "amman": "AMM",
    "beirut": "BEY",
    "tel aviv": "TLV",
    "istanbul": "IST",
    "ankara": "ESB",
    # Europe
    "london": "LHR", "london heathrow": "LHR",
    "london gatwick": "LGW",
    "paris": "CDG", "paris charles de gaulle": "CDG",
    "paris orly": "ORY",
    "amsterdam": "AMS",
    "frankfurt": "FRA",
    "munich": "MUC",
    "berlin": "BER",
    "madrid": "MAD",
    "barcelona": "BCN",
    "rome": "FCO",
    "milan": "MXP",
    "zurich": "ZRH",
    "vienna": "VIE",
    "brussels": "BRU",
    "lisbon": "LIS",
    "stockholm": "ARN",
    "oslo": "OSL",
    "copenhagen": "CPH",
    "helsinki": "HEL",
    "warsaw": "WAW",
    "prague": "PRG",
    "budapest": "BUD",
    "athens": "ATH",
    "dublin": "DUB",
    "edinburgh": "EDI",
    "manchester": "MAN",
    # Oceania
    "sydney": "SYD",
    "melbourne": "MEL",
    "brisbane": "BNE",
    "perth": "PER",
    "adelaide": "ADL",
    "auckland": "AKL",
    "christchurch": "CHC",
    # North America
    "new york": "JFK", "new york jfk": "JFK",
    "new york newark": "EWR", "newark": "EWR",
    "los angeles": "LAX",
    "san francisco": "SFO",
    "chicago": "ORD",
    "miami": "MIA",
    "toronto": "YYZ",
    "vancouver": "YVR",
    "montreal": "YUL",
    "seattle": "SEA",
    "boston": "BOS",
    "dallas": "DFW",
    "houston": "IAH",
    "atlanta": "ATL",
    "denver": "DEN",
    "las vegas": "LAS",
    "honolulu": "HNL",
    "cancun": "CUN",
    "mexico city": "MEX",
    # Africa
    "cairo": "CAI",
    "johannesburg": "JNB",
    "cape town": "CPT",
    "nairobi": "NBO",
    "casablanca": "CMN",
    "addis ababa": "ADD",
    "lagos": "LOS",
    "accra": "ACC",
}

def _to_iata(city: str) -> str:
    """Return IATA from static map → autocomplete fallback → uppercase city (3-letter truncation)."""
    key = city.lower().strip()
    # Exact match
    code = _CITY_TO_IATA.get(key)
    if code:
        return code
    # Partial match: city string contains a known key (e.g. "Tokyo, Japan" → "NRT")
    for known, iata in _CITY_TO_IATA.items():
        if known in key:
            return iata
    # SerpAPI autocomplete
    auto = serp_flights_autocomplete(city)
    if auto:
        return auto
    # Last resort: take first word, uppercase, pad/truncate to 3 chars
    first_word = key.split()[0].upper() if key else city.upper()
    return first_word[:3]


# ─────────────────────────────────────────────────────────────────────────────
# Google Flights
# ─────────────────────────────────────────────────────────────────────────────

_TIME_PREF_TO_HOURS = {
    "midnight":      "0,6",
    "early morning": "6,9",
    "morning":       "9,12",
    "afternoon":     "12,17",
    "evening":       "17,20",
    "night":         "20,24",
}

def _time_pref_to_outbound_times(time_pref: str) -> str | None:
    """Convert e.g. 'Morning (09:00–12:00)' → '9,12' for SerpAPI outbound_times."""
    if not time_pref:
        return None
    lower = time_pref.lower()
    for key, val in _TIME_PREF_TO_HOURS.items():
        if key in lower:
            return val
    return None


def serp_flights(
    origin: str,
    destination: str,
    date: str,
    time_pref: str = "",
    direction: str = "outbound",   # "outbound" or "return" — determines which cache to populate
) -> str | None:
    """
    Search Google Flights via SerpAPI. Includes price insights banner.
    Populates _flight_options_outbound or _flight_options_return for "Change flight" UI.

    Args:
        origin/destination : City name or IATA code — auto-converted to IATA.
        date               : 'YYYY-MM-DD' or 'YYYY-MM-DD to YYYY-MM-DD' (uses first date).
        time_pref          : e.g. 'Morning (09:00–12:00)' — filters departure window.
        direction          : 'outbound' or 'return' — which alternatives cache to fill.
    """
    dep_id = _to_iata(origin)
    arr_id = _to_iata(destination)
    outbound_date = date.split(" to ")[0].strip() if " to " in date else date.strip()

    print(f"[SerpAPI] Google Flights: {origin}({dep_id}) → {destination}({arr_id}) "
          f"on {outbound_date}" + (f" [{time_pref}]" if time_pref else ""))

    params: dict = {
        "departure_id":  dep_id,
        "arrival_id":    arr_id,
        "outbound_date": outbound_date,
        "currency":      "USD",
        "hl":            "en",
        "type":          "2",   # one-way
        "stops":         "0",   # nonstop only — avoids multi-leg flights misread as wrong-city routes
    }
    hours = _time_pref_to_outbound_times(time_pref)
    if hours:
        params["outbound_times"] = hours

    result = _serpapi_request("google_flights", params)
    if not result:
        return None

    flights = result.get("best_flights") or result.get("other_flights") or []

    # If no nonstop flights found, retry without the stops restriction
    if not flights and params.get("stops") == "0":
        print("[SerpAPI] No nonstop flights found — retrying with connections allowed")
        params_with_stops = {k: v for k, v in params.items() if k != "stops"}
        result = _serpapi_request("google_flights", params_with_stops)
        if result:
            flights = result.get("best_flights") or result.get("other_flights") or []
            if flights:
                print(f"[SerpAPI] Found {len(flights)} connecting flight(s) as fallback")
    price_insights = result.get("price_insights", {})

    lines = []
    structured: list[dict] = []

    if price_insights:
        level = price_insights.get("price_level", "")
        pr    = price_insights.get("typical_price_range") or []
        if level and len(pr) == 2:
            lines.append(f"[Price: {level} — typical USD {pr[0]}–{pr[1]}]")

    for f in flights[:5]:
        price = f.get("price")
        legs  = f.get("flights", [])
        if not legs:
            continue
        
        # Use first leg departure + last leg arrival to represent the whole journey
        first_leg = legs[0]
        last_leg  = legs[-1]
        dep       = first_leg.get("departure_airport", {})
        arr       = last_leg.get("arrival_airport", {})
        
        # Validate arrival airport against destination IATA
        actual_arr_id = arr.get("id", "").upper()
        if arr_id and len(arr_id) == 3 and actual_arr_id != arr_id:
            found_dest = False
            for i, lg in enumerate(legs):
                if lg.get("arrival_airport", {}).get("id", "").upper() == arr_id:
                    last_leg = lg
                    arr = lg.get("arrival_airport", {})
                    legs = legs[:i+1] # Truncate legs to the intended destination
                    found_dest = True
                    break
            
            if not found_dest:
                print(f"[SerpAPI] Warning: Flight ends at {actual_arr_id}, but {arr_id} was requested. Skipping.")
                continue

        # Aggregate flight numbers and airlines for multi-leg journeys
        airlines = []
        flight_nos = []
        for lg in legs:
            al = lg.get("airline", "")
            fn = lg.get("flight_number", "")
            if al and al not in airlines:
                airlines.append(al)
            if fn:
                flight_nos.append(fn)
        
        airline_display = " / ".join(airlines)
        flight_no_display = " / ".join(flight_nos)
        
        dep_time  = dep.get("time", "")
        arr_time  = arr.get("time", "")
        dep_name  = dep.get("name", dep.get("id", ""))
        arr_name  = arr.get("name", arr.get("id", ""))
        total_dur = f.get("total_duration") or sum(lg.get("duration") or 0 for lg in legs) or None
        cls       = first_leg.get("travel_class", "")

        # Indicate if it's a connecting flight
        stop_count = len(legs) - 1
        stop_info = f" ({stop_count} stop{'s' if stop_count > 1 else ''})" if stop_count > 0 else " (Nonstop)"

        display = f"{airline_display} {flight_no_display}{stop_info} | {dep_name} → {arr_name} | dep {dep_time} → arr {arr_time}"

        if total_dur:
            display += f" | {total_dur} min"
        if cls:
            display += f" | {cls}"
        if price:
            display += f" | USD {price}"
        lines.append(display)

        structured.append({
            "airline":           airline_display,
            "flight_number":     flight_no_display,
            "departure_airport": dep_name,
            "arrival_airport":   arr_name,
            "departure_time":    dep_time,
            "arrival_time":      arr_time,
            "duration_min":      total_dur,
            "travel_class":      cls,
            "price_usd":         price,
            "display":           display,
        })

    # Populate appropriate cache
    if direction == "return":
        _flight_options_return.clear()
        _flight_options_return.extend(structured)
    else:
        _flight_options_outbound.clear()
        _flight_options_outbound.extend(structured)

    return "\n".join(lines) if lines else None


# ─────────────────────────────────────────────────────────────────────────────
# Google Hotels
# ─────────────────────────────────────────────────────────────────────────────

def serp_hotels(
    destination: str,
    check_in: str = "",
    check_out: str = "",
    preferences: str = "",
) -> str | None:
    """
    Search Google Hotels. Populates _hotel_tokens cache for review follow-ups.
    Returns formatted hotel list or None.
    """
    print(f"[SerpAPI] Google Hotels: {destination} | {check_in}→{check_out} | {preferences}")
    params: dict = {
        "q":        f"hotels in {destination} {preferences}".strip(),
        "hl":       "en",
        "currency": "USD",
    }
    if check_in:
        params["check_in_date"] = check_in
    if check_out:
        params["check_out_date"] = check_out

    result = _serpapi_request("google_hotels", params)
    if not result:
        return None

    properties = result.get("properties", [])
    if not properties:
        return None

    lines = []
    _hotel_options.clear()

    for h in properties[:6]:
        name        = h.get("name", "")
        token       = h.get("property_token", "")
        rating      = h.get("overall_rating", "")
        reviews     = h.get("reviews", "")
        hotel_class = h.get("hotel_class", "")
        price_info  = h.get("rate_per_night", {})
        price_raw   = (price_info.get("lowest") or price_info.get("extracted_lowest") or "").replace("\xa0", " ").strip()
        price       = price_raw.lstrip("$").strip() if price_raw else ""
        description = h.get("description", "")

        if name and token:
            _hotel_tokens[name.lower()] = token

        line = name
        if hotel_class:
            line += f" | {hotel_class}"
        if rating:
            review_str = f" ({reviews} reviews)" if reviews else ""
            line += f" | ⭐ {rating}/5{review_str}"
        if price:
            line += f" | From USD {price}/night"

        _hotel_options.append({
            "name":           name,
            "hotel_class":    hotel_class,
            "rating":         rating,
            "reviews":        reviews,
            "price_per_night_usd": price,
            "property_token": token,
            "description":    description[:100] if description else "",
            "display":        line,
        })
        if description:
            line += f" | {description[:120]}"
        lines.append(line)

    return "\n".join(lines) if lines else None


def serp_hotel_details(property_token: str, check_in: str = "", check_out: str = "") -> str | None:
    """
    Get detailed property info for a hotel (amenities, price range, nearby).
    Requires property_token from serp_hotels() results.
    check_in / check_out in YYYY-MM-DD format — required by SerpAPI.
    """
    if not property_token:
        return None
    params: dict = {"property_token": property_token, "q": "hotel", "hl": "en", "currency": "USD"}
    if check_in:
        params["check_in_date"] = check_in
    if check_out:
        params["check_out_date"] = check_out
    result = _serpapi_request("google_hotels", params)
    if not result:
        return None

    lines = []
    if result.get("name"):
        lines.append(result["name"])
    if result.get("description"):
        lines.append(f"About: {result['description'][:200]}")
    if result.get("overall_rating"):
        lines.append(f"Rating: ⭐ {result['overall_rating']}/5 ({result.get('reviews','')} reviews)")
    pr = result.get("typical_price_range") or []
    if len(pr) == 2:
        lines.append(f"Typical price: USD {pr[0]}–{pr[1]}/night")
    amenities = result.get("amenities", [])
    if amenities:
        lines.append(f"Amenities: {', '.join(str(a) for a in amenities[:12])}")
    nearby = result.get("nearby_places", [])
    if nearby:
        lines.append("Nearby: " + ", ".join(p.get("name", "") for p in nearby[:4]))
    return "\n".join(line for line in lines if line) or None


def serp_hotel_reviews(property_token: str, num: int = 5) -> str | None:
    """
    Fetch top guest reviews for a hotel.
    Requires property_token from serp_hotels() results.
    """
    if not property_token:
        return None
    result = _serpapi_request("google_hotels_reviews", {
        "property_token": property_token,
        "sort_by":        "1",   # most helpful
        "hl":             "en",
    })
    if not result:
        return None

    reviews = result.get("reviews", [])
    lines = []
    for r in reviews[:num]:
        rating  = r.get("rating", "")
        snippet = r.get("snippet", "")
        user    = r.get("user", {}).get("name", "Guest")
        source  = r.get("source", "Google")
        if snippet:
            lines.append(f"⭐{rating} ({source}) {user}: \"{snippet[:150]}\"")
    return "\n".join(lines) if lines else None


def get_hotel_token(hotel_name: str) -> str | None:
    """Look up cached property_token by hotel name (case-insensitive partial match)."""
    if not hotel_name:
        return None
    key = hotel_name.lower().strip()
    # exact match first
    if key in _hotel_tokens:
        return _hotel_tokens[key]
    # partial match
    for name, token in _hotel_tokens.items():
        if key in name or name in key:
            return token
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Google Maps
# ─────────────────────────────────────────────────────────────────────────────

def serp_local(destination: str, query: str) -> str | None:
    """
    Search Google Maps for local places/attractions. Populates _place_data_ids cache.
    Returns formatted place list or None.
    """
    print(f"[SerpAPI] Google Maps: {query} in {destination}")
    result = _serpapi_request("google_maps", {
        "q":    f"{query} in {destination}",
        "hl":   "en",
        "type": "search",
    })
    if not result:
        return None

    places = result.get("local_results", [])
    if not places:
        return None

    lines = []
    for p in places[:8]:
        name    = p.get("title", "")
        data_id = p.get("data_id", "") or p.get("place_id", "")
        rating  = p.get("rating", "")
        reviews = p.get("reviews", "")
        type_   = p.get("type", "")
        address = p.get("address", "")
        hours   = p.get("hours", "")

        if name and data_id:
            _place_data_ids[name.lower()] = data_id

        line = name
        if type_:
            line += f" | {type_}"
        if rating:
            review_str = f" ({reviews} reviews)" if reviews else ""
            line += f" | ⭐ {rating}{review_str}"
        if address:
            line += f" | {address}"
        if hours:
            line += f" | {hours}"
        lines.append(line)

    return "\n".join(lines) if lines else None


def serp_maps_autocomplete(query: str) -> str | None:
    """Resolve place name to data_id via Google Maps Autocomplete. Returns data_id or None."""
    result = _serpapi_request("google_maps_autocomplete", {"q": query, "hl": "en"})
    if not result:
        return None
    for c in result.get("completions", []):
        data_id = c.get("data_id") or c.get("place_id", "")
        if data_id:
            return data_id
    return None


def serp_maps_reviews(place_name_or_data_id: str, num: int = 5) -> str | None:
    """
    Get Google Maps reviews for a place.
    Accepts a place name (auto-resolved via cache → autocomplete) or a raw data_id.
    Returns formatted review snippets or None.
    """
    key = place_name_or_data_id.lower().strip()
    data_id = _place_data_ids.get(key, "")
    if not data_id:
        if ":" in place_name_or_data_id or place_name_or_data_id.startswith("0x"):
            data_id = place_name_or_data_id
        else:
            data_id = serp_maps_autocomplete(place_name_or_data_id) or ""
    if not data_id:
        return None

    print(f"[SerpAPI] Maps reviews: {place_name_or_data_id}")
    # Note: 'num' cannot be used on initial page without topic_id/query/next_page_token
    result = _serpapi_request("google_maps_reviews", {
        "data_id": data_id,
        "hl":      "en",
        "sort_by": "ratingHigh",
    })
    if not result:
        return None

    place_info = result.get("place_info", {})
    reviews    = result.get("reviews", [])
    if not reviews:
        return None

    lines = []
    if place_info.get("rating"):
        lines.append(f"Overall: ⭐{place_info['rating']} "
                     f"({place_info.get('reviews','')} reviews) — {place_info.get('type','')}")
    for r in reviews[:num]:
        snippet = r.get("snippet", "")
        if snippet:
            lines.append(f"⭐{r.get('rating','')} {r.get('user',{}).get('name','Visitor')}: "
                         f"\"{snippet[:150]}\"")
    return "\n".join(lines) if lines else None


def get_place_data_id(place_name: str) -> str | None:
    """Look up cached data_id by place name (case-insensitive)."""
    return _place_data_ids.get(place_name.lower().strip())


# ─────────────────────────────────────────────────────────────────────────────
# TripAdvisor
# ─────────────────────────────────────────────────────────────────────────────

def serp_tripadvisor(query: str, location: str = "") -> str | None:
    """
    Search TripAdvisor via SerpAPI for hotels, attractions, and restaurants.
    More reliable for niche destinations (islands, small towns) than Google Hotels/Maps.
    Returns formatted results with real ratings and prices, or None.

    Args:
        query:    What to search for (e.g. "diving", "hotels", "restaurants")
        location: City or destination name
    """
    search_q = f"{query} {location}".strip() if location else query
    print(f"[SerpAPI] TripAdvisor: {search_q}")
    result = _serpapi_request("tripadvisor", {"q": search_q, "hl": "en"})
    if not result:
        return None

    lines = []
    for section_key in ("hotels", "attractions", "restaurants", "rental_properties"):
        for item in result.get(section_key, [])[:4]:
            name = item.get("name", "")
            if not name:
                continue
            rating_obj  = item.get("rating", {})
            rating_val  = rating_obj.get("rating", "") if isinstance(rating_obj, dict) else rating_obj
            review_cnt  = rating_obj.get("reviews", "") if isinstance(rating_obj, dict) else ""
            price       = item.get("price", "") or item.get("price_level", "")
            category    = item.get("category", "") or section_key.replace("_", " ").title()
            description = item.get("description", "")

            line = f"{name} | {category}"
            if rating_val:
                review_str = f" ({review_cnt} reviews)" if review_cnt else ""
                line += f" | ⭐ {rating_val}{review_str}"
            if price:
                line += f" | {price}"
            if description:
                line += f" | {description[:100]}"
            lines.append(line)

    return "\n".join(lines) if lines else None


# ─────────────────────────────────────────────────────────────────────────────
# Structured data collectors (return list[dict] instead of formatted text)
# ─────────────────────────────────────────────────────────────────────────────

def serp_local_structured(destination: str, query: str) -> list[dict]:
    """
    Google Maps search returning a list of place dicts (not a text string).
    Populates _place_data_ids cache for review follow-ups.
    Each dict: {name, type, rating, reviews, address, price, hours, data_id, lat, lng}
    GPS coordinates (lat/lng) are included for geographic clustering by the planner.
    """
    result = _serpapi_request("google_maps", {
        "q":    f"{query} in {destination}",
        "hl":   "en",
        "type": "search",
    })
    if not result:
        return []

    items = []
    for p in result.get("local_results", [])[:12]:
        name    = p.get("title", "")
        data_id = p.get("data_id", "") or p.get("place_id", "")
        if name and data_id:
            _place_data_ids[name.lower()] = data_id
        price_raw = p.get("price", "")
        gps = p.get("gps_coordinates", {})
        items.append({
            "name":    name,
            "type":    p.get("type", ""),
            "rating":  p.get("rating", ""),
            "reviews": p.get("reviews", ""),
            "address": p.get("address", ""),
            "price":   price_raw,          # e.g. "$", "$$", "$15", or ""
            "hours":   p.get("hours", ""),
            "data_id": data_id,
            "lat":     gps.get("latitude"),
            "lng":     gps.get("longitude"),
        })
    return items


def serp_tripadvisor_structured(query: str, location: str = "") -> list[dict]:
    """
    TripAdvisor search returning a list of item dicts (not a text string).
    SerpAPI TripAdvisor engine returns a flat 'places' list with fields:
      title, place_type, rating (number), reviews (number), description, price.
    Each returned dict: {name, category, rating, reviews, price, description, source}
    """
    search_q = f"{query} {location}".strip() if location else query
    result = _serpapi_request("tripadvisor", {"q": search_q, "hl": "en"})
    if not result:
        return []

    # SerpAPI TripAdvisor uses 'places' with 'title' — fall back to legacy section keys
    raw_places = result.get("places", [])
    if not raw_places:
        # Legacy format fallback
        for section_key in ("attractions", "restaurants", "hotels", "rental_properties"):
            raw_places += [{**item, "_section": section_key}
                           for item in result.get(section_key, [])[:6]]

    items = []
    for item in raw_places[:20]:
        name = item.get("title") or item.get("name", "")
        if not name:
            continue
        # rating may be a plain number or nested dict
        rating_raw = item.get("rating")
        if isinstance(rating_raw, dict):
            rating_val = str(rating_raw.get("rating", ""))
            review_cnt = str(rating_raw.get("reviews", ""))
        else:
            rating_val = str(rating_raw) if rating_raw else ""
            review_cnt = str(item.get("reviews", ""))
        price = (item.get("price") or item.get("price_level") or "").replace("\xa0", " ").strip()
        category = (item.get("place_type") or item.get("category")
                    or item.get("_section", "Attraction")).replace("_", " ").title()
        items.append({
            "name":        name,
            "category":    category,
            "rating":      rating_val,
            "reviews":     review_cnt,
            "price":       price,
            "description": (item.get("description", "") or "")[:120],
            "source":      "tripadvisor",
        })
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Google Travel Explore
# ─────────────────────────────────────────────────────────────────────────────

def serp_travel_explore(
    origin: str = "",
    month: int = 0,
    travel_duration: int = 2,
    max_price: int = 0,
) -> str | None:
    """
    Discover affordable destinations from an origin using Google Travel Explore.

    Args:
        origin:          Origin city or IATA (optional).
        month:           Travel month 1–12 (0 = any within next 6 months).
        travel_duration: 1=weekend, 2=1 week (default), 3=2 weeks.
        max_price:       Max flight price (0 = no limit).
    Returns formatted destination list with flight+hotel prices or None.
    """
    dep_id = _to_iata(origin) if origin else ""
    print(f"[SerpAPI] Travel Explore: from {origin}({dep_id}) | "
          f"month={month} | duration={travel_duration}")

    params: dict = {
        "hl":              "en",
        "currency":        "USD",
        "travel_duration": travel_duration,
    }
    if dep_id:
        params["departure_id"] = dep_id
    if month:
        params["month"] = month
    if max_price:
        params["max_price"] = max_price

    result = _serpapi_request("google_travel_explore", params)
    if not result:
        return None

    destinations = result.get("destinations", [])
    if not destinations:
        return None

    lines = []
    for d in destinations[:8]:
        name         = d.get("name", "")
        country      = d.get("country", "")
        flight_price = d.get("flight_price")
        hotel_price  = d.get("hotel_price")
        airline      = d.get("airline", "")
        start        = d.get("start_date", "")
        end          = d.get("end_date", "")
        stops        = d.get("number_of_stops", -1)

        line = f"{name}, {country}"
        if flight_price:
            line += f" | ✈️ USD {flight_price}"
        if hotel_price:
            line += f" | 🏨 USD {hotel_price}/night"
        if airline:
            line += f" | {airline}"
        if start and end:
            line += f" | {start} – {end}"
        if stops == 0:
            line += " | Nonstop"
        lines.append(line)

    return "\n".join(lines) if lines else None
