"""
backend/db/crud.py — save / load plans from RDS
"""
import hashlib
import hmac
import secrets
from sqlalchemy.orm import Session
from .models import Plan, PlanItinerary, PlanFlight, PlanHotel, PlanExplain, User


def save_plan(db: Session, plan_id: str, state: dict, result: dict, via_debate: bool = False) -> Plan:
    """Insert a new plan row + child rows. Returns the Plan ORM object."""
    itineraries_dict: dict = result.get("itineraries", {})
    flights_out: list      = result.get("flight_options_outbound", [])
    flights_ret: list      = result.get("flight_options_return",   [])
    hotels: list           = result.get("hotel_options",           [])

    plan = Plan(
        plan_id                 = plan_id,
        origin                  = state.get("origin", ""),
        destination             = state.get("destination", ""),
        dates                   = state.get("dates", ""),
        duration                = state.get("duration", ""),
        budget                  = state.get("budget", ""),
        preferences             = state.get("preferences", ""),
        hard_constraints        = state.get("hard_constraints") or None,
        soft_preferences        = state.get("soft_preferences") or None,
        search_queries          = state.get("search_queries")   or None,
        option_meta             = result.get("option_meta"),
        tool_log                = result.get("tool_log"),
        # Planner trace — Agent6 needs these when /explain is called lazily on a saved plan
        planner_decision_trace  = result.get("planner_decision_trace") or None,
        chain_of_thought        = result.get("chain_of_thought") or result.get("planner_chain_of_thought") or None,
        via_debate              = via_debate,
        debate_verdict          = result.get("debate_verdict"),
        debate_history          = result.get("debate_history"),
    )
    db.add(plan)

    # Itinerary options A / B / C
    for option, days in itineraries_dict.items():
        db.add(PlanItinerary(plan_id=plan_id, option=option, days=days))

    # Flights
    for f in flights_out:
        db.add(PlanFlight(plan_id=plan_id, direction="outbound", flight_data=f))
    for f in flights_ret:
        db.add(PlanFlight(plan_id=plan_id, direction="return",   flight_data=f))

    # Hotels
    for h in hotels:
        db.add(PlanHotel(plan_id=plan_id, hotel_data=h))

    db.commit()
    db.refresh(plan)
    return plan


def load_plan(db: Session, plan_id: str) -> dict | None:
    """Reconstruct the full cached dict from DB rows (mirrors _plan_cache structure)."""
    plan: Plan | None = db.query(Plan).filter(Plan.plan_id == plan_id).first()
    if not plan:
        return None

    itineraries = {row.option: row.days for row in plan.itineraries}
    flights_out  = [r.flight_data for r in plan.flights if r.direction == "outbound"]
    flights_ret  = [r.flight_data for r in plan.flights if r.direction == "return"]
    hotels       = [r.hotel_data  for r in plan.hotels]

    return {
        # state fields
        "origin":           plan.origin,
        "destination":      plan.destination,
        "dates":            plan.dates,
        "duration":         plan.duration,
        "budget":           plan.budget,
        "preferences":      plan.preferences,
        "hard_constraints": plan.hard_constraints or {},
        "soft_preferences": plan.soft_preferences or {},
        "search_queries":   plan.search_queries   or [],
        "outbound_time_pref": "",
        "return_time_pref":   "",
        "user_profile":       {},
        # result fields
        "itineraries":              itineraries,
        "option_meta":              plan.option_meta  or {},
        "tool_log":                 plan.tool_log     or [],
        "flight_options_outbound":  flights_out,
        "flight_options_return":    flights_ret,
        "hotel_options":            hotels,
        # Planner trace — returned so Agent6 can generate trace-based explanations
        "planner_decision_trace":   plan.planner_decision_trace or {},
        "chain_of_thought":         plan.chain_of_thought or "",
        "planner_chain_of_thought": plan.chain_of_thought or "",
        "debate_verdict":           plan.debate_verdict,
        "debate_history":           plan.debate_history,
    }


def update_plan_result(db: Session, plan_id: str, revised: dict) -> None:
    """Update an existing plan's itineraries / flights / hotels after a revision."""
    plan: Plan | None = db.query(Plan).filter(Plan.plan_id == plan_id).first()
    if not plan:
        return

    # Overwrite scalar columns
    plan.option_meta            = revised.get("option_meta",            plan.option_meta)
    plan.tool_log               = revised.get("tool_log",               plan.tool_log)
    plan.planner_decision_trace = revised.get("planner_decision_trace", plan.planner_decision_trace)
    plan.chain_of_thought       = revised.get("chain_of_thought") or revised.get("planner_chain_of_thought") or plan.chain_of_thought

    # Delete + re-insert child rows
    for itin in list(plan.itineraries):
        db.delete(itin)
    for flight in list(plan.flights):
        db.delete(flight)
    for hotel in list(plan.hotels):
        db.delete(hotel)
    db.flush()

    for option, days in (revised.get("itineraries") or {}).items():
        db.add(PlanItinerary(plan_id=plan_id, option=option, days=days))
    for f in revised.get("flight_options_outbound", []):
        db.add(PlanFlight(plan_id=plan_id, direction="outbound", flight_data=f))
    for f in revised.get("flight_options_return", []):
        db.add(PlanFlight(plan_id=plan_id, direction="return",   flight_data=f))
    for h in revised.get("hotel_options", []):
        db.add(PlanHotel(plan_id=plan_id, hotel_data=h))

    db.commit()


def save_explain(db: Session, plan_id: str, explain_result: dict) -> None:
    """Save Agent6 explain output. Replaces existing row if any."""
    existing = db.query(PlanExplain).filter(PlanExplain.plan_id == plan_id).first()
    if existing:
        existing.explain_data     = explain_result.get("explain_data")
        existing.chain_of_thought = explain_result.get("chain_of_thought")
        existing.agent_steps      = explain_result.get("agent_steps")
    else:
        db.add(PlanExplain(
            plan_id          = plan_id,
            explain_data     = explain_result.get("explain_data"),
            chain_of_thought = explain_result.get("chain_of_thought"),
            agent_steps      = explain_result.get("agent_steps"),
        ))
    db.commit()


def hash_password(password: str, *, iterations: int = 100_000) -> str:
    """
    Hash a plain password using PBKDF2-SHA256.
    Stored format: pbkdf2_sha256$<iterations>$<salt_hex>$<digest_hex>
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify password against hash created by hash_password().
    Returns False when hash format is invalid.
    """
    try:
        algo, iterations_str, salt_hex, digest_hex = password_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (TypeError, ValueError):
        return False

    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def create_user(db: Session, username: str, password: str) -> User:
    """
    Create user with hashed password.
    Raises ValueError if username already exists.
    """
    existing = get_user_by_username(db, username)
    if existing:
        raise ValueError("Username already exists")

    user = User(
        username=username,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    """
    Return user when username/password are valid; otherwise None.
    """
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user
