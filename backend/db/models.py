"""
backend/db/models.py — ORM models for TravelMind RDS

Tables:
  plans             — one row per /generate or /debate call (owns plan_id)
  plan_itineraries  — 3 rows per plan (option A/B/C), full day-by-day JSON
  plan_flights      — flight options outbound + return rows
  plan_hotels       — hotel options rows
  plan_explains     — Agent6 explain output (1 row per plan)
"""
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Integer, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from .database import Base


# ── plans ─────────────────────────────────────────────────────────────────────

class Plan(Base):
    __tablename__ = "plans"

    plan_id      = Column(String(16),  primary_key=True)
    created_at   = Column(DateTime,    default=lambda: datetime.now(timezone.utc), nullable=False)

    # Trip inputs
    origin       = Column(String(128), nullable=False)
    destination  = Column(String(128), nullable=False)
    dates        = Column(String(64),  nullable=False)
    duration     = Column(String(32),  nullable=False)
    budget       = Column(String(64),  nullable=False)
    preferences  = Column(Text,        nullable=False)

    # Extended inputs (Agent1 format)
    hard_constraints = Column(JSONB, nullable=True)
    soft_preferences = Column(JSONB, nullable=True)
    search_queries   = Column(JSONB, nullable=True)

    # Metadata
    option_meta  = Column(JSONB, nullable=True)   # which option won, costs, etc.
    tool_log     = Column(JSONB, nullable=True)   # list of log strings
    via_debate   = Column(Boolean, default=False) # True if produced by /debate

    # Debate extras
    debate_verdict = Column(JSONB, nullable=True)
    debate_history = Column(JSONB, nullable=True)

    # Relationships
    itineraries = relationship("PlanItinerary", back_populates="plan", cascade="all, delete-orphan")
    flights     = relationship("PlanFlight",    back_populates="plan", cascade="all, delete-orphan")
    hotels      = relationship("PlanHotel",     back_populates="plan", cascade="all, delete-orphan")
    explains    = relationship("PlanExplain",   back_populates="plan", cascade="all, delete-orphan")


# ── plan_itineraries ──────────────────────────────────────────────────────────

class PlanItinerary(Base):
    __tablename__ = "plan_itineraries"

    id       = Column(Integer, primary_key=True, autoincrement=True)
    plan_id  = Column(String(16), ForeignKey("plans.plan_id", ondelete="CASCADE"), nullable=False)
    option   = Column(String(4), nullable=False)   # "A", "B", "C"
    days     = Column(JSONB, nullable=False)        # list of day dicts

    plan = relationship("Plan", back_populates="itineraries")


# ── plan_flights ──────────────────────────────────────────────────────────────

class PlanFlight(Base):
    __tablename__ = "plan_flights"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    plan_id     = Column(String(16), ForeignKey("plans.plan_id", ondelete="CASCADE"), nullable=False)
    direction   = Column(String(8), nullable=False)   # "outbound" | "return"
    flight_data = Column(JSONB, nullable=False)        # full flight dict from SerpAPI

    plan = relationship("Plan", back_populates="flights")


# ── plan_hotels ───────────────────────────────────────────────────────────────

class PlanHotel(Base):
    __tablename__ = "plan_hotels"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    plan_id    = Column(String(16), ForeignKey("plans.plan_id", ondelete="CASCADE"), nullable=False)
    hotel_data = Column(JSONB, nullable=False)   # full hotel dict from SerpAPI

    plan = relationship("Plan", back_populates="hotels")


# ── plan_explains ─────────────────────────────────────────────────────────────

class PlanExplain(Base):
    __tablename__ = "plan_explains"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    plan_id          = Column(String(16), ForeignKey("plans.plan_id", ondelete="CASCADE"), nullable=False)
    created_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    explain_data     = Column(JSONB, nullable=True)   # per-option explanations
    chain_of_thought = Column(Text, nullable=True)    # full CoT string
    agent_steps      = Column(JSONB, nullable=True)

    plan = relationship("Plan", back_populates="explains")
