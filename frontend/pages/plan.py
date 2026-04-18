"""
pages/plan.py — Plan Your Trip page
"""

import html
import json
import os
import uuid
from datetime import date, timedelta

import requests
import streamlit as st

from data.store import EXPLAIN_DATA

# Full-stream URL for LangGraph orchestrator (agents service port 8001 in Docker).
GRAPH_STREAM_URL = os.getenv(
    "AGENTS_GRAPH_STREAM_URL",
    "http://localhost:8001/api/invoke/graph/stream",
).rstrip("/")

TIME_PREF_OPTIONS = [
    "No preference",
    "midnight",
    "early morning",
    "morning",
    "afternoon",
    "evening",
    "night",
]

BUDGET_CURRENCY_OPTIONS = ["SGD", "USD", "EUR", "CNY", "JPY"]

DURATION_OPTIONS = [
    "1 day", "2 days", "3 days", "4 days", "5 days",
    "6 days", "7 days", "10 days", "14 days",
]

PIPELINE_STEPS = [
    {"key": "intent", "icon": "🧠", "name": "Intent & Profile Agent"},
    {"key": "research", "icon": "🔍", "name": "Research Agent"},
    {"key": "planner", "icon": "📅", "name": "Itinerary Planning Agent"},
    {"key": "debate", "icon": "⚔️", "name": "Debate & Critique Agent"},
    {"key": "safety", "icon": "🛡️", "name": "Risk & Safety Agent"},
    {"key": "explain", "icon": "💡", "name": "Explainability Agent"},
]


def _init_agent_status() -> dict:
    return {s["key"]: {"state": "pending", "detail": ""} for s in PIPELINE_STEPS}


def _render_agent_panel(placeholder, status: dict):
    style_map = {
        "pending": {
            "bg": "rgba(255,255,255,0.04)",
            "border": "rgba(255,255,255,0.08)",
            "color": "#7a90b0",
            "label": "pending",
        },
        "running": {
            "bg": "rgba(59,158,255,0.12)",
            "border": "rgba(59,158,255,0.35)",
            "color": "#3b9eff",
            "label": "running",
        },
        "success": {
            "bg": "rgba(16,185,129,0.12)",
            "border": "rgba(16,185,129,0.35)",
            "color": "#10b981",
            "label": "success",
        },
        "error": {
            "bg": "rgba(239,68,68,0.12)",
            "border": "rgba(239,68,68,0.35)",
            "color": "#ef4444",
            "label": "failed",
        },
        "skipped": {
            "bg": "rgba(255,255,255,0.02)",
            "border": "rgba(255,255,255,0.06)",
            "color": "#4a5a72",
            "label": "skipped",
        },
    }

    with placeholder.container():
        st.markdown("**Agent Activity**")
        for step in PIPELINE_STEPS:
            s = status.get(step["key"], {"state": "pending", "detail": ""})
            style = style_map.get(s["state"], style_map["pending"])
            detail = html.escape(s.get("detail") or "")
            detail_html = (
                f"<div style='color:#7a90b0;font-size:12px;margin-top:4px'>{detail}</div>"
                if detail
                else ""
            )
            st.markdown(
                f"""
<div style="
    background:{style['bg']};
    border:1px solid {style['border']};
    border-radius:10px;
    padding:10px 12px;
    margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="color:#e8edf5;font-size:13px">
      <span style="margin-right:6px">{step['icon']}</span>
      <strong>{step['name']}</strong>
    </span>
    <span style="
        color:{style['color']};
        font-size:11px;
        font-family:monospace;
        text-transform:uppercase;
        letter-spacing:0.5px">
      {style['label']}
    </span>
  </div>
  {detail_html}
</div>
""",
                unsafe_allow_html=True,
            )


def _render_summary_row(summary: dict):
    """Compact summary: route uses small, wrapping text (not st.metric)."""
    o = summary.get("origin", "—") or "—"
    d = summary.get("destination", "—") or "—"
    dates = summary.get("dates", "—") or "—"
    budget = summary.get("budget", "—") or "—"
    dur = summary.get("duration", "—") or "—"
    st.markdown(
        f"""
<div style="display:grid;grid-template-columns:1.4fr 1fr 0.85fr 0.75fr;gap:10px;
     margin-top:8px;">
  <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);
       border-radius:12px;padding:10px 12px;min-width:0;">
    <div style="color:#7a90b0;font-size:11px;text-transform:uppercase;
         letter-spacing:0.04em;margin-bottom:4px;">Route</div>
    <div style="color:#e8edf5;font-size:13px;font-weight:500;line-height:1.35;
         word-break:break-word;overflow-wrap:anywhere;">
      {o} → {d}
    </div>
  </div>
  <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);
       border-radius:12px;padding:10px 12px;min-width:0;">
    <div style="color:#7a90b0;font-size:11px;text-transform:uppercase;
         letter-spacing:0.04em;margin-bottom:4px;">Dates</div>
    <div style="color:#e8edf5;font-size:13px;line-height:1.35;
         word-break:break-word;">{dates}</div>
  </div>
  <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);
       border-radius:12px;padding:10px 12px;min-width:0;">
    <div style="color:#7a90b0;font-size:11px;text-transform:uppercase;
         letter-spacing:0.04em;margin-bottom:4px;">Budget</div>
    <div style="color:#e8edf5;font-size:13px;font-weight:500;">{budget}</div>
  </div>
  <div style="background:#111827;border:1px solid rgba(255,255,255,0.07);
       border-radius:12px;padding:10px 12px;min-width:0;">
    <div style="color:#7a90b0;font-size:11px;text-transform:uppercase;
         letter-spacing:0.04em;margin-bottom:4px;">Duration</div>
    <div style="color:#e8edf5;font-size:13px;font-weight:500;">{dur}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _run_graph_stream(progress_placeholder) -> tuple[dict | None, str | None]:
    """
    Consume NDJSON from orchestrator; only updates progress_placeholder (no per-agent greens).
    Returns (final_state_slice, error_message).
    """
    payload = st.session_state.get("_graph_stream_payload")
    if not payload:
        return None, "internal: missing graph payload"

    try:
        with requests.post(
            GRAPH_STREAM_URL,
            json={"state": payload},
            stream=True,
            timeout=600,
        ) as resp:
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                return None, f"HTTP {resp.status_code}: {detail}"

            final_state: dict | None = None
            buf = b""
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "progress":
                        step = obj.get("step", 0)
                        progress_placeholder.markdown(
                            f"⏳ **Orchestrator** — step {step} (streaming…)"
                        )
                    elif obj.get("type") == "done":
                        payload_state = obj.get("state")
                        final_state = payload_state if isinstance(payload_state, dict) else {}
                    elif obj.get("type") == "error":
                        return None, str(obj.get("message", "unknown error"))

            if final_state is None:
                return None, "stream ended without a done event"
            return final_state, None
    except requests.exceptions.RequestException as exc:
        return None, f"connection error: {exc.__class__.__name__}: {exc}"


def _agent_status_from_graph_state(s: dict) -> dict:
    """Build right-panel rows purely from the graph's final state (no demo strings)."""
    status = _init_agent_status()
    threat = s.get("threat_blocked") is True
    err = s.get("error_message")

    if threat:
        status["intent"] = {
            "state": "error",
            "detail": (s.get("threat_detail") or s.get("threat_type") or "blocked at input"),
        }
        for k in ("research", "planner", "debate", "safety", "explain"):
            status[k] = {"state": "skipped", "detail": "workflow stopped"}
        return status

    origin = s.get("origin") or ""
    dest = s.get("destination") or ""
    prefs = (s.get("preferences") or "").strip()
    if origin or dest or s.get("intent_profile_output"):
        status["intent"] = {
            "state": "success",
            "detail": (
                f"{origin or '—'} → {dest or '—'}"
                + (f" · {prefs}" if prefs else "")
            ),
        }
    else:
        status["intent"] = {
            "state": "error" if err else "skipped",
            "detail": (err or "no intent fields in result")[:220],
        }

    research = s.get("research")
    search_hits = s.get("search_results")
    if research or search_hits:
        parts: list[str] = []
        if isinstance(research, dict) and research:
            parts.append("research: " + ",".join(list(research.keys())[:6]))
        if isinstance(search_hits, dict) and search_hits:
            parts.append("search: " + ",".join(list(search_hits.keys())[:6]))
        detail = " · ".join(parts) if parts else "search / research completed"
        status["research"] = {"state": "success", "detail": detail[:220]}
    else:
        status["research"] = {
            "state": "error" if err else "skipped",
            "detail": (err or "no research output")[:220],
        }

    raw_itins = s.get("final_itineraries") or s.get("itineraries")
    if isinstance(raw_itins, dict) and raw_itins:
        n_opt = len(raw_itins)
        status["planner"] = {
            "state": "success",
            "detail": f"{n_opt} option(s) · keys {', '.join(sorted(raw_itins.keys()))}",
        }
    elif err:
        status["planner"] = {"state": "error", "detail": str(err)[:220]}
    else:
        status["planner"] = {"state": "error", "detail": "no itineraries in graph result"}

    valid = s.get("is_valid")
    rounds = s.get("debate_count")
    if valid is True:
        score = s.get("composite_score")
        extra = ""
        if isinstance(score, (int, float)):
            extra = f" score={score:.0f}"
        status["debate"] = {"state": "success", "detail": f"approved{extra}"}
    elif valid is False:
        status["debate"] = {
            "state": "success",
            "detail": f"completed with revisions (rounds≈{rounds})",
        }
    else:
        status["debate"] = {"state": "skipped", "detail": "no debate verdict in state"}

    og = (s.get("output_guard_decision") or "").lower()
    flagged = s.get("output_flagged") is True
    if og == "pass" and not flagged:
        status["safety"] = {"state": "success", "detail": "output guard: pass"}
    elif og or flagged:
        reason = s.get("output_flag_reason") or og or "flagged"
        status["safety"] = {"state": "error", "detail": str(reason)[:220]}
    else:
        status["safety"] = {"state": "skipped", "detail": "no output guard decision"}

    if s.get("explanation") or s.get("explain_data"):
        status["explain"] = {"state": "success", "detail": "explanation generated"}
    else:
        status["explain"] = {"state": "skipped", "detail": "no explain payload"}

    return status


def _itineraries_from_state(s: dict) -> dict:
    raw = s.get("final_itineraries") or s.get("itineraries")
    return raw if isinstance(raw, dict) else {}


def render_explain_modal(key: str):
    d = EXPLAIN_DATA.get(key)
    if not d:
        return
    with st.expander(f"Why was **{d['name']}** recommended?", expanded=True):
        st.markdown("**Preference Matches**")
        for m in d["matches"]:
            st.markdown(f"  {m}")
        st.markdown("---")
        st.markdown("**Collaborative Signal**")
        st.info(d["similar"])
        st.markdown("---")
        st.markdown("**Score Breakdown**")
        for label, score in d["scores"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(score / 100, text=label)
            with col2:
                st.markdown(f"`{score}%`")


def render_itinerary(option: str, itineraries: dict, option_meta: dict):
    days = itineraries.get(option, [])
    if not days:
        st.info("No itinerary data for this option.")
        return

    for day_idx, day in enumerate(days):
        day_title = day.get("day", f"Day {day_idx + 1}")
        day_budget = day.get("budget", "")
        with st.expander(f"{day_title}  —  {day_budget}", expanded=(day_idx == 0)):
            for item in day.get("items", []):
                col_time, col_icon, col_name, col_cost = st.columns([0.8, 0.4, 4, 1.2])
                with col_time:
                    st.markdown(
                        f"<span style='color:#7a90b0;font-size:12px;font-family:monospace'>"
                        f"{item.get('time', '')}</span>",
                        unsafe_allow_html=True,
                    )
                with col_icon:
                    st.markdown(
                        f"<span style='font-size:18px'>{item.get('icon', '')}</span>",
                        unsafe_allow_html=True,
                    )
                with col_name:
                    st.markdown(
                        f"<span style='color:#e8edf5;font-size:13px;font-weight:500'>"
                        f"{item.get('name', '')}</span>",
                        unsafe_allow_html=True,
                    )
                with col_cost:
                    st.markdown(
                        f"<span style='color:#7a90b0;font-size:12px;font-family:monospace'>"
                        f"{item.get('cost', '')}</span>",
                        unsafe_allow_html=True,
                    )


def render():
    st.markdown("### Plan Your Trip")
    st.markdown(
        "<span style='color:#7a90b0;font-size:14px'>"
        "Describe your trip, fill in a few details, and let TravelMind's agents "
        "generate personalized itinerary options for you.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if "agent_status" not in st.session_state or not st.session_state.agent_status:
        st.session_state.agent_status = _init_agent_status()

    col_left, col_right = st.columns([3, 1.5])
    right_panel = col_right.empty()
    _render_agent_panel(right_panel, st.session_state.agent_status)

    with col_left:
        with st.container(border=True):
            user_msg = st.text_area(
                "Describe your trip",
                placeholder=(
                    "e.g. I want to visit Kyoto from Singapore, "
                    "love cultural heritage and local food, vegetarian..."
                ),
                height=80,
                key="plan_user_msg",
            )

            st.markdown(
                "<span style='color:#7a90b0;font-size:12px'>"
                "Fill in the fields below (destination, origin, preferences "
                "will be extracted from your message automatically)</span>",
                unsafe_allow_html=True,
            )

            row1_c1, row1_c2, row1_c3 = st.columns(3)
            with row1_c1:
                budget_amount = st.number_input(
                    "Budget",
                    min_value=0,
                    max_value=100000,
                    value=3000,
                    step=100,
                    key="plan_budget_amount",
                )
            with row1_c2:
                budget_currency = st.selectbox(
                    "Currency",
                    BUDGET_CURRENCY_OPTIONS,
                    index=0,
                    key="plan_budget_currency",
                )
            with row1_c3:
                duration = st.selectbox(
                    "Duration",
                    DURATION_OPTIONS,
                    index=4,
                    key="plan_duration",
                )

            row2_c1, row2_c2, row2_c3 = st.columns(3)
            with row2_c1:
                start_date = st.date_input(
                    "Start date",
                    value=date.today() + timedelta(days=14),
                    key="plan_start_date",
                )
            with row2_c2:
                outbound_pref = st.selectbox(
                    "Outbound time",
                    TIME_PREF_OPTIONS,
                    index=0,
                    key="plan_outbound_pref",
                )
            with row2_c3:
                return_pref = st.selectbox(
                    "Return time",
                    TIME_PREF_OPTIONS,
                    index=0,
                    key="plan_return_pref",
                )

            _, btn_col = st.columns([4, 1.5])
            with btn_col:
                generate = st.button(
                    "Generate Options",
                    type="primary",
                    use_container_width=True,
                )

        if generate:
            if not user_msg.strip():
                st.warning("Please describe your trip first.")
                return

            duration_days = duration.split()[0]
            end_date = start_date + timedelta(days=int(duration_days))
            dates_str = f"{start_date.isoformat()} to {end_date.isoformat()}"
            budget_str = f"{budget_currency} {budget_amount}"
            out_pref = None if outbound_pref == "No preference" else outbound_pref
            ret_pref = None if return_pref == "No preference" else return_pref

            graph_payload = {
                "messages": [{"role": "user", "content": user_msg.strip()}],
                "dates": dates_str,
                "budget": budget_str,
                "duration": duration,
                "outbound_time_pref": out_pref,
                "return_time_pref": ret_pref,
            }
            uid = st.session_state.get("user_id")
            if uid:
                graph_payload["user_id"] = uid

            st.session_state._graph_stream_payload = graph_payload

            with right_panel.container():
                st.markdown("**Agent Activity**")
                st.info("Running LangGraph orchestrator (streaming) — panel updates when finished.")
                prog_ph = st.empty()

            progress_bar = st.progress(0, text="Connecting to agents service…")

            final_state, stream_err = _run_graph_stream(prog_ph)
            progress_bar.progress(1.0, text="Done" if not stream_err else "Failed")
            progress_bar.empty()
            del st.session_state["_graph_stream_payload"]

            if stream_err is not None:
                st.session_state.agent_status = _init_agent_status()
                st.session_state.agent_status["intent"] = {
                    "state": "error",
                    "detail": stream_err or "empty response",
                }
                st.session_state.plan_generated = False
                st.session_state.plan_itineraries = {}
                st.session_state.plan_option_meta = {}
                st.error(stream_err or "Graph stream failed")
                st.rerun()

            st.session_state.agent_status = _agent_status_from_graph_state(final_state)

            itins = _itineraries_from_state(final_state)
            st.session_state.plan_itineraries = itins
            st.session_state.plan_option_meta = final_state.get("option_meta") or {}
            st.session_state.plan_id = final_state.get("session_id") or str(uuid.uuid4())[:8]
            st.session_state.plan_flight_outbound = final_state.get("flight_options_outbound") or []
            st.session_state.plan_flight_return = final_state.get("flight_options_return") or []
            st.session_state.plan_hotel_options = final_state.get("hotel_options") or []

            st.session_state.plan_request_summary = {
                "origin": final_state.get("origin") or "",
                "destination": final_state.get("destination") or "",
                "dates": final_state.get("dates") or dates_str,
                "budget": final_state.get("budget") or budget_str,
                "duration": final_state.get("duration") or duration,
            }
            st.session_state.plan_generated = bool(itins)
            if not itins:
                st.warning(
                    "Orchestrator finished but no itinerary blocks were returned. "
                    "Check agent logs or OPENAI/API keys."
                )
            st.rerun()

        if st.session_state.get("plan_generated"):
            itineraries = st.session_state.get("plan_itineraries", {})
            option_meta = st.session_state.get("plan_option_meta", {})
            summary = st.session_state.get("plan_request_summary", {})

            if summary:
                st.markdown("---")
                _render_summary_row(summary)

            if itineraries:
                st.markdown("#### Choose your preferred itinerary style")
                option_keys = list(itineraries.keys())

                opt_cols = st.columns(len(option_keys))
                for i, key in enumerate(option_keys):
                    meta = option_meta.get(key, {})
                    with opt_cols[i]:
                        selected = st.session_state.selected_option == key
                        label = meta.get("badge", meta.get("label", f"Option {key}"))
                        if st.button(
                            f"{'> ' if selected else ''}{label}",
                            key=f"opt_{key}",
                            use_container_width=True,
                            type="primary" if selected else "secondary",
                        ):
                            st.session_state.selected_option = key
                            st.rerun()

                opt = st.session_state.selected_option
                if opt not in option_keys:
                    opt = option_keys[0]
                    st.session_state.selected_option = opt

                meta = option_meta.get(opt, {})
                desc = meta.get("desc", "")
                budget_badge = meta.get("budget", "")
                style_badge = meta.get("style", "")
                if desc or budget_badge:
                    st.markdown(
                        f"""<div class='tm-card'>
                          {desc} &nbsp;
                          <span class='tm-badge tm-badge-blue'>{budget_badge}</span> &nbsp;
                          <span class='tm-badge tm-badge-purple'>{style_badge}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                render_itinerary(opt, itineraries, option_meta)

                col_confirm, col_view = st.columns([1, 1])
                with col_confirm:
                    if st.button(
                        "Use This Itinerary",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.toast("Itinerary saved to My Trip!")
                with col_view:
                    if st.button("View in My Trip", use_container_width=True):
                        st.info("Switch to the My Trip tab above.")
