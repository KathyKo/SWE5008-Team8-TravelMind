"""
pages/replan.py — Dynamic Re-planning page
"""

import html
import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
BACKEND_REPLAN_URL = os.getenv(
    "BACKEND_REPLAN_URL",
    f"{BACKEND_URL}/agent/replan",
).rstrip("/")


ICON_MAP = {
    "flight": "✈️",
    "hotel": "🏨",
    "activity": "🎯",
    "restaurant": "🍽️",
}


def _call_replan_backend(state_payload: dict) -> dict:
    resp = requests.post(BACKEND_REPLAN_URL, json={"state": state_payload}, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _run_pending_replan_if_needed() -> None:
    pending = st.session_state.get("replan_pending_state")
    if not pending:
        return
    try:
        with st.spinner("Dynamic replan agent is generating updates..."):
            result = _call_replan_backend(pending)
        replanner_output = result.get("replanner_output", {}) if isinstance(result, dict) else {}
        replanned_plan = (replanner_output.get("replanned_plan") or {}).get("plan") or []
        st.session_state.replan_backend_result = replanner_output
        st.session_state.replan_current_days = replanned_plan
        st.session_state.replan_unsatisfied = {}
        st.session_state.replan_error = ""
    except Exception as exc:
        st.session_state.replan_error = str(exc)
        st.session_state.replan_backend_result = {}
        st.session_state.replan_current_days = []
    finally:
        st.session_state.replan_pending_state = None


def _render_itinerary_with_checks(days: list[dict]) -> None:
    unsat = st.session_state.get("replan_unsatisfied", {}) or {}
    backend_result = st.session_state.get("replan_backend_result") or {}
    context = backend_result.get("context") or {}
    locked_item_keys = {str(x) for x in (context.get("locked_item_keys") or []) if str(x)}
    for day_idx, day in enumerate(days):
        day_title = day.get("day", f"Day {day_idx + 1}")
        day_budget = day.get("budget", "")
        expander_label = f"📅 {day_title}  —  {day_budget}" if day_budget else f"📅 {day_title}"
        with st.expander(expander_label, expanded=(day_idx == 0)):
            st.caption(
                "On this page, if you are not satisfied, you can select the itinerary items you want to modify, "
                "and we will update only those selected items. "
                "(Note: items you have already visited cannot be selected.)"
            )
            for item_idx, item in enumerate(day.get("items", [])):
                item_key = str(item.get("key") or f"item_{day_idx}_{item_idx}")
                # Slot id must be unique per row (same itinerary key can repeat across days).
                checkbox_id = f"replan_slot_{day_idx}_{item_idx}"
                is_locked_visited = item_key in locked_item_keys
                raw_icon = item.get("icon", "")
                icon_l = str(raw_icon).lower() if isinstance(raw_icon, str) else ""
                is_flight = icon_l == "flight"
                checked = bool(unsat.get(checkbox_id, False))
                col_check, col_time, col_icon, col_name, col_cost = st.columns([0.5, 0.8, 0.4, 4, 1.2])
                with col_check:
                    new_checked = st.checkbox(
                        "Mark for replan",
                        value=True if is_locked_visited else checked,
                        key=f"replan_chk_{checkbox_id}",
                        label_visibility="collapsed",
                        disabled=is_locked_visited or is_flight,
                    )
                    if (not is_locked_visited) and (new_checked != checked):
                        unsat[checkbox_id] = new_checked
                        st.session_state.replan_unsatisfied = unsat
                name_style = (
                    "text-decoration:line-through;color:#9ca3af"
                    if is_locked_visited
                    else "font-size:13px;font-weight:500"
                )
                time_color = "#9ca3af" if is_locked_visited else "#6b7280"
                icon_opacity = "opacity:0.35" if is_locked_visited else ""
                with col_time:
                    st.markdown(
                        f"<span style='color:{time_color};font-size:12px;font-family:monospace'>{item.get('time', '')}</span>",
                        unsafe_allow_html=True,
                    )
                with col_icon:
                    display_icon = ICON_MAP.get(raw_icon, raw_icon) if isinstance(raw_icon, str) else raw_icon
                    st.markdown(
                        f"<span style='font-size:18px;{icon_opacity}'>{display_icon}</span>",
                        unsafe_allow_html=True,
                    )
                with col_name:
                    st.markdown(
                        f"<span style='{name_style}'>{html.escape(str(item.get('name', '')))}</span>",
                        unsafe_allow_html=True,
                    )
                with col_cost:
                    st.markdown(
                        f"<span style='color:#6b7280;font-size:12px;font-family:monospace'>{item.get('cost', '')}</span>",
                        unsafe_allow_html=True,
                    )


def _build_replan_again_payload(days: list[dict], selected_option: str, prev_round: int) -> dict:
    unsat = st.session_state.get("replan_unsatisfied", {}) or {}
    backend_result = st.session_state.get("replan_backend_result") or {}
    context = backend_result.get("context") or {}
    replace_item_keys: list[str] = []
    replace_slots: list[str] = []
    locked_item_keys: list[str] = [str(x) for x in (context.get("locked_item_keys") or []) if str(x)]
    replace_item_names: list[str] = []
    locked_item_names: list[str] = [str(x) for x in (context.get("locked_item_names") or []) if str(x)]

    for day_idx, day in enumerate(days):
        for item_idx, item in enumerate(day.get("items", [])):
            item_key = str(item.get("key") or f"item_{day_idx}_{item_idx}")
            item_name = str(item.get("name") or "")
            checkbox_id = f"replan_slot_{day_idx}_{item_idx}"
            raw_icon = item.get("icon", "")
            if str(raw_icon).lower() == "flight":
                continue
            if unsat.get(checkbox_id, False):
                replace_item_keys.append(item_key)
                replace_slots.append(f"{day_idx}:{item_idx}")
                if item_name:
                    replace_item_names.append(item_name)

    summary = st.session_state.get("plan_request_summary") or {}
    plan_state = st.session_state.get("plan_state") or {}
    return {
        "plan_id": st.session_state.get("plan_id"),
        "origin": summary.get("origin") or plan_state.get("origin"),
        "destination": summary.get("destination") or plan_state.get("destination"),
        "dates": summary.get("dates") or plan_state.get("dates"),
        "duration": summary.get("duration") or plan_state.get("duration"),
        "budget": summary.get("budget") or plan_state.get("budget"),
        "preferences": plan_state.get("preferences"),
        "itineraries": st.session_state.get("plan_itineraries") or {},
        "option_meta": st.session_state.get("plan_option_meta") or {},
        "research": plan_state.get("research") or {},
        "inventory": plan_state.get("inventory") or {},
        "tool_log": list(plan_state.get("tool_log") or []),
        "compact_attractions": plan_state.get("compact_attractions") or [],
        "compact_restaurants": plan_state.get("compact_restaurants") or [],
        "compact_hotels": plan_state.get("compact_hotels") or [],
        "compact_flights_out": plan_state.get("compact_flights_out") or [],
        "compact_flights_ret": plan_state.get("compact_flights_ret") or [],
        "flight_options_outbound": st.session_state.get("plan_flight_outbound") or plan_state.get("flight_options_outbound") or [],
        "flight_options_return": st.session_state.get("plan_flight_return") or plan_state.get("flight_options_return") or [],
        "hotel_options": st.session_state.get("plan_hotel_options") or plan_state.get("hotel_options") or [],
        "att_list_text": plan_state.get("att_list_text") or "",
        "rest_list_text": plan_state.get("rest_list_text") or "",
        "hotel_list_text": plan_state.get("hotel_list_text") or "",
        "flight_out_text": plan_state.get("flight_out_text") or "",
        "flight_ret_text": plan_state.get("flight_ret_text") or "",
        "hotel_opts": plan_state.get("hotel_opts") or st.session_state.get("plan_hotel_options") or [],
        "replan_request": {
            "selected_option": selected_option,
            "itinerary_days": days,
            "replace_item_keys": replace_item_keys,
            "replace_slots": replace_slots,
            "replace_item_names": replace_item_names,
            "locked_item_keys": locked_item_keys,
            "locked_item_names": locked_item_names,
            "one_round_ban_names": list(context.get("one_round_ban_names_next") or []),
            "round": prev_round + 1,
            "source": "replan_page_followup",
        },
    }


def render():
    st.session_state.setdefault("replan_backend_result", {})
    st.session_state.setdefault("replan_current_days", [])
    st.session_state.setdefault("replan_unsatisfied", {})
    st.session_state.setdefault("replan_error", "")

    _run_pending_replan_if_needed()

    backend_result = st.session_state.get("replan_backend_result") or {}
    context = backend_result.get("context") or {}
    days = st.session_state.get("replan_current_days") or []

    st.markdown("### Dynamic Re-planning")
    st.markdown(
        "<span style='color:#7a90b0;font-size:14px'>"
        "Review the updated itinerary and mark items you still want to change.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.container(border=True):
        c1, c2, c3 = st.columns([0.3, 4, 1])
        with c1:
            st.markdown("🟢")
        with c2:
            st.markdown(f"**{context.get('location', 'Current trip context')}**")
            st.caption(
                f"{context.get('current_day', 'Dynamic replan')} · {context.get('current_time', 'N/A')} · "
                f"Originally: {context.get('original_plan_title', 'Selected option')}"
            )
        with c3:
            st.markdown("🔁 Replanned")

    if st.session_state.get("replan_error"):
        st.error(st.session_state["replan_error"])
        return
    if not days:
        st.info("Go to My Trip, tick items, then click 'Need to Dynamic Replan?'. The replanned result will appear here.")
        return

    st.markdown("---")
    st.markdown("#### Replanned Itinerary")
    _render_itinerary_with_checks(days)

    col_a, col_b = st.columns(2)
    selected_option = str(backend_result.get("winner_option") or st.session_state.get("selected_option", "A"))
    prev_round = int(backend_result.get("round") or 1)

    with col_a:
        if st.button("🔁 Replan Selected Items Again", type="primary", use_container_width=True):
            payload = _build_replan_again_payload(days, selected_option, prev_round)
            if not payload["replan_request"].get("replace_slots") and not payload["replan_request"].get(
                "replace_item_keys"
            ):
                st.warning("Please tick the itinerary items you want to modify first.")
            else:
                st.session_state.replan_pending_state = payload
                st.session_state.replan_error = ""
                st.rerun()
    with col_b:
        if st.button("✅ Use This Replan In My Trip", use_container_width=True):
            all_itins = dict(st.session_state.get("plan_itineraries") or {})
            all_itins[selected_option] = days
            st.session_state.plan_itineraries = all_itins
            st.session_state.selected_option = selected_option
            st.session_state.pending_nav = "my_trip"
            st.toast("Replanned itinerary applied to My Trip.")
            st.rerun()