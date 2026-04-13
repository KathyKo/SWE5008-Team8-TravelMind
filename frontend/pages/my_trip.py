"""
pages/my_trip.py — My Trip page

Displays the confirmed itinerary. Calls /planner/explain lazily on first load
(Agent6 runs here, not during Plan generation — keeps Plan page fast).
"""

import requests
import streamlit as st
from utils import sanitize_cost, sanitize_name, get_item_id

BACKEND_URL = "http://localhost:8000"


def _load_explain(plan_id: str) -> dict:
    """Call /planner/explain and cache result in session state."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/planner/explain",
            json={"plan_id": plan_id},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Could not load explanations: {e}")
        return {}


def _render_explain_panel(item_key: str, explain_data: dict, toggle_key: str):
    d = explain_data.get(item_key)
    if not d:
        st.caption("No explanation available for this item.")
        return
    with st.container(border=True):
        st.markdown(f"##### 💡 Why **{d.get('name', 'this')}**?")

        st.markdown("**Why it was chosen**")
        matches = d.get("matches", [])
        if not matches:
            st.caption("Aligned with your preferences and budget.")
        for m in matches:
            st.markdown(f"✅ {m}")

        rating = d.get("rating", "")
        if rating:
            st.markdown("---")
            st.markdown(f"**Real Rating** &nbsp; {rating}")

        highlights = [h for h in d.get("review_highlights", []) if h]
        if highlights:
            st.markdown("**Guest Reviews**")
            for h in highlights:
                st.caption(f'"{h}"')

        cot = d.get("chain_of_thought", "")
        if cot:
            st.markdown("---")
            st.markdown("**Reasoning**")
            st.caption(cot)

        if st.button("Close", key=f"close_{toggle_key}"):
            st.session_state[toggle_key] = False
            st.rerun()


def render():
    result = st.session_state.get("my_trip_result")
    opt    = st.session_state.get("my_trip_option", "A")

    if not result:
        st.markdown("### My Trip")
        st.info("No itinerary selected yet. Go to the 🗺️ Plan tab to generate and confirm your trip.")
        return

    itineraries = result.get("itineraries",  {})
    option_meta = result.get("option_meta",  {})
    meta        = option_meta.get(opt, {})
    days        = itineraries.get(opt, [])
    plan_id     = st.session_state.get("plan_id", "")
    overrides   = st.session_state.get("item_overrides", {})

    st.markdown(f"### My Trip — {meta.get('label', f'Option {opt}')}")
    st.markdown(
        f"<span style='color:#7a90b0;font-size:14px'>"
        f"{result.get('dates', '')} · {meta.get('style', '')} · {meta.get('budget', '')}"
        f"</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Lazy explain: load Agent6 result on first visit ──────
    if plan_id and not st.session_state.get("explain_loaded"):
        with st.spinner("Loading reasoning from Explainability Agent..."):
            explain_result = _load_explain(plan_id)
        if explain_result:
            st.session_state.explain_data     = explain_result.get("explain_data",     {})
            st.session_state.chain_of_thought = explain_result.get("chain_of_thought", "")
            st.session_state.explain_loaded   = True

    explain_data     = st.session_state.get("explain_data",     {})
    chain_of_thought = st.session_state.get("chain_of_thought", "")

    col_main, col_side = st.columns([3, 1.5])

    with col_main:
        visited = st.session_state.setdefault("visited", {})
        visited_count = sum(1 for v in visited.values() if v)
        
        act1, act2 = st.columns([2, 2])
        with act1:
            st.success(f"✓ {visited_count} places visited")
        with act2:
            if st.button("🔄 Need to Re-plan?", use_container_width=True):
                st.session_state.active_page = "replan"
                st.rerun()

        st.markdown("---")

        if not days:
            st.info("No itinerary data found.")
        else:
            for day_idx, day in enumerate(days):
                with st.expander(f"📅 {day['day']}  —  {day['budget']}", expanded=(day_idx == 0)):
                    st.caption("✓ Check off places you've been")

                    for item_idx, item in enumerate(day.get("items", [])):
                        col_check, col_time, col_icon, col_name, col_cost, col_why = st.columns(
                            [0.5, 0.8, 0.4, 4, 1.2, 1.2]
                        )
                        item_id      = get_item_id(opt, day_idx, item_idx, item["name"])
                        is_checked   = visited.get(item_id, False)
                        display_name = sanitize_name(overrides.get(item_id, item["name"]))
                        display_cost = sanitize_cost(item.get("cost", ""))

                        with col_check:
                            checked = st.checkbox(
                                "", value=is_checked,
                                key=f"trip_chk_{item_id}",
                                label_visibility="collapsed",
                            )
                            if checked != is_checked:
                                visited[item_id] = checked
                                if checked:
                                    st.toast(f"✓ Marked **{display_name}** as visited", icon="✅")
                                st.rerun()

                        with col_time:
                            st.markdown(f"<span style='color:#4a5a72;font-size:12px;font-family:monospace'>{item.get('time','')}</span>", unsafe_allow_html=True)
                        with col_icon:
                            st.markdown(f"<span style='font-size:18px'>{item.get('icon','📍')}</span>", unsafe_allow_html=True)
                        with col_name:
                            label = display_name
                            if is_checked:
                                label = f"~~{label}~~ ✓"
                            st.markdown(f"<span style='font-size:13px;font-weight:500'>{label}</span>", unsafe_allow_html=True)
                        with col_cost:
                            st.markdown(f"<span style='color:#7a90b0;font-size:12px;font-family:monospace'>{display_cost}</span>", unsafe_allow_html=True)
                        with col_why:
                            if item.get("key"):
                                if st.button("Why? →", key=f"trip_why_{item_id}", use_container_width=True):
                                    toggle = f"show_why_{item_id}"
                                    st.session_state[toggle] = not st.session_state.get(toggle, False)

                    # Render explanation panels for this day
                    for item_idx, item in enumerate(day.get("items", [])):
                        item_id = get_item_id(opt, day_idx, item_idx, item["name"])
                        toggle_key = f"show_why_{item_id}"
                        if st.session_state.get(toggle_key, False):
                            _render_explain_panel(item.get("key"), explain_data, toggle_key)

    with col_side:
        with st.container(border=True):
            st.markdown("##### 💡 Agent Reasoning")

            if not st.session_state.get("explain_loaded") and plan_id:
                st.caption("Reasoning will appear once loaded above.")
            elif chain_of_thought:
                st.markdown("**Planning Chain of Thought**")
                for line in chain_of_thought.split("\n"):
                    if line.strip():
                        st.caption(line)
            else:
                st.caption("No reasoning trace available.")

            st.markdown("---")
            st.markdown("**Your Profile**")
            user = st.session_state.get("user", {})
            if user:
                for pref in user.get("prefs", []):
                    st.markdown(
                        f"<span class='tm-badge tm-badge-blue'>{pref}</span>",
                        unsafe_allow_html=True,
                    )
