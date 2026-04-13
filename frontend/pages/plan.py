"""
pages/plan.py — Plan Your Trip page

Collects trip details → calls POST /planner/generate (Agent3 only) → displays 3 itinerary options.
Agent6 (Explainability) is called lazily from My Trip page.
Selected option stored in st.session_state for My Trip page.
"""

import requests
import streamlit as st
from utils import sanitize_cost, sanitize_name, get_item_id, is_flight_item, is_hotel_item

BACKEND_URL = "http://localhost:8000"


# ── Helpers ───────────────────────────────────────────────────

def _call_generate(payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{BACKEND_URL}/planner/generate", json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        st.error("Request timed out — agents are taking too long. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure it is running on port 8000.")
    except Exception as e:
        st.error(f"Backend error: {e}")
    return None


def _render_change_panel(item: dict, item_id: str, result: dict, scoped_key: str = ""):
    key = scoped_key or (item.get("key") or "")
    overrides = st.session_state.setdefault("item_overrides", {})

    if is_flight_item(item):
        is_return = key == "flight_return"
        opts  = result.get("flight_options_return" if is_return else "flight_options_outbound", [])
        label = "Return Flight Options" if is_return else "Outbound Flight Options"
        icon  = "✈️"
    else:
        opts  = result.get("hotel_options", [])
        label = "Hotel Options"
        icon  = "🏨"

    if not opts:
        st.caption("No alternatives available.")
        return

    with st.container(border=True):
        st.markdown(f"##### {icon} {label}")
        current = overrides.get(item_id, item.get("name", ""))
        for i, opt in enumerate(opts):
            display  = opt.get("display", opt.get("name", f"Option {i+1}"))
            selected = current == display or current == opt.get("name", "")
            col_sel, col_info = st.columns([0.5, 8])
            with col_sel:
                if st.button("✓" if selected else "○", key=f"pick_{item_id}_{i}",
                             use_container_width=True,
                             type="primary" if selected else "secondary"):
                    overrides[item_id] = display
                    st.session_state[f"show_change_{key}"] = False
                    st.rerun()
            with col_info:
                st.markdown(f"<span style='font-size:13px'>{display}</span>", unsafe_allow_html=True)
        if st.button("Close", key=f"close_change_{key}_{item_id}"):
            st.session_state[f"show_change_{key}"] = False
            st.rerun()


def _render_itinerary(option: str, itineraries: dict, result: dict = None):
    days = itineraries.get(option, [])
    if not days:
        st.info("No itinerary data for this option.")
        return

    visited   = st.session_state.setdefault("visited", {})
    overrides = st.session_state.setdefault("item_overrides", {})

    for day_idx, day in enumerate(days):
        with st.expander(f"📅 {day['day']}", expanded=(day_idx == 0)):
            for item_idx, item in enumerate(day.get("items", [])):
                col_check, col_time, col_icon, col_name, col_cost, col_action = st.columns(
                    [0.5, 0.8, 0.4, 4, 1.2, 1.4]
                )
                item_id      = get_item_id(option, day_idx, item_idx, item["name"])
                is_checked   = visited.get(item_id, False)
                display_name = sanitize_name(overrides.get(item_id, item["name"]))
                display_cost = sanitize_cost(item.get("cost", ""))

                with col_check:
                    # Sync state immediately on change
                    new_checked = st.checkbox("", value=is_checked, key=f"chk_{item_id}", label_visibility="collapsed")
                    if new_checked != is_checked:
                        visited[item_id] = new_checked
                        st.rerun()

                with col_time:
                    st.markdown(f"<span style='color:#4a5a72;font-size:12px;font-family:monospace'>{item.get('time','')}</span>", unsafe_allow_html=True)
                with col_icon:
                    st.markdown(f"<span style='font-size:18px'>{item.get('icon','📍')}</span>", unsafe_allow_html=True)
                with col_name:
                    changed = item_id in overrides
                    suffix  = " <span style='color:#3b9eff;font-size:10px'>✎ changed</span>" if changed else ""
                    st.markdown(f"<span style='font-size:13px;font-weight:500'>{display_name}</span>{suffix}", unsafe_allow_html=True)
                with col_cost:
                    st.markdown(f"<span style='color:#7a90b0;font-size:12px;font-family:monospace'>{display_cost}</span>", unsafe_allow_html=True)
                with col_action:
                    key = item.get("key") or ""
                    scoped_key = f"{option}_{key}" if key else ""
                    can_change = result and key and (is_flight_item(item) or is_hotel_item(item))
                    if can_change:
                        if st.button("Change →", key=f"chg_{item_id}", use_container_width=True):
                            st.session_state[f"show_change_{scoped_key}"] = not st.session_state.get(f"show_change_{scoped_key}", False)

            for item_idx, item in enumerate(day.get("items", [])):
                key = item.get("key") or ""
                if not key:
                    continue
                scoped_key = f"{option}_{key}"
                item_id = get_item_id(option, day_idx, item_idx, item["name"])
                if st.session_state.get(f"show_change_{scoped_key}", False) and result:
                    _render_change_panel(item, item_id, result, scoped_key=scoped_key)


# ── Main render ───────────────────────────────────────────────

def render():
    st.markdown("### Plan Your Trip")
    st.markdown("<span style='color:#7a90b0;font-size:14px'>Fill in your trip details — agents will generate 3 personalised itinerary options.</span>", unsafe_allow_html=True)
    st.markdown("")

    user = st.session_state.get("user", {})

    col_left, col_right = st.columns([3, 1.5])

    with col_left:
        with st.container(border=True):
            st.markdown("**Trip Details**")

            row1_a, row1_b = st.columns(2)
            with row1_a:
                origin = st.text_input("From (origin city)", placeholder="e.g. Singapore")
            with row1_b:
                destination = st.text_input("To (destination)", placeholder="e.g. Kyoto, Japan")

            row2_a, row2_b = st.columns(2)
            with row2_a:
                dates = st.text_input("Travel dates", placeholder="e.g. 2026-05-01 to 2026-05-05")
            with row2_b:
                duration = st.text_input("Duration", placeholder="e.g. 5 days")

            row3_a, row3_b = st.columns(2)
            with row3_a:
                budget = st.selectbox("Budget style", ["budget", "moderate", "luxury"])
            with row3_b:
                preferences = st.text_input("Preferences", placeholder="e.g. vegetarian, culture, hiking")

            TIME_OPTS = [
                "No preference",
                "Midnight (00:00–06:00)",
                "Early Morning (06:00–09:00)",
                "Morning (09:00–12:00)",
                "Afternoon (12:00–17:00)",
                "Evening (17:00–20:00)",
                "Night (20:00–24:00)",
            ]
            st.markdown("**Departure Time Preference**")
            st.caption("Preferred departure window for outbound & return journey (flight / ferry / train)")
            row4_a, row4_b = st.columns(2)
            with row4_a:
                outbound_time_pref = st.selectbox("Outbound departure time", TIME_OPTS, key="out_time")
            with row4_b:
                return_time_pref = st.selectbox("Return departure time", TIME_OPTS, key="ret_time")

            generate = st.button("Generate Options →", type="primary", use_container_width=True)

        # ── Generate ──────────────────────────────────────────
        if generate:
            if not origin or not destination or not dates or not duration:
                st.warning("Please fill in origin, destination, dates, and duration.")
            else:
                st.session_state.plan_result    = None
                st.session_state.plan_generated = False
                st.session_state.plan_id        = None

                with col_right:
                    st.markdown("**Agent Activity**")
                    with st.spinner("Agents are working..."):
                        payload = {
                            "origin":             origin,
                            "destination":        destination,
                            "dates":              dates,
                            "duration":           duration,
                            "budget":             budget,
                            "preferences":        preferences,
                            "outbound_time_pref": "" if outbound_time_pref == "No preference" else outbound_time_pref,
                            "return_time_pref":   "" if return_time_pref == "No preference" else return_time_pref,
                            "user_profile":       user,
                        }
                        result = _call_generate(payload)

                if result:
                    st.session_state.plan_result     = result
                    st.session_state.plan_generated  = True
                    st.session_state.plan_id         = result.get("plan_id")
                    st.session_state.selected_option = list(result.get("itineraries", {}).keys())[0] if result.get("itineraries") else "A"
                    st.session_state.visited         = {}
                    st.session_state.explain_loaded  = False   # reset so My Trip fetches fresh
                    st.rerun()

        # ── Results ───────────────────────────────────────────
        if st.session_state.get("plan_generated") and st.session_state.get("plan_result"):
            result      = st.session_state.plan_result
            itineraries = result.get("itineraries",  {})
            option_meta = result.get("option_meta",  {})

            st.markdown("---")
            st.markdown("#### Choose your preferred itinerary style")
            st.caption("Click a style below to preview, then confirm at the bottom.")

            opt_cols = st.columns(len(option_meta) or 3)
            for i, (key, meta) in enumerate(option_meta.items()):
                with opt_cols[i]:
                    selected = st.session_state.get("selected_option") == key
                    label    = f"✓ {meta['badge']}" if selected else meta['badge']
                    st.markdown(
                        f"<div style='text-align:center;font-size:11px;color:#7a90b0;margin-bottom:4px'>"
                        f"Option {key}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(label, key=f"opt_{key}", use_container_width=True,
                                 type="primary" if selected else "secondary"):
                        st.session_state.selected_option = key
                        st.rerun()

            opt  = st.session_state.get("selected_option", list(option_meta.keys())[0])
            meta = option_meta.get(opt, {})
            st.markdown(
                f"""<div class='tm-card'>
                  <span style='font-size:13px;color:#c8d4e8'>{meta.get('desc','')}</span> &nbsp;
                  <span class='tm-badge tm-badge-blue'>{meta.get('budget','')}</span> &nbsp;
                  <span class='tm-badge tm-badge-purple'>{meta.get('style','')}</span>
                </div>""",
                unsafe_allow_html=True,
            )

            _render_itinerary(opt, itineraries, result)

            col_confirm, _ = st.columns(2)
            with col_confirm:
                if st.button("✓ Use This Itinerary", type="primary", use_container_width=True):
                    st.session_state.my_trip_option = opt
                    st.session_state.my_trip_result = result
                    st.session_state.active_page    = "my_trip"   # auto-redirect
                    st.rerun()

    # ── Right panel: Agent Activity ───────────────────────────
    with col_right:
        st.markdown("**Agent Activity**")
        result = st.session_state.get("plan_result")
        if result:
            tool_log = result.get("tool_log", [])
            for entry in tool_log[:8]:
                if entry.startswith("[") and not entry.startswith("[planner"):
                    st.success(f"✅ {entry.strip('[]')}")

            if tool_log:
                st.markdown("---")
                with st.expander("🔧 Tool Call Log", expanded=False):
                    for entry in tool_log:
                        st.caption(entry)
        else:
            st.markdown(
                "<div style='text-align:center;padding:40px 0;color:#4a5a72;font-size:13px'>"
                "🤖<br>Agent activity appears here once you start planning.</div>",
                unsafe_allow_html=True,
            )
