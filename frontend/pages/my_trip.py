"""
pages/my_trip.py — My Trip page
"""

import html

import streamlit as st

ICON_MAP = {
    "flight": "✈️",
    "hotel": "🏨",
    "activity": "🎯",
    "restaurant": "🍽️",
}


def _build_replan_request_from_my_trip(days: list[dict], selected_opt: str) -> dict:
    visited = st.session_state.get("visited", {}) or {}
    replace_item_keys: list[str] = []
    locked_item_keys: list[str] = []
    replace_item_names: list[str] = []
    locked_item_names: list[str] = []

    for day_idx, day in enumerate(days):
        for item_idx, item in enumerate(day.get("items", [])):
            item_id = f"trip_{day_idx}_{item.get('name', item_idx)}"
            item_key = str(item.get("key") or f"item_{day_idx}_{item_idx}")
            item_name = str(item.get("name") or "")
            if visited.get(item_id, False):
                replace_item_keys.append(item_key)
                if item_name:
                    replace_item_names.append(item_name)
            else:
                locked_item_keys.append(item_key)
                if item_name:
                    locked_item_names.append(item_name)

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
        "replan_request": {
            "selected_option": selected_opt,
            "itinerary_days": days,
            "replace_item_keys": replace_item_keys,
            "replace_item_names": replace_item_names,
            "locked_item_keys": locked_item_keys,
            "locked_item_names": locked_item_names,
            "round": 1,
            "source": "my_trip_checked_items",
        },
    }


# ─── Right panel ──────────────────────────────────────────────────────────────

def _render_right_panel():
    plan_state = st.session_state.get("plan_state")
    selected_check = (
        st.session_state.get("selected_option_check")
        or (plan_state or {}).get("selected_option_check")
        or {}
    )

    with st.container(border=True):

        # ── User Profile ──────────────────────────────────
        st.markdown("**User Profile**")
        user = st.session_state.get("user") or {}
        username = user.get("name") or user.get("username") or ""
        if username:
            st.markdown(
                f"<span style='font-size:14px;font-weight:700;color:inherit'>{html.escape(username)}</span>",
                unsafe_allow_html=True,
            )

        if plan_state:
            ipo = plan_state.get("intent_profile_output") or {}
            soft = ipo.get("soft_preferences") or {}

            tags = list(soft.get("interest_tags") or [])
            vibe = soft.get("vibe") or ""
            style = soft.get("travel_style") or ""
            prefs = [t for t in ([vibe, style] + tags) if t]
            if prefs:
                tags_html = " ".join(
                    f"<span style='background:rgba(59,130,246,0.15);"
                    f"border:1px solid rgba(59,130,246,0.3);"
                    f"border-radius:12px;padding:2px 8px;"
                    f"font-size:11px;color:#93c5fd;margin-right:4px'>{p}</span>"
                    for p in prefs[:5]
                )
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                for pref in user.get("prefs", []):
                    st.markdown(
                        f"<span class='tm-badge tm-badge-blue'>{pref}</span>",
                        unsafe_allow_html=True,
                    )
        else:
            for pref in user.get("prefs", []):
                st.markdown(
                    f"<span class='tm-badge tm-badge-blue'>{pref}</span>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Fairness & Bias Checks ────────────────────────
        st.markdown("**Fairness & Bias Checks**")
        if selected_check:
            filter_bubble_detected = bool(selected_check.get("filter_bubble_detected"))
            demographic_bias_detected = bool(selected_check.get("demographic_bias_detected"))
            filter_bubble_detail = str(selected_check.get("filter_bubble_detail") or "").strip()
            demographic_bias_detail = str(selected_check.get("demographic_bias_detail") or "").strip()
            cold_start_note = str(selected_check.get("cold_start_note") or "").strip()
            confidence = str(selected_check.get("personalization_confidence") or "").lower()

            if not filter_bubble_detected:
                with st.expander("✅ No filter bubble detected", expanded=False):
                    st.markdown(
                        f"<span style='color:#d1fae5;font-size:13px;line-height:1.65'>"
                        f"{html.escape(filter_bubble_detail or 'No obvious echo-chamber pattern found in this itinerary.')}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
            else:
                with st.expander("⚠ Filter bubble risk detected", expanded=False):
                    st.markdown(
                        f"<span style='color:#fde68a;font-size:13px;line-height:1.65'>"
                        f"{html.escape(filter_bubble_detail or 'This itinerary may over-focus on narrow activity types.')}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

            if not demographic_bias_detected:
                with st.expander("✅ No demographic bias", expanded=False):
                    st.markdown(
                        f"<span style='color:#d1fae5;font-size:13px;line-height:1.65'>"
                        f"{html.escape(demographic_bias_detail or 'No direct demographic bias indicators were found.')}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
            else:
                with st.expander("⚠ Demographic bias risk detected", expanded=False):
                    st.markdown(
                        f"<span style='color:#fde68a;font-size:13px;line-height:1.65'>"
                        f"{html.escape(demographic_bias_detail or 'Potential demographic sensitivity detected; review is recommended.')}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )

            if cold_start_note:
                st.warning(f"⚠ {cold_start_note}")
            elif confidence == "low":
                st.warning("⚠ Cold-start: Limited history - 3 more trips needed for full personalisation")
        elif plan_state:
            is_valid = plan_state.get("is_valid")
            debate_count = plan_state.get("debate_count") or 0
            critique = plan_state.get("critique") or {}
            debate_output = plan_state.get("debate_output") or {}

            if is_valid is True:
                st.markdown(
                    "<div style='background:rgba(16,185,129,0.12);"
                    "border:1px solid rgba(16,185,129,0.35);"
                    "border-radius:8px;padding:8px 12px;margin-bottom:8px'>"
                    "<span style='color:#10b981;font-size:13px'>✓ Approved</span></div>",
                    unsafe_allow_html=True,
                )
            elif debate_count > 0:
                st.markdown(
                    f"<div style='background:rgba(245,158,11,0.12);"
                    f"border:1px solid rgba(245,158,11,0.35);"
                    f"border-radius:8px;padding:8px 12px;margin-bottom:8px'>"
                    f"<span style='color:#f59e0b;font-size:13px'>⚠ Revised ({debate_count} round(s))</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.success("✓ No filter bubble detected")
                st.success("✓ No demographic bias")

            if isinstance(critique, dict) and critique:
                for dim, val in list(critique.items())[:6]:
                    passed = val is True or (
                        isinstance(val, str) and val.lower() in ("pass", "ok", "true", "yes")
                    )
                    color = "#10b981" if passed else "#ef4444"
                    st.markdown(
                        f"<div style='display:flex;align-items:center;margin-top:4px'>"
                        f"<span style='color:{color};margin-right:6px;font-size:10px'>●</span>"
                        f"<span style='color:#e8edf5;font-size:12px'>{html.escape(str(dim))}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            summary = (
                debate_output.get("current_round_summary")
                if isinstance(debate_output, dict)
                else None
            )
            if summary:
                st.markdown(
                    f"<div style='color:#7a90b0;font-size:11px;margin-top:6px;"
                    f"font-style:italic'>{html.escape(str(summary)[:200])}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("✓ No filter bubble detected")
            st.success("✓ No demographic bias")
            st.warning("⚠ Cold-start: Limited history — 3 more trips needed for full personalisation")

        st.markdown("---")

        # ── Why behind the wander ─────────────────────────
        st.markdown("**Why behind the wander**")
        if plan_state:
            summary_block = plan_state.get("summary") or {}
            overall = ""
            day_summaries = {}
            if isinstance(summary_block, dict):
                overall = summary_block.get("overall_summary") or ""
                day_summaries = summary_block.get("day_summaries") or {}

            if not overall:
                # fallback: older pipeline format
                ex = plan_state.get("explain_data") or plan_state.get("explanation") or {}
                if isinstance(ex, dict):
                    overall = ex.get("summary") or ex.get("overall_summary") or ""

            if overall:
                day_html = ""
                if isinstance(day_summaries, dict):
                    for day_key, day_text in day_summaries.items():
                        day_html += (
                            f"<div style='margin-top:6px'>"
                            f"<span style='font-weight:600;font-size:11px'>{html.escape(str(day_key))}:</span> "
                            f"<span style='font-size:11px;color:#6b7280'>{html.escape(str(day_text))}</span>"
                            f"</div>"
                        )
                st.markdown(
                    f"<div style='max-height:260px;overflow-y:auto;padding-right:4px'>"
                    f"<div style='font-size:12px;margin-bottom:6px'>{html.escape(str(overall))}</div>"
                    f"{day_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No explanation available yet.")
        else:
            st.caption("Generate a trip in Planner to see explanations.")


# ─── Main render ──────────────────────────────────────────────────────────────

def render():
    # Use real plan data if available, else fall back to demo data
    plan_itineraries = st.session_state.get("plan_itineraries") or {}
    plan_option_meta = st.session_state.get("plan_option_meta") or {}
    selected_opt = st.session_state.get("selected_option", "A")

    if plan_itineraries:
        days = plan_itineraries.get(selected_opt) or []
        destination = (st.session_state.get("plan_request_summary") or {}).get("destination", "")
        dates = (st.session_state.get("plan_request_summary") or {}).get("dates", "")
        meta = plan_option_meta.get(selected_opt) or {}
        label = meta.get("badge", meta.get("label", f"Option {selected_opt}"))
        title = f"My Trip — {destination}" if destination else "My Trip"
        subtitle = f"{dates} · {label}" if dates else label
    else:
        days = []
        title = "My Trip"
        subtitle = ""

    st.markdown(f"### {title}")
    st.markdown(
        f"<span style='color:#7a90b0;font-size:14px'>{subtitle}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    col_main, col_side = st.columns([3, 1.5])

    with col_main:
        # ── Action bar ───────────────────────────────────────
        visited_count = sum(1 for v in st.session_state.get("visited", {}).values() if v)
        act1, act2 = st.columns(2)
        with act1:
            st.markdown(
                f"<div style='border:1px solid #d1fae5;background:#f0fdf4;"
                f"border-radius:8px;padding:8px 14px;font-size:14px;color:#065f46;"
                f"font-weight:500;line-height:1.8'>✓ {visited_count} places visited</div>",
                unsafe_allow_html=True,
            )
        with act2:
            if st.button("🔄 Need to Re-plan?", use_container_width=True):
                st.session_state._pending_nav = "replan"
                st.rerun()

        st.markdown("---")

        # ── Itinerary ────────────────────────────────────────
        if "visited" not in st.session_state:
            st.session_state.visited = {}
        visited = st.session_state.visited

        for day_idx, day in enumerate(days):
            day_title = day.get("day", f"Day {day_idx + 1}")
            day_budget = day.get("budget", "")
            expander_label = f"📅 {day_title}  —  {day_budget}" if day_budget else f"📅 {day_title}"
            with st.expander(expander_label, expanded=(day_idx == 0)):
                st.caption("✓ Check off places you've been — this trains your personal AI profile")

                for item_idx, item in enumerate(day.get("items", [])):
                    item_id = f"trip_{day_idx}_{item.get('name', item_idx)}"
                    is_checked = visited.get(item_id, False)

                    col_check, col_time, col_icon, col_name, col_cost = st.columns(
                        [0.5, 0.8, 0.4, 4, 1.2]
                    )

                    with col_check:
                        checked = st.checkbox(
                            "Mark as visited",
                            value=is_checked,
                            key=f"trip_chk_{item_id}",
                            label_visibility="collapsed",
                        )
                        if checked != is_checked:
                            st.session_state.visited[item_id] = checked
                            if checked:
                                st.toast(
                                    f"✓ Marked **{item.get('name', '')}** as visited",
                                    icon="✅",
                                )
                            st.rerun()

                    name_style = (
                        "text-decoration:line-through;color:#9ca3af"
                        if checked
                        else "color:inherit"
                    )
                    time_color = "#9ca3af" if checked else "#6b7280"
                    icon_opacity = "opacity:0.35" if checked else ""

                    with col_time:
                        st.markdown(
                            f"<span style='color:{time_color};font-size:12px;"
                            f"font-family:monospace'>{item.get('time', '')}</span>",
                            unsafe_allow_html=True,
                        )
                    with col_icon:
                        raw_icon = item.get("icon", "")
                        display_icon = ICON_MAP.get(raw_icon, raw_icon) if isinstance(raw_icon, str) else raw_icon
                        st.markdown(
                            f"<span style='font-size:18px;{icon_opacity}'>{display_icon}</span>",
                            unsafe_allow_html=True,
                        )
                    with col_name:
                        st.markdown(
                            f"<span style='{name_style};font-size:13px;font-weight:500'>"
                            f"{html.escape(item.get('name', ''))}</span>",
                            unsafe_allow_html=True,
                        )
                    with col_cost:
                        st.markdown(
                            f"<span style='color:#6b7280;font-size:12px;font-family:monospace'>"
                            f"{item.get('cost', '')}</span>",
                            unsafe_allow_html=True,
                        )


    with col_side:
        _render_right_panel()
