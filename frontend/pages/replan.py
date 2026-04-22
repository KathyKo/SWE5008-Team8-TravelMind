"""
pages/replan.py — Dynamic Re-planning page
"""

import time
import os
import requests
import streamlit as st
from data.store import SITUATIONS, TIME_OPTIONS, ALT_OPTIONS, REPLAN_LOG

AGENTS_REPLAN_URL = os.getenv("AGENTS_REPLAN_URL", "http://agents:8001/replan").rstrip("/")


def _build_feedback() -> dict:
    return {
        "situation": st.session_state.replan_situation,
        "time": st.session_state.replan_time,
        "current_day": "Day 3",
        "current_time": "14:00",
        "current_location": "Central Kyoto, near Nishiki Market",
    }


def _call_replan_backend() -> dict:
    payload = {
        "user_feedback": _build_feedback(),
        "user_id": st.session_state.get("user_id", "demo_user"),
    }
    resp = requests.post(AGENTS_REPLAN_URL, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()


def render():
    st.session_state.setdefault("replan_backend_result", {})
    st.session_state.setdefault("replan_alt_options", ALT_OPTIONS)

    context = (st.session_state.get("replan_backend_result", {}) or {}).get("context", {})
    st.markdown("### Dynamic Re-planning")
    st.markdown("<span style='color:#7a90b0;font-size:14px'>Tell us what changed — TravelMind will adapt your itinerary.</span>", unsafe_allow_html=True)
    st.markdown("")

    # ── Location bar ─────────────────────────────────────────
    with st.container(border=True):
        c1, c2, c3 = st.columns([0.3, 4, 1])
        with c1:
            st.markdown("🟢")
        with c2:
            st.markdown(f"**{context.get('location', 'Central Kyoto, near Nishiki Market')}**")
            st.caption(
                f"{context.get('current_day', 'Day 3')} · {context.get('current_time', '14:00')} · "
                f"Originally: {context.get('original_plan_title', 'Debate winner plan')}"
            )
        with c3:
            st.markdown("🌤 26°C")

    st.markdown("---")

    # ── Step 1: Situation ─────────────────────────────────────
    st.markdown("#### 1. What's changed?")
    st.caption("Select the situation that best describes your current state.")

    col1, col2, col3, col4 = st.columns(4)
    situation_cols = [col1, col2, col3, col4]

    for i, sit in enumerate(SITUATIONS):
        with situation_cols[i]:
            selected = st.session_state.replan_situation == sit["key"]
            if st.button(
                f"{sit['emoji']}\n**{sit['label']}**\n{sit['desc']}",
                key=f"sit_{sit['key']}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state.replan_situation = sit["key"]
                st.session_state.replan_time = None
                st.session_state.replan_done = False
                st.session_state.chosen_alt = None
                st.rerun()

    # ── Step 2: Time ──────────────────────────────────────────
    if st.session_state.replan_situation:
        sit_label = next(s["label"] for s in SITUATIONS if s["key"] == st.session_state.replan_situation)
        st.success(f"Selected: **{sit_label}**")
        st.markdown("---")
        st.markdown("#### 2. How much time do you have?")
        st.caption("We'll find alternatives that fit your remaining day.")

        t_col1, t_col2, t_col3 = st.columns(3)
        time_cols = [t_col1, t_col2, t_col3]

        for i, opt in enumerate(TIME_OPTIONS):
            with time_cols[i]:
                selected = st.session_state.replan_time == opt["key"]
                if st.button(
                    f"{opt['emoji']}\n**{opt['label']}**\n{opt['desc']}",
                    key=f"time_{opt['key']}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    st.session_state.replan_time = opt["key"]
                    st.session_state.replan_done = False
                    st.session_state.chosen_alt = None
                    st.rerun()

    # ── Re-plan working ───────────────────────────────────────
    if st.session_state.replan_time and not st.session_state.replan_done:
        st.markdown("---")
        with st.container(border=True):
            st.markdown("🔄 **Re-planning Agent working...**")
            log_placeholder = st.empty()
            log_lines = []
            for tag, msg in REPLAN_LOG:
                color = "#3b9eff" if tag == "INFO" else "#10b981"
                log_lines.append(f"<span style='color:{color};font-family:monospace;font-size:11px'>[{tag}]</span> <span style='font-size:12px;color:#7a90b0'>{msg}</span>")
                log_placeholder.markdown("<br>".join(log_lines), unsafe_allow_html=True)
                time.sleep(0.5)
            try:
                result = _call_replan_backend()
                replanner_output = result.get("replanner_output", {}) if isinstance(result, dict) else {}
                alts = replanner_output.get("alternatives")
                if isinstance(alts, list) and alts:
                    st.session_state.replan_alt_options = alts
                else:
                    st.session_state.replan_alt_options = ALT_OPTIONS
                st.session_state.replan_backend_result = replanner_output
            except Exception as exc:
                st.session_state.replan_alt_options = ALT_OPTIONS
                st.session_state.replan_backend_result = {"error": str(exc)}
        st.session_state.replan_done = True
        st.rerun()

    # ── Alternatives ──────────────────────────────────────────
    if st.session_state.replan_done:
        st.markdown("---")
        st.markdown("#### Nearby Alternatives")
        backend_error = (st.session_state.get("replan_backend_result") or {}).get("error")
        if backend_error:
            st.info(f"后端重规划暂不可用，已回退到演示方案：{backend_error}")

        alt_options = st.session_state.get("replan_alt_options") or ALT_OPTIONS
        for i, alt in enumerate(alt_options):
            with st.container(border=True):
                chosen = st.session_state.chosen_alt == i
                c_icon, c_body, c_btn = st.columns([0.5, 5, 1.5])
                with c_icon:
                    st.markdown(f"<span style='font-size:28px'>{alt['icon']}</span>", unsafe_allow_html=True)
                with c_body:
                    st.markdown(f"**{alt['name']}**")
                    st.caption(alt["desc"])
                    tag_color = "#10b981" if alt["tag"] == "free" else "#7a90b0"
                    st.markdown(
                        f"<span style='font-size:11px;font-family:monospace;color:{tag_color}'>{alt['price']}</span> &nbsp; "
                        f"<span style='font-size:11px;color:#f59e0b'>{alt['rating']}</span> &nbsp; "
                        f"<span style='font-size:11px;color:#4a5a72'>{alt['dist']}</span>",
                        unsafe_allow_html=True,
                    )
                with c_btn:
                    if st.button(
                        "✓ Selected" if chosen else "Select",
                        key=f"alt_{i}",
                        use_container_width=True,
                        type="primary" if chosen else "secondary",
                    ):
                        st.session_state.chosen_alt = i
                        st.rerun()

        # ── Ripple effect ──────────────────────────────────────
        if st.session_state.chosen_alt is not None:
            alt = alt_options[st.session_state.chosen_alt]
            ripple = (st.session_state.get("replan_backend_result") or {}).get("ripple_effect")
            if ripple:
                st.warning(f"📅 **Ripple effect:** {ripple}")
            else:
                st.warning(
                    f"📅 **Ripple effect:** {alt['name']} selected. "
                    "Bamboo Grove moved to Day 4 (09:00). Tenryuji shifted to Day 4 afternoon. "
                    "Budget unchanged — SGD 2,847 total."
                )