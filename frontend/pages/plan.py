"""
pages/plan.py — Plan Your Trip page
"""

import time
import streamlit as st
from data.store import (
    AGENT_STEPS, DEBATE_MESSAGES, ITINERARIES, OPTION_META, EXPLAIN_DATA
)


def render_explain_modal(key: str):
    d = EXPLAIN_DATA.get(key)
    if not d:
        return
    with st.expander(f"💡 Why was **{d['name']}** recommended?", expanded=True):
        st.markdown("**Preference Matches**")
        for m in d["matches"]:
            st.markdown(f"✅ {m}")
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


def render_itinerary(option: str):
    days = ITINERARIES.get(option, [])
    visited = st.session_state.visited

    for day_idx, day in enumerate(days):
        with st.expander(f"📅 {day['day']}  —  {day['budget']}", expanded=(day_idx == 0)):
            st.caption("✓ Check off places you've been — this trains your personal AI profile")
            for item in day["items"]:
                col_check, col_time, col_icon, col_name, col_cost, col_why = st.columns(
                    [0.5, 0.8, 0.4, 4, 1.2, 1.2]
                )
                item_id = f"{option}_{day_idx}_{item['name']}"
                is_checked = visited.get(item_id, False)

                with col_check:
                    if st.checkbox(
                        "Mark as visited",
                        value=is_checked,
                        key=f"chk_{item_id}",
                        label_visibility="collapsed",
                    ):
                        if not is_checked:
                            st.session_state.visited[item_id] = True
                            st.toast(f"✓ Marked **{item['name']}** as visited — AI profile updated", icon="✅")
                    else:
                        st.session_state.visited[item_id] = False

                with col_time:
                    st.markdown(f"<span style='color:#4a5a72;font-size:12px;font-family:monospace'>{item['time']}</span>", unsafe_allow_html=True)
                with col_icon:
                    st.markdown(f"<span style='font-size:18px'>{item['icon']}</span>", unsafe_allow_html=True)
                with col_name:
                    st.markdown(f"<span style='font-size:13px;font-weight:500'>{item['name']}</span>", unsafe_allow_html=True)
                with col_cost:
                    st.markdown(f"<span style='color:#7a90b0;font-size:12px;font-family:monospace'>{item['cost']}</span>", unsafe_allow_html=True)
                with col_why:
                    if item.get("key") and st.button("Why? →", key=f"why_{item_id}", use_container_width=True):
                        st.session_state[f"show_explain_{item['key']}"] = True

            # Show explain panels
            for item in day["items"]:
                if item.get("key") and st.session_state.get(f"show_explain_{item['key']}", False):
                    render_explain_modal(item["key"])
                    if st.button("Close", key=f"close_{item['key']}_{day_idx}"):
                        st.session_state[f"show_explain_{item['key']}"] = False
                        st.rerun()


def render():
    st.markdown("### Plan Your Trip")
    st.markdown("<span style='color:#7a90b0;font-size:14px'>Tell TravelMind what you have in mind — agents will generate multiple itinerary options for you.</span>", unsafe_allow_html=True)
    st.markdown("")

    col_left, col_right = st.columns([3, 1.5])

    with col_left:
        # ── Input ────────────────────────────────────────────
        with st.container(border=True):
            st.markdown("**Your request**")
            st.text_area(
                "query",
                value="I want to visit Kyoto for 5 days, budget SGD 3000, vegetarian, love cultural heritage and local food",
                height=80,
                label_visibility="collapsed",
            )
            tag_col1, tag_col2, tag_col3, tag_col4, _, btn_col = st.columns([1, 1, 1.2, 1, 2, 1.8])
            with tag_col1:
                st.markdown("`🗓 5 days`")
            with tag_col2:
                st.markdown("`💰 SGD 3000`")
            with tag_col3:
                st.markdown("`🌿 Vegetarian`")
            with tag_col4:
                st.markdown("`🏯 Culture`")
            with btn_col:
                generate = st.button("Generate Options →", type="primary", use_container_width=True)

        if generate:
            st.session_state.plan_generated = False
            with col_right:
                st.markdown("**Agent Activity**")
                progress_bar = st.progress(0, text="Starting agents...")
                step_placeholders = [st.empty() for _ in AGENT_STEPS]

            for i, step in enumerate(AGENT_STEPS):
                progress_bar.progress((i + 1) / len(AGENT_STEPS), text=f"{step['name']} working...")
                step_placeholders[i].success(f"{step['icon']} **{step['name']}** — {step['detail']}")
                time.sleep(0.6)

            progress_bar.progress(1.0, text="✅ Complete!")
            st.session_state.plan_generated = True
            st.rerun()

        # ── Results ──────────────────────────────────────────
        if st.session_state.plan_generated:
            st.markdown("---")
            st.markdown("#### Choose your preferred itinerary style")

            opt_cols = st.columns(3)
            for i, (key, meta) in enumerate(OPTION_META.items()):
                with opt_cols[i]:
                    selected = st.session_state.selected_option == key
                    if st.button(
                        f"{'✓ ' if selected else ''}{meta['badge']}",
                        key=f"opt_{key}",
                        use_container_width=True,
                        type="primary" if selected else "secondary",
                    ):
                        st.session_state.selected_option = key
                        st.rerun()

            opt = st.session_state.selected_option
            meta = OPTION_META[opt]
            st.markdown(
                f"""<div class='tm-card'>
                  {meta['desc']} &nbsp;
                  <span class='tm-badge tm-badge-blue'>{meta['budget']}</span> &nbsp;
                  <span class='tm-badge tm-badge-purple'>{meta['style']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

            render_itinerary(opt)

            col_confirm, col_view = st.columns([1, 1])
            with col_confirm:
                if st.button("✓ Use This Itinerary", type="primary", use_container_width=True):
                    st.toast("✅ Itinerary saved to My Trip!", icon="✅")
            with col_view:
                if st.button("View in My Trip →", use_container_width=True):
                    st.info("Switch to the 📅 My Trip tab above.")

    # ── Agent Panel (shown after generation) ─────────────────
    with col_right:
        if st.session_state.plan_generated:
            st.markdown("**Agent Activity**")
            for step in AGENT_STEPS:
                st.success(f"{step['icon']} **{step['name']}** — {step['detail']}")

            st.markdown("---")
            st.markdown("**⚔️ Debate & Critique**")
            for msg_type, sender, text in DEBATE_MESSAGES:
                if msg_type == "critique":
                    st.error(f"🗡 **{sender}**: {text}")
                elif msg_type == "reply":
                    st.info(f"📅 **{sender}**: {text}")
                else:
                    st.success(f"💡 **{sender}**: {text}")
        else:
            st.markdown("**Agent Activity**")
            st.markdown(
                "<div style='text-align:center;padding:40px 0;color:#4a5a72;font-size:13px'>"
                "🤖<br>Agent activity appears here once you start planning.</div>",
                unsafe_allow_html=True,
            )
