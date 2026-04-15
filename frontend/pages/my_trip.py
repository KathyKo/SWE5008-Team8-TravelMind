"""
pages/my_trip.py — My Trip page
"""

import streamlit as st
from data.store import ITINERARIES, EXPLAIN_DATA


def render():
    st.markdown("### My Trip — Kyoto")
    st.markdown("<span style='color:#7a90b0;font-size:14px'>10 Mar – 14 Mar 2026 · Option A: Cultural Focus</span>", unsafe_allow_html=True)
    st.markdown("")

    col_main, col_side = st.columns([3, 1.5])

    with col_main:
        # ── Budget bar ───────────────────────────────────────
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown("**Budget Used**")
                st.progress(0.949, text="SGD 2,847 / 3,000 — SGD 153 remaining")
            with c2:
                st.metric("Spent", "SGD 2,847", delta="-153 under budget", delta_color="normal")

        # ── Action bar ───────────────────────────────────────
        visited_count = sum(1 for v in st.session_state.visited.values() if v)
        act1, act2, act3 = st.columns([2, 2, 3])
        with act1:
            st.success(f"✓ {visited_count} places visited")
        with act2:
            if st.button("🔄 Need to Re-plan?", use_container_width=True):
                st.info("Switch to the 🔄 Re-plan tab above.")

        st.markdown("---")

        # ── Itinerary ────────────────────────────────────────
        days = ITINERARIES["A"]
        visited = st.session_state.visited

        for day_idx, day in enumerate(days):
            with st.expander(f"📅 {day['day']}  —  {day['budget']}", expanded=(day_idx == 0)):
                st.caption("✓ Check off places you've been — this trains your personal AI profile")

                for item in day["items"]:
                    col_check, col_time, col_icon, col_name, col_cost, col_why = st.columns(
                        [0.5, 0.8, 0.4, 4, 1.2, 1.2]
                    )
                    item_id = f"trip_{day_idx}_{item['name']}"
                    is_checked = visited.get(item_id, False)

                    with col_check:
                        checked = st.checkbox(
                            "",
                            value=is_checked,
                            key=f"trip_chk_{item_id}",
                            label_visibility="collapsed",
                        )
                        if checked != is_checked:
                            st.session_state.visited[item_id] = checked
                            if checked:
                                st.toast(f"✓ Marked **{item['name']}** as visited — AI profile updated", icon="✅")

                    with col_time:
                        st.markdown(f"<span style='color:#4a5a72;font-size:12px;font-family:monospace'>{item['time']}</span>", unsafe_allow_html=True)
                    with col_icon:
                        st.markdown(f"<span style='font-size:18px'>{item['icon']}</span>", unsafe_allow_html=True)
                    with col_name:
                        label = item["name"]
                        if visited.get(item_id):
                            label = f"~~{label}~~ ✓"
                        st.markdown(f"<span style='font-size:13px;font-weight:500'>{item['name']}</span>", unsafe_allow_html=True)
                    with col_cost:
                        st.markdown(f"<span style='color:#7a90b0;font-size:12px;font-family:monospace'>{item['cost']}</span>", unsafe_allow_html=True)
                    with col_why:
                        if item.get("key"):
                            if st.button("Why? →", key=f"trip_why_{item_id}", use_container_width=True):
                                st.session_state[f"trip_explain_{item['key']}"] = not st.session_state.get(f"trip_explain_{item['key']}", False)

                # Explain panels
                for item in day["items"]:
                    if item.get("key") and st.session_state.get(f"trip_explain_{item['key']}", False):
                        d = EXPLAIN_DATA[item["key"]]
                        with st.container(border=True):
                            st.markdown(f"##### 💡 Why {d['name']}?")
                            st.markdown("**Preference Matches**")
                            for m in d["matches"]:
                                st.markdown(f"✅ {m}")
                            st.info(f"👥 {d['similar']}")
                            st.markdown("**Score Breakdown**")
                            for label, score in d["scores"]:
                                st.progress(score / 100, text=f"{label}: {score}%")
                            if st.button("Close", key=f"trip_close_{item['key']}_{day_idx}"):
                                st.session_state[f"trip_explain_{item['key']}"] = False
                                st.rerun()

    # ── Explainability Side Panel ─────────────────────────────
    with col_side:
        with st.container(border=True):
            st.markdown("##### 💡 Recommendation Reasoning")
            st.caption("Click **Why? →** on any item to see why it was recommended.")

            st.markdown("---")
            st.markdown("**Your Profile**")
            user = st.session_state.user
            if user:
                for pref in user.get("prefs", []):
                    st.markdown(f"<span class='tm-badge tm-badge-blue'>{pref}</span>", unsafe_allow_html=True)

            st.markdown("")
            st.markdown("**Personalisation Engine**")
            st.progress(0.87, text="Profile confidence: 87%")
            st.caption("Based on your selections and visit check-ins")

            st.markdown("---")
            st.markdown("**Fairness Check**")
            st.success("✓ No filter bubble detected")
            st.success("✓ No demographic bias")
            st.warning("⚠ Cold-start: Limited history — 3 more trips needed for full personalisation")