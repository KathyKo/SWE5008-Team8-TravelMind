"""
pages/security.py — Security Demo page
"""

import streamlit as st
from datetime import datetime
import requests
from data.store import ATTACK_PATTERNS, PRESETS, PIPELINE_STAGES

# ── API Configuration ────────────────────────────────────────
# BACKEND_URL = "http://localhost:8000"  # Change to http://backend:8000 in Docker
BACKEND_URL = "http://backend:8000"

def call_security_check(text: str) -> dict:
    """
    Call the backend security check endpoint.
    Returns the security check result from input_guard_agent.
    """
    try:
        url = f"{BACKEND_URL}/travel/security/check"
        payload = {"text": text, "user_id": st.session_state.get("user_id", "test_user")}
        
        # Log the request
        print(f"[Frontend] 🔍 Sending security check request to {url}")
        print(f"[Frontend] Payload: {payload}")
        
        response = requests.post(
            url,
            json=payload,
            timeout=10,
        )
        
        print(f"[Frontend] ✅ Response status: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        print(f"[Frontend] Response: {result}")
        return result
    except requests.exceptions.ConnectionError as e:
        print(f"[Frontend] ❌ Connection Error: {str(e)}")
        print(f"[Frontend] Backend URL: {BACKEND_URL}")
        # Fallback to local classification if backend is unavailable
        st.warning("⚠️  Backend unavailable — falling back to local simulation")
        return classify_input_local(text)
    except Exception as e:
        print(f"[Frontend] ❌ Error: {str(e)}")
        st.error(f"Error calling security check: {str(e)}")
        return {"threat_blocked": False, "threat_type": "Unknown"}


def classify_input_local(text: str):
    """Local fallback classification for testing without backend."""
    lower = text.lower()
    for pattern in ATTACK_PATTERNS:
        for kw in pattern["keywords"]:
            if kw.lower() in lower:
                return {
                    "threat_blocked": True,
                    "threat_type": pattern["type"],
                    "threat_detail": pattern["reason"],
                    "sanitised_input": text,
                    "security_audit_log": [],
                }
    return {
        "threat_blocked": False,
        "threat_type": "Normal Query",
        "sanitised_input": text,
        "security_audit_log": [],
    }



def render():
    st.markdown("### Security Demo")
    st.markdown("<span style='color:#7a90b0;font-size:14px'>Test TravelMind's Risk & Safety Agent — watch it intercept attacks in real time.</span>", unsafe_allow_html=True)
    st.markdown("")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # ── Attack input ──────────────────────────────────────
        with st.container(border=True):
            st.markdown("<span style='color:#ef4444;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1.5px'>⚡ Attack Input</span>", unsafe_allow_html=True)
            attack_text = st.text_area(
                "attack_input",
                placeholder="Type an attack or pick a preset below...",
                height=90,
                label_visibility="collapsed",
                key="attack_input_area",
            )
            col_status, col_btn = st.columns([3, 1])
            with col_status:
                st.markdown("<span style='font-size:11px;color:#4a5a72;font-family:monospace'>risk_agent v2.1 · active</span>", unsafe_allow_html=True)
            with col_btn:
                send = st.button("⚡ Send", type="primary", use_container_width=True)

        # ── Presets ───────────────────────────────────────────
        st.markdown("<span style='font-size:11px;font-weight:600;color:#4a5a72;text-transform:uppercase;letter-spacing:1.5px'>Quick Presets</span>", unsafe_allow_html=True)
        preset_cols = st.columns(len(PRESETS))
        chosen_preset = None
        for i, (label, payload) in enumerate(PRESETS):
            with preset_cols[i]:
                if st.button(label, key=f"preset_{i}", use_container_width=True):
                    chosen_preset = payload

        if chosen_preset:
            st.session_state["attack_prefill"] = chosen_preset
            st.info(f"Preset loaded: `{chosen_preset[:60]}...`" if len(chosen_preset) > 60 else f"Preset loaded: `{chosen_preset}`")
            # Process immediately via backend
            with st.spinner("🔍 Checking security..."):
                result = call_security_check(chosen_preset)
            
            is_blocked = result.get("threat_blocked", False)
            threat_type = result.get("threat_type", "Unknown")
            threat_detail = result.get("threat_detail", "")
            now = datetime.now().strftime("%H:%M:%S")
            entry = {
                "blocked": is_blocked,
                "type": threat_type,
                "stage": None,  # Will be mapped from threat_type if needed
                "input": chosen_preset[:80] + ("..." if len(chosen_preset) > 80 else ""),
                "reason": threat_detail if is_blocked else "Input clean. Routing to Intent Agent.",
                "time": now,
            }
            st.session_state.security_log.insert(0, entry)
            if is_blocked:
                st.session_state.blocked_count += 1
            else:
                st.session_state.passed_count += 1
            st.rerun()

        # Process manual send
        if send and attack_text.strip():
            with st.spinner("🔍 Checking security..."):
                result = call_security_check(attack_text)
            
            is_blocked = result.get("threat_blocked", False)
            threat_type = result.get("threat_type", "Unknown")
            threat_detail = result.get("threat_detail", "")
            now = datetime.now().strftime("%H:%M:%S")
            entry = {
                "blocked": is_blocked,
                "type": threat_type,
                "stage": None,  # Will be mapped from threat_type if needed
                "input": attack_text[:80] + ("..." if len(attack_text) > 80 else ""),
                "reason": threat_detail if is_blocked else "Input clean. Routing to Intent Agent.",
                "time": now,
            }
            st.session_state.security_log.insert(0, entry)
            if is_blocked:
                st.session_state.blocked_count += 1
            else:
                st.session_state.passed_count += 1
            st.rerun()

        # ── Log ───────────────────────────────────────────────
        st.markdown("---")
        col_log_title, col_log_count = st.columns([3, 1])
        with col_log_title:
            st.markdown("**Interception Log**")
        with col_log_count:
            st.markdown(f"<span style='color:#ef4444;font-family:monospace;font-size:12px'>{st.session_state.blocked_count} blocked</span>", unsafe_allow_html=True)

        if not st.session_state.security_log:
            st.markdown("<div style='text-align:center;padding:32px 0;color:#4a5a72;font-size:13px'>No activity yet.</div>", unsafe_allow_html=True)
        else:
            for entry in st.session_state.security_log[:10]:
                if entry["blocked"]:
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([1.5, 4, 1.2])
                        with c1:
                            st.markdown("🔴 **BLOCKED**")
                        with c2:
                            st.markdown(f"**{entry['type']}**")
                        with c3:
                            st.markdown(f"<span style='font-size:11px;color:#4a5a72;font-family:monospace'>{entry['time']}</span>", unsafe_allow_html=True)
                        st.caption(f'"{entry["input"]}"')
                        st.markdown(f"<span style='font-size:11px;font-family:monospace;color:#ef4444'>→ {entry['reason']}</span>", unsafe_allow_html=True)
                else:
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([1.5, 4, 1.2])
                        with c1:
                            st.markdown("🟢 **PASSED**")
                        with c2:
                            st.markdown(f"**{entry['type']}**")
                        with c3:
                            st.markdown(f"<span style='font-size:11px;color:#4a5a72;font-family:monospace'>{entry['time']}</span>", unsafe_allow_html=True)
                        st.caption(f'"{entry["input"]}"')
                        st.markdown(f"<span style='font-size:11px;font-family:monospace;color:#10b981'>→ {entry['reason']}</span>", unsafe_allow_html=True)

    with col_right:
        # ── Pipeline ──────────────────────────────────────────
        with st.container(border=True):
            st.markdown("**🔐 Security Pipeline**")
            st.caption("All inputs pass through 5 stages before reaching any agent.")
            active_stage = st.session_state.active_pipe_stage

            for i, (name, desc) in enumerate(PIPELINE_STAGES):
                stage_num = i + 1
                is_active = active_stage == stage_num
                color = "#ef4444" if is_active else "#4a5a72"
                bg = "rgba(239,68,68,0.08)" if is_active else "transparent"
                st.markdown(
                    f"""<div style='display:flex;gap:10px;padding:8px;border-radius:8px;background:{bg};margin-bottom:4px;'>
                      <span style='width:24px;height:24px;border-radius:6px;background:{"rgba(239,68,68,0.2)" if is_active else "#1a2235"};
                        display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;
                        color:{color};font-family:monospace;flex-shrink:0;text-align:center;line-height:24px'>{stage_num}</span>
                      <div>
                        <div style='font-size:12px;font-weight:600;color:{"#ef4444" if is_active else "#e8edf5"}'>{name}</div>
                        <div style='font-size:11px;color:#7a90b0'>{desc}</div>
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ── Stats ─────────────────────────────────────────────
        st.markdown("")
        with st.container(border=True):
            st.markdown("**Attack Statistics**")
            s1, s2 = st.columns(2)
            with s1:
                st.metric("🔴 Blocked", st.session_state.blocked_count)
            with s2:
                st.metric("🟢 Passed", st.session_state.passed_count)

            total = st.session_state.blocked_count + st.session_state.passed_count
            if total > 0:
                block_rate = st.session_state.blocked_count / total
                st.progress(block_rate, text=f"Block rate: {block_rate:.0%}")
