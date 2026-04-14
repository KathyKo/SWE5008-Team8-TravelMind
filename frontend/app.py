"""
app.py — TravelMind Streamlit entry point
Run: streamlit run app.py
"""

import streamlit as st
from data.store import USERS

st.set_page_config(
    page_title="TravelMind",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

  /* Accent colours */
  :root {
    --accent: #3b9eff;
    --accent2: #00d4aa;
    --success: #10b981;
    --danger: #ef4444;
    --warn: #f59e0b;
    --purple: #8b5cf6;
    --surface: #111827;
    --surface2: #1a2235;
    --border: rgba(255,255,255,0.07);
    --text-muted: #7a90b0;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 12px 16px;
  }

  /* Buttons */
  .stButton > button {
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.2s;
  }
  .stButton > button:hover { transform: translateY(-1px); }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #7a90b0;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(59,158,255,0.15);
    color: #3b9eff;
  }

  /* Progress bar */
  .stProgress > div > div > div { background: linear-gradient(90deg, #00d4aa, #3b9eff); }

  /* Expander */
  .streamlit-expanderHeader {
    background: #1a2235;
    border-radius: 10px;
  }

  /* Text input */
  .stTextInput > div > div > input,
  .stTextArea textarea {
    background: #1a2235 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #e8edf5 !important;
  }

  /* Selectbox */
  .stSelectbox > div > div {
    background: #1a2235;
    border-radius: 10px;
  }

  /* Info / success / warning boxes */
  .stAlert { border-radius: 10px; }

  /* Custom card */
  .tm-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
  }
  .tm-card:hover { border-color: rgba(255,255,255,0.12); }

  .tm-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
  }
  .tm-badge-blue { background: rgba(59,158,255,0.15); color: #3b9eff; }
  .tm-badge-green { background: rgba(16,185,129,0.15); color: #10b981; }
  .tm-badge-red { background: rgba(239,68,68,0.15); color: #ef4444; }
  .tm-badge-purple { background: rgba(139,92,246,0.15); color: #8b5cf6; }

  .tm-topbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 10px 0 16px 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 20px;
  }
  .tm-logo {
    font-size: 22px;
    font-weight: 800;
    color: #e8edf5;
    letter-spacing: -0.5px;
  }
  .tm-logo span { color: #3b9eff; }
  .tm-user-chip {
    background: #1a2235;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 13px;
    color: #e8edf5;
  }
  .tm-demo-badge {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: #ef4444;
    font-family: monospace;
  }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ───────────────────────────────────────
def init_state():
    defaults = {
        "logged_in": False,
        "user": None,
        "visited": {},          # item_key -> bool
        "selected_option": "A",
        "plan_generated": False,
      "main_section": "🛡️ Security",
        "security_log": [],
        "blocked_count": 0,
        "passed_count": 0,
        "replan_situation": None,
        "replan_time": None,
        "replan_done": False,
        "chosen_alt": None,
        "active_pipe_stage": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Login Screen ─────────────────────────────────────────────
def login_screen():
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; margin-bottom:8px;'>
          <span style='font-size:32px; font-weight:800; color:#e8edf5; letter-spacing:-1px;'>
            🗺️ Travel<span style='color:#3b9eff;'>Mind</span>
          </span>
        </div>
        <div style='text-align:center; color:#7a90b0; font-size:14px; margin-bottom:32px;'>
          Sign in to access your personalised travel planner
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            email = st.text_input("Email", value="alice@example.com", placeholder="you@example.com")
            password = st.text_input("Password", type="password", value="demo123", placeholder="••••••••")
            submitted = st.form_submit_button("Sign In →", use_container_width=True, type="primary")

            if submitted:
                user = USERS.get(email)
                if user and user["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.user = {**user, "email": email}
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try a demo account below.")

        st.markdown("<div style='text-align:center; color:#4a5a72; font-size:12px; margin:16px 0;'>— or sign in as a demo user —</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👩 Alice\nCulture lover", use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.user = {**USERS["alice@example.com"], "email": "alice@example.com"}
                st.rerun()
        with col2:
            if st.button("👨 Bob\nFoodie", use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.user = {**USERS["bob@example.com"], "email": "bob@example.com"}
                st.rerun()
        with col3:
            if st.button("👩 Carol\nAdventure", use_container_width=True):
                st.session_state.logged_in = True
                st.session_state.user = {**USERS["carol@example.com"], "email": "carol@example.com"}
                st.rerun()


# ── Topbar ───────────────────────────────────────────────────
def topbar():
    user = st.session_state.user
    col1, col2, col3, col4 = st.columns([2, 4, 2, 1])
    with col1:
        st.markdown(f"""
        <div class='tm-topbar'>
          <span class='tm-logo'>🗺️ Travel<span>Mind</span></span>
          <span class='tm-user-chip'>{user['avatar']} {user['name']}</span>
          <span class='tm-demo-badge'>● DEMO</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        if st.button("Sign out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_state()
            st.rerun()


# ── Main App ─────────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        login_screen()
        return

    topbar()

    # Import pages here to avoid circular imports
    from pages.plan import render as render_plan
    from pages.my_trip import render as render_trip
    from pages.replan import render as render_replan
    from pages.security import render as render_security

    nav = st.radio(
        "Main navigation",
        ["🗺️ Plan", "📅 My Trip", "🔄 Re-plan", "🛡️ Security"],
        horizontal=True,
        label_visibility="collapsed",
        key="main_section",
    )

    if nav == "🗺️ Plan":
        render_plan()
    elif nav == "📅 My Trip":
        render_trip()
    elif nav == "🔄 Re-plan":
        render_replan()
    else:
        render_security()


if __name__ == "__main__":
    main()
