"""
app.py — TravelMind Streamlit entry point
Run: streamlit run app.py
"""

import os
import requests
import streamlit as st
from data.store import USERS

st.set_page_config(
    page_title="TravelMind",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def _profile_from_username(username: str) -> dict:
    demo_profile = USERS.get(username)
    if demo_profile:
        return {
            "name": demo_profile.get("name", username.split("@")[0]),
            "avatar": demo_profile.get("avatar", "User"),
            "prefs": demo_profile.get("prefs", []),
        }
    return {
        "name": username.split("@")[0] if "@" in username else username,
        "avatar": "User",
        "prefs": [],
    }


def _login_with_backend(username: str, password: str) -> tuple[dict | None, str | None]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/auth/login",
            json={"username": username, "password": password},
            timeout=20,
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Login failed")
            return None, str(detail)
        return resp.json(), None
    except requests.exceptions.RequestException:
        return None, "Backend unavailable"


def _register_with_backend(username: str, password: str) -> tuple[dict | None, str | None]:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/auth/register",
            json={"username": username, "password": password},
            timeout=20,
        )
        if resp.status_code != 200:
            detail = resp.json().get("detail", "Register failed")
            return None, str(detail)
        return resp.json(), None
    except requests.exceptions.RequestException:
        return None, "Backend unavailable"


def _set_login_state(username: str, user_id: str | None) -> None:
    profile = _profile_from_username(username)
    st.session_state.logged_in = True
    st.session_state.user_id = user_id
    st.session_state.user = {
        **profile,
        "email": username,
        "username": username,
    }


def _fill_demo_credentials(username: str) -> None:
    st.session_state.auth_mode_widget = "Sign In"
    st.session_state.auth_mode = "Sign In"
    st.session_state.login_username_input = username
    demo_user = USERS.get(username, {})
    st.session_state.login_password_input = demo_user.get("password", "demo123")

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
    color: #e8edf5 !important;
  }
  .stSelectbox div[data-baseweb="select"] > div {
    background: #1a2235 !important;
    color: #e8edf5 !important;
  }
  .stSelectbox div[data-baseweb="select"] span,
  .stSelectbox div[data-baseweb="select"] input {
    color: #e8edf5 !important;
  }
  /* Dropdown list items */
  div[data-baseweb="popover"] li,
  div[data-baseweb="popover"] [role="option"] {
    background: #1a2235 !important;
    color: #e8edf5 !important;
  }
  div[data-baseweb="popover"] [role="option"]:hover {
    background: #233047 !important;
  }

  /* Number input */
  .stNumberInput input {
    background: #1a2235 !important;
    color: #e8edf5 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
  }
  div[data-testid="stNumberInput"] button {
    background: #2d3a52 !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
    min-width: 2.25rem !important;
  }
  div[data-testid="stNumberInput"] button:hover {
    background: #3b4d6e !important;
    border-color: rgba(59,158,255,0.45) !important;
    color: #ffffff !important;
  }
  div[data-testid="stNumberInput"] button svg,
  div[data-testid="stNumberInput"] button path {
    fill: #f8fafc !important;
    stroke: #f8fafc !important;
  }

  /* Date input */
  .stDateInput input {
    background: #1a2235 !important;
    color: #e8edf5 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
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
        "user_id": None,
        "user": None,
        "auth_mode": "Sign In",
        "auth_mode_widget": "Sign In",
        "auth_mode_pending": None,
        "auth_notice": None,
        "login_prefill": "",
        "login_password_prefill": "",
        "login_username_input": "",
        "login_password_input": "",
        "login_fill_pending_username": None,
        "login_fill_pending_password": None,
        "visited": {},          # item_key -> bool
        "selected_option": "A",
        "plan_generated": False,
        "plan_itineraries": {},
        "plan_option_meta": {},
        "plan_id": None,
        "plan_flight_outbound": [],
        "plan_flight_return": [],
        "plan_hotel_options": [],
        "plan_request_summary": {},
        "agent_status": {},
        "main_section_key": "plan",
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
        if st.session_state.login_fill_pending_username is not None:
            st.session_state.login_username_input = st.session_state.login_fill_pending_username
            st.session_state.login_fill_pending_username = None
        if st.session_state.login_fill_pending_password is not None:
            st.session_state.login_password_input = st.session_state.login_fill_pending_password
            st.session_state.login_fill_pending_password = None

        if st.session_state.auth_mode_pending:
            st.session_state.auth_mode_widget = st.session_state.auth_mode_pending
            st.session_state.auth_mode_pending = None

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; margin-bottom:8px;'>
          <span style='font-size:32px; font-weight:800; color:#e8edf5; letter-spacing:-1px;'>
            Travel<span style='color:#3b9eff;'>Mind</span>
          </span>
        </div>
        <div style='text-align:center; color:#7a90b0; font-size:14px; margin-bottom:32px;'>
          Sign in to access your personalised travel planner
        </div>
        """, unsafe_allow_html=True)

        auth_mode = st.radio(
            "Auth mode",
            ["Sign In", "Register"],
            key="auth_mode_widget",
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.auth_mode = auth_mode

        if auth_mode == "Sign In":
            if st.session_state.auth_notice:
                st.success(st.session_state.auth_notice)
                st.session_state.auth_notice = None

            st.text_input(
                "Username",
                placeholder="username",
                key="login_username_input",
            )
            st.text_input(
                "Password",
                type="password",
                placeholder="At least 6 characters",
                key="login_password_input",
            )
            submitted = st.button(
                "Sign In",
                use_container_width=True,
                type="primary",
                key="login_submit_btn",
            )

            if submitted:
                clean_username = st.session_state.login_username_input.strip()
                current_password = st.session_state.login_password_input
                data, err = _login_with_backend(clean_username, current_password)
                if data:
                    _set_login_state(clean_username, data.get("user_id"))
                    st.rerun()
                elif err == "Backend unavailable":
                    user = USERS.get(clean_username)
                    if user and user["password"] == current_password:
                        st.warning("Backend unavailable. Logged in with local demo mode.")
                        _set_login_state(clean_username, None)
                        st.rerun()
                    else:
                        st.error("Backend unavailable, and demo credential did not match.")
                else:
                    st.error(err or "Invalid credentials.")
        else:
            register_username = st.text_input(
                "Username ",
                value="",
                placeholder="new_username",
                key="register_username_input",
            )
            register_password = st.text_input(
                "Password ",
                type="password",
                value="",
                placeholder="At least 6 characters",
                key="register_password_input",
            )
            submitted = st.button(
                "Create Account",
                use_container_width=True,
                type="primary",
                key="register_submit_btn",
            )

            if submitted:
                clean_username = register_username.strip()
                data, err = _register_with_backend(clean_username, register_password)
                if data:
                    st.session_state.login_prefill = clean_username
                    st.session_state.login_fill_pending_username = clean_username
                    st.session_state.login_fill_pending_password = ""
                    st.session_state.auth_notice = "Register success. Please sign in."
                    st.session_state.auth_mode_pending = "Sign In"
                    st.rerun()
                elif err == "Backend unavailable":
                    st.error("Backend unavailable. Cannot register in local demo mode.")
                else:
                    st.error(err or "Register failed.")

        st.markdown("<div style='text-align:center; color:#4a5a72; font-size:12px; margin:16px 0;'>or sign in as a demo user</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.button(
                "Alice\nCulture lover",
                use_container_width=True,
                key="demo_user_alice",
                on_click=_fill_demo_credentials,
                args=("alice@example.com",),
            )
        with col2:
            st.button(
                "Bob\nFoodie",
                use_container_width=True,
                key="demo_user_bob",
                on_click=_fill_demo_credentials,
                args=("bob@example.com",),
            )
        with col3:
            st.button(
                "Carol\nAdventure",
                use_container_width=True,
                key="demo_user_carol",
                on_click=_fill_demo_credentials,
                args=("carol@example.com",),
            )


# -- Topbar ───────────────────────────────────────────────────
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
        ["plan", "my_trip", "replan", "security"],
        horizontal=True,
        label_visibility="collapsed",
        key="main_section_key",
        format_func=lambda key: {
            "plan": "🗺️ Plan",
            "my_trip": "📅 My Trip",
            "replan": "🔄 Re-plan",
            "security": "🛡️ Security",
        }[key],
    )

    if nav == "plan":
        render_plan()
    elif nav == "my_trip":
        render_trip()
    elif nav == "replan":
        render_replan()
    else:
        render_security()


if __name__ == "__main__":
    main()
