"""
AttendVision - main entry point.

Run with:  streamlit run app.py

This file only handles: page config, DB init, the login gate, and routing
to the selected view. Each view lives in views/ and owns its own UI logic.
"""
import streamlit as st

# ================= ATTENDVISION UI =================

st.set_page_config(
    page_title="AttendVision",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* ---------- MAIN APP ---------- */

.stApp {
    background: linear-gradient(135deg, #f7faff 0%, #eef4ff 100%);
    color: #172033;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* ---------- HEADINGS ---------- */

h1 {
    color: #123b78 !important;
    font-weight: 800 !important;
}

h2 {
    color: #174a87 !important;
    font-weight: 750 !important;
}

h3 {
    color: #1d4f91 !important;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #082852 0%,
        #0d3b78 55%,
        #1254a0 100%
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] button {
    border-radius: 10px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.15);
}

/* ---------- BUTTONS ---------- */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: none;
    background: linear-gradient(
        90deg,
        #1769d1,
        #2989f5
    );
    color: white;
    font-weight: 700;
    padding: 0.65rem 1rem;
    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(23,105,209,0.25);
}

/* ---------- INPUTS ---------- */

.stTextInput input,
.stNumberInput input {
    border-radius: 10px !important;
    border: 1px solid #d5dfef !important;
    background: white !important;
}

div[data-baseweb="select"] {
    border-radius: 10px !important;
}

/* ---------- METRIC CARDS ---------- */

[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e1e8f2;
    box-shadow: 0 4px 16px rgba(30,60,100,0.08);
}

[data-testid="stMetricLabel"] {
    color: #667085 !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #123b78 !important;
    font-weight: 800;
}

/* ---------- DATA TABLE ---------- */

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #e1e8f2;
}

/* ---------- FORMS ---------- */

div[data-testid="stForm"] {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e0e7f1;
    box-shadow: 0 8px 30px rgba(30,60,100,0.10);
}

/* ---------- ALERTS ---------- */

div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---------- CARDS ---------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
    border-color: #dce5f2;
    background: white;
}

/* ---------- DIVIDERS ---------- */

hr {
    border-color: #dce5f2;
}

/* ---------- IMAGES ---------- */

img {
    border-radius: 14px;
}

/* ---------- HIDE STREAMLIT DEFAULT UI ---------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)
import config
import database as db
import auth
from views import dashboard, students, subjects, attendance, history, reports, settings

st.set_page_config(page_title="AttendVision", page_icon="🎓", layout="wide")

db.init_db()
db.seed_demo_data()

NAV_ITEMS = {
    "Dashboard": dashboard,
    "Take Attendance": attendance,
    "Students": students,
    "Subjects": subjects,
    "Attendance History": history,
    "Reports": reports,
    "Settings": settings,
}


def render_login():
    st.title("🎓 AttendVision")
    st.caption("Smart, hour-wise attendance using computer vision.")

    with st.form("login_form"):
        identifier = st.text_input("Username (admin) or Email (faculty)")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")

    if submitted:
        user = auth.login(identifier.strip(), password)
        if user:
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("Invalid credentials.")

    with st.expander("Demo credentials"):
        st.code(
            f"Admin    → username: {config.ADMIN_USERNAME}  password: {config.ADMIN_PASSWORD}\n"
            "Faculty  → email: faculty@example.edu  password: faculty123",
            language="text",
        )


def render_student_dashboard(user):
    st.title("🎓 Student Dashboard")
    st.caption("Your personal attendance overview")

    st.success(f"Welcome, {user['name']} 👋")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Roll Number", user["roll_number"])
    col2.metric("Department", user["department"])
    col3.metric("Year", user["year"])
    col4.metric("Section", user["section"])

    st.divider()

    # Attendance percentage
    attendance_percentage = db.get_student_attendance_percentage(user["id"])

    st.subheader("📊 Attendance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Overall Attendance",
            f"{attendance_percentage}%"
        )

    with col2:
        if attendance_percentage >= 75:
            st.success("✅ Attendance Status: Good")
        else:
            st.warning("⚠️ Attendance below 75%")

    st.divider()

    st.subheader("📋 Student Information")

    st.write(f"**Name:** {user['name']}")
    st.write(f"**Roll Number:** {user['roll_number']}")
    st.write(f"**Department:** {user['department']}")
    st.write(f"**Year:** {user['year']}")
    st.write(f"**Section:** {user['section']}")

    st.divider()

    st.info(
        "Your attendance is automatically updated when you are "
        "recognized by the classroom camera."
    )


def render_app():
    user = st.session_state["user"]

    # ================= STUDENT =================
    if user.get("role") == "student":
        with st.sidebar:
            st.markdown("## 🎓 AttendVision")
            st.caption(f"Student: {user['name']}")

            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()

        render_student_dashboard(user)
        return

    # ================= ADMIN / FACULTY =================
    with st.sidebar:
        st.markdown("## 🎓 AttendVision")
        st.caption(
            f"{user['name']} ({user['role'].title()})"
        )

        choice = st.radio(
            "Navigate",
            list(NAV_ITEMS.keys()),
            label_visibility="collapsed"
        )

        st.divider()

        if st.button("Logout"):
            camera = st.session_state.get("attendance_camera")

            if camera is not None:
                camera.release()

            st.session_state.clear()
            st.rerun()

    NAV_ITEMS[choice].render()


def main():
    if "user" not in st.session_state:
        render_login()
    else:
        render_app()


if __name__ == "__main__":
    main()
