import streamlit as st

import database as db


def render():
    st.title("Settings")
    user = st.session_state["user"]

    if user["role"] == "admin":
        st.subheader("Add Faculty Account")
        with st.form("add_faculty_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            name = c1.text_input("Name *")
            email = c2.text_input("Email *")
            c3, c4 = st.columns(2)
            department = c3.text_input("Department")
            password = c4.text_input("Temporary Password *", type="password")
            submitted = st.form_submit_button("Add Faculty", type="primary")

            if submitted:
                if not (name and email and password):
                    st.error("Name, Email and Password are required.")
                else:
                    try:
                        db.add_faculty(name.strip(), email.strip(), password, department)
                        st.success(f"Added faculty account for {name}.")
                    except Exception as e:
                        st.error(f"Could not add faculty — email may already be in use. ({e})")
        st.divider()

    st.subheader("Privacy & Consent")
    st.markdown(
        "- Face data is used **only** for automated attendance in this system.\n"
        "- Students must give consent, and the college must authorize collection, "
        "before any face is enrolled.\n"
        "- Enrolled face images are stored locally on this machine under `data/students/`, "
        "not sent to any external service.\n"
        "- Recognition results are a best-effort match, not a certainty — low-confidence "
        "matches are shown as **Unknown** and are never auto-marked present.\n"
        "- Retain only the student data this attendance system needs, and follow your "
        "institution's data retention and deletion policy."
    )
