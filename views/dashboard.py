import streamlit as st
import pandas as pd
from datetime import date

import database as db


def render():
    st.title("Dashboard")

    stats = db.get_dashboard_stats(date.today().isoformat())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Today's Classes", stats["todays_classes"])
    c2.metric("Total Students", stats["total_students"])
    c3.metric("Present Today", stats["present_today"])
    c4.metric("Absent Today", stats["absent_today"])
    c5.metric("Attendance %", f"{stats['attendance_percentage']}%")

    st.divider()
    st.subheader("Recent Attendance")

    history = db.get_attendance_history()[:15]
    if not history:
        st.info("No attendance has been recorded yet. Start a session from **Take Attendance**.")
        return

    df = pd.DataFrame(history)[
        ["date", "hour", "subject_name", "roll_number", "student_name", "status", "confidence"]
    ]
    df.columns = ["Date", "Hour", "Subject", "Roll No.", "Student", "Status", "Confidence"]
    st.dataframe(df, use_container_width=True, hide_index=True)
