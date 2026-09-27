import streamlit as st
import pandas as pd

import database as db


def render():
    st.title("Reports")

    subjects = db.get_subjects()
    students = db.get_students()

    report_type = st.radio(
        "Report type", ["Daily Attendance", "Subject Attendance", "Student Attendance", "Monthly Attendance"],
        horizontal=True,
    )

    date_filter = subject_id = student_id = None

    if report_type == "Daily Attendance":
        d = st.date_input("Date")
        date_filter = d.isoformat() if d else None

    elif report_type == "Subject Attendance":
        if not subjects:
            st.info("No subjects yet.")
            return
        choice = st.selectbox("Subject", [f"{s['subject_code']} — {s['subject_name']}" for s in subjects])
        subject_id = next(s["id"] for s in subjects if f"{s['subject_code']} — {s['subject_name']}" == choice)

    elif report_type == "Student Attendance":
        if not students:
            st.info("No students yet.")
            return
        choice = st.selectbox("Student", [f"{s['roll_number']} — {s['name']}" for s in students])
        student_id = next(s["id"] for s in students if f"{s['roll_number']} — {s['name']}" == choice)
        pct = db.get_student_attendance_percentage(student_id)
        st.metric("Attendance %", f"{pct}%")

    else:  # Monthly Attendance
        month = st.text_input("Month (YYYY-MM)", value=pd.Timestamp.today().strftime("%Y-%m"))

    if report_type == "Monthly Attendance":
        records = [r for r in db.get_attendance_history() if r["date"].startswith(month)]
    else:
        records = db.get_attendance_history(date_filter=date_filter, subject_id=subject_id, student_id=student_id)

    if not records:
        st.info("No records match this report.")
        return

    df = pd.DataFrame(records)[
        ["date", "hour", "subject_code", "subject_name", "roll_number", "student_name", "status"]
    ]
    df.columns = ["Date", "Hour", "Code", "Subject", "Roll No.", "Student", "Status"]
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Export CSV", data=csv_bytes,
        file_name=f"attendvision_{report_type.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
