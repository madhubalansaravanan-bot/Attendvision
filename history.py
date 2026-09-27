import streamlit as st
import pandas as pd

import database as db


def render():
    st.title("Attendance History")

    subjects = db.get_subjects()
    students = db.get_students()

    c1, c2, c3, c4 = st.columns(4)
    date_filter = c1.date_input("Date", value=None)
    subject_choice = c2.selectbox(
        "Subject", ["All"] + [f"{s['subject_code']} — {s['subject_name']}" for s in subjects]
    )
    hour_filter = c3.selectbox("Hour", ["All"] + list(range(1, 13)))
    section_filter = c4.text_input("Section")

    student_choice = st.selectbox(
        "Student", ["All"] + [f"{s['roll_number']} — {s['name']}" for s in students]
    )

    subject_id = None
    if subject_choice != "All":
        subject_id = next(s["id"] for s in subjects if f"{s['subject_code']} — {s['subject_name']}" == subject_choice)

    student_id = None
    if student_choice != "All":
        student_id = next(s["id"] for s in students if f"{s['roll_number']} — {s['name']}" == student_choice)

    records = db.get_attendance_history(
        date_filter=date_filter.isoformat() if date_filter else None,
        subject_id=subject_id,
        hour=None if hour_filter == "All" else hour_filter,
        student_id=student_id,
        section=section_filter or None,
    )

    if not records:
        st.info("No attendance records match these filters.")
        return

    df = pd.DataFrame(records)[
        ["date", "hour", "subject_code", "subject_name", "roll_number", "student_name",
         "status", "confidence", "timestamp"]
    ]
    df.columns = ["Date", "Hour", "Code", "Subject", "Roll No.", "Student", "Status",
                  "Match Distance", "Recorded At"]
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"{len(df)} record(s).")
