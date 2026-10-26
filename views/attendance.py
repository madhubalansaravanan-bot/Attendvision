
"""AttendVision attendance workflow with face-recognition diagnostics."""

from datetime import date

import cv2
import numpy as np
import streamlit as st

import database as db
from vision.detector import FaceDetector
from vision.enrollment import count_samples, retrain_all


def render():
    st.title("📷 Take Attendance")

    if not st.session_state.get("attendance_running"):
        render_setup()
    else:
        render_live()


def render_setup():
    subjects = db.get_subjects()

    if not subjects:
        st.warning("No subjects exist yet. Add one under Subjects first.")
        return

    st.subheader("Start Attendance Session")

    with st.form("start_attendance_form"):
        col1, col2 = st.columns(2)

        department = col1.text_input("Department", value="Aerospace")
        section = col2.text_input("Class / Section", value="A")

        subject_options = {
            f"{s['subject_code']} — {s['subject_name']}": s["id"]
            for s in subjects
        }

        subject_label = st.selectbox(
            "Subject", list(subject_options.keys())
        )

        col3, col4 = st.columns(2)
        session_date = col3.date_input("Date", value=date.today())

        hour = col4.number_input(
            "Hour", min_value=1, max_value=12, value=1, step=1
        )

        start = st.form_submit_button(
            "START ATTENDANCE", type="primary"
        )

    if not start:
        return

    students = db.get_students(
        department=department.strip(),
        section=section.strip(),
    )

    if not students:
        st.error(
            f"No students found for {department} - Section {section}. "
            "Check the department and section values in your database."
        )
        return

    user = st.session_state.get("user")

    if not user:
        st.error("Your login session expired. Please log in again.")
        return

    # --------------------------------------------------------
    # DIAGNOSTICS: Check persistent samples and train the model.
    # --------------------------------------------------------

    st.subheader("Face Recognition Diagnostics")

    storage_counts = {}

    for student in students:
        roll = str(student["roll_number"])

        try:
            stored_count = count_samples(roll)
            storage_counts[roll] = stored_count

            st.write(
                f"**{roll} — {student['name']}**: "
                f"{stored_count} stored sample(s)"
            )

        except Exception as exc:
            storage_counts[roll] = None
            st.error(
                f"Storage check failed for {roll}: "
                f"{type(exc).__name__}: {exc}"
            )

    try:
        with st.spinner(
            "Downloading stored face images and training recognizer..."
        ):
            recognizer, samples = retrain_all(students)

        st.write(
            "Students with readable samples:",
            len(samples),
        )
        st.write(
            "Recognizer trained:",
            recognizer.is_trained,
        )

        if not recognizer.is_trained:
            st.error(
                "The recognizer was not trained. "
                "Check the sample counts and storage errors above."
            )
            return

    except Exception as exc:
        st.error(
            f"Face model loading failed: {type(exc).__name__}: {exc}"
        )
        st.info(
            "Check that SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "are configured in Streamlit Secrets, and that the "
            "'face-samples' bucket contains the expected image files."
        )
        return

    # --------------------------------------------------------
    # CREATE OR REUSE ATTENDANCE SESSION.
    # --------------------------------------------------------

    subject_id = subject_options[subject_label]
    session_date_str = session_date.isoformat()

    try:
        existing = db.find_open_session(
            subject_id,
            session_date_str,
            int(hour),
            section.strip(),
        )

        if existing:
            session_id = existing["id"]
        else:
            session_id = db.create_session(
                subject_id,
                user["id"],
                session_date_str,
                int(hour),
                section.strip(),
            )

    except Exception as exc:
        st.error(f"Could not create attendance session: {exc}")
        return

    # Store the trained recognizer and session information.
    st.session_state["attendance_recognizer"] = recognizer
    st.session_state["attendance_running"] = True
    st.session_state["attendance_session_id"] = session_id
    st.session_state["attendance_students"] = {
        student["id"]: student for student in students
    }
    st.session_state["attendance_subject_label"] = subject_label
    st.session_state["attendance_hour"] = int(hour)
    st.session_state["attendance_date"] = session_date_str

    st.success("Face model loaded and attendance session is ready.")
    st.rerun()


def render_live():
    session_id = st.session_state["attendance_session_id"]
    subject_label = st.session_state["attendance_subject_label"]
    hour = st.session_state["attendance_hour"]
    session_date = st.session_state["attendance_date"]
    students = st.session_state["attendance_students"]

    st.subheader(f"{subject_label} — Hour {hour} — {session_date}")

    st.info(
        "Take a clear classroom photo. The system will detect "
        "faces and attempt to match them with enrolled students."
    )

    recognizer = st.session_state.get("attendance_recognizer")

    if recognizer is None or not recognizer.is_trained:
        st.error(
            "The face model is unavailable. End this session "
            "and start attendance again to reload stored samples."
        )
        return

    picture = st.camera_input(
        "Take Attendance Photo",
        key=f"attendance_camera_{session_id}",
    )

    if picture is not None:
        process_attendance_photo(picture, session_id, students)

    st.divider()

    try:
        records = db.get_session_records(session_id)
    except Exception as exc:
        st.error(f"Could not load attendance records: {exc}")
        records = []

    present_ids = {
        record["student_id"] for record in records
    }

    col1, col2 = st.columns(2)
    col1.metric("Present", len(present_ids))
    col2.metric(
        "Remaining",
        max(len(students) - len(present_ids), 0),
    )

    st.subheader("📋 Live Attendance")

    if records:
        for record in records[::-1]:
            st.success(
                f"✅ {record['student_name']} "
                f"({record['roll_number']})"
            )
    else:
        st.info("No students recognized yet.")

    st.divider()

    if st.button("⏹ End Attendance Session", type="primary"):
        try:
            db.end_session(session_id)
            absent_students = db.get_absent_students(session_id)
        except Exception as exc:
            st.error(f"Could not close attendance session: {exc}")
            return

        clear_attendance_state()

        st.session_state["last_absent_students"] = absent_students
        st.session_state["last_absent_session_id"] = session_id

        st.success("Attendance session ended.")
        st.rerun()


def process_attendance_photo(picture, session_id, students):
    image_bytes = picture.getvalue()

    frame = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        st.error("Could not read the captured image.")
        return

    try:
        detector = FaceDetector()
        recognizer = st.session_state.get("attendance_recognizer")

        if recognizer is None or not recognizer.is_trained:
            st.error(
                "Face model unavailable. Please end the session "
                "and start it again."
            )
            return

        boxes = detector.detect(frame)

    except Exception as exc:
        st.error(
            f"Face detection or recognition error: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    if len(boxes) == 0:
        st.warning(
            "No faces detected. Try better lighting and ensure "
            "faces are clearly visible."
        )
        st.image(frame, channels="BGR", caption="Captured image")
        return

    recognized_names = []
    unknown_count = 0

    for box in boxes:
        x, y, w, h = box

        face_gray = detector.crop_face(frame, box)

        try:
            student_id, distance = recognizer.predict(face_gray)
        except Exception as exc:
            st.error(f"Face prediction failed: {exc}")
            continue

        # Never mark a face unless its predicted ID belongs to
        # the students selected for this attendance session.
        if student_id is not None and student_id in students:
            student = students[student_id]

            try:
                db.mark_attendance(
                    session_id,
                    student_id,
                    float(distance),
                )
            except Exception as exc:
                st.error(
                    f"Could not save attendance for "
                    f"{student['name']}: {exc}"
                )
                continue

            recognized_names.append(student["name"])

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 200, 0),
                3,
            )

            cv2.putText(
                frame,
                f"{student['name']} ({student['roll_number']})",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
            )

        else:
            unknown_count += 1

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                "Unknown",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    st.image(
        frame,
        channels="BGR",
        caption="Attendance Result",
    )

    for name in dict.fromkeys(recognized_names):
        st.success(f"✅ Attendance marked: {name}")

    if unknown_count:
        st.warning(
            f"{unknown_count} unknown face(s) detected."
        )


def clear_attendance_state():
    keys = [
        "attendance_running",
        "attendance_session_id",
        "attendance_students",
        "attendance_subject_label",
        "attendance_hour",
        "attendance_date",
        "attendance_recognizer",
    ]

    for key in keys:
        st.session_state.pop(key, None)
        
