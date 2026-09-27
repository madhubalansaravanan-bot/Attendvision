from datetime import date

import cv2
import numpy as np
import streamlit as st

import config
import database as db
from vision.detector import FaceDetector
from vision.recognizer import FaceRecognizer


def render():
    st.title("📷 Take Attendance")

    if not st.session_state.get("attendance_running"):
        render_setup()
    else:
        render_live()


def render_setup():

    subjects = db.get_subjects()

    if not subjects:
        st.warning(
            "No subjects exist yet. Add one under Subjects first."
        )
        return

    st.subheader("Start Attendance Session")

    with st.form("start_attendance_form"):

        col1, col2 = st.columns(2)

        department = col1.text_input(
            "Department",
            value="Aerospace"
        )

        section = col2.text_input(
            "Class / Section",
            value="A"
        )

        subject_options = {
            f"{s['subject_code']} — {s['subject_name']}": s["id"]
            for s in subjects
        }

        subject_label = st.selectbox(
            "Subject",
            list(subject_options.keys())
        )

        col3, col4 = st.columns(2)

        session_date = col3.date_input(
            "Date",
            value=date.today()
        )

        hour = col4.number_input(
            "Hour",
            min_value=1,
            max_value=12,
            value=1,
            step=1
        )

        start = st.form_submit_button(
            "START ATTENDANCE",
            type="primary"
        )

    if not start:
        return

    # Get students
    students = db.get_students(
        department=department,
        section=section
    )

    if not students:
        st.error(
            f"No students found for {department} - Section {section}."
        )
        return

    # Load face recognizer
    recognizer = FaceRecognizer()

    if not recognizer.is_trained:
        st.error(
            "❌ No enrolled faces found. "
            "Enroll at least one student's face first."
        )
        return

    user = st.session_state.get("user")

    if not user:
        st.error("User session expired. Please log in again.")
        return

    subject_id = subject_options[subject_label]

    session_date_str = session_date.isoformat()

    # Check for existing open session
    existing = db.find_open_session(
        subject_id,
        session_date_str,
        int(hour),
        section
    )

    try:

        if existing:

            session_id = existing["id"]

        else:

            session_id = db.create_session(
                subject_id,
                user["id"],
                session_date_str,
                int(hour),
                section
            )

    except Exception as e:

        st.error(
            f"Could not create attendance session: {e}"
        )
        return

    # Store everything needed by the live page
    st.session_state["attendance_running"] = True

    st.session_state["attendance_session_id"] = session_id

    st.session_state["attendance_students"] = {
        s["id"]: s
        for s in students
    }

    st.session_state["attendance_subject_label"] = (
        subject_label
    )

    st.session_state["attendance_hour"] = int(hour)

    st.session_state["attendance_date"] = (
        session_date_str
    )

    st.rerun()


def render_live():

    session_id = st.session_state[
        "attendance_session_id"
    ]

    subject_label = st.session_state[
        "attendance_subject_label"
    ]

    hour = st.session_state[
        "attendance_hour"
    ]

    session_date = st.session_state[
        "attendance_date"
    ]

    students = st.session_state[
        "attendance_students"
    ]

    st.subheader(
        f"{subject_label} — Hour {hour} — {session_date}"
    )

    st.info(
        "📷 Take a clear classroom photo. "
        "The system will detect and recognize enrolled students."
    )

    # Browser camera
    picture = st.camera_input(
        "Take Attendance Photo",
        key=f"attendance_camera_{session_id}"
    )

    if picture is not None:

        process_attendance_photo(
            picture,
            session_id,
            students
        )

    st.divider()

    # Attendance summary
    records = db.get_session_records(
        session_id
    )

    present_ids = {
        record["student_id"]
        for record in records
    }

    col1, col2 = st.columns(2)

    col1.metric(
        "Present",
        len(present_ids)
    )

    col2.metric(
        "Remaining",
        max(
            len(students) - len(present_ids),
            0
        )
    )

    st.subheader("📋 Live Attendance")

    if records:

        for record in records[::-1]:

            st.success(
                f"✅ {record['student_name']} "
                f"({record['roll_number']})"
            )

    else:

        st.info(
            "No students recognized yet."
        )

    st.divider()

    if st.button(
        "⏹ End Attendance Session",
        type="primary"
    ):

        db.end_session(
            session_id
        )

        clear_attendance_state()

        st.success(
            "Attendance session ended."
        )

        st.rerun()


def process_attendance_photo(
    picture,
    session_id,
    students
):

    image_bytes = picture.getvalue()

    frame = cv2.imdecode(
        np.frombuffer(
            image_bytes,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if frame is None:

        st.error(
            "❌ Could not read the captured image."
        )

        return

    # Face detector
    try:

        detector = FaceDetector()

    except Exception as e:

        st.error(
            f"❌ Face detector error: {e}"
        )

        return

    # Face recognizer
    try:

        recognizer = FaceRecognizer()

    except Exception as e:

        st.error(
            f"❌ Face recognizer error: {e}"
        )

        return

    if not recognizer.is_trained:

        st.error(
            "❌ Face recognition model is not trained."
        )

        return

    # Detect faces
    try:

        boxes = detector.detect(
            frame
        )

    except Exception as e:

        st.error(
            f"❌ Face detection failed: {e}"
        )

        return

    if len(boxes) == 0:

        st.warning(
            "⚠️ No face detected. "
            "Make sure the face is clearly visible "
            "and there is enough lighting."
        )

        st.image(
            frame,
            channels="BGR",
            caption="Captured image"
        )

        return

    recognized_names = []

    unknown_count = 0

    # Process every detected face
    for box in boxes:

        x, y, w, h = box

        face_gray = detector.crop_face(
            frame,
            box
        )

        student_id, distance = (
            recognizer.predict(
                face_gray
            )
        )

        # Recognized
        if (
            student_id is not None
            and student_id in students
        ):

            student = students[
                student_id
            ]

            name = student["name"]

            roll_number = student[
                "roll_number"
            ]

            # Mark attendance
            marked = db.mark_attendance(
                session_id,
                student_id,
                float(distance)
            )

            recognized_names.append(
                name
            )

            # Green box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 200, 0),
                3
            )

            cv2.putText(
                frame,
                f"{name} ({roll_number})",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2
            )

        # Unknown face
        else:

            unknown_count += 1

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "Unknown",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # Show processed image
    st.image(
        frame,
        channels="BGR",
        caption="Attendance Result"
    )

    # Results
    if recognized_names:

        unique_names = list(
            dict.fromkeys(
                recognized_names
            )
        )

        for name in unique_names:

            st.success(
                f"✅ Attendance marked: {name}"
            )

    if unknown_count > 0:

        st.warning(
            f"⚠️ {unknown_count} "
            "unknown face(s) detected."
        )


def clear_attendance_state():

    keys = [
        "attendance_running",
        "attendance_session_id",
        "attendance_students",
        "attendance_subject_label",
        "attendance_hour",
        "attendance_date"
    ]

    for key in keys:

        st.session_state.pop(
            key,
            None
        )
