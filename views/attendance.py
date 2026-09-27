import threading
from datetime import date

import av
import cv2
import streamlit as st


import config
import database as db
from vision.detector import FaceDetector
from vision.recognizer import FaceRecognizer


RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
}


class AttendanceProcessor:
    def __init__(self, recognizer, detector, students, session_id):
        self.recognizer = recognizer
        self.detector = detector
        self.students = students
        self.session_id = session_id

        self.lock = threading.Lock()
        self.recognized_ids = set()
        self.match_streak = {}

    def process(self, frame):
        img = frame.to_ndarray(format="bgr24")

        boxes = self.detector.detect(img)

        for box in boxes:
            x, y, w, h = box

            face_gray = self.detector.crop_face(img, box)
            student_id, distance = self.recognizer.predict(face_gray)

            if student_id is not None and student_id in self.students:

                label = (
                    f"{self.students[student_id]['name']} "
                    f"({self.students[student_id]['roll_number']})"
                )

                color = (0, 200, 0)

                with self.lock:
                    self.match_streak[student_id] = (
                        self.match_streak.get(student_id, 0) + 1
                    )

                    if (
                        self.match_streak[student_id]
                        >= config.CONSECUTIVE_MATCH_FRAMES_REQUIRED
                    ):
                        self.recognized_ids.add(student_id)

                        try:
                            db.mark_attendance(
                                self.session_id,
                                student_id,
                                float(distance)
                            )
                        except Exception:
                            pass

            else:
                label = "Unknown"
                color = (0, 0, 220)

            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                img,
                label,
                (x, max(y - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


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
            "No subjects exist yet. Add one under **Subjects** first."
        )
        return

    with st.form("start_attendance_form"):

        c1, c2 = st.columns(2)

        department = c1.text_input(
            "Department",
            value="Aerospace"
        )

        section = c2.text_input(
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

        c3, c4 = st.columns(2)

        session_date = c3.date_input(
            "Date",
            value=date.today()
        )

        hour = c4.number_input(
            "Hour",
            min_value=1,
            max_value=12,
            value=1,
            step=1
        )

        start = st.form_submit_button(
            "START CAMERA",
            type="primary"
        )

    if not start:
        return

    students = db.get_students(
        department=department,
        section=section
    )

    if not students:
        st.error(
            f"No students found for {department} - Section {section}."
        )
        return

    recognizer = FaceRecognizer()

    if not recognizer.is_trained:
        st.error(
            "No enrolled faces found. "
            "Enroll a student's face first."
        )
        return

    user = st.session_state["user"]

    subject_id = subject_options[subject_label]
    session_date_str = session_date.isoformat()

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

        st.error(f"Could not create attendance session: {e}")
        return

    detector = FaceDetector()

    processor = AttendanceProcessor(
        recognizer,
        detector,
        {s["id"]: s for s in students},
        session_id
    )

    st.session_state["attendance_running"] = True
    st.session_state["attendance_session_id"] = session_id
    st.session_state["attendance_processor"] = processor
    st.session_state["attendance_students"] = {
        s["id"]: s for s in students
    }
    st.session_state["attendance_subject_label"] = subject_label
    st.session_state["attendance_hour"] = int(hour)
    st.session_state["attendance_date"] = session_date_str

    st.rerun()


def render_live():

    session_id = st.session_state["attendance_session_id"]

    st.subheader(
        f"{st.session_state['attendance_subject_label']} "
        f"— Hour {st.session_state['attendance_hour']} "
        f"— {st.session_state['attendance_date']}"
    )

    st.info(
        "Click START below and allow camera permission when your browser asks."
    )

    processor = st.session_state["attendance_processor"]

    def video_callback(frame):
        return processor.process(frame)

    ctx = webrtc_streamer(
        key=f"attendance-{session_id}",
        video_frame_callback=video_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    st.divider()

    records = db.get_session_records(session_id)

    present_ids = {
        r["student_id"]
        for r in records
    }

    students = st.session_state["attendance_students"]

    col1, col2 = st.columns(2)

    col1.metric(
        "Present",
        len(present_ids)
    )

    col2.metric(
        "Remaining",
        max(len(students) - len(present_ids), 0)
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

        db.end_session(session_id)

        for key in [
            "attendance_running",
            "attendance_session_id",
            "attendance_processor",
            "attendance_students",
            "attendance_subject_label",
            "attendance_hour",
            "attendance_date"
        ]:
            st.session_state.pop(key, None)

        st.success("Attendance session ended.")
        st.rerun()
