import time
from datetime import date

import cv2
import streamlit as st

import config
import database as db
from vision.camera import Camera, CameraError
from vision.detector import FaceDetector
from vision.recognizer import FaceRecognizer

REFRESH_SECONDS = 0.15


def render():
    st.title("Take Attendance")

    if not st.session_state.get("attendance_running"):
        _render_setup()
    else:
        _render_live_session()


def _render_setup():
    subjects = db.get_subjects()
    if not subjects:
        st.warning("No subjects exist yet. Add one under **Subjects** first.")
        return

    with st.form("start_attendance_form"):
        c1, c2 = st.columns(2)
        department = c1.text_input("Department", value="Computer Science")
        section = c2.text_input("Class / Section", value="A")

        subj_options = {f"{s['subject_code']} — {s['subject_name']}": s["id"] for s in subjects}
        subject_label = st.selectbox("Subject", list(subj_options.keys()))

        c3, c4 = st.columns(2)
        session_date = c3.date_input("Date", value=date.today())
        hour = c4.number_input("Hour", min_value=1, max_value=12, value=1, step=1)

        start = st.form_submit_button("START CAMERA", type="primary")

    if not start:
        return

    students = db.get_students(department=department, section=section)
    if not students:
        st.warning("No students found for this department/section. Add students first.")
        return

    recognizer = FaceRecognizer()
    if not recognizer.is_trained:
        st.error(
            "No enrolled faces yet — recognition can't run. Enroll at least one student's "
            "face under **Students → Enroll Face** first."
        )
        return

    user = st.session_state["user"]
    subject_id = subj_options[subject_label]
    session_date_str = session_date.isoformat()

    existing = db.find_open_session(subject_id, session_date_str, int(hour), section)
    session_id = existing["id"] if existing else db.create_session(
        subject_id, user["id"] or 0, session_date_str, int(hour), section
    )

    try:
        camera = Camera().open()
    except CameraError as e:
        st.error(f"Camera error: {e}")
        return

    st.session_state["attendance_running"] = True
    st.session_state["attendance_session_id"] = session_id
    st.session_state["attendance_camera"] = camera
    st.session_state["attendance_recognizer"] = recognizer
    st.session_state["attendance_students"] = {s["id"]: s for s in students}
    st.session_state["attendance_match_streak"] = {}
    st.session_state["attendance_subject_label"] = subject_label
    st.session_state["attendance_hour"] = int(hour)
    st.session_state["attendance_date"] = session_date_str
    st.rerun()


def _render_live_session():
    session_id = st.session_state["attendance_session_id"]

    header_l, header_r = st.columns([3, 1])
    header_l.subheader(
        f"{st.session_state['attendance_subject_label']} — Hour {st.session_state['attendance_hour']} "
        f"— {st.session_state['attendance_date']}"
    )
    stop = header_r.button("⏹ End Session", type="primary")

    if stop:
        _end_session()
        st.rerun()
        return

    col_cam, col_list = st.columns([2, 1])
    frame_slot = col_cam.empty()
    status_slot = col_cam.empty()

    camera = st.session_state["attendance_camera"]
    detector_key = "attendance_detector"
    if detector_key not in st.session_state:
        st.session_state[detector_key] = FaceDetector()
    detector = st.session_state[detector_key]
    recognizer = st.session_state["attendance_recognizer"]
    students = st.session_state["attendance_students"]
    streaks = st.session_state["attendance_match_streak"]

    try:
        frame = camera.read_frame()
    except CameraError as e:
        st.error(f"Camera error: {e}. Ending session safely.")
        _end_session()
        return

    boxes = detector.detect(frame)
    display_frame = frame.copy()
    seen_ids_this_frame = set()

    for box in boxes:
        x, y, w, h = box
        face_gray = detector.crop_face(frame, box)
        student_id, distance = recognizer.predict(face_gray)

        if student_id is not None and student_id in students:
            label = f"{students[student_id]['name']} ({students[student_id]['roll_number']})"
            color = (0, 200, 0)
            seen_ids_this_frame.add(student_id)
            streaks[student_id] = streaks.get(student_id, 0) + 1

            if streaks[student_id] >= config.CONSECUTIVE_MATCH_FRAMES_REQUIRED:
                db.mark_attendance(session_id, student_id, float(distance))
        else:
            label = "Unknown"
            color = (0, 0, 220)

        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(display_frame, label, (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # decay streaks for students not seen this frame, so a brief occlusion
    # doesn't instantly re-arm re-marking (they're already marked anyway)
    for sid in list(streaks.keys()):
        if sid not in seen_ids_this_frame:
            streaks[sid] = max(streaks[sid] - 1, 0)

    frame_slot.image(display_frame, channels="BGR")
    if not boxes:
        status_slot.info("No faces in frame.")
    else:
        status_slot.empty()

    with col_list:
        st.markdown("**LIVE ATTENDANCE**")
        records = db.get_session_records(session_id)
        present_ids = {r["student_id"] for r in records}

        st.metric("Present", len(present_ids))
        st.metric("Remaining", max(len(students) - len(present_ids), 0))

        if records:
            for r in records[::-1][:12]:
                st.success(f"✓ {r['student_name']} ({r['roll_number']})", icon="✅")
        else:
            st.caption("No students recognized yet.")

    time.sleep(REFRESH_SECONDS)
    st.rerun()


def _end_session():
    session_id = st.session_state.get("attendance_session_id")
    camera = st.session_state.get("attendance_camera")

    if camera is not None:
        camera.release()

    if session_id is not None:
        db.end_session(session_id)
        records = db.get_session_records(session_id)
        st.session_state["last_session_summary"] = {
            "session_id": session_id,
            "present": len(records),
            "total": len(st.session_state.get("attendance_students", {})),
        }

    for key in ("attendance_running", "attendance_session_id", "attendance_camera",
                "attendance_recognizer", "attendance_students", "attendance_match_streak",
                "attendance_detector"):
        st.session_state.pop(key, None)
