import time
import cv2
import streamlit as st
import pandas as pd

import config
import database as db
from vision.camera import Camera, CameraError
from vision.detector import FaceDetector
from vision.enrollment import save_face_sample, count_samples, retrain_all


def render():
    st.title("Students")

    tab_list, tab_add, tab_enroll = st.tabs(["All Students", "Add / Edit / Delete", "Enroll Face"])

    with tab_list:
        _render_list()

    with tab_add:
        _render_add_edit_delete()

    with tab_enroll:
        _render_enrollment()


def _render_list():
    students = db.get_students()
    if not students:
        st.info("No students yet. Add one in the **Add / Edit / Delete** tab.")
        return

    df = pd.DataFrame(students)
    df["face_enrolled"] = df["face_enrolled"].map({1: "✓ Enrolled", 0: "Not enrolled"})
    df = df[["roll_number", "name", "department", "year", "section", "email", "face_enrolled"]]
    df.columns = ["Roll No.", "Name", "Department", "Year", "Section", "Email", "Face Status"]
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_add_edit_delete():
    st.subheader("Add Student")
    with st.form("add_student_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        roll_number = c1.text_input("Roll Number *")
        name = c2.text_input("Name *")
        c3, c4, c5 = st.columns(3)
        department = c3.text_input("Department", value="Computer Science")
        year = c4.text_input("Year", value="1")
        section = c5.text_input("Section", value="A")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Add Student", type="primary")

        if submitted:
            if not roll_number or not name:
                st.error("Roll Number and Name are required.")
            else:
                try:
                    db.add_student(roll_number.strip(), name.strip(), department, year, section, email)
                    st.success(f"Added {name} ({roll_number}).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not add student — roll number may already exist. ({e})")

    st.divider()
    st.subheader("Edit / Delete Student")
    students = db.get_students()
    if not students:
        st.info("No students to edit yet.")
        return

    options = {f"{s['roll_number']} — {s['name']}": s["id"] for s in students}
    choice = st.selectbox("Select a student", list(options.keys()))
    student = db.get_student(options[choice])

    with st.form("edit_student_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name", value=student["name"])
        email = c2.text_input("Email", value=student["email"] or "")
        c3, c4, c5 = st.columns(3)
        department = c3.text_input("Department", value=student["department"] or "")
        year = c4.text_input("Year", value=student["year"] or "")
        section = c5.text_input("Section", value=student["section"] or "")

        col_save, col_delete = st.columns(2)
        save = col_save.form_submit_button("Save Changes", type="primary")
        delete = col_delete.form_submit_button("Delete Student")

        if save:
            db.update_student(student["id"], name, department, year, section, email)
            st.success("Student updated.")
            st.rerun()

        if delete:
            db.delete_student(student["id"])
            st.warning(f"Deleted {student['name']}.")
            st.rerun()


def _render_enrollment():
    st.subheader("Enroll Face")
    st.caption(
        "Captures a short burst of face samples from the webcam and (re)trains the "
        "recognizer. Make sure the student has given consent before enrolling, per "
        "the college's data policy."
    )

    students = db.get_students()
    if not students:
        st.info("Add a student first.")
        return

    options = {f"{s['roll_number']} — {s['name']}": s for s in students}
    choice = st.selectbox("Select student to enroll", list(options.keys()), key="enroll_select")
    student = options[choice]

    existing = count_samples(student["roll_number"])
    st.write(f"Existing samples on disk: **{existing}**")

    if st.button("Start Capture", type="primary", key="start_capture"):
        _capture_samples(student)


def _capture_samples(student):
    target = config.FACE_SAMPLES_PER_STUDENT
    frame_slot = st.empty()
    status_slot = st.empty()
    progress = st.progress(0)

    try:
        detector = FaceDetector()
        with Camera() as cam:
            captured = count_samples(student["roll_number"])
            start_count = captured
            attempts = 0
            max_attempts = target * 20  # safety valve so a bad angle can't loop forever

            while captured - start_count < target and attempts < max_attempts:
                attempts += 1
                frame = cam.read_frame()
                boxes = detector.detect(frame)

                display_frame = frame.copy()
                for (x, y, w, h) in boxes:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                frame_slot.image(display_frame, channels="BGR", caption="Enrollment preview")

                if len(boxes) == 1:
                    face = detector.crop_face(frame, boxes[0])
                    captured += 1
                    save_face_sample(student["roll_number"], face, captured)
                elif len(boxes) == 0:
                    status_slot.warning("No face detected — face the camera directly.")
                else:
                    status_slot.warning("Multiple faces detected — only one person should be in frame.")

                progress.progress(min((captured - start_count) / target, 1.0))
                time.sleep(0.08)

        frame_slot.empty()

        if captured - start_count < target:
            st.warning(
                f"Only captured {captured - start_count}/{target} samples before stopping. "
                "You can run capture again to add more."
            )
        else:
            status_slot.success(f"Captured {target} new samples.")

        with st.spinner("Retraining recognizer on all enrolled students..."):
            all_students = db.get_students()
            _, samples = retrain_all(all_students)
            if student["id"] in samples:
                db.set_face_enrolled(student["id"], True)

        st.success(f"{student['name']} is enrolled and the recognizer has been retrained.")

    except CameraError as e:
        st.error(f"Camera error: {e}")
