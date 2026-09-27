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
    st.subheader("📷 Capture Face Samples")

    st.info(
        "Allow camera access, take clear photos of your face, "
        "and capture multiple angles."
    )

    picture = st.camera_input(
        "Take a face photo",
        key=f"face_camera_{student['id']}",
        resolution="720p"
    )

    if picture is not None:
        import numpy as np
        import cv2

        bytes_data = picture.getvalue()

        frame = cv2.imdecode(
            np.frombuffer(bytes_data, np.uint8),
            cv2.IMREAD_COLOR
        )

        detector = FaceDetector()
        faces = detector.detect(frame)

        if len(faces) == 0:
            st.error("❌ No face detected. Please try again.")
            return

        if len(faces) > 1:
            st.warning("⚠️ Multiple faces detected. Keep only one face visible.")
            return

        face = detector.crop_face(frame, faces[0])

        student_dir = os.path.join(
            config.STUDENT_FACES_DIR,
            str(student["id"])
        )

        os.makedirs(student_dir, exist_ok=True)

        existing = len(os.listdir(student_dir))
        filename = os.path.join(
            student_dir,
            f"{existing + 1}.jpg"
        )

        cv2.imwrite(filename, face)

        st.success(
            f"✅ Face sample {existing + 1} captured successfully."
        )

        st.image(frame, channels="BGR")

        if st.button("🔄 Capture Another Photo"):
            st.rerun()
