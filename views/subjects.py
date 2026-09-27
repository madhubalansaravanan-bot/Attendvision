import streamlit as st
import pandas as pd

import database as db


def render():
    st.title("Subjects")

    tab_list, tab_manage = st.tabs(["All Subjects", "Add / Edit / Delete"])

    with tab_list:
        subjects = db.get_subjects()
        if not subjects:
            st.info("No subjects yet.")
        else:
            df = pd.DataFrame(subjects)[["subject_code", "subject_name", "department", "year", "semester"]]
            df.columns = ["Code", "Name", "Department", "Year", "Semester"]
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_manage:
        st.subheader("Add Subject")
        with st.form("add_subject_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            code = c1.text_input("Subject Code *")
            name = c2.text_input("Subject Name *")
            c3, c4, c5 = st.columns(3)
            department = c3.text_input("Department", value="Computer Science")
            year = c4.text_input("Year", value="1")
            semester = c5.text_input("Semester", value="1")
            submitted = st.form_submit_button("Add Subject", type="primary")

            if submitted:
                if not code or not name:
                    st.error("Subject Code and Name are required.")
                else:
                    try:
                        db.add_subject(code.strip(), name.strip(), department, year, semester)
                        st.success(f"Added {name} ({code}).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not add subject — code may already exist. ({e})")

        st.divider()
        st.subheader("Edit / Delete Subject")
        subjects = db.get_subjects()
        if not subjects:
            st.info("No subjects to edit yet.")
            return

        options = {f"{s['subject_code']} — {s['subject_name']}": s for s in subjects}
        choice = st.selectbox("Select a subject", list(options.keys()))
        subject = options[choice]

        with st.form("edit_subject_form"):
            name = st.text_input("Subject Name", value=subject["subject_name"])
            c1, c2, c3 = st.columns(3)
            department = c1.text_input("Department", value=subject["department"] or "")
            year = c2.text_input("Year", value=subject["year"] or "")
            semester = c3.text_input("Semester", value=subject["semester"] or "")

            col_save, col_delete = st.columns(2)
            save = col_save.form_submit_button("Save Changes", type="primary")
            delete = col_delete.form_submit_button("Delete Subject")

            if save:
                db.update_subject(subject["id"], name, department, year, semester)
                st.success("Subject updated.")
                st.rerun()

            if delete:
                db.delete_subject(subject["id"])
                st.warning(f"Deleted {subject['subject_name']}.")
                st.rerun()
