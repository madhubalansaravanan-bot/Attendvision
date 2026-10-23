
"""Persistent face-sample storage using a private Supabase Storage bucket."""
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from supabase import create_client

import config
from vision.recognizer import FaceRecognizer

BUCKET = "face-samples"


def _storage():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "must be configured in Streamlit Secrets."
        )
    return create_client(url, key).storage.from_(BUCKET)


def student_dir(roll_number):
    # Retained for compatibility; samples are stored in Supabase.
    return config.STUDENTS_DIR / roll_number


def save_face_sample(roll_number, face_gray, sample_index):
    path = f"{roll_number}/{sample_index:03d}.jpg"
    ok, encoded = cv2.imencode(".jpg", face_gray)
    if not ok:
        raise RuntimeError("Could not encode face sample.")

    _storage().upload(
        path,
        encoded.tobytes(),
        {"content-type": "image/jpeg", "upsert": "true"},
    )
    return path


def count_samples(roll_number):
    files = _storage().list(path=roll_number)
    return sum(
        1 for item in files
        if item.get("name", "").lower().endswith(".jpg")
    )


def retrain_all(students):
    samples_by_student_id = {}
    storage = _storage()

    for student in students:
        roll = student["roll_number"]
        try:
            files = storage.list(path=roll)
        except Exception:
            continue

        images = []
        for item in files:
            name = item.get("name", "")
            if not name.lower().endswith(".jpg"):
                continue
            remote_path = f"{roll}/{name}"
            try:
                content = storage.download(remote_path)
                image = cv2.imdecode(
                    np.frombuffer(content, np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
                if image is not None:
                    images.append(image)
            except Exception:
                continue

        if images:
            samples_by_student_id[student["id"]] = images

    recognizer = FaceRecognizer()
    if samples_by_student_id:
        recognizer.train(samples_by_student_id)

    return recognizer, samples_by_student_id
