
"""Persistent face-sample storage using Supabase Storage."""

import cv2
import numpy as np
import streamlit as st
from supabase import create_client

from vision.recognizer import FaceRecognizer

BUCKET = "face-samples"


def _storage():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not key:
        raise RuntimeError(
            "Configure SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY in Streamlit Secrets."
        )

    return create_client(url, key).storage.from_(BUCKET)


def student_dir(roll_number):
    """Compatibility helper; actual samples are stored in Supabase."""
    import config
    return config.STUDENTS_DIR / str(roll_number)


def save_face_sample(roll_number, face_gray, sample_index):
    roll = str(roll_number).strip()

    if not roll or "/" in roll or ".." in roll:
        raise ValueError("Invalid student roll number.")

    ok, encoded = cv2.imencode(".jpg", face_gray)

    if not ok:
        raise RuntimeError("Could not encode face sample.")

    path = f"{roll}/{sample_index:03d}.jpg"

    _storage().upload(
        path,
        encoded.tobytes(),
        {
            "content-type": "image/jpeg",
            "upsert": "true",
        },
    )

    return path


def count_samples(roll_number):
    files = _storage().list(path=str(roll_number))

    return sum(
        1
        for item in files
        if item.get("name", "").lower().endswith(".jpg")
    )


def retrain_all(students):
    """Download persistent samples and train a fresh recognizer."""
    storage = _storage()
    samples_by_student_id = {}
    errors = []

    for student in students:
        roll = str(student["roll_number"])
        images = []

        try:
            files = storage.list(path=roll)

            for item in files:
                name = item.get("name", "")

                if not name.lower().endswith(".jpg"):
                    continue

                remote_path = f"{roll}/{name}"

                try:
                    content = storage.download(remote_path)

                    image = cv2.imdecode(
                        np.frombuffer(content, dtype=np.uint8),
                        cv2.IMREAD_GRAYSCALE,
                    )

                    if image is not None and image.size:
                        images.append(image)
                    else:
                        errors.append(
                            f"{roll}/{name}: image could not be decoded"
                        )

                except Exception as exc:
                    errors.append(
                        f"{roll}/{name}: download failed ({exc})"
                    )

        except Exception as exc:
            errors.append(f"{roll}: Storage listing failed ({exc})")

        if images:
            samples_by_student_id[student["id"]] = images

    if not samples_by_student_id:
        detail = "; ".join(errors[:3])

        raise RuntimeError(
            "No readable face samples were downloaded from Supabase. "
            + (f"Details: {detail}" if detail else
               "Check the bucket, credentials and stored image paths.")
        )

    recognizer = FaceRecognizer()
    recognizer.train(samples_by_student_id)

    return recognizer, samples_by_student_id
    
