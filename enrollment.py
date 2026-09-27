"""
AttendVision - enrollment helpers.

Face samples are stored on disk as grayscale JPEGs under
data/students/<roll_number>/, one folder per student. This keeps the
database free of large binary blobs (the `face_data` field in the spec's
schema is realized here as "face_enrolled" flag + files on disk) and
makes it trivial to retrain the recognizer from scratch at any time.
"""
import cv2

import config
from vision.detector import FaceDetector
from vision.recognizer import FaceRecognizer


def student_dir(roll_number):
    d = config.STUDENTS_DIR / roll_number
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_face_sample(roll_number, face_gray, sample_index):
    path = student_dir(roll_number) / f"{sample_index:03d}.jpg"
    cv2.imwrite(str(path), face_gray)
    return path


def count_samples(roll_number):
    d = config.STUDENTS_DIR / roll_number
    if not d.exists():
        return 0
    return len(list(d.glob("*.jpg")))


def retrain_all(students):
    """students: list of student dicts (must include 'id' and 'roll_number').
    Rebuilds the recognizer from every enrolled student's saved samples."""
    detector = FaceDetector()  # noqa: F841 (kept for API symmetry / future use)
    samples_by_student_id = {}

    for student in students:
        roll = student["roll_number"]
        d = config.STUDENTS_DIR / roll
        if not d.exists():
            continue
        images = []
        for img_path in sorted(d.glob("*.jpg")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        if images:
            samples_by_student_id[student["id"]] = images

    recognizer = FaceRecognizer()
    if samples_by_student_id:
        recognizer.train(samples_by_student_id)
    return recognizer, samples_by_student_id
