"""
AttendVision - configuration.

Reads settings from environment variables where sensible, with safe
local-development defaults so the prototype runs out of the box.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
STUDENTS_DIR = DATA_DIR / "students"      # enrolled face images, one folder per roll number
MODELS_DIR = DATA_DIR / "models"          # trained LBPH recognizer
ATTENDANCE_DIR = DATA_DIR / "attendance"  # exported CSV reports

for _d in (DATA_DIR, STUDENTS_DIR, MODELS_DIR, ATTENDANCE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Environment variables (see .env.example) ---------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", str(DATA_DIR / "attendvision.db"))
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# --- Face recognition tuning ---------------------------------------------
# LBPHFaceRecognizer.predict() returns a DISTANCE, not a similarity score:
# LOWER = more confident. Anything above this threshold is reported as
# "Unknown" and is never auto-marked present.
RECOGNITION_CONFIDENCE_THRESHOLD = 70.0

# Minimum accepted number of consecutive frames a face must be confidently
# matched in before we mark attendance, to reduce false positives from a
# single lucky frame.
CONSECUTIVE_MATCH_FRAMES_REQUIRED = 3

# Number of face samples captured per student during enrollment.
FACE_SAMPLES_PER_STUDENT = 25

FACE_IMG_SIZE = (200, 200)  # normalized grayscale face size used for training/prediction
