"""
AttendVision - face recognition.

Uses OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer,
part of opencv-contrib. It's lightweight, needs no GPU or external model
download, and is a standard choice for small-scale attendance prototypes.

IMPORTANT: LBPH.predict() returns a DISTANCE where LOWER means a closer
match. We convert that into a simple confident/not-confident decision
using config.RECOGNITION_CONFIDENCE_THRESHOLD, and this module never
claims certainty - callers should always treat a match as "best guess".
"""
import json
import cv2
import numpy as np

import config


class FaceRecognizer:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}  # int label -> student_id
        self._trained = False
        self.model_path = config.MODELS_DIR / "lbph_model.yml"
        self.label_map_path = config.MODELS_DIR / "label_map.json"
        self._try_load()

    def _try_load(self):
        if self.model_path.exists() and self.label_map_path.exists():
            self.model.read(str(self.model_path))
            with open(self.label_map_path) as f:
                self.label_map = {int(k): v for k, v in json.load(f).items()}
            self._trained = True

    @property
    def is_trained(self):
        return self._trained

    def train(self, samples_by_student_id: dict):
        """samples_by_student_id: {student_id: [grayscale face images]}.
        Trains from scratch on everything currently enrolled."""
        faces, labels = [], []
        label_map = {}
        next_label = 0

        for student_id, images in samples_by_student_id.items():
            if not images:
                continue
            label_map[next_label] = student_id
            for img in images:
                resized = cv2.resize(img, config.FACE_IMG_SIZE)
                faces.append(resized)
                labels.append(next_label)
            next_label += 1

        if not faces:
            raise ValueError("No enrolled face samples to train on.")

        self.model.train(faces, np.array(labels))
        self.label_map = label_map
        self._trained = True

        self.model.write(str(self.model_path))
        with open(self.label_map_path, "w") as f:
            json.dump(label_map, f)

    def predict(self, face_gray):
        """Returns (student_id_or_None, confidence_distance).
        student_id is None when the match isn't confident enough - callers
        must treat that as Unknown and must not auto-mark attendance."""
        if not self._trained:
            return None, None

        resized = cv2.resize(face_gray, config.FACE_IMG_SIZE)
        label, distance = self.model.predict(resized)

        if distance > config.RECOGNITION_CONFIDENCE_THRESHOLD:
            return None, distance

        student_id = self.label_map.get(label)
        return student_id, distance
