
"""AttendVision face recognition using OpenCV LBPH."""

import json
import cv2
import numpy as np
import config


class FaceRecognizer:
    def __init__(self):
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "OpenCV face module is unavailable. "
                "Install opencv-contrib-python-headless."
            )

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        self._trained = False

        self.model_path = config.MODELS_DIR / "lbph_model.yml"
        self.label_map_path = config.MODELS_DIR / "label_map.json"

        self._try_load()

    def _try_load(self):
        if not self.model_path.exists() or not self.label_map_path.exists():
            return

        try:
            with open(self.label_map_path, encoding="utf-8") as f:
                raw_map = json.load(f)

            self.model.read(str(self.model_path))

            self.label_map = {
                int(label): student_id
                for label, student_id in raw_map.items()
            }

            self._trained = bool(self.label_map)

        except (
            cv2.error,
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            self.label_map = {}
            self._trained = False

    @property
    def is_trained(self):
        return self._trained

    def train(self, samples_by_student_id: dict):
        """Train from grayscale images keyed by database student ID."""
        faces = []
        labels = []
        label_map = {}
        next_label = 0

        for student_id, images in samples_by_student_id.items():
            valid_images = []

            for image in images:
                if image is None or image.size == 0:
                    continue

                if image.ndim == 3:
                    image = cv2.cvtColor(
                        image, cv2.COLOR_BGR2GRAY
                    )

                resized = cv2.resize(
                    image, tuple(config.FACE_IMG_SIZE)
                )

                valid_images.append(resized)

            if not valid_images:
                continue

            label_map[next_label] = student_id

            for image in valid_images:
                faces.append(image)
                labels.append(next_label)

            next_label += 1

        if not faces:
            self._trained = False
            raise ValueError(
                "No readable face samples were available for training."
            )

        self.model.train(faces, np.asarray(labels, dtype=np.int32))
        self.label_map = label_map
        self._trained = True

        # Local files are a cache only. Supabase face samples remain
        # the persistent source used to rebuild the model.
        self.model.write(str(self.model_path))

        temporary_path = self.label_map_path.with_suffix(".json.tmp")
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(self.label_map, f)

        temporary_path.replace(self.label_map_path)

    def predict(self, face_gray):
        """Return (student_id or None, LBPH distance)."""
        if not self._trained:
            return None, None

        if face_gray is None or face_gray.size == 0:
            return None, None

        if face_gray.ndim == 3:
            face_gray = cv2.cvtColor(
                face_gray, cv2.COLOR_BGR2GRAY
            )

        resized = cv2.resize(
            face_gray, tuple(config.FACE_IMG_SIZE)
        )

        label, distance = self.model.predict(resized)

        if distance > config.RECOGNITION_CONFIDENCE_THRESHOLD:
            return None, float(distance)

        return self.label_map.get(label), float(distance)
        
