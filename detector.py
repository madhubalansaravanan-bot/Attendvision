"""
AttendVision - face detection.

Uses OpenCV's bundled Haar cascade classifier. This is intentionally the
simplest reliable option for a local prototype (no extra model downloads,
runs fast on CPU). It can be swapped for a DNN-based detector later
without changing the rest of the pipeline, since it just returns boxes.
"""
import cv2


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

    def detect(self, frame_bgr):
        """Returns a list of (x, y, w, h) boxes for every face found."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
        )
        return [tuple(map(int, f)) for f in faces]

    @staticmethod
    def crop_face(frame_bgr, box):
        x, y, w, h = box
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return gray[y:y + h, x:x + w]
