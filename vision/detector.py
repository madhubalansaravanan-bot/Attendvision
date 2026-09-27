import os
import cv2


class FaceDetector:

    def __init__(self):

        possible_paths = [
            os.path.join(
                os.path.dirname(__file__),
                "haarcascade_frontalface_default.xml"
            ),

            os.path.join(
                cv2.data.haarcascades,
                "haarcascade_frontalface_default.xml"
            )
        ]

        cascade_path = None

        for path in possible_paths:
            if os.path.exists(path):
                cascade_path = path
                break

        if cascade_path is None:
            raise RuntimeError(
                "Haar cascade file not found. "
                "Please add haarcascade_frontalface_default.xml "
                "to the vision folder."
            )

      self.cascade = cv2.CascadeClassifier()

if not self.cascade.load(cascade_path):
    raise RuntimeError(
        f"Could not load Haar cascade from {cascade_path}"
    )
    def detect(self, frame_bgr):

        gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        return [
            tuple(map(int, face))
            for face in faces
        ]

    @staticmethod
    def crop_face(frame_bgr, box):

        x, y, w, h = box

        gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY
        )

        return gray[
            y:y + h,
            x:x + w
        ]
