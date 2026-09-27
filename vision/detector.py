import cv2


class FaceDetector:

    def __init__(self):

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.cascade = cv2.CascadeClassifier(
            cascade_path
        )

        if self.cascade.empty():
            raise RuntimeError(
                "Could not load OpenCV built-in Haar cascade."
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
