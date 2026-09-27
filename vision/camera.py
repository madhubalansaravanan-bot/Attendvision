"""
AttendVision - webcam wrapper.

Thin wrapper around cv2.VideoCapture with explicit error states so the UI
layer can show a clear message instead of crashing when no camera is
present or it gets disconnected mid-session.
"""
import cv2


class CameraError(Exception):
    pass


class Camera:
    def __init__(self, index=0):
        self.index = index
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            self.cap = None
            raise CameraError(
                f"Could not open webcam at index {self.index}. "
                "Check that a camera is connected and not in use by another app."
            )
        return self

    def read_frame(self):
        """Returns a single BGR frame. Raises CameraError if the camera
        disconnects mid-session so the caller can stop the session safely."""
        if self.cap is None:
            raise CameraError("Camera is not open.")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError("Lost connection to the webcam.")
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
