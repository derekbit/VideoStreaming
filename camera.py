import cv2
import numpy as np
import time
from yolo import process_frame

scaling_factor = 0.5

class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
    def __del__(self):
        self.cap.release()
    def get_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
            interpolation=cv2.INTER_AREA)

        frame = process_frame(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
