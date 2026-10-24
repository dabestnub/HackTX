"""Camera frame -> MediaPipe hand landmarks -> letter probabilities, matching the repo's app.py."""
import os
from dataclasses import dataclass

# makes cv2.imdecode refuse decompression bombs before allocating
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(640 * 480))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .classifier import Classifier, features
from .config import MODELS


@dataclass
class Recognition:
    probs: np.ndarray | None  # 26 softmax values, or None when no hand is seen
    landmarks: list | None    # 21 [x, y] points, 0-1, in the mirrored (selfie) view


class Recognizer:
    """One per player; the HandLandmarker is never shared between players."""

    def __init__(self, classifier: Classifier, task_path=MODELS / "hand_landmarker.task", max_w=640, max_h=480):
        options = vision.HandLandmarkerOptions(  # same settings as app.py:30-37
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.classifier = classifier
        self.max_w, self.max_h = max_w, max_h

    def process(self, jpeg: bytes) -> Recognition | None:
        """None when the bytes are not a JPEG within the size limit (Q16)."""
        try:
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None
        except cv2.error:
            frame = None
        if frame is None or frame.shape[1] > self.max_w or frame.shape[0] > self.max_h:
            return None
        frame = cv2.flip(frame, 1)  # app.py:105 mirrors every frame before MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return Recognition(None, None)
        hand = result.hand_landmarks[0]  # first hand only, handedness ignored, as the repo does
        x = features([[lm.x, lm.y, lm.z] for lm in hand])
        return Recognition(self.classifier.probs(x), [[round(lm.x, 3), round(lm.y, 3)] for lm in hand])

    def close(self):
        self.detector.close()
