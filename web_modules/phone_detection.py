import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
MODEL_FILENAME = "efficientdet_lite0.tflite"

class PhoneDetector:
    def __init__(self, model_dir="."):
        self.model_path = os.path.join(model_dir, MODEL_FILENAME)
        self._ensure_model()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5,
            running_mode=vision.RunningMode.IMAGE,
            max_results=5,
            category_allowlist=["cell phone"]
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def _ensure_model(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading object detection model to {self.model_path}...")
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download model: {e}")
                # Create a dummy file or raise error?
                # Without model we can't run.
                raise e

    def detect(self, frame: np.ndarray) -> bool:
        """
        Detects if a cell phone is present in the frame.
        Args:
            frame: BGR numpy array from OpenCV.
        Returns:
            True if cell phone detected, False otherwise.
        """
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        # We used category_allowlist=["cell phone"], so any result is a phone.
        # But let's verify categories just in case or if we remove allowlist later.
        for detection in detection_result.detections:
            for category in detection.categories:
                if category.category_name == "cell phone" and category.score > 0.5:
                    return True
        return False

