"""
Face detection using MediaPipe Face Landmarker (Tasks API).
Returns face landmarks and bounding box for skin segmentation.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from pathlib import Path
import urllib.request
import os

from app.core.models import BoundingBox
from app.config import DetectionConfig


# Model URL and local path
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"


def ensure_model_downloaded():
    """Download the face landmarker model if not present."""
    if MODEL_PATH.exists():
        return str(MODEL_PATH)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face landmarker model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")
    return str(MODEL_PATH)


@dataclass
class FaceDetectionResult:
    """Result of face detection."""
    landmarks: np.ndarray  # Shape: (478, 2) - normalized coordinates
    landmarks_pixels: np.ndarray  # Shape: (478, 2) - pixel coordinates
    bbox: BoundingBox
    confidence: float


class FaceDetector:
    """Detect faces and extract landmarks using MediaPipe Face Landmarker."""

    # Key landmark indices for facial features
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    # Face oval (silhouette) - indices for face boundary
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]

    # Left eye
    LEFT_EYE = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
        386, 385, 384, 398
    ]

    # Right eye
    RIGHT_EYE = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
        159, 160, 161, 246
    ]

    # Left eyebrow
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

    # Right eyebrow
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    # Lips (outer)
    LIPS_OUTER = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
        270, 269, 267, 0, 37, 39, 40, 185
    ]

    # Lips (inner)
    LIPS_INNER = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
        310, 311, 312, 13, 82, 81, 80, 191
    ]

    # Nostrils ONLY - just the dark nostril openings, NOT the nose surface
    # Keep nose skin available for spot detection
    # MediaPipe landmarks around nostril openings only
    NOSTRILS = [
        # Left nostril opening
        219, 218,       # Left nostril inner edge
        235,            # Left nostril bottom
        # Right nostril opening
        439, 438,       # Right nostril inner edge
        455,            # Right nostril bottom
        # Center bottom (between nostrils)
        2,              # Nose bottom center point
    ]

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()

        # Ensure model is downloaded
        model_path = ensure_model_downloaded()

        # Create FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=self.config.face_detection_confidence,
            min_face_presence_confidence=self.config.face_detection_confidence,
            min_tracking_confidence=self.config.face_detection_confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> FaceDetectionResult | None:
        """
        Detect face in image and return landmarks.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            FaceDetectionResult or None if no face detected
        """
        h, w = image.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Process image
        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        # Get first face landmarks
        face_landmarks = results.face_landmarks[0]

        # Convert to numpy array of normalized coordinates
        landmarks = np.array([
            [lm.x, lm.y] for lm in face_landmarks
        ])

        # Convert to pixel coordinates
        landmarks_pixels = landmarks.copy()
        landmarks_pixels[:, 0] *= w
        landmarks_pixels[:, 1] *= h
        landmarks_pixels = landmarks_pixels.astype(np.int32)

        # Calculate bounding box from face oval
        face_oval_points = landmarks_pixels[self.FACE_OVAL]
        x_min, y_min = face_oval_points.min(axis=0)
        x_max, y_max = face_oval_points.max(axis=0)

        # Add some padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        bbox = BoundingBox(
            x=int(x_min),
            y=int(y_min),
            width=int(x_max - x_min),
            height=int(y_max - y_min),
        )

        # Estimate confidence (use detection confidence if available)
        confidence = 0.9  # Default high confidence if face mesh succeeded

        return FaceDetectionResult(
            landmarks=landmarks,
            landmarks_pixels=landmarks_pixels,
            bbox=bbox,
            confidence=confidence,
        )

    def get_feature_points(
        self, landmarks_pixels: np.ndarray, feature: str
    ) -> np.ndarray:
        """Get pixel coordinates for a specific facial feature."""
        indices_map = {
            "face_oval": self.FACE_OVAL,
            "left_eye": self.LEFT_EYE,
            "right_eye": self.RIGHT_EYE,
            "left_eyebrow": self.LEFT_EYEBROW,
            "right_eyebrow": self.RIGHT_EYEBROW,
            "lips_outer": self.LIPS_OUTER,
            "lips_inner": self.LIPS_INNER,
            "nostrils": self.NOSTRILS,
        }

        indices = indices_map.get(feature, [])
        return landmarks_pixels[indices]

    def close(self):
        """Release resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()