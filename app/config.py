"""
Configuration parameters for pigmentation spot detection.
All thresholds are tunable via Streamlit sidebar.
"""

from dataclasses import dataclass


@dataclass
class DetectionConfig:
    """Tunable detection parameters."""

    # Skin color detection (YCrCb color space)
    skin_cr_min: int = 133
    skin_cr_max: int = 173
    skin_cb_min: int = 77
    skin_cb_max: int = 127

    # Spot detection thresholds (LAB color space)
    # Delta from mean skin tone to consider as spot
    spot_b_threshold: float = 3.0  # Lower = more sensitive to brown
    spot_l_threshold: float = 6.0  # Lower = more sensitive to darkness

    # Classification thresholds (combined delta)
    light_min: float = 1.0
    light_max: float = 6.0
    medium_min: float = 6.0
    medium_max: float = 12.0
    dark_min: float = 12.0

    # Spot size filtering
    min_spot_area: int = 15  # Catch smaller spots
    max_spot_area: int = 2000  # Individual spots, not large regions

    # Morphological operations
    morph_kernel_size: int = 3

    # Face detection
    face_detection_confidence: float = 0.5

    # Exclusion zone expansion (pixels to add around facial features)
    eye_expansion: int = 15
    eyebrow_expansion: int = 8
    lip_expansion: int = 5
    nostril_expansion: int = 8  # Exclude dark nostril openings only, keep nose skin


# Default configuration instance
DEFAULT_CONFIG = DetectionConfig()


# Visualization colors (BGR format for OpenCV)
COLORS = {
    "light": (0, 255, 255),
    "medium": (0, 165, 255),
    "dark": (0, 0, 255),
    "debug_skin": (0, 255, 0),
    "debug_face": (255, 0, 0),
}