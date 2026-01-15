"""
Image loading and conversion utilities.
"""

import cv2
import numpy as np
from PIL import Image
import io


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Load image from bytes (e.g., from Streamlit file uploader).

    Args:
        data: Raw image bytes

    Returns:
        BGR image as numpy array (OpenCV format)
    """
    # Use PIL to handle various formats
    pil_image = Image.open(io.BytesIO(data))

    # Convert to RGB if necessary
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert to numpy array
    rgb_array = np.array(pil_image)

    # Convert RGB to BGR for OpenCV
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    return bgr_array


def load_image_from_path(path: str) -> np.ndarray | None:
    """
    Load image from file path.

    Args:
        path: Path to image file

    Returns:
        BGR image as numpy array or None if failed
    """
    image = cv2.imread(path)
    return image


def resize_image(
    image: np.ndarray,
    max_dimension: int = 1920,
) -> tuple[np.ndarray, float]:
    """
    Resize image if larger than max dimension while preserving aspect ratio.

    Args:
        image: Input image
        max_dimension: Maximum width or height

    Returns:
        Tuple of (resized image, scale factor)
    """
    h, w = image.shape[:2]
    scale = 1.0

    if max(h, w) > max_dimension:
        if w > h:
            scale = max_dimension / w
        else:
            scale = max_dimension / h

        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image, scale


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB (PIL/Streamlit)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR (OpenCV)."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def encode_image_to_bytes(image: np.ndarray, format: str = "PNG") -> bytes:
    """
    Encode image to bytes for download.

    Args:
        image: BGR image
        format: Output format (PNG, JPEG)

    Returns:
        Encoded image bytes
    """
    # Convert to RGB for PIL
    rgb_image = bgr_to_rgb(image)
    pil_image = Image.fromarray(rgb_image)

    # Encode to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return buffer.getvalue()