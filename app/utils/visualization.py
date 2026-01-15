"""
Visualization utilities for drawing annotations on images.
"""

import cv2
import numpy as np

from app.config import COLORS
from app.core.models import DetectionResult, DetectedSpot, SpotClassification


def draw_spot_annotations(
    image: np.ndarray,
    result: DetectionResult,
    draw_labels: bool = True,
    draw_contours: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes and labels for detected spots.

    Args:
        image: BGR image to annotate
        result: Detection result with spots
        draw_labels: Whether to draw classification labels
        draw_contours: Whether to draw actual contours instead of bboxes

    Returns:
        Annotated image copy
    """
    annotated = image.copy()

    for spot in result.spots:
        color = _get_spot_color(spot.classification)

        if draw_contours and spot.contour is not None:
            # Draw contour
            cv2.drawContours(annotated, [spot.contour], -1, color, 2)
        else:
            # Draw bounding box
            bbox = spot.bbox
            cv2.rectangle(
                annotated,
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                color,
                2,
            )

        if draw_labels:
            # Draw label background
            label = f"#{spot.id} {spot.classification.value}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            label_x = spot.bbox.x
            label_y = spot.bbox.y - 5

            # Ensure label is within image bounds
            if label_y - text_h < 0:
                label_y = spot.bbox.y + spot.bbox.height + text_h + 5

            # Draw background rectangle
            cv2.rectangle(
                annotated,
                (label_x, label_y - text_h - 2),
                (label_x + text_w + 4, label_y + 2),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                annotated,
                label,
                (label_x + 2, label_y),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                thickness,
            )

    return annotated


def create_debug_visualization(
    image: np.ndarray,
    result: DetectionResult,
    show_skin_mask: bool = True,
    show_face_landmarks: bool = True,
    show_face_bbox: bool = True,
) -> np.ndarray:
    """
    Create debug visualization with overlays.

    Args:
        image: Original BGR image
        result: Detection result with debug info
        show_skin_mask: Overlay skin mask in green
        show_face_landmarks: Draw face mesh landmarks
        show_face_bbox: Draw face bounding box

    Returns:
        Debug visualization image
    """
    debug_img = image.copy()
    debug_info = result.debug_info

    # Overlay skin mask
    if show_skin_mask and debug_info.skin_mask is not None:
        skin_overlay = np.zeros_like(debug_img)
        skin_overlay[debug_info.skin_mask > 0] = COLORS["debug_skin"]
        debug_img = cv2.addWeighted(debug_img, 0.7, skin_overlay, 0.3, 0)

    # Draw face landmarks
    if show_face_landmarks and debug_info.face_landmarks is not None:
        for point in debug_info.face_landmarks:
            cv2.circle(debug_img, tuple(point), 1, COLORS["debug_face"], -1)

    # Draw face bounding box
    if show_face_bbox and debug_info.face_bbox is not None:
        bbox = debug_info.face_bbox
        cv2.rectangle(
            debug_img,
            (bbox.x, bbox.y),
            (bbox.x + bbox.width, bbox.y + bbox.height),
            COLORS["debug_face"],
            2,
        )

    return debug_img


def create_comparison_image(
    original: np.ndarray,
    annotated: np.ndarray,
    debug: np.ndarray | None = None,
) -> np.ndarray:
    """
    Create side-by-side comparison image.

    Args:
        original: Original image
        annotated: Annotated result image
        debug: Optional debug visualization

    Returns:
        Combined comparison image
    """
    # Ensure all images have same height
    h = original.shape[0]

    images = [original, annotated]
    if debug is not None:
        images.append(debug)

    # Resize if needed to match heights
    resized = []
    for img in images:
        if img.shape[0] != h:
            scale = h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_w, h))
        resized.append(img)

    # Add labels
    labeled = []
    labels = ["Original", "Detected Spots", "Debug View"]
    for i, img in enumerate(resized):
        img_copy = img.copy()
        cv2.putText(
            img_copy,
            labels[i],
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        labeled.append(img_copy)

    # Concatenate horizontally
    return np.hstack(labeled)


def create_spot_legend() -> np.ndarray:
    """Create a legend image explaining spot colors."""
    legend_h = 80
    legend_w = 300
    legend = np.ones((legend_h, legend_w, 3), dtype=np.uint8) * 255

    items = [
        ("Light", COLORS["light"]),
        ("Medium", COLORS["medium"]),
        ("Dark", COLORS["dark"]),
    ]

    y = 25
    for label, color in items:
        # Draw color box
        cv2.rectangle(legend, (10, y - 15), (30, y + 5), color, -1)
        cv2.rectangle(legend, (10, y - 15), (30, y + 5), (0, 0, 0), 1)

        # Draw label
        cv2.putText(
            legend,
            label,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        y += 25

    return legend


def _get_spot_color(classification: SpotClassification) -> tuple[int, int, int]:
    """Get BGR color for spot classification."""
    return COLORS.get(classification.value, (255, 255, 255))