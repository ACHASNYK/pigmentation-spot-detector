"""
Data models for pigmentation spot detection results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from datetime import datetime
import numpy as np


class SpotClassification(Enum):
    """Classification of spot intensity."""
    LIGHT = "light"
    MEDIUM = "medium"
    DARK = "dark"


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class DetectedSpot:
    """A single detected pigmentation spot."""
    id: int
    bbox: BoundingBox
    center: tuple[int, int]
    area: int
    classification: SpotClassification
    color_delta_l: float  # Lightness delta from mean
    color_delta_b: float  # Brown (B-channel) delta from mean
    combined_delta: float  # Combined score for classification
    contour: np.ndarray | None = None  # Original contour points

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "center": {"x": self.center[0], "y": self.center[1]},
            "area": self.area,
            "classification": self.classification.value,
            "color_delta_l": round(self.color_delta_l, 2),
            "color_delta_b": round(self.color_delta_b, 2),
            "combined_delta": round(self.combined_delta, 2),
        }


@dataclass
class DebugInfo:
    """Debug information for analysis and tuning."""
    face_detected: bool
    face_bbox: BoundingBox | None
    face_confidence: float
    mean_skin_color_lab: tuple[float, float, float] | None  # L, A, B
    skin_pixel_count: int
    total_face_pixels: int
    skin_coverage_percent: float
    spots_before_filtering: int
    processing_steps: list[str] = field(default_factory=list)
    # These are stored separately, not serialized to JSON
    skin_mask: np.ndarray | None = None
    face_landmarks: Any | None = None

    def to_dict(self) -> dict:
        return {
            "face_detected": self.face_detected,
            "face_bbox": self.face_bbox.to_dict() if self.face_bbox else None,
            "face_confidence": round(self.face_confidence, 3),
            "mean_skin_color_lab": {
                "L": round(self.mean_skin_color_lab[0], 2),
                "A": round(self.mean_skin_color_lab[1], 2),
                "B": round(self.mean_skin_color_lab[2], 2),
            } if self.mean_skin_color_lab else None,
            "skin_pixel_count": self.skin_pixel_count,
            "total_face_pixels": self.total_face_pixels,
            "skin_coverage_percent": round(self.skin_coverage_percent, 2),
            "spots_before_filtering": self.spots_before_filtering,
            "processing_steps": self.processing_steps,
        }


@dataclass
class DetectionResult:
    """Complete detection result for a single image."""
    image_name: str
    spots: list[DetectedSpot]
    debug_info: DebugInfo
    processing_time_ms: float
    config_used: dict
    # Not serialized
    annotated_image: np.ndarray | None = None
    debug_visualization: np.ndarray | None = None

    @property
    def spot_count(self) -> int:
        return len(self.spots)

    @property
    def spots_by_category(self) -> dict[str, int]:
        counts = {"light": 0, "medium": 0, "dark": 0}
        for spot in self.spots:
            counts[spot.classification.value] += 1
        return counts

    def to_dict(self) -> dict:
        return {
            "image_name": self.image_name,
            "spot_count": self.spot_count,
            "spots_by_category": self.spots_by_category,
            "spots": [spot.to_dict() for spot in self.spots],
            "debug_info": self.debug_info.to_dict(),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "config_used": self.config_used,
        }


@dataclass
class BatchReport:
    """Combined report for multiple images."""
    timestamp: str
    total_images: int
    successful_images: int
    failed_images: int
    results: list[DetectionResult]
    summary: dict

    @classmethod
    def create(cls, results: list[DetectionResult], failed_count: int = 0) -> "BatchReport":
        total_spots = {"light": 0, "medium": 0, "dark": 0}
        total_count = 0

        for result in results:
            for category, count in result.spots_by_category.items():
                total_spots[category] += count
                total_count += count

        summary = {
            "total_spots_detected": total_count,
            "spots_by_category": total_spots,
            "average_spots_per_image": round(total_count / len(results), 2) if results else 0,
        }

        return cls(
            timestamp=datetime.now().isoformat(),
            total_images=len(results) + failed_count,
            successful_images=len(results),
            failed_images=failed_count,
            results=results,
            summary=summary,
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "total_images": self.total_images,
            "successful_images": self.successful_images,
            "failed_images": self.failed_images,
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
        }