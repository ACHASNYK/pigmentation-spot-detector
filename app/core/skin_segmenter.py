"""
Skin segmentation module.
Creates a binary mask of skin-only regions, excluding facial features.
"""

import cv2
import numpy as np

from app.config import DetectionConfig
from app.core.face_detector import FaceDetector, FaceDetectionResult


class SkinSegmenter:
    """Create skin-only mask by excluding facial features."""

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()

    def create_skin_mask(
        self,
        image: np.ndarray,
        face_result: FaceDetectionResult,
    ) -> tuple[np.ndarray, dict]:
        """
        Create a binary mask of skin-only regions.

        Args:
            image: BGR image
            face_result: Face detection result with landmarks

        Returns:
            Tuple of (skin_mask, debug_info_dict)
        """
        h, w = image.shape[:2]
        landmarks = face_result.landmarks_pixels

        debug_info = {
            "steps": [],
            "face_pixels": 0,
            "skin_pixels": 0,
        }

        # Step 1: Create face region mask from face oval
        face_mask = self._create_face_region_mask(landmarks, (h, w))
        debug_info["steps"].append("Created face region mask from landmarks")
        debug_info["face_pixels"] = int(np.sum(face_mask > 0))

        # Step 2: Create exclusion masks for facial features
        exclusion_mask = self._create_exclusion_mask(landmarks, (h, w))
        debug_info["steps"].append("Created exclusion mask for eyes, brows, lips, nose")

        # Step 3: Apply color-based skin filtering
        skin_color_mask = self._create_skin_color_mask(image)
        debug_info["steps"].append("Applied YCrCb skin color filtering")

        # Step 4: Create hair exclusion mask (dark regions that are likely hair)
        hair_mask = self._create_hair_exclusion_mask(image, face_mask)
        debug_info["steps"].append("Created hair exclusion mask (LAB-based dark region detection)")

        # Step 5: Combine masks
        # skin = face_region AND skin_color AND NOT(exclusions) AND NOT(hair)
        skin_mask = cv2.bitwise_and(face_mask, skin_color_mask)
        skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(exclusion_mask))
        skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(hair_mask))
        debug_info["steps"].append("Combined masks: face AND skin_color AND NOT(exclusions) AND NOT(hair)")

        # Step 6: Light erosion to clean up tiny noise (reduced to preserve valid skin)
        erode_kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv2.erode(skin_mask, erode_kernel, iterations=1)
        debug_info["steps"].append("Light erosion to remove noise")

        # Step 7: Clean up with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        debug_info["steps"].append("Applied morphological cleanup")

        debug_info["skin_pixels"] = int(np.sum(skin_mask > 0))

        return skin_mask, debug_info

    def _create_face_region_mask(
        self, landmarks: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray:
        """Create mask from face oval landmarks with INWARD contraction.

        IMPORTANT: We CONTRACT the face oval inward to avoid hair edges.
        This is more reliable than trying to detect hair by color.
        The contraction keeps the mask away from the hairline on all sides.
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Get face oval points
        face_oval_indices = FaceDetector.FACE_OVAL
        face_points = landmarks[face_oval_indices].copy().astype(np.float64)

        # Calculate face center (centroid)
        center_x = np.mean(face_points[:, 0])
        center_y = np.mean(face_points[:, 1])

        # Calculate face dimensions
        y_min = face_points[:, 1].min()
        y_max = face_points[:, 1].max()
        x_min = face_points[:, 0].min()
        x_max = face_points[:, 0].max()
        face_height = y_max - y_min
        face_width = x_max - x_min

        # NEW APPROACH: Create flat-top forehead instead of triangular extension
        # Separate top points from rest to create proper forehead region

        adjusted_points = []
        top_points = []

        for i in range(len(face_points)):
            px, py = face_points[i]

            # Calculate relative position in face
            rel_y = (py - y_min) / face_height  # 0 = top, 1 = bottom
            rel_x = (px - center_x) / (face_width / 2)  # -1 = left edge, 0 = center, 1 = right edge
            abs_rel_x = abs(rel_x)  # 0 at center, 1 at edges

            # For TOP points: collect separately for forehead reconstruction
            if rel_y < 0.25:  # Top 25% of face (forehead area)
                top_points.append((px, py, abs_rel_x))

            # For MIDDLE points: CONTRACT sides to avoid hair
            elif rel_y < 0.70:  # Middle section (eyes to cheeks)
                side_contraction = abs_rel_x * 0.10  # Up to 10% at edges
                dx = center_x - px
                dy = center_y - py
                new_x = px + dx * side_contraction
                new_y = py + dy * side_contraction
                adjusted_points.append([new_x, new_y])

            # For BOTTOM points: minimal adjustment (chin/jaw - no hair)
            else:
                contraction = 0.02
                dx = center_x - px
                dy = center_y - py
                new_x = px + dx * contraction
                new_y = py + dy * contraction
                adjusted_points.append([new_x, new_y])

        # Process TOP points to create flat forehead
        if top_points:
            # Find the highest point (minimum y) in the center region
            center_top_points = [(px, py) for px, py, abs_rx in top_points if abs_rx < 0.5]
            if center_top_points:
                # Extend upward from the topmost center point
                min_y = min(py for _, py in center_top_points)
                forehead_top_y = min_y - 0.20 * face_height  # Extend 20% upward
                forehead_top_y = max(0, forehead_top_y)  # Don't go above image
            else:
                # Fallback if no center points
                min_y = min(py for _, py, _ in top_points)
                forehead_top_y = min_y - 0.15 * face_height
                forehead_top_y = max(0, forehead_top_y)

            # Create flat top line and side transitions
            for px, py, abs_rel_x in sorted(top_points, key=lambda p: p[0]):  # Sort by x
                if abs_rel_x < 0.70:  # Central/mid forehead - create flat top
                    # Gradually transition from top edge to flat line
                    blend = abs_rel_x / 0.70  # 0 at center, 1 at 0.70
                    new_y = forehead_top_y * (1 - blend) + py * blend
                    adjusted_points.append([px, new_y])
                else:  # Sides - contract inward to avoid hair
                    contraction = 0.08
                    dx = center_x - px
                    dy = center_y - py
                    new_x = px + dx * contraction
                    new_y = py + dy * contraction
                    adjusted_points.append([new_x, new_y])

        # Convert to numpy array
        adjusted_points = np.array(adjusted_points, dtype=np.float64)

        # Ensure points stay within image bounds
        adjusted_points[:, 0] = np.clip(adjusted_points[:, 0], 0, w - 1)
        adjusted_points[:, 1] = np.clip(adjusted_points[:, 1], 0, h - 1)

        contracted_points = adjusted_points

        # Convert to int for drawing
        contracted_points = contracted_points.astype(np.int32)

        # Create convex hull and fill
        hull = cv2.convexHull(contracted_points)
        cv2.fillConvexPoly(mask, hull, 255)

        return mask

    def _create_exclusion_mask(
        self, landmarks: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray:
        """Create mask for facial features to exclude (eyes, brows, lips, nose)."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Exclude left eye
        self._add_feature_to_mask(
            mask, landmarks, FaceDetector.LEFT_EYE,
            expansion=self.config.eye_expansion
        )

        # Exclude right eye
        self._add_feature_to_mask(
            mask, landmarks, FaceDetector.RIGHT_EYE,
            expansion=self.config.eye_expansion
        )

        # Exclude left eyebrow
        self._add_feature_to_mask(
            mask, landmarks, FaceDetector.LEFT_EYEBROW,
            expansion=self.config.eyebrow_expansion
        )

        # Exclude right eyebrow
        self._add_feature_to_mask(
            mask, landmarks, FaceDetector.RIGHT_EYEBROW,
            expansion=self.config.eyebrow_expansion
        )

        # Exclude lips (outer region)
        self._add_feature_to_mask(
            mask, landmarks, FaceDetector.LIPS_OUTER,
            expansion=self.config.lip_expansion
        )

        # Exclude nostrils only (not the whole nose)
        self._add_feature_to_mask(
            mask, landmarks, FaceDetector.NOSTRILS,
            expansion=self.config.nostril_expansion
        )

        return mask

    def _add_feature_to_mask(
        self,
        mask: np.ndarray,
        landmarks: np.ndarray,
        indices: list[int],
        expansion: int = 0,
    ) -> None:
        """Add a facial feature region to the exclusion mask."""
        points = landmarks[indices]

        # Create convex hull
        if len(points) >= 3:
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            # Expand the region if needed
            if expansion > 0:
                kernel = np.ones((expansion * 2 + 1, expansion * 2 + 1), np.uint8)
                # Only dilate the newly added region
                temp_mask = np.zeros_like(mask)
                cv2.fillConvexPoly(temp_mask, hull, 255)
                dilated = cv2.dilate(temp_mask, kernel, iterations=1)
                mask[:] = cv2.bitwise_or(mask, dilated)

    def _create_skin_color_mask(self, image: np.ndarray) -> np.ndarray:
        """Create mask based on skin color in YCrCb color space."""
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # Define skin color range
        lower = np.array([0, self.config.skin_cr_min, self.config.skin_cb_min])
        upper = np.array([255, self.config.skin_cr_max, self.config.skin_cb_max])

        # Create mask
        mask = cv2.inRange(ycrcb, lower, upper)

        return mask

    def _create_hair_exclusion_mask(
        self, image: np.ndarray, face_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create mask for hair regions to exclude from skin detection.

        IMPORTANT: Only detect hair at the BOUNDARIES/EDGES of the face mask,
        not in the interior where pigmentation spots exist. This prevents
        false positives where dark pigmentation spots are mistaken for hair.
        """
        h, w = image.shape[:2]
        hair_mask = np.zeros((h, w), dtype=np.uint8)

        # Step 1: Create boundary region (only edges of face mask)
        # Hair intrusion only happens at boundaries, not in face interior
        kernel = np.ones((15, 15), np.uint8)
        face_eroded = cv2.erode(face_mask, kernel, iterations=1)
        boundary_region = cv2.bitwise_xor(face_mask, face_eroded)

        # Step 2: Only analyze pixels in boundary region
        boundary_pixels = boundary_region > 0

        if np.sum(boundary_pixels) == 0:
            return hair_mask

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L_channel = lab[:, :, 0].astype(np.float32)

        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        S_channel = hsv[:, :, 1].astype(np.float32)
        V_channel = hsv[:, :, 2].astype(np.float32)

        # Calculate statistics from full face (not just boundary)
        face_pixels = face_mask > 0
        if np.sum(face_pixels) == 0:
            return hair_mask

        face_L_values = L_channel[face_pixels]
        mean_L = np.mean(face_L_values)
        std_L = np.std(face_L_values)

        # Hair detection criteria - ONLY apply to boundary region:
        # 1. Very dark regions (much stricter threshold)
        dark_threshold = min(60, mean_L - 2.0 * std_L)  # Stricter: only VERY dark
        dark_regions = L_channel < dark_threshold

        # 2. Gray/black hair with low saturation
        very_dark = L_channel < 70  # Much stricter than before (was 100)
        low_saturation = S_channel < 40  # Lower threshold (was 60)
        gray_hair = very_dark & low_saturation

        # 3. Extremely dark value in HSV
        very_dark_v = V_channel < 50  # Stricter (was 70)

        # Combine criteria - ONLY in boundary region
        potential_hair = (dark_regions | gray_hair | very_dark_v) & boundary_pixels

        # Create initial hair mask
        hair_mask[potential_hair] = 255

        # Small dilation to catch immediate hair edges
        dilate_kernel = np.ones((3, 3), np.uint8)
        hair_mask = cv2.dilate(hair_mask, dilate_kernel, iterations=1)

        # Keep only within original face region
        hair_mask = cv2.bitwise_and(hair_mask, face_mask)

        return hair_mask

    def calculate_mean_skin_color(
        self, image: np.ndarray, skin_mask: np.ndarray
    ) -> tuple[float, float, float] | None:
        """
        Calculate mean skin color in LAB color space.

        Returns:
            Tuple of (L, A, B) mean values or None if no skin pixels
        """
        if np.sum(skin_mask > 0) == 0:
            return None

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Get mean values for skin pixels only
        mean_l = np.mean(lab[:, :, 0][skin_mask > 0])
        mean_a = np.mean(lab[:, :, 1][skin_mask > 0])
        mean_b = np.mean(lab[:, :, 2][skin_mask > 0])

        return (float(mean_l), float(mean_a), float(mean_b))