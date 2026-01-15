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

        # Step 6: Erode mask to stay away from boundaries (prevents edge artifacts)
        erode_kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.erode(skin_mask, erode_kernel, iterations=1)
        debug_info["steps"].append("Eroded mask to stay away from hair/boundary edges")

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

        # ADJUST the face oval based on position:
        # - Top-CENTER (forehead): EXTEND UPWARD to capture more forehead skin
        # - Top-SIDES (temples/hairline): CONTRACT to avoid hair
        # - Sides (sideburns): CONTRACT moderately
        # - Bottom (chin): minimal change

        adjusted_points = face_points.copy()

        for i in range(len(adjusted_points)):
            px, py = adjusted_points[i]

            # Calculate relative position in face
            rel_y = (py - y_min) / face_height  # 0 = top, 1 = bottom
            rel_x = (px - center_x) / (face_width / 2)  # -1 = left edge, 0 = center, 1 = right edge
            abs_rel_x = abs(rel_x)  # 0 at center, 1 at edges

            # For TOP points: EXTEND upward at center, CONTRACT at sides
            if rel_y < 0.25:  # Top 25% of face (forehead area)
                # Extension factor: positive at center, becomes contraction at sides
                # At center (abs_rel_x=0): extend upward by 15%
                # At sides (abs_rel_x=1): contract by 10%
                if abs_rel_x < 0.4:  # Center region - EXTEND upward
                    extension = 0.15 * (1 - abs_rel_x / 0.4)  # 15% at center, 0% at 0.4
                    adjusted_points[i, 1] = py - extension * face_height  # Move UP
                else:  # Side region - CONTRACT to avoid hair
                    contraction = 0.08 * (abs_rel_x - 0.4) / 0.6  # 0% at 0.4, 8% at edge
                    dx = center_x - px
                    dy = center_y - py
                    adjusted_points[i, 0] = px + dx * contraction
                    adjusted_points[i, 1] = py + dy * contraction

            # For MIDDLE points: CONTRACT sides to avoid hair
            elif rel_y < 0.70:  # Middle section (eyes to cheeks)
                # Contract based on how far from center (sides have more hair)
                side_contraction = abs_rel_x * 0.10  # Up to 10% at edges
                dx = center_x - px
                dy = center_y - py
                adjusted_points[i, 0] = px + dx * side_contraction
                adjusted_points[i, 1] = py + dy * side_contraction

            # For BOTTOM points: minimal adjustment (chin/jaw - no hair)
            else:
                contraction = 0.02  # Very slight contraction
                dx = center_x - px
                dy = center_y - py
                adjusted_points[i, 0] = px + dx * contraction
                adjusted_points[i, 1] = py + dy * contraction

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

        Hair is typically characterized by:
        - Low L value in LAB (darkness) - usually < 80
        - Low saturation in HSV
        - Different color properties than skin

        This helps exclude hair strands that fall within the face oval,
        especially on side-angle views where hairline overlaps with face mask.
        """
        h, w = image.shape[:2]
        hair_mask = np.zeros((h, w), dtype=np.uint8)

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L_channel = lab[:, :, 0].astype(np.float32)

        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        S_channel = hsv[:, :, 1].astype(np.float32)
        V_channel = hsv[:, :, 2].astype(np.float32)

        # Only analyze within face region
        face_pixels = face_mask > 0

        if np.sum(face_pixels) == 0:
            return hair_mask

        # Calculate statistics for face region to determine adaptive thresholds
        face_L_values = L_channel[face_pixels]
        mean_L = np.mean(face_L_values)
        std_L = np.std(face_L_values)

        # Hair detection criteria:
        # 1. Very dark regions (L < 80 or significantly darker than mean)
        dark_threshold = min(80, mean_L - 1.5 * std_L)
        dark_regions = L_channel < dark_threshold

        # 2. Also catch moderately dark with low saturation (gray/black hair)
        # Hair often has lower saturation than skin
        moderately_dark = L_channel < 100
        low_saturation = S_channel < 60  # Low color saturation
        gray_hair = moderately_dark & low_saturation

        # 3. Very dark value in HSV (catches dark hair that might have odd LAB values)
        very_dark_v = V_channel < 70

        # Combine hair detection criteria
        potential_hair = (dark_regions | gray_hair | very_dark_v) & face_pixels

        # Create initial hair mask
        hair_mask[potential_hair] = 255

        # Dilate hair regions slightly to catch edges
        dilate_kernel = np.ones((3, 3), np.uint8)
        hair_mask = cv2.dilate(hair_mask, dilate_kernel, iterations=2)

        # Keep only within face region
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