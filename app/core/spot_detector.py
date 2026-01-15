"""
Pigmentation spot detection and classification.
Uses LAB color space analysis to detect brownish spots.
Supports multiple detection modes: standard, hyper-sensitive, and blob detection.
"""

import cv2
import numpy as np
from dataclasses import asdict
from enum import Enum
from skimage.feature import blob_log, blob_dog
from skimage import img_as_float

from app.config import DetectionConfig
from app.core.models import (
    DetectedSpot,
    SpotClassification,
    BoundingBox,
    DebugInfo,
    DetectionResult,
)
from app.core.face_detector import FaceDetector, FaceDetectionResult
from app.core.skin_segmenter import SkinSegmenter
import time


class DetectionMode(Enum):
    """Detection pipeline modes."""
    STANDARD = "standard"  # Original threshold-based detection
    HYPER_SENSITIVE = "hyper_sensitive"  # Catch any brown deviation
    BLOB_DETECTION = "blob_detection"  # Multi-scale blob detection (LoG/DoG)


class SpotDetector:
    """Detect and classify pigmentation spots on skin."""

    def __init__(self, config: DetectionConfig | None = None, mode: DetectionMode = DetectionMode.STANDARD):
        self.config = config or DetectionConfig()
        self.mode = mode
        self.face_detector = FaceDetector(self.config)
        self.skin_segmenter = SkinSegmenter(self.config)

    def detect(self, image: np.ndarray, image_name: str = "unknown") -> DetectionResult:
        """
        Detect pigmentation spots in an image.

        Args:
            image: BGR image (OpenCV format)
            image_name: Name of the image for reporting

        Returns:
            DetectionResult with spots and debug info
        """
        start_time = time.time()
        processing_steps = []

        # Step 1: Detect face
        processing_steps.append("Starting face detection")
        face_result = self.face_detector.detect(image)

        if face_result is None:
            # No face detected
            debug_info = DebugInfo(
                face_detected=False,
                face_bbox=None,
                face_confidence=0.0,
                mean_skin_color_lab=None,
                skin_pixel_count=0,
                total_face_pixels=0,
                skin_coverage_percent=0.0,
                spots_before_filtering=0,
                processing_steps=["Face detection failed - no face found"],
            )
            return DetectionResult(
                image_name=image_name,
                spots=[],
                debug_info=debug_info,
                processing_time_ms=(time.time() - start_time) * 1000,
                config_used=self._config_to_dict(),
            )

        processing_steps.append(f"Face detected with confidence {face_result.confidence:.2f}")

        # Step 2: Create skin mask
        processing_steps.append("Creating skin segmentation mask")
        skin_mask, seg_debug = self.skin_segmenter.create_skin_mask(image, face_result)
        processing_steps.extend(seg_debug["steps"])

        # Step 3: Calculate mean skin color
        mean_skin_lab = self.skin_segmenter.calculate_mean_skin_color(image, skin_mask)

        if mean_skin_lab is None:
            debug_info = DebugInfo(
                face_detected=True,
                face_bbox=face_result.bbox,
                face_confidence=face_result.confidence,
                mean_skin_color_lab=None,
                skin_pixel_count=0,
                total_face_pixels=seg_debug["face_pixels"],
                skin_coverage_percent=0.0,
                spots_before_filtering=0,
                processing_steps=processing_steps + ["No skin pixels found after masking"],
                skin_mask=skin_mask,
                face_landmarks=face_result.landmarks_pixels,
            )
            return DetectionResult(
                image_name=image_name,
                spots=[],
                debug_info=debug_info,
                processing_time_ms=(time.time() - start_time) * 1000,
                config_used=self._config_to_dict(),
            )

        processing_steps.append(f"Mean skin color (LAB): L={mean_skin_lab[0]:.1f}, A={mean_skin_lab[1]:.1f}, B={mean_skin_lab[2]:.1f}")

        # Step 4: Detect spots based on mode
        if self.mode == DetectionMode.BLOB_DETECTION:
            processing_steps.append("Detecting spots using multi-scale blob detection (LoG)")
            spots, spots_before_filter = self._detect_spots_blob(image, skin_mask, mean_skin_lab)
        elif self.mode == DetectionMode.HYPER_SENSITIVE:
            processing_steps.append("Detecting spots using hyper-sensitive mode (catch all brown)")
            spots, spots_before_filter = self._detect_spots_hyper_sensitive(image, skin_mask, mean_skin_lab)
        else:
            processing_steps.append("Detecting pigmentation spots using standard LAB analysis")
            spots, spots_before_filter = self._detect_spots(image, skin_mask, mean_skin_lab)
        processing_steps.append(f"Found {spots_before_filter} potential spots, {len(spots)} after filtering")

        # Calculate skin coverage
        skin_pixel_count = seg_debug["skin_pixels"]
        face_pixel_count = seg_debug["face_pixels"]
        skin_coverage = (skin_pixel_count / face_pixel_count * 100) if face_pixel_count > 0 else 0

        # Create debug info
        debug_info = DebugInfo(
            face_detected=True,
            face_bbox=face_result.bbox,
            face_confidence=face_result.confidence,
            mean_skin_color_lab=mean_skin_lab,
            skin_pixel_count=skin_pixel_count,
            total_face_pixels=face_pixel_count,
            skin_coverage_percent=skin_coverage,
            spots_before_filtering=spots_before_filter,
            processing_steps=processing_steps,
            skin_mask=skin_mask,
            face_landmarks=face_result.landmarks_pixels,
        )

        processing_time = (time.time() - start_time) * 1000

        return DetectionResult(
            image_name=image_name,
            spots=spots,
            debug_info=debug_info,
            processing_time_ms=processing_time,
            config_used=self._config_to_dict(),
        )

    def _detect_spots(
        self,
        image: np.ndarray,
        skin_mask: np.ndarray,
        mean_skin_lab: tuple[float, float, float],
    ) -> tuple[list[DetectedSpot], int]:
        """
        Detect brownish spots using LAB color space analysis.

        IMPORTANT: True pigmentation spots have POSITIVE delta_B (more brown).
        Shadows have NEGATIVE delta_B (less brown) - these are filtered out.

        Returns:
            Tuple of (filtered spots list, count before filtering)
        """
        mean_l, mean_a, mean_b = mean_skin_lab

        # Convert image to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Calculate deviation from mean skin tone
        # Brownish spots have: higher B value (more yellow/brown) and/or lower L (darker)
        delta_b = lab[:, :, 2] - mean_b  # Positive = more brown/yellow
        delta_l = mean_l - lab[:, :, 0]  # Positive = darker than mean

        # Create spot candidate mask
        spot_mask = np.zeros_like(skin_mask)

        # FIXED CRITERIA for pigmentation detection:
        # 1. Primary: Brownish spots (positive delta_B above threshold)
        brownish_spots = delta_b > self.config.spot_b_threshold

        # 2. Secondary: Darker spots that are NOT shadows (delta_L high but delta_B not negative)
        #    Shadows are dark but have NEGATIVE delta_B (less brown than skin)
        darker_spots = (delta_l > self.config.spot_l_threshold) & (delta_b > -2)

        # 3. Also catch subtle spots: slightly brown AND slightly darker
        subtle_spots = (delta_b > 2) & (delta_l > 3)

        # Combined: any of the above criteria (within skin region)
        combined_spots = (brownish_spots | darker_spots | subtle_spots) & (skin_mask > 0)
        spot_mask[combined_spots] = 255

        # Morphological cleanup to reduce noise
        kernel = np.ones(
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
            np.uint8
        )
        spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, kernel)
        spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_CLOSE, kernel)

        # Find connected components (spots)
        contours, _ = cv2.findContours(
            spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        spots_before_filter = len(contours)
        spots = []

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.config.min_spot_area:
                continue
            if area > self.config.max_spot_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = BoundingBox(x=x, y=y, width=w, height=h)

            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            # Calculate average color delta for this spot
            spot_region_mask = np.zeros_like(skin_mask)
            cv2.drawContours(spot_region_mask, [contour], -1, 255, -1)

            spot_pixels = spot_region_mask > 0
            avg_delta_l = float(np.mean(delta_l[spot_pixels])) if np.any(spot_pixels) else 0
            avg_delta_b = float(np.mean(delta_b[spot_pixels])) if np.any(spot_pixels) else 0

            # Combined delta for classification
            # True pigmentation: positive delta_B (more brown) AND/OR positive delta_L (darker)
            # Use weighted combination
            combined_delta = (avg_delta_l * 0.4) + (avg_delta_b * 0.6)

            # CRITICAL: Filter out non-pigmentation
            # 1. Negative combined_delta = not a real spot
            # 2. Very negative delta_B with only slight darkness = shadow
            if combined_delta < 1.0:
                continue  # Too weak to be considered pigmentation
            if avg_delta_b < -2 and avg_delta_l < 20:
                continue  # Shadow (dark but not brown)

            # Classify spot
            classification = self._classify_spot(combined_delta)

            spot = DetectedSpot(
                id=len(spots) + 1,
                bbox=bbox,
                center=(cx, cy),
                area=int(area),
                classification=classification,
                color_delta_l=avg_delta_l,
                color_delta_b=avg_delta_b,
                combined_delta=combined_delta,
                contour=contour,
            )
            spots.append(spot)

        # Sort spots by combined_delta (most prominent first)
        spots.sort(key=lambda s: s.combined_delta, reverse=True)

        # Re-assign IDs after sorting
        for i, spot in enumerate(spots):
            spot.id = i + 1

        return spots, spots_before_filter

    def _detect_spots_hyper_sensitive(
        self,
        image: np.ndarray,
        skin_mask: np.ndarray,
        mean_skin_lab: tuple[float, float, float],
    ) -> tuple[list[DetectedSpot], int]:
        """
        Hyper-sensitive spot detection - catches ANY brown deviation.
        Uses percentile-based detection instead of fixed thresholds.

        Returns:
            Tuple of (filtered spots list, count before filtering)
        """
        mean_l, mean_a, mean_b = mean_skin_lab

        # Convert image to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Calculate deviation from mean skin tone
        delta_b = lab[:, :, 2] - mean_b  # Positive = more brown/yellow
        delta_l = mean_l - lab[:, :, 0]  # Positive = darker than mean

        # Combined score for each pixel
        combined_score = (delta_l * 0.4) + (delta_b * 0.6)

        # Apply skin mask
        skin_pixels_mask = skin_mask > 0

        # Get scores only for skin pixels
        skin_scores = combined_score[skin_pixels_mask]

        if len(skin_scores) == 0:
            return [], 0

        # HYPER-SENSITIVE: Use config thresholds to derive sensitivity
        # Calculate base threshold from config (scaled down for hyper-sensitivity)
        # Lower config values = lower threshold = more sensitive
        base_threshold = (self.config.spot_l_threshold * 0.4 + self.config.spot_b_threshold * 0.6) / 8.0
        min_threshold = max(0.15, base_threshold)  # Lower floor for more sensitivity

        # Also use percentile to catch top deviating pixels
        # Adjust percentile based on sensitivity (lower threshold = lower percentile = more spots)
        percentile_cutoff = 75 + (base_threshold * 4)  # Range ~75-95 depending on sensitivity
        percentile_cutoff = min(95, max(75, percentile_cutoff))
        percentile_threshold = np.percentile(skin_scores, percentile_cutoff)

        print(f"Hyper-sensitive thresholds: min={min_threshold:.2f}, percentile_cutoff={percentile_cutoff:.1f}%, effective={min(min_threshold, percentile_threshold):.2f}")

        # Use the lower of the two thresholds for maximum sensitivity
        effective_threshold = min(min_threshold, percentile_threshold)

        # Create spot mask: any pixel above threshold AND positive delta_B (not shadow)
        spot_mask = np.zeros_like(skin_mask)
        potential_spots = (combined_score > effective_threshold) & skin_pixels_mask & (delta_b > -1)
        spot_mask[potential_spots] = 255

        # Light morphological cleanup (smaller kernel to preserve small spots)
        kernel = np.ones((2, 2), np.uint8)
        spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, kernel)
        spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_CLOSE, kernel)

        # Find connected components
        contours, _ = cv2.findContours(
            spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        spots_before_filter = len(contours)
        spots = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Very permissive area filter
            if area < 5:  # Minimum 5 pixels
                continue
            if area > self.config.max_spot_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = BoundingBox(x=x, y=y, width=w, height=h)

            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            # Calculate average color delta for this spot
            spot_region_mask = np.zeros_like(skin_mask)
            cv2.drawContours(spot_region_mask, [contour], -1, 255, -1)

            spot_pixels = spot_region_mask > 0
            avg_delta_l = float(np.mean(delta_l[spot_pixels])) if np.any(spot_pixels) else 0
            avg_delta_b = float(np.mean(delta_b[spot_pixels])) if np.any(spot_pixels) else 0
            combined_delta = (avg_delta_l * 0.4) + (avg_delta_b * 0.6)

            # Minimal filtering: only remove clear shadows
            if avg_delta_b < -2 and avg_delta_l < 10:
                continue  # Clear shadow

            # Classify spot
            classification = self._classify_spot(combined_delta)

            spot = DetectedSpot(
                id=len(spots) + 1,
                bbox=bbox,
                center=(cx, cy),
                area=int(area),
                classification=classification,
                color_delta_l=avg_delta_l,
                color_delta_b=avg_delta_b,
                combined_delta=combined_delta,
                contour=contour,
            )
            spots.append(spot)

        # Sort by combined delta
        spots.sort(key=lambda s: s.combined_delta, reverse=True)

        # Re-assign IDs
        for i, spot in enumerate(spots):
            spot.id = i + 1

        return spots, spots_before_filter

    def _detect_spots_blob(
        self,
        image: np.ndarray,
        skin_mask: np.ndarray,
        mean_skin_lab: tuple[float, float, float],
    ) -> tuple[list[DetectedSpot], int]:
        """
        Detect spots using multi-scale blob detection (Laplacian of Gaussian).

        LoG finds blobs at multiple scales - perfect for detecting spots of varying sizes.

        IMPORTANT: We erode the skin mask and smooth non-skin areas to prevent
        false positives at skin boundaries (hair, clothing, background).

        Returns:
            Tuple of (filtered spots list, count before filtering)
        """
        mean_l, mean_a, mean_b = mean_skin_lab

        # Convert to LAB for analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # STEP 1: Erode skin mask AGGRESSIVELY to stay away from boundaries
        # Increased erosion to prevent hair edge and boundary artifacts
        erode_kernel = np.ones((11, 11), np.uint8)  # Larger kernel
        eroded_skin_mask = cv2.erode(skin_mask, erode_kernel, iterations=3)  # More iterations

        if np.sum(eroded_skin_mask > 0) < 1000:
            print("Warning: Eroded skin mask too small, using lighter erosion")
            erode_kernel = np.ones((5, 5), np.uint8)
            eroded_skin_mask = cv2.erode(skin_mask, erode_kernel, iterations=2)

        # Create a "spot likelihood" image
        # Higher values = more likely to be a pigmentation spot
        delta_b = lab[:, :, 2] - mean_b  # Positive = more brown
        delta_l = mean_l - lab[:, :, 0]  # Positive = darker

        # Combined score - spots are darker AND/OR more brown
        spot_likelihood = (delta_l * 0.4) + (delta_b * 0.6)

        # STEP 2: Handle non-skin areas WITHOUT creating sharp edges
        # Instead of setting to -100 (creates edges), set to median skin value
        # and apply blur to smooth the transition
        skin_pixels = spot_likelihood[eroded_skin_mask > 0]
        if len(skin_pixels) > 0:
            median_value = np.median(skin_pixels)
        else:
            return [], 0

        # Create smoothed version - set non-skin to median (neutral) value
        spot_likelihood_smooth = spot_likelihood.copy()
        spot_likelihood_smooth[eroded_skin_mask == 0] = median_value

        # Apply Gaussian blur to smooth any remaining boundary effects
        spot_likelihood_smooth = cv2.GaussianBlur(
            spot_likelihood_smooth.astype(np.float32), (5, 5), 1.5
        )

        # Normalize to 0-1 range for blob detection (only using skin region stats)
        min_val = np.percentile(skin_pixels, 5)  # 5th percentile
        max_val = np.percentile(skin_pixels, 95)  # 95th percentile

        if max_val - min_val < 0.5:
            print("Warning: Low variance in skin region")
            return [], 0

        normalized = (spot_likelihood_smooth - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)

        # STEP 3: Zero out non-skin areas AFTER normalization
        # (won't create false edges because values are already smoothed)
        normalized[eroded_skin_mask == 0] = 0.5  # Set to neutral 0.5, not 0

        # Calculate sensitivity-based thresholds from config
        # Lower config values = more sensitive detection
        base_sensitivity = (self.config.spot_l_threshold * 0.4 + self.config.spot_b_threshold * 0.6)

        # LoG threshold: lower = more blobs detected (range ~0.03 to 0.10)
        log_threshold = 0.03 + (base_sensitivity * 0.008)
        log_threshold = max(0.03, min(0.12, log_threshold))  # Clamp to reasonable range

        # Combined delta threshold for filtering
        combined_delta_threshold = base_sensitivity / 3.0
        combined_delta_threshold = max(0.5, min(3.0, combined_delta_threshold))

        print(f"Blob detection sensitivity: base={base_sensitivity:.2f}, LoG_thresh={log_threshold:.3f}, delta_thresh={combined_delta_threshold:.2f}")

        # Run blob detection at multiple scales
        print(f"Running LoG blob detection (threshold={log_threshold:.3f}, based on sensitivity)...")
        blobs_log = blob_log(
            normalized,
            min_sigma=3,      # Min ~6 pixel diameter spots
            max_sigma=25,     # Max ~50 pixel diameter
            num_sigma=10,     # Number of scales
            threshold=log_threshold,   # Sensitivity-adjusted threshold
            overlap=0.3,
        )

        print(f"LoG found {len(blobs_log)} blobs")

        # Use only LoG results (DoG was adding too many duplicates)
        all_blobs = []

        # Add LoG blobs (format: y, x, sigma)
        for blob in blobs_log:
            y, x, sigma = blob
            radius = sigma * np.sqrt(2)  # LoG radius formula
            all_blobs.append((int(y), int(x), radius, 'log'))

        print(f"Total blobs before filtering: {len(all_blobs)}")

        spots_before_filter = len(all_blobs)
        spots = []

        for blob_y, blob_x, radius, method in all_blobs:
            # Check if blob center is in skin region
            if blob_y < 0 or blob_y >= skin_mask.shape[0]:
                continue
            if blob_x < 0 or blob_x >= skin_mask.shape[1]:
                continue
            if skin_mask[blob_y, blob_x] == 0:
                continue

            # Create circular mask for this blob
            r = max(int(radius), 2)
            y_min = max(0, blob_y - r)
            y_max = min(skin_mask.shape[0], blob_y + r + 1)
            x_min = max(0, blob_x - r)
            x_max = min(skin_mask.shape[1], blob_x + r + 1)

            # Create blob mask
            blob_mask = np.zeros_like(skin_mask)
            cv2.circle(blob_mask, (blob_x, blob_y), r, 255, -1)
            blob_mask = cv2.bitwise_and(blob_mask, skin_mask)

            area = int(np.sum(blob_mask > 0))
            if area < 3:
                continue

            # Calculate color properties for this blob
            mask_pixels = blob_mask > 0
            avg_delta_l = float(np.mean(delta_l[mask_pixels]))
            avg_delta_b = float(np.mean(delta_b[mask_pixels]))
            combined_delta = (avg_delta_l * 0.4) + (avg_delta_b * 0.6)

            # Get absolute L value for hair detection
            avg_L = float(np.mean(lab[:, :, 0][mask_pixels]))

            # FILTERING for blob detection:

            # 1. HAIR FILTER (IMPROVED): Multiple conditions to catch hair
            #    Hair characteristics: very dark, not brownish, high contrast with skin
            is_likely_hair = False

            # Case A: Very dark absolute value with extreme darkness delta
            if avg_L < 90 and avg_delta_l > 18:
                is_likely_hair = True

            # Case B: Extremely dark (hair is usually < 70 in L channel)
            if avg_L < 70:
                is_likely_hair = True

            # Case C: Dark AND not brownish (negative or neutral B delta)
            if avg_L < 100 and avg_delta_l > 15 and avg_delta_b < 3:
                is_likely_hair = True

            # Case D: Check if near mask boundary (edge proximity filter)
            # Hair often appears at the boundary between skin and non-skin
            boundary_check_radius = max(r + 5, 10)
            y_check_min = max(0, blob_y - boundary_check_radius)
            y_check_max = min(skin_mask.shape[0], blob_y + boundary_check_radius)
            x_check_min = max(0, blob_x - boundary_check_radius)
            x_check_max = min(skin_mask.shape[1], blob_x + boundary_check_radius)

            region_around = skin_mask[y_check_min:y_check_max, x_check_min:x_check_max]
            if region_around.size > 0:
                skin_ratio = np.sum(region_around > 0) / region_around.size
                # If less than 70% of surrounding area is skin, likely at boundary
                if skin_ratio < 0.7 and avg_delta_l > 12:
                    is_likely_hair = True

            if is_likely_hair:
                continue  # Skip hair/boundary artifacts

            # 2. Must have minimum combined deviation (not just noise)
            #    Threshold derived from config sensitivity (lower = more sensitive)
            combined_delta_threshold = base_sensitivity / 3.0  # Range ~0.7 to 3.0
            combined_delta_threshold = max(0.5, min(3.0, combined_delta_threshold))
            if combined_delta < combined_delta_threshold:
                continue  # Too weak to be a real spot

            # 3. Must be actually brownish (positive delta_b) OR moderately darker
            #    Shadows are dark but have negative delta_b
            if avg_delta_b < -1 and avg_delta_l < 10:
                continue  # Shadow or noise, not pigmentation

            # 4. Apply min_spot_area from config
            if area < self.config.min_spot_area:
                continue

            # Create bounding box
            bbox = BoundingBox(
                x=x_min,
                y=y_min,
                width=x_max - x_min,
                height=y_max - y_min,
            )

            # Get contour for visualization
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0] if contours else None

            # Classify
            classification = self._classify_spot(combined_delta)

            spot = DetectedSpot(
                id=len(spots) + 1,
                bbox=bbox,
                center=(blob_x, blob_y),
                area=area,
                classification=classification,
                color_delta_l=avg_delta_l,
                color_delta_b=avg_delta_b,
                combined_delta=combined_delta,
                contour=contour,
            )
            spots.append(spot)

        print(f"After filtering: {len(spots)} spots")

        # Sort by combined delta
        spots.sort(key=lambda s: s.combined_delta, reverse=True)

        # Re-assign IDs
        for i, spot in enumerate(spots):
            spot.id = i + 1

        return spots, spots_before_filter

    def _classify_spot(self, combined_delta: float) -> SpotClassification:
        """Classify spot based on combined color delta."""
        if combined_delta >= self.config.dark_min:
            return SpotClassification.DARK
        elif combined_delta >= self.config.medium_min:
            return SpotClassification.MEDIUM
        else:
            return SpotClassification.LIGHT

    def _config_to_dict(self) -> dict:
        """Convert config to dictionary for reporting."""
        return {
            "spot_b_threshold": self.config.spot_b_threshold,
            "spot_l_threshold": self.config.spot_l_threshold,
            "light_max": self.config.light_max,
            "medium_max": self.config.medium_max,
            "dark_min": self.config.dark_min,
            "min_spot_area": self.config.min_spot_area,
            "max_spot_area": self.config.max_spot_area,
        }

    def close(self):
        """Release resources."""
        self.face_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()