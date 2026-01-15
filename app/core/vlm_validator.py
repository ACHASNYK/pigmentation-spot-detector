"""
VLM (Vision-Language Model) validation for detected spots.
Uses Claude to validate whether detected spots are real pigmentation or artifacts.

Zone-based approach:
- SAFE ZONE: Spots clearly on face center - keep without AI
- HAIRLINE ZONE: Spots at face edges - reject without AI
- AMBIGUOUS ZONE: Spots near edges - verify with AI individually
"""

import base64
import cv2
import numpy as np
from dataclasses import dataclass
from enum import Enum
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from app.core.models import DetectedSpot


class SpotZone(Enum):
    SAFE = "safe"           # Center of face - definitely real spots
    AMBIGUOUS = "ambiguous" # Near edges - need AI verification
    HAIRLINE = "hairline"   # At face edges - definitely artifacts


@dataclass
class ValidationResult:
    """Result of VLM validation for a spot."""
    spot_id: int
    is_valid_spot: bool
    confidence: str  # "high", "medium", "low"
    spot_type: str  # "freckle", "mole", "age_spot", "artifact", "unknown"
    reason: str


def is_vlm_available() -> bool:
    """Check if VLM validation is available (API key set)."""
    if not ANTHROPIC_AVAILABLE:
        return False
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    return api_key is not None and len(api_key) > 0


def classify_spot_zone(
    spot: DetectedSpot,
    face_bbox: dict,
    image_width: int,
    image_height: int,
) -> tuple[SpotZone, str]:
    """
    Classify spot into zones based on position RELATIVE TO FACE (not image).

    Args:
        spot: Detected spot
        face_bbox: Face bounding box with x, y, width, height
        image_width: Full image width
        image_height: Full image height

    Returns:
        Tuple of (zone, reason)
    """
    cx, cy = spot.center

    # Calculate position relative to FACE bounding box
    face_x = face_bbox.get('x', 0)
    face_y = face_bbox.get('y', 0)
    face_w = face_bbox.get('width', image_width)
    face_h = face_bbox.get('height', image_height)

    # Relative position within face (0-1)
    rel_x = (cx - face_x) / face_w if face_w > 0 else 0.5
    rel_y = (cy - face_y) / face_h if face_h > 0 else 0.5

    # Clamp to valid range
    rel_x = max(0, min(1, rel_x))
    rel_y = max(0, min(1, rel_y))

    # HAIRLINE ZONE - definitely artifacts (very edge of face)
    # Top 5% of face (very top edge)
    if rel_y < 0.05:
        return SpotZone.HAIRLINE, f"At very top edge (y={rel_y:.2f})"

    # Top corners (temple area) - top 12%, outer 12%
    if rel_y < 0.12 and (rel_x < 0.12 or rel_x > 0.88):
        return SpotZone.HAIRLINE, f"At temple corner (x={rel_x:.2f}, y={rel_y:.2f})"

    # Side edges - outer 5% only (very edge)
    if rel_x < 0.05 or rel_x > 0.95:
        return SpotZone.HAIRLINE, f"At side edge (x={rel_x:.2f})"

    # SAFE ZONE - definitely real spots (most of face area)
    # Middle 80% horizontally AND below top 10% AND above bottom 5%
    # This includes forehead, cheeks, nose, chin - basically all face skin
    if 0.10 < rel_x < 0.90 and 0.10 < rel_y < 0.95:
        return SpotZone.SAFE, f"In face area (x={rel_x:.2f}, y={rel_y:.2f})"

    # AMBIGUOUS ZONE - need AI verification (only narrow edge bands)
    # Upper forehead band (between hairline and safe zone)
    if rel_y < 0.10:
        return SpotZone.AMBIGUOUS, f"Upper forehead edge (y={rel_y:.2f})"

    # Side bands (between edge and safe zone)
    if rel_x < 0.10 or rel_x > 0.90:
        return SpotZone.AMBIGUOUS, f"Near face edge (x={rel_x:.2f})"

    # Bottom chin area
    if rel_y > 0.95:
        return SpotZone.AMBIGUOUS, f"Chin edge (y={rel_y:.2f})"

    # Default to safe if nothing else matches
    return SpotZone.SAFE, f"Within face (x={rel_x:.2f}, y={rel_y:.2f})"


def encode_image_to_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode BGR image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.standard_b64encode(buffer).decode('utf-8')


def extract_spot_region(
    image: np.ndarray,
    spot: DetectedSpot,
    padding: int = 60,
    min_size: int = 100,
) -> np.ndarray:
    """Extract a region around the spot for VLM analysis."""
    h, w = image.shape[:2]
    bbox = spot.bbox

    x1 = max(0, bbox.x - padding)
    y1 = max(0, bbox.y - padding)
    x2 = min(w, bbox.x + bbox.width + padding)
    y2 = min(h, bbox.y + bbox.height + padding)

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < min_size:
        expand = (min_size - crop_w) // 2
        x1 = max(0, x1 - expand)
        x2 = min(w, x2 + expand)

    if crop_h < min_size:
        expand = (min_size - crop_h) // 2
        y1 = max(0, y1 - expand)
        y2 = min(h, y2 + expand)

    return image[y1:y2, x1:x2].copy()


def create_context_image(
    image: np.ndarray,
    spot: DetectedSpot,
    max_size: int = 600,
) -> np.ndarray:
    """Create context image with spot marked by bright circle."""
    h, w = image.shape[:2]

    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        context_img = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        context_img = image.copy()

    cx = int(spot.center[0] * scale)
    cy = int(spot.center[1] * scale)
    radius = max(20, int(max(spot.bbox.width, spot.bbox.height) * scale / 2) + 15)

    # Draw bright green circle with crosshairs
    cv2.circle(context_img, (cx, cy), radius, (0, 255, 0), 3)
    cv2.line(context_img, (cx - radius - 10, cy), (cx + radius + 10, cy), (0, 255, 0), 2)
    cv2.line(context_img, (cx, cy - radius - 10), (cx, cy + radius + 10), (0, 255, 0), 2)

    return context_img


def validate_single_spot_with_ai(
    client: "anthropic.Anthropic",
    image: np.ndarray,
    spot: DetectedSpot,
    zone_reason: str,
) -> ValidationResult:
    """
    Validate a single ambiguous spot using Claude vision.
    Sends TWO images: context (full face) + closeup.
    """
    # Create images
    context_img = create_context_image(image, spot, max_size=500)
    closeup = extract_spot_region(image, spot, padding=80, min_size=120)

    # Encode
    context_b64 = encode_image_to_base64(context_img, quality=80)
    closeup_b64 = encode_image_to_base64(closeup, quality=85)

    prompt = f"""Analyze this face image. The GREEN CIRCLE marks a detected area that may be a pigmentation spot.

LOCATION CONTEXT: {zone_reason}

I'm showing you:
1. FIRST IMAGE: Full face with the spot marked by GREEN CIRCLE + crosshairs
2. SECOND IMAGE: Closeup of the marked area

Determine if this is:
- A REAL PIGMENTATION SPOT (freckle, mole, age spot) - round/oval brown mark ON THE SKIN
- A FALSE POSITIVE (hair strand, shadow, skin fold, noise)

CRITICAL: Look carefully at:
- Is there visible HAIR near the marked area? Hair strands are thin dark lines.
- Is this ON SKIN or at the boundary where hair meets skin?
- Does it have the round/oval shape typical of pigmentation spots?

Respond in EXACTLY this format:
VERDICT: SPOT or ARTIFACT
TYPE: freckle/mole/age_spot/hair/shadow/other
REASON: One sentence explanation"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": context_b64,
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": closeup_b64,
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        response_text = response.content[0].text

        # Parse response
        verdict = "SPOT"
        spot_type = "unknown"
        reason = "Could not parse response"

        for line in response_text.split('\n'):
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                verdict = "ARTIFACT" if "ARTIFACT" in v else "SPOT"
            elif line.upper().startswith("TYPE:"):
                spot_type = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        is_valid = verdict == "SPOT"

        return ValidationResult(
            spot_id=spot.id,
            is_valid_spot=is_valid,
            confidence="high" if is_valid else "medium",
            spot_type=spot_type,
            reason=f"AI: {reason}",
        )

    except Exception as e:
        # On error, keep the spot (conservative)
        return ValidationResult(
            spot_id=spot.id,
            is_valid_spot=True,
            confidence="low",
            spot_type="unknown",
            reason=f"AI error: {str(e)}",
        )


def test_api_access(client: "anthropic.Anthropic") -> tuple[bool, str]:
    """Test if the API is accessible before processing."""
    try:
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say OK"}]
        )
        return True, ""
    except anthropic.BadRequestError as e:
        error_msg = str(e)
        if "credit balance" in error_msg.lower():
            return False, "API credits exhausted. Please add credits at https://console.anthropic.com/settings/billing"
        return False, f"API error: {error_msg}"
    except anthropic.AuthenticationError:
        return False, "Invalid API key. Please check your ANTHROPIC_API_KEY."
    except Exception as e:
        return False, f"API connection failed: {str(e)}"


def validate_spots_batch(
    image: np.ndarray,
    spots: list[DetectedSpot],
    progress_callback=None,
    error_callback=None,
    face_bbox: dict = None,
    max_workers: int = 3,
) -> tuple[list[DetectedSpot], list[ValidationResult]]:
    """
    Validate spots using zone-based approach.

    Zones:
    - SAFE: Keep without AI (center of face)
    - HAIRLINE: Reject without AI (edge of face)
    - AMBIGUOUS: Verify with AI individually
    """
    if not is_vlm_available():
        return spots, []

    if not spots:
        return [], []

    h, w = image.shape[:2]

    # Default face bbox if not provided
    if face_bbox is None:
        face_bbox = {'x': 0, 'y': 0, 'width': w, 'height': h}

    validated_spots = []
    validation_results = []
    ambiguous_spots = []

    # Phase 1: Classify spots into zones
    for spot in spots:
        zone, reason = classify_spot_zone(spot, face_bbox, w, h)

        if zone == SpotZone.SAFE:
            # Keep without AI
            validated_spots.append(spot)
            validation_results.append(ValidationResult(
                spot_id=spot.id,
                is_valid_spot=True,
                confidence="high",
                spot_type="freckle",
                reason=f"Safe zone: {reason}",
            ))
        elif zone == SpotZone.HAIRLINE:
            # Reject without AI
            validation_results.append(ValidationResult(
                spot_id=spot.id,
                is_valid_spot=False,
                confidence="high",
                spot_type="hair",
                reason=f"Hairline zone: {reason}",
            ))
        else:  # AMBIGUOUS
            ambiguous_spots.append((spot, reason))

    safe_count = sum(1 for r in validation_results if r.is_valid_spot)
    hairline_count = len(validation_results) - safe_count

    # Debug logging
    print(f"Zone classification: {len(spots)} total spots")
    print(f"  - SAFE zone (kept): {safe_count}")
    print(f"  - HAIRLINE zone (rejected): {hairline_count}")
    print(f"  - AMBIGUOUS zone (need AI): {len(ambiguous_spots)}")
    print(f"  Face bbox: x={face_bbox.get('x')}, y={face_bbox.get('y')}, "
          f"w={face_bbox.get('width')}, h={face_bbox.get('height')}")

    if progress_callback:
        progress_callback(len(validation_results), len(spots))

    # Phase 2: AI verification for ambiguous spots only
    if ambiguous_spots:
        client = anthropic.Anthropic()

        # Test API access
        api_ok, error_msg = test_api_access(client)
        if not api_ok:
            if error_callback:
                error_callback(error_msg)
            # Keep ambiguous spots if API fails
            for spot, _ in ambiguous_spots:
                validated_spots.append(spot)
            return validated_spots, validation_results

        # Process ambiguous spots with AI (in parallel)
        processed = len(validation_results)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_spot = {
                executor.submit(
                    validate_single_spot_with_ai,
                    client, image, spot, reason
                ): spot
                for spot, reason in ambiguous_spots
            }

            for future in as_completed(future_to_spot):
                spot = future_to_spot[future]
                try:
                    result = future.result()
                    validation_results.append(result)

                    if result.is_valid_spot:
                        validated_spots.append(spot)

                    processed += 1
                    if progress_callback:
                        progress_callback(processed, len(spots))

                except Exception as e:
                    # On error, keep the spot
                    validated_spots.append(spot)
                    validation_results.append(ValidationResult(
                        spot_id=spot.id,
                        is_valid_spot=True,
                        confidence="low",
                        spot_type="unknown",
                        reason=f"Error: {str(e)}",
                    ))

    return validated_spots, validation_results


def create_validation_summary(results: list[ValidationResult]) -> dict:
    """Create a summary of validation results."""
    total = len(results)
    if total == 0:
        return {
            "total_checked": 0,
            "valid_spots": 0,
            "artifacts_filtered": 0,
            "filter_rate": 0.0,
        }

    valid = sum(1 for r in results if r.is_valid_spot)
    artifacts = total - valid

    # Count by zone
    safe_zone = sum(1 for r in results if "Safe zone" in r.reason)
    hairline_zone = sum(1 for r in results if "Hairline zone" in r.reason)
    ai_verified = sum(1 for r in results if r.reason.startswith("AI:"))

    # Count by type
    type_counts = {}
    for r in results:
        type_counts[r.spot_type] = type_counts.get(r.spot_type, 0) + 1

    return {
        "total_checked": total,
        "valid_spots": valid,
        "artifacts_filtered": artifacts,
        "safe_zone": safe_zone,
        "hairline_zone": hairline_zone,
        "ai_verified": ai_verified,
        "filter_rate": (artifacts / total) * 100 if total > 0 else 0,
        "by_type": type_counts,
    }
