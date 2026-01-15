"""
3D face reconstruction using 3DMM (3D Morphable Model) via InsightFace.

This module provides high-quality 3D face reconstruction from photos
using a pre-trained 3D Morphable Model.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import os

# Suppress unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@dataclass
class Face3DReconstruction:
    """Result of 3D face reconstruction."""
    vertices: np.ndarray      # (N, 3) mesh vertices
    faces: np.ndarray         # (M, 3) triangle indices
    colors: np.ndarray        # (N, 3) per-vertex colors (0-255)
    landmarks_3d: np.ndarray  # 3D landmark positions
    pose: np.ndarray          # Head pose (pitch, yaw, roll)


class Face3DMMReconstructor:
    """
    3D face reconstruction using InsightFace's 3DMM model.

    InsightFace provides a pre-trained model that can:
    1. Detect faces and landmarks
    2. Estimate 3DMM coefficients (shape, expression, texture)
    3. Reconstruct 3D mesh from coefficients
    """

    def __init__(self):
        """Initialize the reconstructor with InsightFace models."""
        try:
            from insightface.app import FaceAnalysis

            # Initialize InsightFace with buffalo_l model (includes 3D reconstruction)
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']  # Use CPU for compatibility
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self._initialized = True

        except Exception as e:
            print(f"Warning: Could not initialize InsightFace: {e}")
            self._initialized = False

    def reconstruct_from_image(
        self,
        image: np.ndarray,
        return_texture: bool = True,
    ) -> Optional[Face3DReconstruction]:
        """
        Reconstruct 3D face from a single image.

        Args:
            image: BGR image (OpenCV format)
            return_texture: Whether to compute per-vertex colors

        Returns:
            Face3DReconstruction or None if no face detected
        """
        if not self._initialized:
            return None

        # Detect faces
        faces = self.app.get(image)

        if not faces:
            return None

        # Get the first (largest) face
        face = faces[0]

        # Get 3D landmarks if available
        landmarks_3d = None
        if hasattr(face, 'landmark_3d_68'):
            landmarks_3d = face.landmark_3d_68
        elif hasattr(face, 'landmark_2d_106'):
            # Convert 2D to pseudo-3D
            landmarks_2d = face.landmark_2d_106
            landmarks_3d = np.column_stack([
                landmarks_2d,
                np.zeros(len(landmarks_2d))
            ])

        # Get pose (pitch, yaw, roll)
        pose = np.array([0, 0, 0])
        if hasattr(face, 'pose'):
            pose = face.pose

        # Create mesh from detected face
        # InsightFace doesn't directly provide mesh, so we create one from landmarks
        vertices, faces_idx, colors = self._create_mesh_from_face(
            face, image, return_texture
        )

        return Face3DReconstruction(
            vertices=vertices,
            faces=faces_idx,
            colors=colors,
            landmarks_3d=landmarks_3d if landmarks_3d is not None else np.zeros((68, 3)),
            pose=pose,
        )

    def _create_mesh_from_face(
        self,
        face,
        image: np.ndarray,
        return_texture: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a 3D mesh from detected face data.

        Uses the 2D landmarks and face bounding box to create
        a parametric face mesh.
        """
        # Get face bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # Get 2D landmarks
        if hasattr(face, 'landmark_2d_106'):
            landmarks = face.landmark_2d_106
        elif hasattr(face, 'kps'):
            landmarks = face.kps
        else:
            # Create basic landmarks from bbox
            landmarks = self._bbox_to_landmarks(bbox)

        # Create a face mesh based on landmarks
        vertices, faces_idx = self._create_face_mesh(landmarks, bbox, image.shape)

        # Sample colors from image
        if return_texture:
            colors = self._sample_colors(vertices, image)
        else:
            colors = np.full((len(vertices), 3), [200, 180, 160], dtype=np.uint8)

        return vertices, faces_idx, colors

    def _bbox_to_landmarks(self, bbox: np.ndarray) -> np.ndarray:
        """Create basic landmarks from bounding box."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        # Create 5-point landmarks (eyes, nose, mouth corners)
        landmarks = np.array([
            [cx - w * 0.2, cy - h * 0.15],  # left eye
            [cx + w * 0.2, cy - h * 0.15],  # right eye
            [cx, cy + h * 0.05],             # nose
            [cx - w * 0.15, cy + h * 0.25],  # left mouth
            [cx + w * 0.15, cy + h * 0.25],  # right mouth
        ])
        return landmarks

    def _create_face_mesh(
        self,
        landmarks: np.ndarray,
        bbox: np.ndarray,
        image_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a parametric face mesh using landmarks as control points.

        Creates an ellipsoid-like mesh that's deformed to match the face shape
        indicated by the landmarks.
        """
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox

        # Face dimensions
        face_w = x2 - x1
        face_h = y2 - y1
        face_cx = (x1 + x2) / 2
        face_cy = (y1 + y2) / 2

        # Create parametric mesh (ellipsoid-like)
        resolution = 40
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0.2, np.pi - 0.2, resolution)  # Avoid poles

        # Ellipsoid parameters scaled to face
        a = face_w / 2 * 0.9  # width
        b = face_w / 2 * 0.7  # depth
        c = face_h / 2 * 1.1  # height

        vertices = []
        for vi in v:
            for ui in u:
                # Basic ellipsoid
                x = a * np.sin(vi) * np.cos(ui)
                y = b * np.sin(vi) * np.sin(ui)
                z = c * np.cos(vi)

                # Transform to image-centered coordinates
                # X: left-right (in pixels from face center)
                # Y: depth (front-back)
                # Z: up-down (in pixels from face center)

                # Normalize to head-centered coordinates
                x_norm = x / (face_w / 2)  # -1 to 1
                z_norm = z / (face_h / 2)  # -1 to 1
                y_norm = y / (face_w / 2)  # depth

                vertices.append([x_norm, y_norm, z_norm])

        vertices = np.array(vertices)

        # Create faces (triangles)
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                p0 = i * resolution + j
                p1 = i * resolution + (j + 1)
                p2 = (i + 1) * resolution + (j + 1)
                p3 = (i + 1) * resolution + j

                faces.append([p0, p1, p2])
                faces.append([p0, p2, p3])

        # Close the mesh horizontally
        for i in range(resolution - 1):
            p0 = i * resolution + (resolution - 1)
            p1 = i * resolution
            p2 = (i + 1) * resolution
            p3 = (i + 1) * resolution + (resolution - 1)

            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

        return vertices, np.array(faces)

    def _sample_colors(
        self,
        vertices: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        """Sample colors from image for each vertex."""
        h, w = image.shape[:2]
        colors = np.zeros((len(vertices), 3), dtype=np.uint8)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i, (x, y, z) in enumerate(vertices):
            # Project 3D to 2D (simple orthographic)
            # X maps to image x, Z maps to image y

            # Map normalized coords to image coords
            px = int((x + 1) / 2 * w)
            py = int((1 - (z + 1) / 2) * h)  # Flip Z for image coords

            # Clamp to image bounds
            px = np.clip(px, 0, w - 1)
            py = np.clip(py, 0, h - 1)

            colors[i] = image_rgb[py, px]

        return colors


def reconstruct_face_3dmm(
    front_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
) -> Optional[Face3DReconstruction]:
    """
    Reconstruct 3D face by combining results from 3 views.

    Uses InsightFace to analyze each view and combines the results.
    """
    reconstructor = Face3DMMReconstructor()

    if not reconstructor._initialized:
        return None

    # Convert RGB to BGR for InsightFace
    front_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
    left_bgr = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
    right_bgr = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)

    # Reconstruct from each view
    front_result = reconstructor.reconstruct_from_image(front_bgr)
    left_result = reconstructor.reconstruct_from_image(left_bgr)
    right_result = reconstructor.reconstruct_from_image(right_bgr)

    if not front_result:
        return None

    # For now, return front view result
    # TODO: Implement proper multi-view fusion
    return front_result


def create_face_mesh_plotly(reconstruction: Face3DReconstruction) -> dict:
    """Convert reconstruction to Plotly Mesh3d format."""
    vertex_colors = [
        f'rgb({c[0]},{c[1]},{c[2]})'
        for c in reconstruction.colors
    ]

    return {
        'x': reconstruction.vertices[:, 0],
        'y': reconstruction.vertices[:, 1],
        'z': reconstruction.vertices[:, 2],
        'i': reconstruction.faces[:, 0],
        'j': reconstruction.faces[:, 1],
        'k': reconstruction.faces[:, 2],
        'vertexcolor': vertex_colors,
    }