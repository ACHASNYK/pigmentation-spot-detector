"""
3D head reconstruction from multi-view photos using landmark-based deformation.

This module reconstructs a personalized 3D head mesh by:
1. Extracting MediaPipe landmarks from front/left/right photos
2. Triangulating 3D landmark positions from multiple views
3. Deforming a template mesh using RBF (Radial Basis Function) interpolation
4. Sampling colors from photos for per-vertex coloring
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay
from dataclasses import dataclass
from typing import Optional
import cv2

from app.utils.visualization_3d import ViewAngle


@dataclass
class ReconstructedMesh:
    """A reconstructed 3D head mesh."""
    vertices: np.ndarray  # (N, 3) vertex positions
    faces: np.ndarray     # (M, 3) triangle indices
    colors: np.ndarray    # (N, 3) per-vertex RGB colors (0-255)
    landmarks_3d: np.ndarray  # (478, 3) 3D landmark positions


def create_template_head_mesh(resolution: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a template head mesh (ellipsoid) with UV mapping.

    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle indices
    """
    # Create ellipsoid parameters (head proportions)
    a = 0.85   # width (ear to ear)
    b = 1.0    # depth (front to back)
    c = 1.3    # height (chin to top)

    # Generate vertices
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0.1, np.pi - 0.1, resolution)  # Avoid poles

    vertices = []
    for vi in v:
        for ui in u:
            x = a * np.sin(vi) * np.cos(ui)
            y = b * np.sin(vi) * np.sin(ui)
            z = c * np.cos(vi)
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    # Generate faces (triangles)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Quad vertices
            p0 = i * resolution + j
            p1 = i * resolution + (j + 1)
            p2 = (i + 1) * resolution + (j + 1)
            p3 = (i + 1) * resolution + j

            # Two triangles per quad
            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    # Wrap around horizontally
    for i in range(resolution - 1):
        p0 = i * resolution + (resolution - 1)
        p1 = i * resolution
        p2 = (i + 1) * resolution
        p3 = (i + 1) * resolution + (resolution - 1)

        faces.append([p0, p1, p2])
        faces.append([p0, p2, p3])

    faces = np.array(faces)

    return vertices, faces


def landmarks_to_3d_positions(
    landmarks_front: np.ndarray,  # (478, 2) normalized coords
    landmarks_left: np.ndarray,
    landmarks_right: np.ndarray,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """
    Estimate 3D landmark positions from 3 orthogonal views.

    Uses a weighted combination approach:
    - X coordinate: primarily from front view, refined by side views
    - Y coordinate: primarily from side views
    - Z coordinate: from vertical position in all views

    Returns:
        landmarks_3d: (478, 3) array of 3D positions
    """
    n_landmarks = landmarks_front.shape[0]
    landmarks_3d = np.zeros((n_landmarks, 3))

    # Head dimensions for scaling
    head_width = 0.85 * 2   # full width
    head_depth = 1.0 * 2    # full depth
    head_height = 1.3 * 2   # full height

    for i in range(n_landmarks):
        # Get normalized coordinates (0-1) for each view
        fx, fy = landmarks_front[i]  # front: x is left-right, y is up-down
        lx, ly = landmarks_left[i]   # left: x is front-back, y is up-down
        rx, ry = landmarks_right[i]  # right: x is back-front, y is up-down

        # X coordinate (left-right): from front view
        # Map 0-1 to -width/2 to +width/2
        x = (fx - 0.5) * head_width

        # Y coordinate (front-back): from side views
        # Left view: x=0 is back, x=1 is front
        # Right view: x=0 is front, x=1 is back
        y_from_left = (lx - 0.5) * head_depth
        y_from_right = (0.5 - rx) * head_depth  # inverted

        # Average the Y estimates (could use visibility weighting)
        y = (y_from_left + y_from_right) / 2

        # Z coordinate (up-down): average from all views
        # y=0 is top, y=1 is bottom in image coords
        z_from_front = (0.5 - fy) * head_height
        z_from_left = (0.5 - ly) * head_height
        z_from_right = (0.5 - ry) * head_height

        z = (z_from_front + z_from_left + z_from_right) / 3

        landmarks_3d[i] = [x, y, z]

    return landmarks_3d


def estimate_landmark_visibility(
    landmark_3d: np.ndarray,
    view: ViewAngle,
) -> float:
    """
    Estimate how visible a landmark is from a given view angle.
    Based on the normal direction of the landmark on the head surface.
    """
    x, y, z = landmark_3d

    # Approximate surface normal (pointing outward from ellipsoid center)
    normal = np.array([x / 0.85**2, y / 1.0**2, z / 1.3**2])
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # View directions
    if view == ViewAngle.FRONT:
        view_dir = np.array([0, -1, 0])  # looking from -Y direction
    elif view == ViewAngle.LEFT:
        view_dir = np.array([1, 0, 0])   # looking from +X direction
    elif view == ViewAngle.RIGHT:
        view_dir = np.array([-1, 0, 0])  # looking from -X direction
    else:
        view_dir = np.array([0, -1, 0])

    # Visibility is dot product (1 = fully visible, 0 = edge, -1 = occluded)
    visibility = np.dot(normal, view_dir)
    return max(0, visibility)


def triangulate_landmarks_weighted(
    landmarks_front: np.ndarray,
    landmarks_left: np.ndarray,
    landmarks_right: np.ndarray,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """
    Triangulate 3D landmarks using visibility-weighted combination.

    More sophisticated than simple averaging - weights each view's
    contribution based on estimated visibility.
    """
    n_landmarks = landmarks_front.shape[0]
    landmarks_3d = np.zeros((n_landmarks, 3))

    # Head dimensions
    head_width = 0.85 * 2
    head_depth = 1.0 * 2
    head_height = 1.3 * 2

    # First pass: get rough 3D positions
    rough_3d = landmarks_to_3d_positions(
        landmarks_front, landmarks_left, landmarks_right,
        img_width, img_height
    )

    # Second pass: refine with visibility weighting
    for i in range(n_landmarks):
        fx, fy = landmarks_front[i]
        lx, ly = landmarks_left[i]
        rx, ry = landmarks_right[i]

        # Estimate visibility from each view
        vis_front = estimate_landmark_visibility(rough_3d[i], ViewAngle.FRONT)
        vis_left = estimate_landmark_visibility(rough_3d[i], ViewAngle.LEFT)
        vis_right = estimate_landmark_visibility(rough_3d[i], ViewAngle.RIGHT)

        # Add small epsilon to avoid division by zero
        total_vis = vis_front + vis_left + vis_right + 1e-8

        # X coordinate: front view is most reliable
        x_front = (fx - 0.5) * head_width
        landmarks_3d[i, 0] = x_front

        # Y coordinate: side views weighted by visibility
        y_left = (lx - 0.5) * head_depth
        y_right = (0.5 - rx) * head_depth

        if vis_left + vis_right > 0.1:
            landmarks_3d[i, 1] = (y_left * vis_left + y_right * vis_right) / (vis_left + vis_right + 1e-8)
        else:
            # Fall back to front view depth estimation (less accurate)
            landmarks_3d[i, 1] = 0.3  # slight forward bias for face

        # Z coordinate: weighted average
        z_front = (0.5 - fy) * head_height
        z_left = (0.5 - ly) * head_height
        z_right = (0.5 - ry) * head_height

        landmarks_3d[i, 2] = (
            z_front * vis_front +
            z_left * vis_left +
            z_right * vis_right
        ) / total_vis

    return landmarks_3d


def deform_mesh_rbf(
    template_vertices: np.ndarray,
    source_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    smoothing: float = 0.1,
) -> np.ndarray:
    """
    Deform template mesh vertices using RBF interpolation.

    Args:
        template_vertices: (N, 3) original mesh vertices
        source_landmarks: (K, 3) landmark positions on template
        target_landmarks: (K, 3) target landmark positions
        smoothing: RBF smoothing parameter (higher = smoother)

    Returns:
        deformed_vertices: (N, 3) deformed mesh vertices
    """
    # Compute displacement vectors at landmarks
    displacements = target_landmarks - source_landmarks

    # Create RBF interpolators for each axis
    rbf_x = RBFInterpolator(
        source_landmarks, displacements[:, 0],
        kernel='thin_plate_spline',
        smoothing=smoothing,
    )
    rbf_y = RBFInterpolator(
        source_landmarks, displacements[:, 1],
        kernel='thin_plate_spline',
        smoothing=smoothing,
    )
    rbf_z = RBFInterpolator(
        source_landmarks, displacements[:, 2],
        kernel='thin_plate_spline',
        smoothing=smoothing,
    )

    # Interpolate displacements for all vertices
    dx = rbf_x(template_vertices)
    dy = rbf_y(template_vertices)
    dz = rbf_z(template_vertices)

    # Apply deformation
    deformed_vertices = template_vertices.copy()
    deformed_vertices[:, 0] += dx
    deformed_vertices[:, 1] += dy
    deformed_vertices[:, 2] += dz

    return deformed_vertices


def get_template_landmark_positions(n_landmarks: int = 478) -> np.ndarray:
    """
    Get approximate 3D positions of MediaPipe landmarks on the template ellipsoid.

    This creates a mapping from landmark indices to positions on the template.
    Uses the canonical face model layout.
    """
    # For now, create approximate positions based on face region
    # In a full implementation, this would use the actual MediaPipe face mesh topology

    landmarks = np.zeros((n_landmarks, 3))

    # Key landmark indices (from MediaPipe)
    # Face oval
    face_oval_indices = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]

    # Distribute landmarks on face region of ellipsoid
    # This is a simplified approximation
    for i in range(n_landmarks):
        # Map index to spherical coordinates on front face
        # Use a pseudo-random but deterministic distribution
        t = i / n_landmarks

        # Concentrate landmarks on front face (y > 0)
        theta = (i % 50) / 50 * np.pi - np.pi/2  # -90 to 90 degrees (front half)
        phi = 0.3 + (i // 50) / 10 * 0.5  # vertical position

        # Convert to Cartesian on ellipsoid
        x = 0.85 * np.sin(phi) * np.cos(theta) * 0.8  # scale down for face region
        y = 1.0 * np.sin(phi) * np.sin(theta) * 0.8
        z = 1.3 * np.cos(phi)

        # Bias towards front
        y = max(y, 0.2)

        landmarks[i] = [x, y, z]

    return landmarks


def sample_vertex_colors(
    vertices: np.ndarray,
    front_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
) -> np.ndarray:
    """
    Sample colors for each vertex from the most appropriate view.

    Args:
        vertices: (N, 3) vertex positions
        front_img: RGB image from front view
        left_img: RGB image from left view
        right_img: RGB image from right view

    Returns:
        colors: (N, 3) RGB colors for each vertex
    """
    n_vertices = vertices.shape[0]
    colors = np.zeros((n_vertices, 3), dtype=np.uint8)

    h, w = front_img.shape[:2]

    # Head dimensions
    head_width = 0.85 * 2
    head_depth = 1.0 * 2
    head_height = 1.3 * 2

    for i in range(n_vertices):
        x, y, z = vertices[i]

        # Determine best view based on vertex position
        vis_front = estimate_landmark_visibility(vertices[i], ViewAngle.FRONT)
        vis_left = estimate_landmark_visibility(vertices[i], ViewAngle.LEFT)
        vis_right = estimate_landmark_visibility(vertices[i], ViewAngle.RIGHT)

        # Project to image coordinates for each view
        # Front view: X -> image x, Z -> image y
        u_front = int((x / head_width + 0.5) * w)
        v_front = int((0.5 - z / head_height) * h)

        # Left view: Y -> image x, Z -> image y
        u_left = int((y / head_depth + 0.5) * w)
        v_left = int((0.5 - z / head_height) * h)

        # Right view: -Y -> image x, Z -> image y
        u_right = int((0.5 - y / head_depth) * w)
        v_right = int((0.5 - z / head_height) * h)

        # Clamp to image bounds
        u_front = np.clip(u_front, 0, w - 1)
        v_front = np.clip(v_front, 0, h - 1)
        u_left = np.clip(u_left, 0, w - 1)
        v_left = np.clip(v_left, 0, h - 1)
        u_right = np.clip(u_right, 0, w - 1)
        v_right = np.clip(v_right, 0, h - 1)

        # Sample colors from each view
        color_front = front_img[v_front, u_front]
        color_left = left_img[v_left, u_left]
        color_right = right_img[v_right, u_right]

        # Weighted blend based on visibility
        total_vis = vis_front + vis_left + vis_right + 1e-8

        blended_color = (
            color_front.astype(float) * vis_front +
            color_left.astype(float) * vis_left +
            color_right.astype(float) * vis_right
        ) / total_vis

        colors[i] = np.clip(blended_color, 0, 255).astype(np.uint8)

    return colors


def reconstruct_head(
    landmarks_front: np.ndarray,
    landmarks_left: np.ndarray,
    landmarks_right: np.ndarray,
    front_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    mesh_resolution: int = 40,
    rbf_smoothing: float = 0.5,
) -> ReconstructedMesh:
    """
    Full 3D head reconstruction pipeline.

    Args:
        landmarks_*: (478, 2) normalized landmark coordinates from each view
        *_img: RGB images from each view
        mesh_resolution: Resolution of template mesh
        rbf_smoothing: Smoothing parameter for RBF deformation

    Returns:
        ReconstructedMesh with vertices, faces, colors, and 3D landmarks
    """
    h, w = front_img.shape[:2]

    # Step 1: Create template mesh
    template_vertices, faces = create_template_head_mesh(mesh_resolution)

    # Step 2: Get template landmark positions
    template_landmarks = get_template_landmark_positions(landmarks_front.shape[0])

    # Step 3: Triangulate target 3D landmarks from views
    target_landmarks = triangulate_landmarks_weighted(
        landmarks_front, landmarks_left, landmarks_right,
        w, h
    )

    # Step 4: Select subset of landmarks for deformation (key facial features)
    # Using face oval and some interior points for stability
    key_indices = list(range(0, 478, 10))  # Every 10th landmark

    source_subset = template_landmarks[key_indices]
    target_subset = target_landmarks[key_indices]

    # Step 5: Deform mesh using RBF
    deformed_vertices = deform_mesh_rbf(
        template_vertices,
        source_subset,
        target_subset,
        smoothing=rbf_smoothing,
    )

    # Step 6: Sample colors from images
    colors = sample_vertex_colors(
        deformed_vertices,
        front_img, left_img, right_img,
    )

    return ReconstructedMesh(
        vertices=deformed_vertices,
        faces=faces,
        colors=colors,
        landmarks_3d=target_landmarks,
    )


def mesh_to_plotly_data(mesh: ReconstructedMesh) -> dict:
    """
    Convert reconstructed mesh to Plotly Mesh3d format.
    """
    # Convert colors to plotly format
    vertex_colors = [
        f'rgb({c[0]},{c[1]},{c[2]})'
        for c in mesh.colors
    ]

    return {
        'x': mesh.vertices[:, 0],
        'y': mesh.vertices[:, 1],
        'z': mesh.vertices[:, 2],
        'i': mesh.faces[:, 0],
        'j': mesh.faces[:, 1],
        'k': mesh.faces[:, 2],
        'vertexcolor': vertex_colors,
    }