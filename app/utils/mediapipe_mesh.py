"""
3D face mesh extraction and reconstruction using MediaPipe.

MediaPipe Face Mesh provides:
- 478 3D landmarks with (x, y, z) coordinates
- Canonical face mesh triangulation
- UV coordinates for texture mapping

This module properly extracts and renders the MediaPipe face mesh.
"""

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path

@dataclass
class MediaPipeMesh:
    """A 3D face mesh extracted from MediaPipe."""
    vertices: np.ndarray      # (478, 3) 3D coordinates
    faces: np.ndarray         # (N, 3) triangle indices
    colors: np.ndarray        # (478, 3) per-vertex RGB colors
    uv_coords: np.ndarray     # (478, 2) UV texture coordinates


def create_triangulation_from_landmarks(
    landmarks_2d: np.ndarray,
    max_edge_length: float = 0.12,
    min_edge_length: float = 0.002,
) -> np.ndarray:
    """
    Create triangulation using Delaunay on actual 2D landmark positions.

    Filters out large triangles that span across face boundaries and
    degenerate triangles that are too small.

    Args:
        landmarks_2d: (N, 2) array of normalized landmark positions
        max_edge_length: Maximum allowed edge length (normalized coords)
        min_edge_length: Minimum edge length to avoid degenerate triangles
    """
    from scipy.spatial import Delaunay

    # Run Delaunay triangulation on the 2D landmark positions
    tri = Delaunay(landmarks_2d)

    # Filter out triangles with edges that are too long or too short
    filtered_triangles = []

    for simplex in tri.simplices:
        # Get the three vertices of this triangle
        v0, v1, v2 = landmarks_2d[simplex[0]], landmarks_2d[simplex[1]], landmarks_2d[simplex[2]]

        # Calculate edge lengths
        edge1 = np.linalg.norm(v1 - v0)
        edge2 = np.linalg.norm(v2 - v1)
        edge3 = np.linalg.norm(v0 - v2)

        # Calculate triangle area (to filter degenerate triangles)
        area = 0.5 * np.abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))

        # Keep triangle only if all edges are in valid range and area is reasonable
        if (edge1 < max_edge_length and edge2 < max_edge_length and edge3 < max_edge_length and
            edge1 > min_edge_length and edge2 > min_edge_length and edge3 > min_edge_length and
            area > 0.00001):
            filtered_triangles.append(simplex)

    return np.array(filtered_triangles)


def extract_mediapipe_mesh(
    image: np.ndarray,
    detector,
) -> Optional[MediaPipeMesh]:
    """
    Extract the 3D face mesh from an image using MediaPipe.

    Args:
        image: BGR image
        detector: MediaPipe FaceDetector instance

    Returns:
        MediaPipeMesh or None if no face detected
    """
    from app.core.face_detector import FaceDetector

    h, w = image.shape[:2]

    # Detect face and get landmarks
    result = detector.detect(image)
    if result is None:
        return None

    # Get 3D landmarks (MediaPipe provides z-depth)
    # landmarks array is (478, 2) normalized, but we can estimate depth

    # For proper 3D, we need to use the full MediaPipe Face Mesh with z coords
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Get the face landmarks with 3D coordinates
    face_landmarks = detector.landmarker.detect(mp_image).face_landmarks

    if not face_landmarks:
        return None

    # Extract 3D vertices
    vertices = []
    for lm in face_landmarks[0]:
        # x, y are normalized (0-1), z is relative depth
        # Scale to reasonable 3D coordinates
        x = (lm.x - 0.5) * 2  # -1 to 1
        y = (lm.y - 0.5) * 2  # -1 to 1 (will flip for proper orientation)
        z = lm.z * 2          # Depth scaling

        vertices.append([x, -z, -y])  # Reorder for proper 3D orientation

    vertices = np.array(vertices)

    # Get triangulation using actual 2D landmark positions
    # This creates proper triangles that follow the face structure
    faces = create_triangulation_from_landmarks(result.landmarks)

    # Sample colors from image
    colors = sample_vertex_colors(vertices, image, result.landmarks)

    # Create UV coordinates (use normalized landmark positions)
    uv_coords = result.landmarks.copy()

    return MediaPipeMesh(
        vertices=vertices,
        faces=faces,
        colors=colors,
        uv_coords=uv_coords,
    )


def sample_vertex_colors(
    vertices: np.ndarray,
    image: np.ndarray,
    landmarks_2d: np.ndarray,
    sample_radius: int = 3,
) -> np.ndarray:
    """
    Sample colors from image for each vertex using 2D landmark positions.
    Uses area averaging for smoother colors instead of single pixel sampling.

    Args:
        vertices: 3D vertex positions
        image: BGR source image
        landmarks_2d: Normalized 2D landmark positions
        sample_radius: Radius for area sampling (pixels)
    """
    h, w = image.shape[:2]
    n_vertices = len(vertices)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    colors = np.zeros((n_vertices, 3), dtype=np.uint8)

    for i in range(min(n_vertices, len(landmarks_2d))):
        # Use 2D landmark positions for sampling
        u, v = landmarks_2d[i]
        px = u * w
        py = v * h

        # Sample area around the point for smoother color
        color_sum = np.zeros(3, dtype=np.float64)
        count = 0

        for dy in range(-sample_radius, sample_radius + 1):
            for dx in range(-sample_radius, sample_radius + 1):
                sx = int(px + dx)
                sy = int(py + dy)

                if 0 <= sx < w and 0 <= sy < h:
                    color_sum += image_rgb[sy, sx].astype(np.float64)
                    count += 1

        if count > 0:
            colors[i] = (color_sum / count).astype(np.uint8)
        else:
            # Fallback to center pixel
            cx = int(np.clip(px, 0, w - 1))
            cy = int(np.clip(py, 0, h - 1))
            colors[i] = image_rgb[cy, cx]

    return colors


def fuse_multi_view_meshes(
    front_mesh: MediaPipeMesh,
    left_mesh: Optional[MediaPipeMesh],
    right_mesh: Optional[MediaPipeMesh],
    front_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
) -> MediaPipeMesh:
    """
    Fuse meshes from multiple views into a single textured mesh.

    Uses visibility-based blending to combine vertex positions and colors
    from different views.
    """
    n_vertices = len(front_mesh.vertices)

    # Start with front mesh as base
    fused_vertices = front_mesh.vertices.copy()
    fused_colors = front_mesh.colors.copy().astype(float)

    # Weight accumulator for color blending
    weights = np.ones(n_vertices)

    # Blend left view if available
    if left_mesh is not None:
        for i in range(n_vertices):
            # Calculate visibility weight based on vertex position
            # Vertices on the left side (negative X) should use left view more
            x = front_mesh.vertices[i, 0]
            left_weight = max(0, -x)  # Higher for left side

            if left_weight > 0.1:
                fused_colors[i] = (
                    fused_colors[i] * weights[i] +
                    left_mesh.colors[i].astype(float) * left_weight
                )
                weights[i] += left_weight

    # Blend right view if available
    if right_mesh is not None:
        for i in range(n_vertices):
            x = front_mesh.vertices[i, 0]
            right_weight = max(0, x)  # Higher for right side

            if right_weight > 0.1:
                fused_colors[i] = (
                    fused_colors[i] * weights[i] +
                    right_mesh.colors[i].astype(float) * right_weight
                )
                weights[i] += right_weight

    # Normalize colors
    for i in range(n_vertices):
        if weights[i] > 0:
            fused_colors[i] /= weights[i]

    fused_colors = np.clip(fused_colors, 0, 255).astype(np.uint8)

    return MediaPipeMesh(
        vertices=fused_vertices,
        faces=front_mesh.faces,
        colors=fused_colors,
        uv_coords=front_mesh.uv_coords,
    )


def mesh_to_plotly(mesh: MediaPipeMesh) -> dict:
    """Convert MediaPipeMesh to Plotly Mesh3d format."""
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


def extend_mesh_with_head(
    mesh: MediaPipeMesh,
    add_forehead: bool = True,
    add_back: bool = True,
    forehead_rows: int = 5,
    back_segments: int = 12,
) -> MediaPipeMesh:
    """
    Extend the face mesh with forehead extension and back of head.

    Args:
        mesh: Original MediaPipe face mesh
        add_forehead: Add extra rows above forehead
        add_back: Add back-of-head ellipsoid cap
        forehead_rows: Number of vertex rows to add for forehead
        back_segments: Resolution for back-of-head cap
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    colors = mesh.colors.copy()
    uv_coords = mesh.uv_coords.copy()

    vertex_offset = len(vertices)

    # Find face dimensions
    x_range = (np.min(vertices[:, 0]), np.max(vertices[:, 0]))
    z_range = (np.min(vertices[:, 2]), np.max(vertices[:, 2]))
    y_min = np.min(vertices[:, 1])

    # Add forehead extension
    if add_forehead:
        # Find top edge vertices (highest Z values)
        z_threshold = z_range[1] - 0.1 * (z_range[1] - z_range[0])
        top_mask = vertices[:, 2] > z_threshold
        top_indices = np.where(top_mask)[0]

        if len(top_indices) > 3:
            top_verts = vertices[top_indices]
            top_colors = colors[top_indices]

            # Sort by X for consistent ordering
            sort_idx = np.argsort(top_verts[:, 0])
            top_verts = top_verts[sort_idx]
            top_colors = top_colors[sort_idx]
            top_orig_idx = top_indices[sort_idx]

            new_verts = []
            new_colors = []
            new_uv = []

            for row in range(1, forehead_rows + 1):
                row_factor = row / forehead_rows
                z_offset = 0.15 * row_factor  # Move up
                # NEGATIVE Y = backward (away from camera, curve of head)
                y_offset = -0.12 * row_factor  # Move BACK (negative Y direction)

                for i, (v, c) in enumerate(zip(top_verts, top_colors)):
                    # Curve inward at edges (head gets narrower at top)
                    x_factor = 1.0 - 0.2 * row_factor * (abs(v[0]) / max(abs(x_range[0]), abs(x_range[1])))

                    new_v = [
                        v[0] * x_factor,
                        v[1] + y_offset,  # Move backward
                        v[2] + z_offset   # Move up
                    ]
                    new_verts.append(new_v)
                    new_colors.append(c)  # Use same color as edge

                    # UV: same X, smaller Y (higher on image)
                    orig_uv = uv_coords[top_orig_idx[i]]
                    new_uv.append([orig_uv[0], max(0, orig_uv[1] - 0.03 * row)])

            if new_verts:
                new_verts = np.array(new_verts)
                new_colors = np.array(new_colors)
                new_uv = np.array(new_uv)

                # Create faces connecting forehead rows
                n_per_row = len(top_verts)
                forehead_faces = []

                # Connect first row to original mesh top edge
                for i in range(n_per_row - 1):
                    orig_i = top_orig_idx[i]
                    orig_i1 = top_orig_idx[i + 1]
                    new_i = vertex_offset + i
                    new_i1 = vertex_offset + i + 1

                    forehead_faces.append([orig_i, new_i, new_i1])
                    forehead_faces.append([orig_i, new_i1, orig_i1])

                # Connect subsequent forehead rows
                for row in range(forehead_rows - 1):
                    base = vertex_offset + row * n_per_row
                    for i in range(n_per_row - 1):
                        p0 = base + i
                        p1 = base + i + 1
                        p2 = base + n_per_row + i + 1
                        p3 = base + n_per_row + i

                        forehead_faces.append([p0, p3, p2])
                        forehead_faces.append([p0, p2, p1])

                vertices = np.vstack([vertices, new_verts])
                colors = np.vstack([colors, new_colors])
                uv_coords = np.vstack([uv_coords, new_uv])
                faces = np.vstack([faces, np.array(forehead_faces)])

                vertex_offset = len(vertices)

    # Add back of head
    if add_back:
        back_verts, back_faces, back_colors = create_back_head_cap(
            vertices, colors, back_segments
        )

        if len(back_verts) > 0:
            vertices = np.vstack([vertices, back_verts])
            colors = np.vstack([colors, back_colors])

            # UV for back (will show skin tone)
            back_uv = np.zeros((len(back_verts), 2))
            back_uv[:, 0] = 0.5  # Center U
            back_uv[:, 1] = 0.5  # Center V
            uv_coords = np.vstack([uv_coords, back_uv])

            # Offset face indices
            back_faces = back_faces + vertex_offset
            faces = np.vstack([faces, back_faces])

    return MediaPipeMesh(
        vertices=vertices,
        faces=faces,
        colors=colors,
        uv_coords=uv_coords,
    )


def create_back_head_cap(
    face_vertices: np.ndarray,
    face_colors: np.ndarray,
    n_segments: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a back-of-head ellipsoid cap that connects seamlessly to face boundary.

    Coordinate system (from MediaPipe transform [x, -z, -y]):
    - X: left (-) to right (+)
    - Y: back (-) to front (+)  <-- positive Y is towards camera/front
    - Z: bottom (-) to top (+)

    Back of head needs NEGATIVE Y values (behind the face).
    """
    # Find face mesh boundary and dimensions
    x_min, x_max = np.min(face_vertices[:, 0]), np.max(face_vertices[:, 0])
    z_min, z_max = np.min(face_vertices[:, 2]), np.max(face_vertices[:, 2])
    y_max = np.max(face_vertices[:, 1])  # Front of face (nose tip)
    y_min = np.min(face_vertices[:, 1])  # Back of face surface

    # Head dimensions - match face mesh size
    head_width = (x_max - x_min) / 2
    head_height = (z_max - z_min) / 2
    head_depth = head_width * 1.2  # Head is slightly deeper than wide

    center_x = (x_min + x_max) / 2
    center_z = (z_min + z_max) / 2

    vertices = []
    colors = []

    # Average skin color from face
    avg_color = np.mean(face_colors, axis=0).astype(np.float64)

    # Generate back-of-head as connected ellipsoid cap
    # Start from face boundary (y_min) and curve backward
    n_rings = n_segments

    for i in range(n_rings):
        # Ring parameter: 0 = at face boundary, 1 = back of head
        t = i / (n_rings - 1)

        # Ring Y position: starts at y_min (face), curves to back
        # Use cosine for smooth curve
        ring_y = y_min - head_depth * (1 - np.cos(t * np.pi / 2))

        # Ring radius: starts at face width, shrinks toward back
        # At t=0 (face boundary), radius = head_width
        # At t=1 (back), radius = 0
        ring_radius = head_width * np.cos(t * np.pi / 2)

        # Ring Z center: follows head curvature
        ring_z_offset = head_height * 0.2 * np.sin(t * np.pi)

        for j in range(n_segments):
            # Theta goes around the ring: -90 to +90 degrees (back half only)
            theta = np.pi * (-0.5 + j / (n_segments - 1))

            x = center_x + ring_radius * np.sin(theta)
            y = ring_y
            z = center_z + ring_z_offset + head_height * np.cos(theta) * (1 - t * 0.3)

            vertices.append([x, y, z])

            # Color: gradually darker toward back
            shade = 0.95 - 0.15 * t
            variation = np.random.uniform(-3, 3, 3)
            back_color = np.clip(avg_color * shade + variation, 0, 255).astype(np.uint8)
            colors.append(back_color)

    if len(vertices) < 4:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    vertices = np.array(vertices)
    colors = np.array(colors)

    # Create faces by connecting adjacent rings
    faces = []
    for i in range(n_rings - 1):
        for j in range(n_segments - 1):
            # Current ring indices
            p0 = i * n_segments + j
            p1 = i * n_segments + j + 1
            # Next ring indices
            p2 = (i + 1) * n_segments + j + 1
            p3 = (i + 1) * n_segments + j

            # Two triangles per quad
            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    faces = np.array(faces) if faces else np.array([]).reshape(0, 3)

    return vertices, faces, colors


def map_spots_to_mesh_coords(
    spots_2d: list,
    mesh: MediaPipeMesh,
    img_width: int,
    img_height: int,
) -> list:
    """
    Map 2D spot positions to 3D mesh coordinates.

    Uses the mesh UV coordinates to find the closest landmark,
    then uses that landmark's 3D position for the spot.
    """
    from scipy.spatial import cKDTree

    if len(spots_2d) == 0:
        return []

    # Build KD-tree of UV coordinates for fast nearest neighbor lookup
    uv_tree = cKDTree(mesh.uv_coords)

    mapped_spots = []
    for spot in spots_2d:
        # Get normalized 2D position of spot center
        norm_x = spot.center[0] / img_width
        norm_y = spot.center[1] / img_height

        # Find nearest mesh vertex
        dist, idx = uv_tree.query([norm_x, norm_y])

        # Get 3D position from mesh
        vertex_3d = mesh.vertices[idx]

        # Offset slightly outward (along Y axis which is depth)
        # Y positive = toward camera, so ADD offset to move spots in front of mesh
        offset_factor = 0.05  # Increased offset for better visibility
        spot_3d = {
            'x': vertex_3d[0],
            'y': vertex_3d[1] + offset_factor,  # Move toward camera (positive Y)
            'z': vertex_3d[2],
            'classification': spot.classification.value,
            'id': spot.id,
            'area': spot.area,
            'source_view': getattr(spot, 'source_view', 'front'),
        }
        mapped_spots.append(spot_3d)

    return mapped_spots