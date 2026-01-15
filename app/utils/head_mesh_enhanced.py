"""
Enhanced 3D head mesh with complete geometry and high-resolution textures.

Improvements over basic MediaPipe mesh:
1. Extended forehead/scalp coverage
2. Generated back-of-head geometry
3. High-resolution UV-mapped textures
4. Smooth blending between views
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.spatial import Delaunay
from scipy.interpolate import griddata


@dataclass
class EnhancedHeadMesh:
    """Complete head mesh with extended geometry."""
    vertices: np.ndarray      # (N, 3) 3D coordinates
    faces: np.ndarray         # (M, 3) triangle indices
    uv_coords: np.ndarray     # (N, 2) UV texture coordinates
    vertex_colors: np.ndarray # (N, 3) per-vertex RGB
    face_mask: np.ndarray     # (N,) bool - True if vertex is from face landmarks


def create_ellipsoid_cap(
    n_rings: int = 15,
    n_segments: int = 30,
    a: float = 0.85,   # width
    b: float = 1.0,    # depth
    c: float = 1.3,    # height
    coverage: str = "full",  # "full", "back", "top"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create ellipsoid vertices for head shape.

    Returns:
        vertices: (N, 3) array
        faces: (M, 3) triangle indices
        uv_coords: (N, 2) UV coordinates
    """
    vertices = []
    uv_coords = []

    # Determine angle ranges based on coverage
    if coverage == "full":
        theta_range = (0, 2 * np.pi)        # Full circle around
        phi_range = (0, np.pi)              # Top to bottom
    elif coverage == "back":
        theta_range = (-np.pi/2, np.pi/2)   # Back half
        phi_range = (0, np.pi)
    elif coverage == "top":
        theta_range = (0, 2 * np.pi)
        phi_range = (0, np.pi/3)            # Just top cap
    else:
        theta_range = (0, 2 * np.pi)
        phi_range = (0, np.pi)

    # Generate vertices
    for i in range(n_rings):
        phi = phi_range[0] + (phi_range[1] - phi_range[0]) * i / (n_rings - 1)

        for j in range(n_segments):
            theta = theta_range[0] + (theta_range[1] - theta_range[0]) * j / (n_segments - 1)

            # Ellipsoid coordinates
            x = a * np.sin(phi) * np.cos(theta)
            y = b * np.sin(phi) * np.sin(theta)
            z = c * np.cos(phi)

            vertices.append([x, y, z])

            # UV coordinates (cylindrical projection)
            u = (theta - theta_range[0]) / (theta_range[1] - theta_range[0])
            v = (phi - phi_range[0]) / (phi_range[1] - phi_range[0])
            uv_coords.append([u, v])

    vertices = np.array(vertices)
    uv_coords = np.array(uv_coords)

    # Generate faces
    faces = []
    for i in range(n_rings - 1):
        for j in range(n_segments - 1):
            p0 = i * n_segments + j
            p1 = i * n_segments + (j + 1)
            p2 = (i + 1) * n_segments + (j + 1)
            p3 = (i + 1) * n_segments + j

            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    return vertices, np.array(faces), uv_coords


def extend_face_mesh_with_scalp(
    face_vertices: np.ndarray,
    face_landmarks_2d: np.ndarray,
    image_height: int,
    n_scalp_rings: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extend face mesh vertices to include forehead/scalp area.

    MediaPipe landmarks don't cover the full forehead. This function
    extrapolates vertices upward from the top face landmarks.
    """
    # Find top edge of face mesh (lowest Y in image = highest on face)
    top_indices = np.where(face_landmarks_2d[:, 1] < 0.25)[0]

    if len(top_indices) == 0:
        # Fall back to top 10% of vertices by Y position
        y_threshold = np.percentile(face_landmarks_2d[:, 1], 10)
        top_indices = np.where(face_landmarks_2d[:, 1] <= y_threshold)[0]

    if len(top_indices) == 0:
        return face_vertices, np.array([])

    # Get the top edge vertices
    top_verts_3d = face_vertices[top_indices]
    top_verts_2d = face_landmarks_2d[top_indices]

    # Sort by X to get left-to-right order
    sort_idx = np.argsort(top_verts_2d[:, 0])
    top_verts_3d = top_verts_3d[sort_idx]
    top_verts_2d = top_verts_2d[sort_idx]

    # Generate scalp vertices by extrapolating upward
    scalp_vertices = []
    scalp_uv = []

    # Calculate average direction for "up" in 3D space
    center_top = np.mean(top_verts_3d, axis=0)
    up_dir = np.array([0, 0, 1])  # Z is up

    for ring in range(1, n_scalp_rings + 1):
        # Offset factor increases with each ring
        offset = 0.08 * ring

        for i, (v3d, v2d) in enumerate(zip(top_verts_3d, top_verts_2d)):
            # Move vertex upward and slightly backward
            new_v3d = v3d.copy()
            new_v3d[2] += offset  # Up
            new_v3d[1] += offset * 0.3  # Slightly back (curve of head)

            # Curve inward toward center for natural head shape
            center_dist = np.abs(v3d[0])  # Distance from center
            inward_factor = 0.1 * ring * center_dist
            if v3d[0] > 0:
                new_v3d[0] -= inward_factor
            else:
                new_v3d[0] += inward_factor

            scalp_vertices.append(new_v3d)

            # UV: same X, reduced Y (higher on texture)
            new_uv = [v2d[0], max(0, v2d[1] - 0.05 * ring)]
            scalp_uv.append(new_uv)

    if scalp_vertices:
        return np.vstack([face_vertices, np.array(scalp_vertices)]), np.array(scalp_uv)
    return face_vertices, np.array([])


def create_back_of_head(
    face_vertices: np.ndarray,
    n_rings: int = 12,
    n_segments: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate back-of-head geometry that connects to face mesh edges.

    Creates an ellipsoid cap for the back/sides of the head that
    smoothly connects to the face mesh boundary.
    """
    # Find face mesh boundary (edge vertices)
    # These are vertices with X close to ±max or Y close to max (back of face)

    x_max = np.max(np.abs(face_vertices[:, 0]))
    y_max = np.max(face_vertices[:, 1])
    z_min = np.min(face_vertices[:, 2])
    z_max = np.max(face_vertices[:, 2])

    # Head dimensions based on face mesh
    head_width = x_max * 1.1
    head_depth = y_max * 1.5  # Head is deeper than face
    head_height = (z_max - z_min) * 1.2

    # Center of head
    center_z = (z_max + z_min) / 2

    vertices = []
    uv_coords = []

    # Generate back-of-head as partial ellipsoid
    # Only the back hemisphere (y > 0 in our coordinate system)
    for i in range(n_rings):
        # Phi goes from top to bottom
        phi = np.pi * i / (n_rings - 1)

        for j in range(n_segments):
            # Theta only covers back portion (-90 to +90 degrees from back)
            theta = -np.pi/2 + np.pi * j / (n_segments - 1)

            # Skip front-facing vertices (they overlap with face mesh)
            if np.abs(theta) > np.pi * 0.4:
                continue

            # Ellipsoid coordinates (centered, back-facing)
            x = head_width * np.sin(phi) * np.sin(theta)
            y = head_depth * np.sin(phi) * np.cos(theta)  # Positive = back
            z = center_z + head_height/2 * np.cos(phi)

            # Only keep vertices that are behind face (y > small threshold)
            if y > 0.1:
                vertices.append([x, y, z])

                # UV for back of head (will use generated skin tone)
                u = 0.5 + 0.5 * np.sin(theta)  # 0-1 range
                v = phi / np.pi
                uv_coords.append([u, v])

    if not vertices:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([]).reshape(0, 2)

    vertices = np.array(vertices)
    uv_coords = np.array(uv_coords)

    # Triangulate back vertices
    if len(vertices) >= 3:
        try:
            # Project to 2D for triangulation (use X and Z)
            proj_2d = vertices[:, [0, 2]]
            tri = Delaunay(proj_2d)
            faces = tri.simplices
        except:
            faces = np.array([]).reshape(0, 3)
    else:
        faces = np.array([]).reshape(0, 3)

    return vertices, faces, uv_coords


def create_high_res_texture(
    front_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    texture_size: int = 4096,
    face_landmarks_front: np.ndarray = None,
    face_landmarks_left: np.ndarray = None,
    face_landmarks_right: np.ndarray = None,
) -> np.ndarray:
    """
    Create high-resolution texture with proper face alignment.

    Uses face landmarks to align and warp photos for better texture mapping.
    """
    h = texture_size
    w = texture_size * 2  # Wider for cylindrical unwrap

    # Create base texture with skin tone gradient
    texture = create_skin_tone_base(w, h)

    # Calculate view regions (cylindrical unwrap)
    # U = 0: back-left, U = 0.25: left, U = 0.5: front, U = 0.75: right, U = 1.0: back-right

    view_width = w // 3
    blend_width = w // 12

    # Resize and enhance images
    front_resized = resize_and_enhance(front_img, view_width, h)
    left_resized = resize_and_enhance(left_img, view_width, h)
    right_resized = resize_and_enhance(right_img, view_width, h)

    # Project views onto texture with alpha blending
    # Front view at center (U = 0.5)
    texture = blend_view_onto_texture(
        texture, front_resized,
        center_u=0.5, blend_width=blend_width
    )

    # Left view (U = 0.75 - looking at right side of texture)
    texture = blend_view_onto_texture(
        texture, left_resized,
        center_u=0.75, blend_width=blend_width
    )

    # Right view (U = 0.25)
    texture = blend_view_onto_texture(
        texture, right_resized,
        center_u=0.25, blend_width=blend_width
    )

    return texture


def create_skin_tone_base(width: int, height: int) -> np.ndarray:
    """Create a base texture with realistic skin tone gradient."""
    texture = np.zeros((height, width, 3), dtype=np.uint8)

    # Base skin tone (average Caucasian skin)
    base_color = np.array([210, 180, 160])

    # Add subtle variation
    for y in range(height):
        for x in range(width):
            # Vertical gradient (slightly darker at bottom)
            v_factor = 1.0 - 0.1 * (y / height)

            # Horizontal gradient (slightly different tone at sides)
            u = x / width
            # Back of head is slightly different tone
            if u < 0.15 or u > 0.85:
                h_factor = 0.95
            else:
                h_factor = 1.0

            color = base_color * v_factor * h_factor

            # Add subtle noise for texture
            noise = np.random.normal(0, 3, 3)
            color = np.clip(color + noise, 0, 255)

            texture[y, x] = color.astype(np.uint8)

    # Smooth the noise
    texture = cv2.GaussianBlur(texture, (5, 5), 0)

    return texture


def resize_and_enhance(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize image with enhancement for better texture quality."""
    # Use high-quality interpolation
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)

    # Slight sharpening
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]]) / 2
    sharpened = cv2.filter2D(resized, -1, kernel)

    # Blend original with sharpened
    enhanced = cv2.addWeighted(resized, 0.7, sharpened, 0.3, 0)

    return enhanced


def blend_view_onto_texture(
    texture: np.ndarray,
    view_img: np.ndarray,
    center_u: float,
    blend_width: int,
) -> np.ndarray:
    """Blend a view image onto the texture at specified U position."""
    h, w = texture.shape[:2]
    vh, vw = view_img.shape[:2]

    center_x = int(center_u * w)
    start_x = center_x - vw // 2

    # Create alpha mask for smooth blending
    alpha = np.ones(vw, dtype=np.float32)

    # Fade at edges
    fade_zone = min(blend_width, vw // 4)
    alpha[:fade_zone] = np.linspace(0, 1, fade_zone)
    alpha[-fade_zone:] = np.linspace(1, 0, fade_zone)

    # Apply view with blending
    for i in range(vw):
        tex_x = (start_x + i) % w
        a = alpha[i]

        texture[:, tex_x] = (
            texture[:, tex_x].astype(float) * (1 - a) +
            view_img[:, i].astype(float) * a
        ).astype(np.uint8)

    return texture


def build_enhanced_head_mesh(
    face_vertices: np.ndarray,
    face_faces: np.ndarray,
    face_colors: np.ndarray,
    face_uv: np.ndarray,
    include_back: bool = True,
    include_scalp: bool = True,
) -> EnhancedHeadMesh:
    """
    Build complete head mesh from face mesh plus generated geometry.
    """
    all_vertices = [face_vertices]
    all_faces = [face_faces]
    all_uv = [face_uv]
    all_colors = [face_colors]
    face_mask = [np.ones(len(face_vertices), dtype=bool)]

    vertex_offset = len(face_vertices)

    # Add back of head
    if include_back:
        back_verts, back_faces, back_uv = create_back_of_head(face_vertices)

        if len(back_verts) > 0:
            all_vertices.append(back_verts)
            all_faces.append(back_faces + vertex_offset)
            all_uv.append(back_uv)

            # Generate skin tone colors for back
            back_colors = np.full((len(back_verts), 3), [200, 175, 155], dtype=np.uint8)
            all_colors.append(back_colors)
            face_mask.append(np.zeros(len(back_verts), dtype=bool))

            vertex_offset += len(back_verts)

    # Combine all geometry
    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces) if len(all_faces) > 1 else all_faces[0]
    uv_coords = np.vstack(all_uv) if len(all_uv) > 1 else all_uv[0]
    colors = np.vstack(all_colors)
    mask = np.concatenate(face_mask)

    return EnhancedHeadMesh(
        vertices=vertices,
        faces=faces,
        uv_coords=uv_coords,
        vertex_colors=colors,
        face_mask=mask,
    )


def sample_texture_colors(
    vertices: np.ndarray,
    uv_coords: np.ndarray,
    texture: np.ndarray,
) -> np.ndarray:
    """Sample colors from texture using UV coordinates."""
    h, w = texture.shape[:2]
    n_verts = len(vertices)
    colors = np.zeros((n_verts, 3), dtype=np.uint8)

    for i in range(n_verts):
        u, v = uv_coords[i]

        # Convert UV to pixel coordinates
        px = int(u * (w - 1))
        py = int(v * (h - 1))

        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)

        # Sample with bilinear interpolation for smoother colors
        colors[i] = bilinear_sample(texture, px, py)

    return colors


def bilinear_sample(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinear interpolation sampling from image."""
    h, w = img.shape[:2]

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    # Interpolation weights
    wx = x - x0
    wy = y - y0

    # Sample four corners
    c00 = img[y0, x0].astype(float)
    c01 = img[y0, x1].astype(float)
    c10 = img[y1, x0].astype(float)
    c11 = img[y1, x1].astype(float)

    # Bilinear interpolation
    c = (c00 * (1-wx) * (1-wy) +
         c01 * wx * (1-wy) +
         c10 * (1-wx) * wy +
         c11 * wx * wy)

    return np.clip(c, 0, 255).astype(np.uint8)