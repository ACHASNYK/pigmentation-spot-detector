"""
3D textured head visualization module.
Creates a head mesh with photo textures mapped from front/left/right views.
"""

import numpy as np
from PIL import Image
from dataclasses import dataclass
from enum import Enum
import io
import base64

from app.utils.visualization_3d import ViewAngle, Spot3D


def create_head_mesh(
    resolution: int = 50,
    a: float = 0.85,  # width (ear to ear)
    b: float = 1.0,   # depth (front to back)
    c: float = 1.3,   # height (chin to top)
):
    """
    Create an ellipsoid head mesh with UV coordinates for texture mapping.

    The mesh is oriented with:
    - X axis: left-right (ears)
    - Y axis: front-back (nose direction is +Y)
    - Z axis: up-down (top of head is +Z)

    Requires pyvista to be installed.
    """
    import pyvista as pv

    # Create parametric ellipsoid
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    # Create mesh grid
    u_grid, v_grid = np.meshgrid(u, v)

    # Ellipsoid coordinates
    x = a * np.sin(v_grid) * np.cos(u_grid)
    y = b * np.sin(v_grid) * np.sin(u_grid)
    z = c * np.cos(v_grid)

    # Flatten for point cloud
    points = np.column_stack([
        x.flatten(),
        y.flatten(),
        z.flatten()
    ])

    # Create faces (quads converted to triangles)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Current quad vertices
            p0 = i * resolution + j
            p1 = i * resolution + (j + 1)
            p2 = (i + 1) * resolution + (j + 1)
            p3 = (i + 1) * resolution + j

            # Two triangles per quad
            faces.append([3, p0, p1, p2])
            faces.append([3, p0, p2, p3])

    faces = np.array(faces).flatten()

    # Create PyVista mesh
    mesh = pv.PolyData(points, faces)

    # Add UV coordinates for texture mapping
    # U: horizontal angle (0-1 around the head)
    # V: vertical position (0-1 from top to bottom)
    uv = np.column_stack([
        u_grid.flatten() / (2 * np.pi),  # U: 0 to 1
        v_grid.flatten() / np.pi,         # V: 0 to 1
    ])
    mesh.active_texture_coordinates = uv

    return mesh


def create_composite_texture(
    front_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    texture_width: int = 1024,
    texture_height: int = 512,
) -> np.ndarray:
    """
    Create a composite texture from three view images.

    The texture is laid out as a cylindrical unwrap:
    - Left side of texture: right view (looking at left side of face)
    - Center of texture: front view
    - Right side of texture: left view (looking at right side of face)

    Returns RGB numpy array suitable for texture mapping.
    """
    # Resize images to fit in texture
    section_width = texture_width // 3

    # Convert and resize each image
    front_pil = Image.fromarray(front_img).resize((section_width, texture_height))
    left_pil = Image.fromarray(left_img).resize((section_width, texture_height))
    right_pil = Image.fromarray(right_img).resize((section_width, texture_height))

    # Create composite texture
    composite = Image.new('RGB', (texture_width, texture_height), (200, 180, 160))

    # Place images - the UV mapping goes: right view -> front -> left view
    # as we go around the head from U=0 to U=1
    composite.paste(right_pil, (0, 0))
    composite.paste(front_pil, (section_width, 0))
    composite.paste(left_pil, (section_width * 2, 0))

    return np.array(composite)


def create_blended_texture(
    front_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    texture_width: int = 1024,
    texture_height: int = 512,
    blend_width: int = 50,
) -> np.ndarray:
    """
    Create a blended composite texture with smooth transitions between views.
    """
    section_width = texture_width // 3

    # Convert and resize each image
    front_pil = Image.fromarray(front_img).resize((section_width + blend_width * 2, texture_height))
    left_pil = Image.fromarray(left_img).resize((section_width + blend_width * 2, texture_height))
    right_pil = Image.fromarray(right_img).resize((section_width + blend_width * 2, texture_height))

    front_arr = np.array(front_pil).astype(float)
    left_arr = np.array(left_pil).astype(float)
    right_arr = np.array(right_pil).astype(float)

    # Create base skin-toned texture
    composite = np.full((texture_height, texture_width, 3), [200, 180, 160], dtype=float)

    # Create blend masks
    def create_blend_mask(width, blend_w):
        mask = np.ones(width)
        # Fade in at start
        mask[:blend_w] = np.linspace(0, 1, blend_w)
        # Fade out at end
        mask[-blend_w:] = np.linspace(1, 0, blend_w)
        return mask[:, np.newaxis, np.newaxis]

    img_width = section_width + blend_width * 2
    blend_mask = create_blend_mask(img_width, blend_width)

    # Place right view (U = 0 to ~0.33)
    start = 0
    end = section_width + blend_width
    composite[:, start:end] = (
        composite[:, start:end] * (1 - blend_mask[:end-start].T.reshape(1, -1, 1)) +
        right_arr[:, :end-start] * blend_mask[:end-start].T.reshape(1, -1, 1)
    )

    # Place front view (U = ~0.33 to ~0.66)
    start = section_width - blend_width
    end = section_width * 2 + blend_width
    weight = blend_mask.T.reshape(1, -1, 1)
    for i, x in enumerate(range(start, min(end, texture_width))):
        if i < img_width:
            w = blend_mask[i, 0, 0]
            composite[:, x] = composite[:, x] * (1 - w) + front_arr[:, i] * w

    # Place left view (U = ~0.66 to 1.0)
    start = section_width * 2 - blend_width
    for i, x in enumerate(range(start, texture_width)):
        if i < img_width:
            w = blend_mask[i, 0, 0]
            composite[:, x] = composite[:, x] * (1 - w) + left_arr[:, i] * w

    return np.clip(composite, 0, 255).astype(np.uint8)


def project_photo_to_texture_simple(
    front_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    texture_width: int = 2048,
    texture_height: int = 1024,
) -> np.ndarray:
    """
    Simple projection: place photos side by side with smooth blending.

    Texture UV layout (cylindrical unwrap):
    - U = 0.0: Back of head (behind left ear)
    - U = 0.25: Left side of face (left view)
    - U = 0.5: Front of face (front view)
    - U = 0.75: Right side of face (right view)
    - U = 1.0: Back of head (behind right ear)
    """
    # Target: 2048 wide texture split into regions
    # Back(left) | Left | Front | Right | Back(right)
    #   256     | 512  |  512  |  512  |   256

    back_width = texture_width // 8
    view_width = texture_width // 4
    blend_zone = texture_width // 16  # Smooth transition zone

    # Resize images
    h = texture_height
    front_resized = np.array(Image.fromarray(front_img).resize((view_width, h)))
    left_resized = np.array(Image.fromarray(left_img).resize((view_width, h)))
    right_resized = np.array(Image.fromarray(right_img).resize((view_width, h)))

    # Create base texture (skin tone)
    texture = np.full((h, texture_width, 3), [210, 185, 165], dtype=np.uint8)

    # Calculate regions (cylindrical unwrap going counter-clockwise when viewed from top)
    # Starting from back of head going around: back -> right -> front -> left -> back

    # Right view center at U = 0.25 (from front's perspective, this is the right side)
    # But in cylindrical unwrap, U=0.25 is left side of face
    # Let's correct:
    # U=0: back, U=0.25: right ear area, U=0.5: front, U=0.75: left ear area

    # Place photos with blending
    def place_with_blend(texture, img, center_u, img_width):
        center_x = int(center_u * texture_width)
        start_x = center_x - img_width // 2
        end_x = start_x + img_width

        # Handle wraparound at edges
        for i in range(img_width):
            tex_x = (start_x + i) % texture_width
            img_x = i

            # Calculate blend weight (fade at edges)
            edge_dist = min(i, img_width - 1 - i)
            weight = min(1.0, edge_dist / blend_zone) if blend_zone > 0 else 1.0

            # Blend
            texture[:, tex_x] = (
                texture[:, tex_x].astype(float) * (1 - weight) +
                img[:, img_x].astype(float) * weight
            ).astype(np.uint8)

        return texture

    # Place views
    texture = place_with_blend(texture, front_resized, 0.5, view_width)   # Front at U=0.5
    texture = place_with_blend(texture, left_resized, 0.75, view_width)   # Left at U=0.75
    texture = place_with_blend(texture, right_resized, 0.25, view_width)  # Right at U=0.25

    return texture


def add_spots_to_texture(
    texture: np.ndarray,
    spots_3d: list[Spot3D],
    spot_radius: int = 8,
) -> np.ndarray:
    """
    Add spot markers to the texture image.

    Converts 3D spot coordinates to UV coordinates and draws circles on texture.
    """
    h, w = texture.shape[:2]
    texture = texture.copy()

    color_map = {
        'light': (255, 255, 0),      # Yellow
        'medium': (255, 165, 0),     # Orange
        'dark': (255, 0, 0),         # Red
    }

    for spot in spots_3d:
        # Convert 3D coordinates to UV
        # U = atan2(y, x) / (2*pi) + 0.5 (to get 0-1 range with front at 0.5)
        theta = np.arctan2(spot.y, spot.x)
        u = (theta / (2 * np.pi)) + 0.5
        u = u % 1.0  # Wrap around

        # V = acos(z / r) / pi (0 at top, 1 at bottom)
        r = np.sqrt(spot.x**2 + spot.y**2 + spot.z**2)
        if r > 0:
            v = np.arccos(np.clip(spot.z / r, -1, 1)) / np.pi
        else:
            v = 0.5

        # Convert to pixel coordinates
        px = int(u * w) % w
        py = int(v * h)
        py = np.clip(py, 0, h - 1)

        # Draw spot circle
        color = color_map.get(spot.classification, (255, 255, 255))

        # Scale radius by spot area
        radius = max(3, min(15, int(np.sqrt(spot.area) / 2)))

        # Draw filled circle with border
        for dy in range(-radius - 1, radius + 2):
            for dx in range(-radius - 1, radius + 2):
                dist = np.sqrt(dx**2 + dy**2)
                tx = (px + dx) % w
                ty = py + dy
                if 0 <= ty < h:
                    if dist <= radius:
                        # Inside: spot color
                        texture[ty, tx] = color
                    elif dist <= radius + 1:
                        # Border: black outline
                        texture[ty, tx] = (0, 0, 0)

    return texture


def create_pyvista_textured_head(
    front_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    spots_3d: list[Spot3D],
    show_spots_on_texture: bool = True,
):
    """
    Create a PyVista head mesh with photo texture and spots.

    Returns the mesh and texture objects for rendering.
    Requires pyvista to be installed.
    """
    import pyvista as pv

    # Create head mesh
    mesh = create_head_mesh(resolution=60)

    # Create composite texture
    texture_img = project_photo_to_texture_simple(
        front_img, left_img, right_img,
        texture_width=2048,
        texture_height=1024,
    )

    # Add spots to texture if requested
    if show_spots_on_texture and spots_3d:
        texture_img = add_spots_to_texture(texture_img, spots_3d)

    # Create PyVista texture
    texture = pv.Texture(texture_img)

    return mesh, texture


def render_textured_head_to_html(
    mesh,
    texture,
    spots_3d: list[Spot3D] = None,
) -> str:
    """
    Render the textured head to an HTML string for embedding in Streamlit.
    Requires pyvista to be installed.
    """
    import pyvista as pv

    # Create plotter
    plotter = pv.Plotter(off_screen=True)

    # Add textured mesh
    plotter.add_mesh(mesh, texture=texture, smooth_shading=True)

    # Add spot markers as spheres if provided
    if spots_3d:
        color_map = {
            'light': 'yellow',
            'medium': 'orange',
            'dark': 'red',
        }

        for spot in spots_3d:
            # Create small sphere at spot location (slightly outside mesh)
            center = np.array([spot.x, spot.y, spot.z])
            # Push slightly outward
            center = center * 1.02

            radius = max(0.02, min(0.06, spot.area / 2000))
            sphere = pv.Sphere(radius=radius, center=center)

            color = color_map.get(spot.classification, 'white')
            plotter.add_mesh(sphere, color=color, opacity=0.9)

    # Set camera for front view
    plotter.camera_position = [(0, -3, 0.5), (0, 0, 0), (0, 0, 1)]

    # Export to HTML
    return plotter.export_html(None)


def create_spot_spheres(spots_3d: list[Spot3D]) -> list[dict]:
    """
    Create sphere data for spots to overlay on mesh.
    """
    spheres = []

    color_map = {
        'light': [1.0, 1.0, 0.0],      # Yellow
        'medium': [1.0, 0.65, 0.0],    # Orange
        'dark': [1.0, 0.0, 0.0],       # Red
    }

    for spot in spots_3d:
        # Push spot position slightly outward from mesh surface
        pos = np.array([spot.x, spot.y, spot.z])
        pos = pos * 1.03  # 3% outside the surface

        radius = max(0.02, min(0.08, np.sqrt(spot.area) / 50))

        spheres.append({
            'center': pos.tolist(),
            'radius': radius,
            'color': color_map.get(spot.classification, [1, 1, 1]),
            'classification': spot.classification,
            'id': spot.id,
            'source_view': spot.source_view.value,
        })

    return spheres