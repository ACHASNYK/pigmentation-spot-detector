"""
3D visualization module for pigmentation spot mapping.
Creates a 3D head model with spots mapped from front/left/right photos.
"""

import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from enum import Enum


class ViewAngle(Enum):
    """Photo view angles."""
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Spot3D:
    """A spot mapped to 3D coordinates."""
    id: int
    x: float
    y: float
    z: float
    classification: str
    area: int
    combined_delta: float
    source_view: ViewAngle


def create_cylinder_head(radius: float = 1.0, height: float = 1.5, resolution: int = 50):
    """
    Create a cylindrical approximation of a head.

    Returns vertices for a cylinder that can display spots.
    """
    # Create cylinder mesh
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(-height/2, height/2, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)

    return x_grid, y_grid, z_grid


def create_ellipsoid_head(a: float = 0.8, b: float = 1.0, c: float = 1.2, resolution: int = 50):
    """
    Create an ellipsoid approximation of a head.
    a = width (ear to ear)
    b = depth (front to back)
    c = height (chin to top)

    Returns mesh coordinates.
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z


def map_2d_to_cylinder(
    x_2d: int,
    y_2d: int,
    img_width: int,
    img_height: int,
    view: ViewAngle,
    radius: float = 1.0,
    height: float = 1.5
) -> tuple[float, float, float]:
    """
    Map 2D image coordinates to 3D cylinder coordinates.

    Front view: theta around 0 (facing +Y direction)
    Left view: theta around -pi/2 (facing +X direction)
    Right view: theta around +pi/2 (facing -X direction)
    """
    # Normalize coordinates to -1 to 1 range
    x_norm = (x_2d / img_width) * 2 - 1  # -1 (left) to 1 (right)
    y_norm = (y_2d / img_height) * 2 - 1  # -1 (top) to 1 (bottom)

    # Map Y to Z (vertical position on cylinder)
    z = -y_norm * (height / 2)  # Flip because image Y is inverted

    # Map X to theta based on view angle
    if view == ViewAngle.FRONT:
        # Front view: center at theta=0 (facing +Y)
        # X maps to a range around theta=0
        theta = x_norm * (np.pi / 3)  # ~60 degree spread

    elif view == ViewAngle.LEFT:
        # Left side of face: theta around -pi/2
        theta = -np.pi/2 + x_norm * (np.pi / 3)

    elif view == ViewAngle.RIGHT:
        # Right side of face: theta around +pi/2
        theta = np.pi/2 + x_norm * (np.pi / 3)

    # Convert to Cartesian
    x_3d = radius * np.cos(theta)
    y_3d = radius * np.sin(theta)
    z_3d = z

    return x_3d, y_3d, z_3d


def map_2d_to_ellipsoid(
    x_2d: int,
    y_2d: int,
    img_width: int,
    img_height: int,
    view: ViewAngle,
    a: float = 0.8,
    b: float = 1.0,
    c: float = 1.2
) -> tuple[float, float, float]:
    """
    Map 2D image coordinates to 3D ellipsoid coordinates.
    """
    # Normalize coordinates
    x_norm = (x_2d / img_width) * 2 - 1
    y_norm = (y_2d / img_height) * 2 - 1

    # Map Y to phi (polar angle from top)
    # y_norm = -1 (top) -> phi near 0
    # y_norm = 1 (bottom) -> phi near pi
    phi = (y_norm + 1) * np.pi / 2  # 0 to pi range, centered on face area
    phi = np.clip(phi, 0.2, np.pi - 0.2)  # Avoid poles

    # Map X and view to theta (azimuthal angle)
    if view == ViewAngle.FRONT:
        theta = x_norm * (np.pi / 4)  # ~45 degree spread around front
    elif view == ViewAngle.LEFT:
        theta = -np.pi/2 + x_norm * (np.pi / 4)
    elif view == ViewAngle.RIGHT:
        theta = np.pi/2 + x_norm * (np.pi / 4)

    # Convert to Cartesian (ellipsoid)
    x_3d = a * np.sin(phi) * np.cos(theta)
    y_3d = b * np.sin(phi) * np.sin(theta)
    z_3d = c * np.cos(phi)

    return x_3d, y_3d, z_3d


def map_spots_to_3d(
    spots: list,
    img_width: int,
    img_height: int,
    view: ViewAngle,
    use_ellipsoid: bool = True
) -> list[Spot3D]:
    """
    Convert 2D spots to 3D coordinates.
    """
    spots_3d = []

    for spot in spots:
        cx, cy = spot.center

        if use_ellipsoid:
            x, y, z = map_2d_to_ellipsoid(cx, cy, img_width, img_height, view)
        else:
            x, y, z = map_2d_to_cylinder(cx, cy, img_width, img_height, view)

        spot_3d = Spot3D(
            id=spot.id,
            x=x,
            y=y,
            z=z,
            classification=spot.classification.value,
            area=spot.area,
            combined_delta=spot.combined_delta,
            source_view=view,
        )
        spots_3d.append(spot_3d)

    return spots_3d


def create_3d_head_figure(
    spots_3d: list[Spot3D],
    use_ellipsoid: bool = True,
    show_mesh: bool = True
) -> go.Figure:
    """
    Create a 3D Plotly figure with head mesh and mapped spots.
    """
    fig = go.Figure()

    # Add head mesh (semi-transparent)
    if show_mesh:
        if use_ellipsoid:
            x, y, z = create_ellipsoid_head()
        else:
            x, y, z = create_cylinder_head()

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, 'rgb(255, 220, 185)'], [1, 'rgb(255, 200, 165)']],
            opacity=0.3,
            showscale=False,
            name='Head',
            hoverinfo='skip',
        ))

    # Color mapping for classifications
    color_map = {
        'light': 'yellow',
        'medium': 'orange',
        'dark': 'red',
    }

    # Group spots by classification for legend
    for classification in ['light', 'medium', 'dark']:
        class_spots = [s for s in spots_3d if s.classification == classification]

        if not class_spots:
            continue

        # Add spots as 3D scatter
        fig.add_trace(go.Scatter3d(
            x=[s.x for s in class_spots],
            y=[s.y for s in class_spots],
            z=[s.z for s in class_spots],
            mode='markers',
            marker=dict(
                size=[max(3, min(12, s.area / 50)) for s in class_spots],
                color=color_map[classification],
                opacity=0.8,
                line=dict(width=1, color='black'),
            ),
            name=f'{classification.capitalize()} spots ({len(class_spots)})',
            text=[f"Spot #{s.id}<br>View: {s.source_view.value}<br>Score: {s.combined_delta:.1f}<br>Area: {s.area}px"
                  for s in class_spots],
            hoverinfo='text',
        ))

    # Configure layout for X-axis rotation
    fig.update_layout(
        title='3D Head with Pigmentation Spots',
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            aspectmode='data',
            camera=dict(
                eye=dict(x=0, y=-2, z=0.5),  # Front view
                up=dict(x=0, y=0, z=1),
            ),
            dragmode='orbit',
        ),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    return fig


def create_rotation_animation(
    spots_3d: list[Spot3D],
    use_ellipsoid: bool = True,
    frames: int = 36
) -> go.Figure:
    """
    Create an animated 3D figure that rotates around the X axis.
    """
    fig = create_3d_head_figure(spots_3d, use_ellipsoid)

    # Create frames for rotation animation
    animation_frames = []
    for i in range(frames):
        angle = (i / frames) * 2 * np.pi
        # Rotate camera around the model (X-axis rotation means camera moves in Y-Z plane)
        eye_y = -2 * np.cos(angle)
        eye_z = 2 * np.sin(angle) + 0.5

        frame = go.Frame(
            layout=dict(
                scene=dict(
                    camera=dict(
                        eye=dict(x=0, y=eye_y, z=eye_z),
                        up=dict(x=0, y=0, z=1),
                    )
                )
            ),
            name=str(i)
        )
        animation_frames.append(frame)

    fig.frames = animation_frames

    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                xanchor='left',
                buttons=[
                    dict(
                        label='Rotate',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=100, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                        )]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                        )]
                    ),
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        method='animate',
                        args=[[str(i)], dict(
                            mode='immediate',
                            frame=dict(duration=100, redraw=True),
                        )],
                        label=f'{int(i * 360 / frames)}°'
                    )
                    for i in range(0, frames, frames // 8)  # Show 8 labels
                ],
                x=0.1,
                len=0.8,
                xanchor='left',
                y=-0.05,
                currentvalue=dict(
                    prefix='Angle: ',
                    visible=True,
                    xanchor='center'
                ),
            )
        ]
    )

    return fig