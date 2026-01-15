"""
Streamlit application for pigmentation spot detection.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import streamlit as st
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import DetectionConfig
from app.core.spot_detector import SpotDetector, DetectionMode
from app.core.models import BatchReport, DetectionResult
from app.utils.image_utils import (
    load_image_from_bytes,
    bgr_to_rgb,
    encode_image_to_bytes,
)
from app.utils.visualization import (
    draw_spot_annotations,
    create_debug_visualization,
    create_spot_legend,
)
from app.utils.visualization_3d import (
    ViewAngle,
    map_spots_to_3d,
    create_3d_head_figure,
    create_rotation_animation,
    create_ellipsoid_head,
)
from app.utils.visualization_3d_textured import (
    project_photo_to_texture_simple,
    add_spots_to_texture,
)
from app.utils.head_reconstruction import (
    reconstruct_head,
    mesh_to_plotly_data,
    ReconstructedMesh,
)
from app.utils.mediapipe_mesh import (
    extract_mediapipe_mesh,
    fuse_multi_view_meshes,
    mesh_to_plotly,
    map_spots_to_mesh_coords,
    extend_mesh_with_head,
    MediaPipeMesh,
)
from app.core.face_detector import FaceDetector
from app.core.vlm_validator import (
    is_vlm_available,
    validate_spots_batch,
    create_validation_summary,
)


def main():
    st.set_page_config(
        page_title="Pigmentation Spot Detector",
        page_icon="🔬",
        layout="wide",
    )

    st.title("Pigmentation Spot Detector")
    st.markdown("Detect and classify pigmentation spots on human face skin")

    # Sidebar configuration
    config, detection_mode = create_sidebar_config()

    # Create tabs for 2D and 3D modes
    tab_2d, tab_3d = st.tabs(["2D Detection", "3D Head Mapping"])

    with tab_2d:
        st.header("2D Spot Detection")
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload face images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Upload one or more face images (front, left side, or right side views)",
            key="uploader_2d",
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")

            # Show thumbnails of uploaded images
            st.subheader("Uploaded Images")
            thumb_cols = st.columns(min(len(uploaded_files), 4))
            for i, uploaded_file in enumerate(uploaded_files):
                col_idx = i % 4
                with thumb_cols[col_idx]:
                    # Reset file pointer and show thumbnail
                    uploaded_file.seek(0)
                    st.image(uploaded_file, caption=uploaded_file.name[:20], width=150)
                    uploaded_file.seek(0)  # Reset for later processing

            # Process button
            if st.button("🔍 Detect Spots", type="primary", key="detect_2d"):
                process_images(uploaded_files, config, detection_mode)

    with tab_3d:
        st.header("3D Head Mapping")
        st.markdown("""
        Upload **3 photos** of the same face from different angles to create a 3D visualization:
        - **Front view** - face looking directly at camera
        - **Left view** - left side of face (right cheek visible)
        - **Right view** - right side of face (left cheek visible)
        """)

        # Three file uploaders for different angles
        col1, col2, col3 = st.columns(3)

        with col1:
            front_file = st.file_uploader(
                "Front View",
                type=["jpg", "jpeg", "png", "webp"],
                key="front_upload",
                help="Face looking directly at camera",
            )
            if front_file:
                st.image(front_file, caption="Front", use_container_width=True)

        with col2:
            left_file = st.file_uploader(
                "Left View",
                type=["jpg", "jpeg", "png", "webp"],
                key="left_upload",
                help="Left side of face visible",
            )
            if left_file:
                st.image(left_file, caption="Left", width="stretch")

        with col3:
            right_file = st.file_uploader(
                "Right View",
                type=["jpg", "jpeg", "png", "webp"],
                key="right_upload",
                help="Right side of face visible",
            )
            if right_file:
                st.image(right_file, caption="Right", width="stretch")

        # Process 3D button
        if front_file and left_file and right_file:
            if st.button("🌐 Generate 3D Head Model", type="primary", key="generate_3d"):
                process_3d_visualization(
                    front_file, left_file, right_file,
                    config, detection_mode
                )


def create_sidebar_config() -> tuple[DetectionConfig, DetectionMode]:
    """Create sidebar with tunable parameters."""
    st.sidebar.header("Detection Settings")

    # Pipeline selection
    st.sidebar.subheader("Detection Pipeline")
    pipeline_options = {
        "Standard (LAB thresholds)": DetectionMode.STANDARD,
        "Hyper-Sensitive (catch all)": DetectionMode.HYPER_SENSITIVE,
        "Blob Detection (LoG/DoG)": DetectionMode.BLOB_DETECTION,
    }
    selected_pipeline = st.sidebar.radio(
        "Select detection method",
        options=list(pipeline_options.keys()),
        index=1,  # Default to hyper-sensitive
        help="""
        **Standard**: Uses fixed color thresholds (faster, less sensitive)

        **Hyper-Sensitive**: Catches ANY brown deviation using percentile-based detection (recommended)

        **Blob Detection**: Uses multi-scale Laplacian of Gaussian (LoG) and Difference of Gaussians (DoG) to find spots at all sizes
        """,
    )
    detection_mode = pipeline_options[selected_pipeline]

    st.sidebar.divider()
    st.sidebar.subheader("Spot Detection Sensitivity")
    spot_b_threshold = st.sidebar.slider(
        "Brown threshold (B-channel)",
        min_value=1.0,
        max_value=15.0,
        value=1.0,  # Minimum for max sensitivity
        step=0.5,
        help="Lower = more sensitive to brownish spots",
    )

    spot_l_threshold = st.sidebar.slider(
        "Darkness threshold (L-channel)",
        min_value=1.0,
        max_value=20.0,
        value=1.0,  # Minimum for max sensitivity
        step=0.5,
        help="Lower = more sensitive to darker spots",
    )

    st.sidebar.subheader("Spot Size Filtering")
    min_spot_area = st.sidebar.slider(
        "Minimum spot area (pixels)",
        min_value=5,
        max_value=100,
        value=15,
        step=5,
        help="Filter out spots smaller than this",
    )

    max_spot_area = st.sidebar.slider(
        "Maximum spot area (pixels)",
        min_value=500,
        max_value=10000,
        value=2000,
        step=100,
        help="Filter out regions larger than this (individual spots only)",
    )

    st.sidebar.subheader("Classification Thresholds")
    light_max = st.sidebar.slider(
        "Light/Medium boundary",
        min_value=1.0,
        max_value=12.0,
        value=2.0,  # Lower for more sensitive classification
        step=0.5,
    )

    medium_max = st.sidebar.slider(
        "Medium/Dark boundary",
        min_value=4.0,
        max_value=20.0,
        value=5.0,  # Lower for more sensitive classification
        step=0.5,
    )

    st.sidebar.subheader("Exclusion Zones")
    eye_expansion = st.sidebar.slider(
        "Eye exclusion (pixels)",
        min_value=3,
        max_value=30,
        value=15,
        step=2,
        help="Expand eye exclusion zone",
    )

    eyebrow_expansion = st.sidebar.slider(
        "Eyebrow exclusion (pixels)",
        min_value=2,
        max_value=20,
        value=16,
        step=2,
    )

    lip_expansion = st.sidebar.slider(
        "Lip exclusion (pixels)",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        help="Expand lip exclusion zone",
    )

    nostril_expansion = st.sidebar.slider(
        "Nostril exclusion (pixels)",
        min_value=0,
        max_value=20,
        value=8,
        step=1,
        help="Expand nostril exclusion zone",
    )

    st.sidebar.subheader("Skin Detection (YCrCb)")
    skin_cr_min = st.sidebar.slider("Cr min", 100, 160, 133)
    skin_cr_max = st.sidebar.slider("Cr max", 160, 200, 173)
    skin_cb_min = st.sidebar.slider("Cb min", 50, 100, 77)
    skin_cb_max = st.sidebar.slider("Cb max", 100, 160, 127)

    # AI Verification section
    st.sidebar.divider()
    st.sidebar.subheader("AI Verification (Claude)")

    vlm_available = is_vlm_available()
    if not vlm_available:
        st.sidebar.warning("Set ANTHROPIC_API_KEY to enable AI verification")

    enable_ai_verification = st.sidebar.checkbox(
        "Enable AI verification",
        value=False,
        disabled=not vlm_available,
        help="Use Claude to validate detected spots and filter false positives (hair, shadows, etc.)",
    )

    # Store in session state for access in process_images
    st.session_state['enable_ai_verification'] = enable_ai_verification and vlm_available

    config = DetectionConfig(
        spot_b_threshold=spot_b_threshold,
        spot_l_threshold=spot_l_threshold,
        min_spot_area=min_spot_area,
        max_spot_area=max_spot_area,
        light_max=light_max,
        medium_min=light_max,
        medium_max=medium_max,
        dark_min=medium_max,
        skin_cr_min=skin_cr_min,
        skin_cr_max=skin_cr_max,
        skin_cb_min=skin_cb_min,
        skin_cb_max=skin_cb_max,
        eye_expansion=eye_expansion,
        eyebrow_expansion=eyebrow_expansion,
        lip_expansion=lip_expansion,
        nostril_expansion=nostril_expansion,
    )
    return config, detection_mode


def process_images(uploaded_files, config: DetectionConfig, mode: DetectionMode):
    """Process uploaded images and display results."""
    results: list[DetectionResult] = []
    failed_count = 0
    all_validation_summaries = []

    # Check if AI verification is enabled
    enable_ai = st.session_state.get('enable_ai_verification', False)

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Show mode being used
    mode_names = {
        DetectionMode.STANDARD: "Standard (LAB thresholds)",
        DetectionMode.HYPER_SENSITIVE: "Hyper-Sensitive",
        DetectionMode.BLOB_DETECTION: "Blob Detection (LoG/DoG)",
    }
    st.info(f"Using detection mode: **{mode_names[mode]}**" + (" + AI Verification" if enable_ai else ""))

    # Initialize detector with selected mode
    with SpotDetector(config, mode=mode) as detector:
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")

            try:
                # Load image at FULL RESOLUTION (no downscaling)
                image_bytes = uploaded_file.read()
                image = load_image_from_bytes(image_bytes)
                h, w = image.shape[:2]
                status_text.text(f"Processing {uploaded_file.name} at {w}x{h} (full resolution)...")

                # Detect spots
                result = detector.detect(image, uploaded_file.name)
                original_count = len(result.spots)

                # AI Verification (if enabled)
                if enable_ai and result.spots:
                    status_text.text(f"AI verifying {len(result.spots)} spots in {uploaded_file.name}...")

                    # Create progress callback
                    ai_progress = st.empty()
                    ai_error_container = st.empty()

                    def update_ai_progress(current, total):
                        ai_progress.text(f"AI verification: {current}/{total} spots...")

                    def handle_ai_error(error_msg):
                        ai_error_container.error(f"AI Verification Error: {error_msg}")

                    # Extract face bbox for zone-based validation
                    face_bbox = None
                    if result.debug_info.face_bbox is not None:
                        fb = result.debug_info.face_bbox
                        face_bbox = {'x': fb.x, 'y': fb.y, 'width': fb.width, 'height': fb.height}

                    # Validate spots using zone-based approach
                    validated_spots, validation_results = validate_spots_batch(
                        image, result.spots,
                        progress_callback=update_ai_progress,
                        error_callback=handle_ai_error,
                        face_bbox=face_bbox,
                    )
                    ai_progress.empty()

                    # Update result with validated spots
                    filtered_count = original_count - len(validated_spots)
                    result.spots = validated_spots
                    # spot_count is a computed property, no need to set it

                    # Store validation summary
                    summary = create_validation_summary(validation_results)
                    summary['image_name'] = uploaded_file.name
                    all_validation_summaries.append(summary)

                    status_text.text(f"AI filtered {filtered_count} artifacts from {uploaded_file.name}")

                # Generate visualizations (no labels, just colored boxes)
                result.annotated_image = draw_spot_annotations(image, result, draw_labels=False)
                result.debug_visualization = create_debug_visualization(image, result)

                results.append(result)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                failed_count += 1

            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("Processing complete!")
    progress_bar.empty()

    # Display results
    if results:
        display_results(results, failed_count, all_validation_summaries if enable_ai else None)


def display_results(results: list[DetectionResult], failed_count: int, validation_summaries: list = None):
    """Display detection results."""
    # Create batch report
    batch_report = BatchReport.create(results, failed_count)

    # AI Verification summary (if used)
    if validation_summaries:
        st.header("AI Verification Results (Zone-Based)")
        total_checked = sum(s['total_checked'] for s in validation_summaries)
        total_filtered = sum(s['artifacts_filtered'] for s in validation_summaries)
        total_valid = sum(s['valid_spots'] for s in validation_summaries)
        safe_zone = sum(s.get('safe_zone', 0) for s in validation_summaries)
        hairline_zone = sum(s.get('hairline_zone', 0) for s in validation_summaries)
        ai_verified = sum(s.get('ai_verified', 0) for s in validation_summaries)

        # Main metrics row
        ai_cols = st.columns(4)
        with ai_cols[0]:
            st.metric("Total Spots", total_checked)
        with ai_cols[1]:
            st.metric("Valid Spots", total_valid)
        with ai_cols[2]:
            st.metric("Artifacts Filtered", total_filtered)
        with ai_cols[3]:
            filter_rate = (total_filtered / total_checked * 100) if total_checked > 0 else 0
            st.metric("Filter Rate", f"{filter_rate:.1f}%")

        # Zone breakdown row
        st.markdown("**Zone Breakdown:**")
        zone_cols = st.columns(3)
        with zone_cols[0]:
            st.metric("Safe Zone", safe_zone, help="Center of face - kept without AI")
        with zone_cols[1]:
            st.metric("Hairline Zone", hairline_zone, help="Face edges - rejected without AI")
        with zone_cols[2]:
            st.metric("AI Verified", ai_verified, help="Ambiguous spots - sent to Claude")

        # Show artifact types found
        all_types = {}
        for summary in validation_summaries:
            for spot_type, count in summary.get('by_type', {}).items():
                all_types[spot_type] = all_types.get(spot_type, 0) + count

        if all_types:
            with st.expander("Detected Types"):
                st.json(all_types)

    # Summary section
    st.header("Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Images Processed", batch_report.successful_images)
    with col2:
        st.metric("Total Spots", batch_report.summary["total_spots_detected"])
    with col3:
        st.metric("Avg Spots/Image", batch_report.summary["average_spots_per_image"])
    with col4:
        if failed_count > 0:
            st.metric("Failed", failed_count)

    # Spots by category
    st.subheader("Spots by Category")
    spots_by_cat = batch_report.summary["spots_by_category"]
    cat_cols = st.columns(3)
    with cat_cols[0]:
        st.metric("Light", spots_by_cat["light"], help="Subtle pigmentation")
    with cat_cols[1]:
        st.metric("Medium", spots_by_cat["medium"], help="Moderate pigmentation")
    with cat_cols[2]:
        st.metric("Dark", spots_by_cat["dark"], help="Strong pigmentation")

    # Legend
    st.image(bgr_to_rgb(create_spot_legend()), caption="Color Legend", width=200)

    # Individual results
    st.header("Individual Results")

    for result in results:
        with st.expander(f"📷 {result.image_name} - {result.spot_count} spots", expanded=True):
            display_single_result(result)

    # Download section
    st.header("Export Results")
    col1, col2 = st.columns(2)

    with col1:
        # JSON download
        json_data = json.dumps(batch_report.to_dict(), indent=2)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_data,
            file_name="pigmentation_report.json",
            mime="application/json",
        )

    with col2:
        # Download annotated images info
        st.info(f"💡 Annotated images can be downloaded individually from each result above")


def create_zoomable_image(img_array: np.ndarray, title: str = "", key: str = ""):
    """Create a zoomable image using Plotly with pan/zoom controls."""
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(img_array)
    height, width = img_array.shape[:2]

    # Create figure with the image
    fig = go.Figure()

    # Add image as a layout image (background)
    fig.add_layout_image(
        dict(
            source=pil_img,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=width,
            sizey=height,
            sizing="stretch",
            layer="below"
        )
    )

    # Configure axes
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[0, width],
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[height, 0],  # Inverted for image coordinates
        scaleanchor="x",
        scaleratio=1,
    )

    # Configure layout
    fig.update_layout(
        title=title,
        width=None,  # Auto width
        height=600,
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        dragmode="pan",  # Default to pan mode
        modebar=dict(
            bgcolor="rgba(255,255,255,0.8)",
            orientation="h",
        ),
    )

    # Configure modebar buttons for zoom controls
    config = {
        "modeBarButtonsToAdd": ["drawrect", "eraseshape"],
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "displayModeBar": True,
        "scrollZoom": True,  # Enable scroll wheel zoom
    }

    return fig, config


def display_single_result(result: DetectionResult):
    """Display results for a single image."""

    # Zoomable annotated image section
    st.subheader("Detected Spots (Zoomable)")
    st.caption("Use mouse wheel to zoom, drag to pan. Double-click to reset view.")

    if result.annotated_image is not None:
        fig, config = create_zoomable_image(
            bgr_to_rgb(result.annotated_image),
            key=f"annotated_{result.image_name}"
        )
        st.plotly_chart(fig, use_container_width=True, config=config)

        # Download button for annotated image
        img_bytes = encode_image_to_bytes(result.annotated_image)
        st.download_button(
            label="Download annotated image",
            data=img_bytes,
            file_name=f"annotated_{result.image_name}",
            mime="image/png",
            key=f"download_{result.image_name}",
        )

    # Debug view in expander (smaller, non-zoomable)
    with st.expander("Debug View"):
        if result.debug_visualization is not None:
            st.image(
                bgr_to_rgb(result.debug_visualization),
                width="stretch",
            )

    # Metrics
    met_cols = st.columns(5)
    with met_cols[0]:
        st.metric("Total Spots", result.spot_count)
    with met_cols[1]:
        st.metric("Processing Time", f"{result.processing_time_ms:.0f}ms")
    with met_cols[2]:
        if result.debug_info.mean_skin_color_lab:
            st.metric("Skin L*", f"{result.debug_info.mean_skin_color_lab[0]:.1f}")
    with met_cols[3]:
        st.metric("Skin Coverage", f"{result.debug_info.skin_coverage_percent:.1f}%")
    with met_cols[4]:
        # Show image dimensions (original quality preserved)
        if result.annotated_image is not None:
            h, w = result.annotated_image.shape[:2]
            st.metric("Resolution", f"{w}x{h}")

    # Spots table
    if result.spots:
        st.subheader("Detected Spots Details")

        # Convert to table data
        table_data = []
        for spot in result.spots:
            table_data.append({
                "ID": spot.id,
                "Classification": spot.classification.value,
                "Area (px)": spot.area,
                "Center X": spot.center[0],
                "Center Y": spot.center[1],
                "Delta L": f"{spot.color_delta_l:.1f}",
                "Delta B": f"{spot.color_delta_b:.1f}",
                "Score": f"{spot.combined_delta:.1f}",
            })

        st.dataframe(table_data, use_container_width=True)

    # Debug info expander
    with st.expander("🔧 Debug Information"):
        st.json(result.debug_info.to_dict())

    # JSON for this image
    with st.expander("📄 JSON Data"):
        st.json(result.to_dict())


def create_plotly_textured_head(texture_img: np.ndarray, spots_3d: list):
    """Create a Plotly 3D surface with texture colors mapped to vertices."""
    # Create ellipsoid mesh
    x, y, z = create_ellipsoid_head(a=0.85, b=1.0, c=1.3, resolution=60)

    # Sample colors from texture based on UV coordinates
    h, w = texture_img.shape[:2]
    resolution = x.shape[0]

    # Create color array for surface
    colors = np.zeros((resolution, resolution, 3), dtype=np.uint8)

    for i in range(resolution):
        for j in range(resolution):
            # Calculate UV from position on ellipsoid
            theta = np.arctan2(y[i, j], x[i, j])
            u = (theta / (2 * np.pi)) + 0.5
            u = u % 1.0

            # V from z position
            r = np.sqrt(x[i, j]**2 + y[i, j]**2 + z[i, j]**2)
            if r > 0:
                v = np.arccos(np.clip(z[i, j] / r, -1, 1)) / np.pi
            else:
                v = 0.5

            # Sample texture
            tx = int(u * (w - 1))
            ty = int(v * (h - 1))
            colors[i, j] = texture_img[ty, tx]

    # Convert colors to plotly format (0-1 range)
    surfacecolor = (colors[:, :, 0].astype(float) * 65536 +
                    colors[:, :, 1].astype(float) * 256 +
                    colors[:, :, 2].astype(float))

    fig = go.Figure()

    # Add textured surface
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=surfacecolor,
        colorscale=[
            [0, 'rgb(0,0,0)'],
            [0.5, 'rgb(128,128,128)'],
            [1, 'rgb(255,255,255)']
        ],
        showscale=False,
        opacity=0.95,
        name='Head',
        hoverinfo='skip',
    ))

    # Add spots as scatter points
    color_map = {
        'light': 'yellow',
        'medium': 'orange',
        'dark': 'red',
    }

    for classification in ['light', 'medium', 'dark']:
        class_spots = [s for s in spots_3d if s.classification == classification]
        if class_spots:
            fig.add_trace(go.Scatter3d(
                x=[s.x * 1.05 for s in class_spots],
                y=[s.y * 1.05 for s in class_spots],
                z=[s.z * 1.05 for s in class_spots],
                mode='markers',
                marker=dict(
                    size=[max(4, min(10, s.area / 80)) for s in class_spots],
                    color=color_map[classification],
                    opacity=0.9,
                    line=dict(width=1, color='black'),
                ),
                name=f'{classification.capitalize()} ({len(class_spots)})',
                text=[f"Spot #{s.id}<br>View: {s.source_view.value}" for s in class_spots],
                hoverinfo='text',
            ))

    fig.update_layout(
        title='3D Textured Head (Plotly)',
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            aspectmode='data',
            camera=dict(eye=dict(x=0, y=-2, z=0.5)),
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    st.plotly_chart(fig, use_container_width=True, key="plotly_textured_fallback")


def create_reconstructed_mesh_view(
    front_img: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    spots_2d_by_view: dict,  # {view_name: (spots_list, img_width, img_height)}
):
    """
    Create a 3D visualization using MediaPipe's 3D face mesh.
    Uses the actual 3D coordinates from MediaPipe Face Mesh.

    Args:
        front_img, left_img, right_img: RGB images
        spots_2d_by_view: Dict mapping view name to (spots, width, height)
    """
    import cv2

    # Mesh enhancement options
    with st.expander("Mesh Enhancement Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            add_forehead = st.checkbox("Extend forehead/scalp", value=True,
                help="Add geometry above the face to complete the forehead")
        with col2:
            forehead_rows = st.slider("Forehead extension", 2, 10, 6,
                help="Number of vertex rows to add for forehead")

    # Back of head disabled - face only
    add_back = False

    st.info("Extracting 3D face mesh from MediaPipe...")

    # Convert RGB to BGR for face detector
    front_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
    left_bgr = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
    right_bgr = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)

    # Extract MediaPipe mesh from each view
    with FaceDetector() as detector:
        front_mesh = extract_mediapipe_mesh(front_bgr, detector)
        left_mesh = extract_mediapipe_mesh(left_bgr, detector)
        right_mesh = extract_mediapipe_mesh(right_bgr, detector)

    if front_mesh is None:
        st.warning("Could not extract face mesh from front view.")
        return

    # Fuse meshes from multiple views
    fused_mesh = fuse_multi_view_meshes(
        front_mesh, left_mesh, right_mesh,
        front_img, left_img, right_img,
    )

    base_verts = len(fused_mesh.vertices)
    base_faces = len(fused_mesh.faces)

    # Extend mesh with forehead and back of head
    if add_forehead or add_back:
        fused_mesh = extend_mesh_with_head(
            fused_mesh,
            add_forehead=add_forehead,
            add_back=add_back,
            forehead_rows=forehead_rows,
            back_segments=16,
        )

    st.success(f"Mesh created: {len(fused_mesh.vertices)} vertices ({base_verts} face + {len(fused_mesh.vertices) - base_verts} extended), {len(fused_mesh.faces)} triangles")

    # Map 2D spots to 3D mesh coordinates for front view only (best alignment)
    all_mapped_spots = []
    if 'front' in spots_2d_by_view:
        spots, w, h = spots_2d_by_view['front']
        mapped = map_spots_to_mesh_coords(spots, front_mesh, w, h)
        all_mapped_spots.extend(mapped)

    # Create Plotly visualization
    fig = go.Figure()

    # Add the face mesh
    mesh_data = mesh_to_plotly(fused_mesh)

    fig.add_trace(go.Mesh3d(
        x=mesh_data['x'],
        y=mesh_data['y'],
        z=mesh_data['z'],
        i=mesh_data['i'],
        j=mesh_data['j'],
        k=mesh_data['k'],
        vertexcolor=mesh_data['vertexcolor'],
        opacity=1.0,  # Fully opaque - no transparency
        name='Face Mesh',
        hoverinfo='skip',
        flatshading=False,
        lighting=dict(
            ambient=0.7,
            diffuse=0.9,
            specular=0.2,
            roughness=0.7,
        ),
        lightposition=dict(x=0, y=-1000, z=500),
    ))

    # Add spots as scatter points (now properly mapped to mesh coords)
    color_map = {
        'light': 'yellow',
        'medium': 'orange',
        'dark': 'red',
    }

    for classification in ['light', 'medium', 'dark']:
        class_spots = [s for s in all_mapped_spots if s['classification'] == classification]
        if class_spots:
            fig.add_trace(go.Scatter3d(
                x=[s['x'] for s in class_spots],
                y=[s['y'] for s in class_spots],
                z=[s['z'] for s in class_spots],
                mode='markers',
                marker=dict(
                    size=[max(4, min(10, s['area'] / 80)) for s in class_spots],
                    color=color_map[classification],
                    opacity=0.9,
                    line=dict(width=1, color='black'),
                ),
                name=f'{classification.capitalize()} ({len(class_spots)})',
                text=[f"Spot #{s['id']}" for s in class_spots],
                hoverinfo='text',
            ))

    # Add option to show mesh wireframe
    with st.expander("Mesh Display Options", expanded=False):
        show_wireframe = st.checkbox("Show wireframe", value=False)

    if show_wireframe:
        # Add wireframe as lines (limit for performance)
        max_wireframe_faces = min(800, len(fused_mesh.faces))
        for face in fused_mesh.faces[:max_wireframe_faces]:
            for i in range(3):
                v1 = fused_mesh.vertices[face[i]]
                v2 = fused_mesh.vertices[face[(i+1) % 3]]
                fig.add_trace(go.Scatter3d(
                    x=[v1[0], v2[0]],
                    y=[v1[1], v2[1]],
                    z=[v1[2], v2[2]],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip',
                ))

    fig.update_layout(
        title='3D Face Mesh (MediaPipe)',
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            aspectmode='data',
            # Frontal view: camera at positive Y looking at face
            camera=dict(
                eye=dict(x=0, y=2.5, z=0),  # Directly in front
                up=dict(x=0, y=0, z=1),      # Z is up
            ),
        ),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    st.plotly_chart(fig, use_container_width=True, key="plotly_reconstructed")

    # Show mesh statistics
    with st.expander("Mesh Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Vertices", len(fused_mesh.vertices))
        with col2:
            st.metric("Face Vertices", base_verts)
        with col3:
            st.metric("Total Triangles", len(fused_mesh.faces))
        with col4:
            st.metric("Views Used", sum([1 for m in [front_mesh, left_mesh, right_mesh] if m is not None]))


def process_3d_visualization(
    front_file,
    left_file,
    right_file,
    config: DetectionConfig,
    mode: DetectionMode
):
    """Process 3 photos and create 3D head visualization with spots."""
    st.info("Processing images and mapping spots to 3D model...")

    progress_bar = st.progress(0)
    all_spots_3d = []
    raw_images = {}  # Store RGB images for texture mapping
    spots_2d_by_view = {}  # Store 2D spots for mesh mapping

    # Process each view
    views = [
        (front_file, ViewAngle.FRONT, "Front"),
        (left_file, ViewAngle.LEFT, "Left"),
        (right_file, ViewAngle.RIGHT, "Right"),
    ]

    results = []

    with SpotDetector(config, mode=mode) as detector:
        for i, (file, view_angle, view_name) in enumerate(views):
            progress_bar.progress((i + 1) / 4)

            # Reset file pointer
            file.seek(0)

            # Load and process image
            image_bytes = file.read()
            image = load_image_from_bytes(image_bytes)
            h, w = image.shape[:2]

            # Store RGB image for texture mapping
            raw_images[view_name.lower()] = bgr_to_rgb(image)

            # Detect spots
            result = detector.detect(image, f"{view_name} view")
            result.annotated_image = draw_spot_annotations(image, result, draw_labels=False)
            results.append((view_name, result))

            # Store 2D spots with image dimensions for mesh mapping
            spots_2d_by_view[view_name.lower()] = (result.spots, w, h)

            # Map spots to 3D (for ellipsoid visualization)
            spots_3d = map_spots_to_3d(
                result.spots,
                img_width=w,
                img_height=h,
                view=view_angle,
                use_ellipsoid=True,
            )
            all_spots_3d.extend(spots_3d)

            st.write(f"**{view_name}**: {len(result.spots)} spots detected")

    progress_bar.progress(1.0)
    progress_bar.empty()

    # Display summary
    st.success(f"Total spots mapped to 3D: **{len(all_spots_3d)}**")

    # Count by classification
    light_count = sum(1 for s in all_spots_3d if s.classification == 'light')
    medium_count = sum(1 for s in all_spots_3d if s.classification == 'medium')
    dark_count = sum(1 for s in all_spots_3d if s.classification == 'dark')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Light Spots", light_count)
    with col2:
        st.metric("Medium Spots", medium_count)
    with col3:
        st.metric("Dark Spots", dark_count)

    # Create sub-tabs for different 3D views
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "Spots on Ellipsoid",
        "Textured Head",
        "Reconstructed Mesh"
    ])

    with viz_tab1:
        # Original ellipsoid visualization
        st.subheader("3D Head Model (Spots Only)")
        st.caption("Drag to rotate, scroll to zoom. Click legend items to show/hide spot categories.")

        # Create the 3D figure with animation
        fig = create_rotation_animation(all_spots_3d, use_ellipsoid=True)
        st.plotly_chart(fig, use_container_width=True, key="plotly_ellipsoid")

    with viz_tab2:
        # Textured head visualization using Plotly (PyVista has macOS threading issues)
        st.subheader("3D Textured Head Model")
        st.caption("Photos mapped as texture with spots overlaid. Drag to rotate.")

        try:
            # Create composite texture from photos
            texture_img = project_photo_to_texture_simple(
                raw_images['front'],
                raw_images['left'],
                raw_images['right'],
            )
            texture_with_spots = add_spots_to_texture(texture_img, all_spots_3d)

            # Show the composite texture
            with st.expander("View Texture Map (Cylindrical Unwrap)", expanded=False):
                st.caption("Left: Right side of face | Center: Front | Right: Left side of face")
                st.image(texture_with_spots, width="stretch")

            # Show interactive 3D visualization using Plotly
            create_plotly_textured_head(texture_with_spots, all_spots_3d)

        except Exception as e:
            st.error(f"Error creating textured visualization: {str(e)}")

    with viz_tab3:
        # Landmark-based 3D reconstruction
        st.subheader("3D Reconstructed Head (MediaPipe)")
        st.caption("Mesh using MediaPipe 478 landmarks. Drag to rotate.")

        try:
            create_reconstructed_mesh_view(
                raw_images['front'],
                raw_images['left'],
                raw_images['right'],
                spots_2d_by_view,  # Pass 2D spots for proper mesh alignment
            )
        except Exception as e:
            st.error(f"Error creating reconstructed mesh: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # Show individual view results
    st.subheader("Individual View Results")

    cols = st.columns(3)
    for i, (view_name, result) in enumerate(results):
        with cols[i]:
            st.markdown(f"**{view_name} View** - {result.spot_count} spots")
            if result.annotated_image is not None:
                st.image(
                    bgr_to_rgb(result.annotated_image),
                    width="stretch",
                )

    # 3D spots table
    if all_spots_3d:
        with st.expander("3D Spot Coordinates"):
            table_data = []
            for spot in all_spots_3d:
                table_data.append({
                    "ID": spot.id,
                    "View": spot.source_view.value,
                    "Classification": spot.classification,
                    "X": f"{spot.x:.3f}",
                    "Y": f"{spot.y:.3f}",
                    "Z": f"{spot.z:.3f}",
                    "Area": spot.area,
                    "Score": f"{spot.combined_delta:.1f}",
                })
            st.dataframe(table_data, use_container_width=True)


if __name__ == "__main__":
    main()