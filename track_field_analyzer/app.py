"""
Track & Field Form Analyzer - Streamlit Application

A computer vision tool for analyzing sprint form using pose estimation.
Supports 60m-200m sprint events with focus on block starts, drive phase,
and acceleration mechanics.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Import our modules
from src.io.video import (
    load_video_from_uploaded_file,
    load_image_from_uploaded_file,
    get_video_properties,
    sample_frames,
    cleanup_temp_file,
)
from src.pose.mediapipe_pose import PoseEstimator, PoseResult
from src.analysis.angles import extract_joint_angles, get_hip_height_normalized
from src.analysis.metrics import (
    compute_frame_metrics,
    aggregate_metrics,
    load_target_ranges,
    FrameMetrics,
)
from src.analysis.phases import SprintPhase, get_phase_description
from src.viz.overlay import annotate_frame


# Page configuration
st.set_page_config(
    page_title="Track & Field Form Analyzer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .phase-set { color: #ff6464; }
    .phase-drive { color: #ffa500; }
    .phase-acceleration { color: #ffff00; }
    .phase-max-velocity { color: #64ff64; }
    .feedback-item {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .stProgress .st-bo {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize session state variables."""
    if "processed_frames" not in st.session_state:
        st.session_state.processed_frames = []
    if "frame_metrics" not in st.session_state:
        st.session_state.frame_metrics = []
    if "current_frame_idx" not in st.session_state:
        st.session_state.current_frame_idx = 0
    if "video_properties" not in st.session_state:
        st.session_state.video_properties = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False


def render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Analysis Settings")
        
        st.markdown("### Frame Sampling")
        sample_rate = st.slider(
            "Process every Nth frame",
            min_value=1,
            max_value=10,
            value=3,
            help="Higher values = faster processing, less detail"
        )
        
        max_frames = st.number_input(
            "Maximum frames to process",
            min_value=10,
            max_value=500,
            value=100,
            help="Limit processing for long videos"
        )
        
        st.markdown("### Pose Detection")
        model_complexity = st.selectbox(
            "Model complexity",
            options=[0, 1, 2],
            index=1,
            help="0=Lite, 1=Full, 2=Heavy (more accurate but slower)"
        )
        
        min_confidence = st.slider(
            "Minimum detection confidence",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.1,
        )
        
        st.markdown("### Visualization")
        show_angles = st.checkbox("Show angle values", value=True)
        show_frame_info = st.checkbox("Show frame info", value=True)
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This tool analyzes sprint form using AI pose estimation.
        
        **Supported events:** 60m, 100m, 200m
        
        **Phases detected:**
        - 🔴 Set Position
        - 🟠 Drive Phase
        - 🟡 Acceleration
        - 🟢 Max Velocity
        
        *Target ranges based on sprint biomechanics literature.
        Individual optimal ranges vary.*
        """)
    
    return {
        "sample_rate": sample_rate,
        "max_frames": max_frames,
        "model_complexity": model_complexity,
        "min_confidence": min_confidence,
        "show_angles": show_angles,
        "show_frame_info": show_frame_info,
    }


def process_video(
    video_path: str,
    settings: dict[str, Any],
    progress_bar: Any,
    status_text: Any,
) -> tuple[list[np.ndarray], list[FrameMetrics]]:
    """Process video and return annotated frames and metrics."""
    
    # Load target config
    target_config = load_target_ranges()
    
    # Get video properties
    props = get_video_properties(video_path)
    st.session_state.video_properties = props
    
    status_text.text(f"Video: {props['width']}x{props['height']}, "
                     f"{props['fps']:.1f} FPS, {props['duration_sec']:.1f}s")
    
    processed_frames = []
    frame_metrics_list = []
    
    # Initialize pose estimator
    with PoseEstimator(
        static_image_mode=False,
        model_complexity=settings["model_complexity"],
        min_detection_confidence=settings["min_confidence"],
        min_tracking_confidence=settings["min_confidence"],
    ) as estimator:
        
        # Calculate total frames to process
        total_frames = min(
            props["frame_count"] // settings["sample_rate"],
            settings["max_frames"]
        )
        
        frame_generator = sample_frames(
            video_path,
            sample_rate=settings["sample_rate"],
            max_frames=settings["max_frames"],
        )
        
        for i, (frame_idx, frame_rgb) in enumerate(frame_generator):
            # Update progress
            progress = (i + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx}... ({i + 1}/{total_frames})")
            
            # Run pose estimation
            pose_result = estimator.process_frame(frame_rgb)
            
            if pose_result is not None:
                # Compute metrics
                metrics = compute_frame_metrics(
                    frame_index=frame_idx,
                    fps=props["fps"],
                    landmarks=pose_result.landmarks,
                    target_config=target_config,
                    visibility_threshold=settings["min_confidence"],
                )
                
                # Annotate frame
                annotated = annotate_frame(
                    frame=frame_rgb,
                    landmarks=pose_result.landmarks,
                    angles=metrics.angles,
                    phase=metrics.phase,
                    frame_index=frame_idx,
                    timestamp=metrics.timestamp_sec,
                    draw_angles=settings["show_angles"],
                    draw_info=settings["show_frame_info"],
                    visibility_threshold=settings["min_confidence"],
                )
                
                processed_frames.append(annotated)
                frame_metrics_list.append(metrics)
            else:
                # No pose detected, add original frame
                processed_frames.append(frame_rgb)
    
    return processed_frames, frame_metrics_list


def process_image(
    image: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, FrameMetrics | None]:
    """Process single image and return annotated frame and metrics."""
    
    target_config = load_target_ranges()
    
    with PoseEstimator(
        static_image_mode=True,
        model_complexity=settings["model_complexity"],
        min_detection_confidence=settings["min_confidence"],
    ) as estimator:
        
        pose_result = estimator.process_frame(image)
        
        if pose_result is not None:
            metrics = compute_frame_metrics(
                frame_index=0,
                fps=1.0,
                landmarks=pose_result.landmarks,
                target_config=target_config,
                visibility_threshold=settings["min_confidence"],
            )
            
            annotated = annotate_frame(
                frame=image,
                landmarks=pose_result.landmarks,
                angles=metrics.angles,
                phase=metrics.phase,
                frame_index=0,
                timestamp=0.0,
                draw_angles=settings["show_angles"],
                draw_info=settings["show_frame_info"],
                visibility_threshold=settings["min_confidence"],
            )
            
            return annotated, metrics
        
        return image, None


def render_metrics_panel(
    frame_metrics: list[FrameMetrics],
    aggregated: dict[str, Any],
) -> None:
    """Render the metrics analysis panel."""
    
    st.markdown("## 📊 Analysis Results")
    
    # Phase distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Phase Distribution")
        phase_dist = aggregated.get("phase_distribution", {})
        if phase_dist:
            phase_df = pd.DataFrame([
                {"Phase": SprintPhase(k).display_name, "Frames": v}
                for k, v in phase_dist.items()
            ])
            st.bar_chart(phase_df.set_index("Phase"))
    
    with col2:
        st.markdown("### Average Angles")
        avg_angles = aggregated.get("avg_angles", {})
        if avg_angles:
            # Display in a nicer format
            angle_display = {
                "Left Knee": avg_angles.get("left_knee"),
                "Right Knee": avg_angles.get("right_knee"),
                "Left Hip": avg_angles.get("left_hip"),
                "Right Hip": avg_angles.get("right_hip"),
                "Left Elbow": avg_angles.get("left_elbow"),
                "Right Elbow": avg_angles.get("right_elbow"),
                "Trunk Lean": avg_angles.get("trunk_lean"),
            }
            
            for name, value in angle_display.items():
                if value is not None:
                    st.metric(name, f"{value:.1f}°")
    
    # Phase sequence
    st.markdown("### Phase Sequence")
    phase_seq = aggregated.get("phase_sequence", [])
    if phase_seq:
        seq_text = " → ".join([
            f"{SprintPhase(p['phase']).display_name} ({p['timestamp']:.2f}s)"
            for p in phase_seq
        ])
        st.info(seq_text)
    
    # Feedback
    st.markdown("### 💡 Feedback & Recommendations")
    feedback_list = aggregated.get("overall_feedback", [])
    if feedback_list:
        for fb in feedback_list:
            st.markdown(f"""
            <div class="feedback-item">
                {fb}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Form looks good! No major issues detected.")
    
    # Detailed metrics table
    with st.expander("📋 Detailed Frame-by-Frame Metrics"):
        if frame_metrics:
            metrics_data = [fm.to_dict() for fm in frame_metrics]
            df = pd.DataFrame(metrics_data)
            st.dataframe(df, use_container_width=True)


def render_frame_viewer(processed_frames: list[np.ndarray]) -> None:
    """Render the frame viewer with navigation."""
    
    if not processed_frames:
        return
    
    st.markdown("## 🎬 Frame Viewer")
    
    total_frames = len(processed_frames)
    
    # Frame navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("⬅️ Previous"):
            if st.session_state.current_frame_idx > 0:
                st.session_state.current_frame_idx -= 1
    
    with col2:
        frame_idx = st.slider(
            "Frame",
            0, total_frames - 1,
            st.session_state.current_frame_idx,
            key="frame_slider"
        )
        st.session_state.current_frame_idx = frame_idx
    
    with col3:
        if st.button("Next ➡️"):
            if st.session_state.current_frame_idx < total_frames - 1:
                st.session_state.current_frame_idx += 1
    
    # Display current frame
    current_frame = processed_frames[st.session_state.current_frame_idx]
    st.image(current_frame, caption=f"Frame {st.session_state.current_frame_idx + 1} of {total_frames}")


def main() -> None:
    """Main application entry point."""
    
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">🏃 Track & Field Form Analyzer</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered sprint form analysis for 60m-200m events</p>',
                unsafe_allow_html=True)
    
    # Sidebar
    settings = render_sidebar()
    
    # Main content
    tab1, tab2 = st.tabs(["📹 Video Analysis", "📷 Image Analysis"])
    
    with tab1:
        st.markdown("### Upload a Video")
        st.markdown("Supported formats: MP4, MOV, AVI")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi"],
            key="video_uploader"
        )
        
        if uploaded_video is not None:
            # Show video preview
            st.video(uploaded_video)
            
            if st.button("🚀 Analyze Video", type="primary"):
                # Reset state
                st.session_state.processed_frames = []
                st.session_state.frame_metrics = []
                st.session_state.current_frame_idx = 0
                st.session_state.processing_complete = False
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Save uploaded file to temp location
                    status_text.text("Loading video...")
                    video_path = load_video_from_uploaded_file(uploaded_video)
                    
                    # Process video
                    processed_frames, frame_metrics = process_video(
                        video_path, settings, progress_bar, status_text
                    )
                    
                    # Store results
                    st.session_state.processed_frames = processed_frames
                    st.session_state.frame_metrics = frame_metrics
                    st.session_state.processing_complete = True
                    
                    # Cleanup
                    cleanup_temp_file(video_path)
                    
                    status_text.text("✅ Processing complete!")
                    progress_bar.progress(1.0)
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    status_text.text("❌ Processing failed")
        
        # Show results if processing is complete
        if st.session_state.processing_complete and st.session_state.frame_metrics:
            st.markdown("---")
            
            # Frame viewer
            render_frame_viewer(st.session_state.processed_frames)
            
            st.markdown("---")
            
            # Metrics panel
            aggregated = aggregate_metrics(st.session_state.frame_metrics)
            render_metrics_panel(st.session_state.frame_metrics, aggregated)
    
    with tab2:
        st.markdown("### Upload an Image")
        st.markdown("Analyze a single frame (e.g., block start position)")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
        
        if uploaded_image is not None:
            # Load and display original
            try:
                image = load_image_from_uploaded_file(uploaded_image)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Original")
                    st.image(image, use_container_width=True)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing..."):
                        annotated, metrics = process_image(image, settings)
                    
                    with col2:
                        st.markdown("#### Analyzed")
                        st.image(annotated, use_container_width=True)
                    
                    if metrics:
                        st.markdown("---")
                        st.markdown("### Analysis Results")
                        
                        # Display phase
                        phase = metrics.phase
                        st.markdown(f"**Detected Phase:** {phase.display_name}")
                        st.info(get_phase_description(phase))
                        
                        # Display angles
                        st.markdown("#### Joint Angles")
                        angle_cols = st.columns(4)
                        angle_names = [
                            ("Left Knee", "left_knee"),
                            ("Right Knee", "right_knee"),
                            ("Left Hip", "left_hip"),
                            ("Right Hip", "right_hip"),
                            ("Left Elbow", "left_elbow"),
                            ("Right Elbow", "right_elbow"),
                            ("Trunk Lean", "trunk_lean"),
                        ]
                        
                        for i, (display_name, key) in enumerate(angle_names):
                            value = metrics.angles.get(key)
                            if value is not None and not pd.isna(value):
                                with angle_cols[i % 4]:
                                    st.metric(display_name, f"{value:.1f}°")
                        
                        # Feedback
                        if metrics.feedback:
                            st.markdown("#### Feedback")
                            for fb in metrics.feedback:
                                st.warning(fb)
                    else:
                        st.warning("No pose detected in image. "
                                   "Ensure full body is visible.")
                        
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")


if __name__ == "__main__":
    main()
