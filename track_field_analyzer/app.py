"""
Track & Field Form Analyzer - Streamlit Application

A computer vision tool for analyzing sprint form using pose estimation.
Supports 60m-200m sprint events with focus on block starts, drive phase,
and acceleration mechanics.

Inspired by the Ochy app design - modern, single-page analysis view.

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
    initial_sidebar_state="collapsed",
)

# Modern Dark Theme CSS (Ochy-inspired)
st.markdown("""
<style>
    /* Dark theme background */
    .stApp {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
    }
    
    /* Main container */
    .main .block-container {
        padding: 1rem 2rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .app-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .app-subtitle {
        color: #8b8b9a;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    
    /* Video container */
    .video-container {
        background: #1e1e32;
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Score card styling */
    .score-card {
        background: linear-gradient(135deg, #1e1e32 0%, #2a2a42 100%);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .score-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .score-label {
        color: #8b8b9a;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .score-value {
        font-size: 2rem;
        font-weight: 700;
        color: #fff;
    }
    
    .score-value.good { color: #22c55e; }
    .score-value.okay { color: #eab308; }
    .score-value.poor { color: #ef4444; }
    
    /* Status badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-good {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid #22c55e;
    }
    
    .badge-okay {
        background: rgba(234, 179, 8, 0.2);
        color: #eab308;
        border: 1px solid #eab308;
    }
    
    .badge-poor {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    /* Metric detail */
    .metric-detail {
        color: #a0a0b0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        line-height: 1.4;
    }
    
    /* Total score card */
    .total-score-card {
        background: linear-gradient(135deg, #2a2a42 0%, #1e1e32 100%);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid rgba(124, 58, 237, 0.3);
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.15);
    }
    
    .total-score-label {
        color: #8b8b9a;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .total-score-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Phase indicator */
    .phase-indicator {
        background: linear-gradient(135deg, #1e1e32 0%, #2a2a42 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .phase-name {
        font-size: 1.25rem;
        font-weight: 600;
        color: #fff;
    }
    
    .phase-description {
        color: #8b8b9a;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #8b8b9a;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7c3aed, #00d4ff);
        color: #fff !important;
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        padding: 0.5rem 0;
    }
    
    /* Navigation buttons */
    .nav-button {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        color: #fff;
        padding: 0.5rem 1rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .nav-button:hover {
        background: rgba(255,255,255,0.2);
    }
    
    /* Timeline */
    .timeline-container {
        background: rgba(0,0,0,0.3);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
    }
    
    /* Feedback section */
    .feedback-section {
        background: linear-gradient(135deg, #1e1e32 0%, #2a2a42 100%);
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .feedback-title {
        color: #fff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .feedback-item {
        background: rgba(234, 179, 8, 0.1);
        border-left: 3px solid #eab308;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        color: #e0e0e0;
    }
    
    .feedback-item.good {
        background: rgba(34, 197, 94, 0.1);
        border-left-color: #22c55e;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #7c3aed, #00d4ff);
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
    if "aggregated_metrics" not in st.session_state:
        st.session_state.aggregated_metrics = None


def calculate_form_score(metrics: FrameMetrics | None, aggregated: dict | None) -> tuple[int, str]:
    """Calculate overall form score based on metrics."""
    if metrics is None and aggregated is None:
        return 0, "poor"
    
    score = 50  # Base score
    
    if metrics:
        # Adjust based on phase detection
        if metrics.phase == SprintPhase.SET:
            trunk_lean = abs(metrics.angles.get("trunk_lean", 0))
            if 40 <= trunk_lean <= 55:
                score += 20
            elif 30 <= trunk_lean <= 60:
                score += 10
            
            # Knee angles
            left_knee = metrics.angles.get("left_knee", 0)
            right_knee = metrics.angles.get("right_knee", 0)
            front_knee = min(left_knee or 180, right_knee or 180)
            if 90 <= front_knee <= 110:
                score += 15
            elif 80 <= front_knee <= 120:
                score += 8
        
        elif metrics.phase in [SprintPhase.DRIVE, SprintPhase.ACCELERATION]:
            trunk_lean = abs(metrics.angles.get("trunk_lean", 0))
            if metrics.phase == SprintPhase.DRIVE:
                if 30 <= trunk_lean <= 50:
                    score += 25
                elif 20 <= trunk_lean <= 55:
                    score += 15
            else:
                if 15 <= trunk_lean <= 35:
                    score += 25
                elif 10 <= trunk_lean <= 40:
                    score += 15
        
        elif metrics.phase == SprintPhase.MAX_VELOCITY:
            trunk_lean = abs(metrics.angles.get("trunk_lean", 0))
            if trunk_lean <= 15:
                score += 25
            elif trunk_lean <= 25:
                score += 15
        
        # Penalize for unknown phase
        if metrics.phase == SprintPhase.UNKNOWN:
            score -= 15
    
    # Clamp score
    score = max(0, min(100, score))
    
    # Determine rating
    if score >= 75:
        rating = "good"
    elif score >= 50:
        rating = "okay"
    else:
        rating = "poor"
    
    return score, rating


def get_metric_score(value: float | None, target_min: float, target_max: float, optimal: float) -> tuple[int, str]:
    """Calculate score for a specific metric."""
    if value is None or pd.isna(value):
        return 0, "poor"
    
    value = abs(value)  # Use absolute for angles like trunk lean
    
    if target_min <= value <= target_max:
        # Within range - calculate how close to optimal
        if value == optimal:
            score = 100
        else:
            range_size = target_max - target_min
            distance_from_optimal = abs(value - optimal)
            score = max(60, 100 - int((distance_from_optimal / range_size) * 40))
        rating = "good"
    elif target_min - 10 <= value <= target_max + 10:
        # Close to range
        score = 50
        rating = "okay"
    else:
        # Outside range
        score = 30
        rating = "poor"
    
    return score, rating


def render_score_card(label: str, value: float | None, unit: str, score: int, rating: str, detail: str = "") -> None:
    """Render a metric score card."""
    badge_class = f"badge-{rating}"
    value_class = rating
    
    value_display = f"{value:.0f}" if value and not pd.isna(value) else "—"
    
    st.markdown(f"""
    <div class="score-card">
        <div class="score-header">
            <span class="score-label">{label}</span>
            <span class="badge {badge_class}">{rating.title()}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="score-value {value_class}">{score}%</span>
            <span style="color: #fff; font-size: 1.5rem; font-weight: 600;">{value_display}{unit}</span>
        </div>
        {f'<div class="metric-detail">{detail}</div>' if detail else ''}
    </div>
    """, unsafe_allow_html=True)


def render_analysis_view() -> None:
    """Render the main analysis view with video and metrics side by side."""
    
    if not st.session_state.processing_complete:
        return
    
    processed_frames = st.session_state.processed_frames
    frame_metrics = st.session_state.frame_metrics
    
    if not processed_frames or not frame_metrics:
        st.warning("No analysis data available.")
        return
    
    # Get aggregated metrics
    if st.session_state.aggregated_metrics is None:
        st.session_state.aggregated_metrics = aggregate_metrics(frame_metrics)
    aggregated = st.session_state.aggregated_metrics
    
    # Get current frame metrics
    current_idx = min(st.session_state.current_frame_idx, len(frame_metrics) - 1)
    current_metrics = frame_metrics[current_idx] if current_idx < len(frame_metrics) else None
    
    # Calculate overall score
    total_score, total_rating = calculate_form_score(current_metrics, aggregated)
    
    # Layout: Video on left, metrics on right
    col_video, col_metrics = st.columns([3, 2])
    
    with col_video:
        # Frame navigation
        nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
        
        with nav_col1:
            if st.button("◀ Prev", use_container_width=True):
                if st.session_state.current_frame_idx > 0:
                    st.session_state.current_frame_idx -= 1
                    st.rerun()
        
        with nav_col2:
            frame_idx = st.slider(
                "Frame",
                0, len(processed_frames) - 1,
                st.session_state.current_frame_idx,
                label_visibility="collapsed"
            )
            if frame_idx != st.session_state.current_frame_idx:
                st.session_state.current_frame_idx = frame_idx
                st.rerun()
        
        with nav_col3:
            if st.button("Next ▶", use_container_width=True):
                if st.session_state.current_frame_idx < len(processed_frames) - 1:
                    st.session_state.current_frame_idx += 1
                    st.rerun()
        
        # Display current frame
        current_frame = processed_frames[st.session_state.current_frame_idx]
        st.image(current_frame, use_container_width=True)
        
        # Timeline info
        if current_metrics:
            st.markdown(f"""
            <div class="timeline-container">
                <div style="display: flex; justify-content: space-between; color: #8b8b9a;">
                    <span>Frame {current_metrics.frame_index}</span>
                    <span>{current_metrics.timestamp_sec:.2f}s</span>
                    <span>{st.session_state.current_frame_idx + 1} / {len(processed_frames)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_metrics:
        # Tabs for different views
        tab_analysis, tab_metrics, tab_details = st.tabs(["📊 Analysis", "📐 Metrics", "📋 Details"])
        
        with tab_analysis:
            # Total score card
            st.markdown(f"""
            <div class="total-score-card">
                <div class="total-score-label">🏆 Total Score</div>
                <div class="total-score-value">{total_score}%</div>
                <span class="badge badge-{total_rating}" style="margin-top: 0.5rem;">{total_rating.title()}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Phase indicator
            if current_metrics:
                phase = current_metrics.phase
                phase_colors = {
                    SprintPhase.SET: "#ff6464",
                    SprintPhase.DRIVE: "#ffa500",
                    SprintPhase.ACCELERATION: "#eab308",
                    SprintPhase.MAX_VELOCITY: "#22c55e",
                    SprintPhase.UNKNOWN: "#8b8b9a",
                }
                phase_color = phase_colors.get(phase, "#8b8b9a")
                
                st.markdown(f"""
                <div class="phase-indicator">
                    <div class="phase-name" style="color: {phase_color};">
                        {phase.display_name}
                    </div>
                    <div class="phase-description">
                        {get_phase_description(phase)[:100]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Key metrics as score cards
            if current_metrics:
                angles = current_metrics.angles
                
                # Trunk lean
                trunk_lean = angles.get("trunk_lean")
                trunk_score, trunk_rating = get_metric_score(trunk_lean, 15, 50, 35)
                render_score_card(
                    "Trunk Lean",
                    abs(trunk_lean) if trunk_lean else None,
                    "°",
                    trunk_score,
                    trunk_rating,
                    "Forward lean angle from vertical"
                )
                
                # Front knee (min of left/right)
                left_knee = angles.get("left_knee")
                right_knee = angles.get("right_knee")
                front_knee = min(left_knee or 180, right_knee or 180) if (left_knee or right_knee) else None
                knee_score, knee_rating = get_metric_score(front_knee, 90, 140, 110)
                render_score_card(
                    "Knee Drive",
                    front_knee,
                    "°",
                    knee_score,
                    knee_rating,
                    "Front knee angle for power output"
                )
        
        with tab_metrics:
            # All angles in a clean layout
            if current_metrics:
                angles = current_metrics.angles
                
                st.markdown("### Joint Angles")
                
                metric_data = [
                    ("Left Knee", angles.get("left_knee"), "°"),
                    ("Right Knee", angles.get("right_knee"), "°"),
                    ("Left Hip", angles.get("left_hip"), "°"),
                    ("Right Hip", angles.get("right_hip"), "°"),
                    ("Left Elbow", angles.get("left_elbow"), "°"),
                    ("Right Elbow", angles.get("right_elbow"), "°"),
                    ("Trunk Lean", abs(angles.get("trunk_lean", 0)) if angles.get("trunk_lean") else None, "°"),
                ]
                
                for i in range(0, len(metric_data), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(metric_data):
                            label, value, unit = metric_data[i + j]
                            with col:
                                if value is not None and not pd.isna(value):
                                    st.metric(label, f"{value:.1f}{unit}")
                                else:
                                    st.metric(label, "—")
                
                st.markdown("---")
                st.markdown("### Averages")
                
                avg_angles = aggregated.get("avg_angles", {})
                for key, value in avg_angles.items():
                    display_name = key.replace("_", " ").title()
                    st.markdown(f"**{display_name}:** {value:.1f}°")
        
        with tab_details:
            # Phase sequence
            st.markdown("### Phase Sequence")
            phase_seq = aggregated.get("phase_sequence", [])
            if phase_seq:
                for p in phase_seq:
                    phase = SprintPhase(p['phase'])
                    st.markdown(f"- **{phase.display_name}** at {p['timestamp']:.2f}s")
            
            st.markdown("---")
            
            # Feedback
            st.markdown("### 💡 Recommendations")
            feedback_list = aggregated.get("overall_feedback", [])
            if feedback_list:
                for fb in feedback_list:
                    st.markdown(f"""
                    <div class="feedback-item">
                        {fb}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feedback-item good">
                    ✅ Form looks good! No major issues detected.
                </div>
                """, unsafe_allow_html=True)


def process_video(
    video_path: str,
    settings: dict[str, Any],
    progress_bar: Any,
    status_text: Any,
) -> tuple[list[np.ndarray], list[FrameMetrics]]:
    """Process video and return annotated frames and metrics."""
    
    target_config = load_target_ranges()
    props = get_video_properties(video_path)
    st.session_state.video_properties = props
    
    status_text.text(f"Video: {props['width']}x{props['height']}, "
                     f"{props['fps']:.1f} FPS, {props['duration_sec']:.1f}s")
    
    processed_frames = []
    frame_metrics_list = []
    
    with PoseEstimator(
        static_image_mode=False,
        model_complexity=settings["model_complexity"],
        min_detection_confidence=settings["min_confidence"],
        min_tracking_confidence=settings["min_confidence"],
    ) as estimator:
        
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
            progress = min((i + 1) / max(total_frames, 1), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing frame {frame_idx}... ({i + 1}/{total_frames})")
            
            pose_result = estimator.process_frame(frame_rgb)
            
            if pose_result is not None:
                metrics = compute_frame_metrics(
                    frame_index=frame_idx,
                    fps=props["fps"],
                    landmarks=pose_result.landmarks,
                    target_config=target_config,
                    visibility_threshold=settings["min_confidence"],
                )
                
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
                processed_frames.append(frame_rgb)
    
    return processed_frames, frame_metrics_list


def main() -> None:
    """Main application entry point."""
    
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">🏃 Sprint Form Analyzer</h1>
        <p class="app-subtitle">AI-powered analysis for 60m-200m events</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have analysis results to show
    if st.session_state.processing_complete:
        # Show analysis view
        render_analysis_view()
        
        # Button to upload new video
        st.markdown("---")
        if st.button("📹 Analyze New Video", type="secondary"):
            st.session_state.processing_complete = False
            st.session_state.processed_frames = []
            st.session_state.frame_metrics = []
            st.session_state.aggregated_metrics = None
            st.rerun()
    else:
        # Show upload UI
        st.markdown("### 📹 Upload Your Sprint Video")
        st.markdown("Supported formats: MP4, MOV, AVI")
        
        # Settings in expander
        with st.expander("⚙️ Analysis Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                sample_rate = st.slider("Sample every Nth frame", 1, 10, 3)
                max_frames = st.number_input("Max frames to process", 10, 500, 100)
            with col2:
                model_complexity = st.selectbox("Model quality", [0, 1, 2], index=1,
                    format_func=lambda x: ["Lite (fast)", "Full", "Heavy (accurate)"][x])
                min_confidence = st.slider("Detection confidence", 0.3, 0.9, 0.5)
        
        settings = {
            "sample_rate": sample_rate,
            "max_frames": max_frames,
            "model_complexity": model_complexity,
            "min_confidence": min_confidence,
            "show_angles": True,
            "show_frame_info": True,
        }
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi"],
            label_visibility="collapsed"
        )
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            if st.button("🚀 Analyze Video", type="primary", use_container_width=True):
                st.session_state.processed_frames = []
                st.session_state.frame_metrics = []
                st.session_state.current_frame_idx = 0
                st.session_state.aggregated_metrics = None
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Loading video...")
                    video_path = load_video_from_uploaded_file(uploaded_video)
                    
                    processed_frames, frame_metrics = process_video(
                        video_path, settings, progress_bar, status_text
                    )
                    
                    st.session_state.processed_frames = processed_frames
                    st.session_state.frame_metrics = frame_metrics
                    st.session_state.processing_complete = True
                    
                    cleanup_temp_file(video_path)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Analysis complete!")
                    
                    # Rerun to show analysis view
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    status_text.text("❌ Processing failed")


if __name__ == "__main__":
    main()
