"""
Sprint Form Analyzer - Premium Sports Dashboard
A best-in-class biomechanics analysis tool for sprinters (60m-200m).

Design inspired by Ochy app - clean, actionable, trustworthy.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.io.video import (
    load_video_from_uploaded_file,
    get_video_properties,
    sample_frames,
    cleanup_temp_file,
)
from src.pose.mediapipe_pose import PoseEstimator
from src.analysis.metrics import (
    compute_frame_metrics,
    aggregate_metrics,
    load_target_ranges,
    FrameMetrics,
)
from src.analysis.phases import SprintPhase, get_phase_description
from src.viz.overlay import annotate_frame

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Sprint Form Analyzer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS - Premium Sports Dashboard Theme
# =============================================================================
st.markdown("""
<style>
/* =========================== */
/* PREMIUM SPORTS THEME        */
/* =========================== */

/* Hide default elements */
#MainMenu, footer, .stDeployButton {display: none;}

/* Typography */
h1, h2, h3 {font-weight: 600 !important;}

/* Hero Score Card */
.hero-score {
    background: linear-gradient(135deg, #1a1a2e 0%, #0d0d1a 100%);
    border: 2px solid #00d4ff;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.15);
}

.hero-score-value {
    font-size: 4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

.hero-score-label {
    color: #a0a0a0;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}

.hero-summary {
    color: #e8e8e8;
    font-size: 1rem;
    margin-top: 1rem;
    line-height: 1.5;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.metric-label {
    color: #a0a0a0;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #e8e8e8;
    margin-right: 0.5rem;
}

.metric-unit {
    color: #a0a0a0;
    font-size: 1rem;
}

/* Status Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-good {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.4);
}

.badge-okay {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-poor {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

/* Context Bar (target range) */
.context-bar {
    height: 8px;
    background: linear-gradient(90deg, 
        #ef4444 0%, 
        #f59e0b 20%, 
        #22c55e 40%, 
        #22c55e 60%, 
        #f59e0b 80%, 
        #ef4444 100%);
    border-radius: 4px;
    margin: 0.75rem 0;
    position: relative;
}

.context-marker {
    position: absolute;
    width: 4px;
    height: 16px;
    background: #fff;
    border-radius: 2px;
    top: -4px;
    transform: translateX(-50%);
    box-shadow: 0 0 10px rgba(255,255,255,0.5);
}

/* Coaching Cue */
.coaching-cue {
    color: #a0a0a0;
    font-size: 0.85rem;
    line-height: 1.4;
    margin-top: 0.5rem;
    padding-left: 0.5rem;
    border-left: 2px solid #00d4ff;
}

/* Focus Area Cards */
.focus-card {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.25rem;
    height: 100%;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.focus-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.focus-title {
    color: #e8e8e8;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.focus-value {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.focus-cue {
    color: #a0a0a0;
    font-size: 0.85rem;
    line-height: 1.4;
}

/* Phase Badge */
.phase-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 600;
}

/* Video Container */
.video-container {
    background: #0a0a14;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Info Box */
.info-box {
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 12px;
    padding: 1rem;
    color: #e8e8e8;
}

.warning-box {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 12px;
    padding: 1rem;
    color: #e8e8e8;
}

/* Sidebar styling */
.sidebar-header {
    color: #00d4ff;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem;
    color: #a0a0a0;
}

.empty-state h3 {
    color: #e8e8e8;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    defaults = {
        "processed_frames": [],
        "frame_metrics": [],
        "current_frame_idx": 0,
        "processing_complete": False,
        "aggregated_metrics": None,
        "video_properties": None,
        # User profile
        "user_event": "100m",
        "user_level": "Intermediate",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_rating(value: float, optimal: float, good_range: float = 10, okay_range: float = 20) -> tuple[str, str]:
    """Get rating based on distance from optimal."""
    if value is None or pd.isna(value):
        return "poor", "—"
    
    diff = abs(value - optimal)
    if diff <= good_range:
        return "good", "✓ Good"
    elif diff <= okay_range:
        return "okay", "⚠ Okay"
    else:
        return "poor", "✗ Needs Work"


def calculate_form_score(metrics: FrameMetrics | None) -> tuple[float, str, str]:
    """Calculate overall form score (0-10)."""
    if metrics is None:
        return 0, "poor", "No pose detected"
    
    score = 5.0  # Base
    notes = []
    
    angles = metrics.angles
    trunk = abs(angles.get("trunk_lean", 0) or 0)
    
    # Phase-appropriate scoring
    if metrics.phase == SprintPhase.SET:
        if 40 <= trunk <= 55:
            score += 2
            notes.append("good forward lean")
        elif 30 <= trunk <= 60:
            score += 1
    elif metrics.phase == SprintPhase.DRIVE:
        if 30 <= trunk <= 50:
            score += 2.5
            notes.append("strong drive angle")
        elif 20 <= trunk <= 55:
            score += 1.5
    elif metrics.phase == SprintPhase.ACCELERATION:
        if 15 <= trunk <= 35:
            score += 2
        elif 10 <= trunk <= 40:
            score += 1
    elif metrics.phase == SprintPhase.MAX_VELOCITY:
        if trunk <= 15:
            score += 2.5
            notes.append("good upright posture")
        elif trunk <= 25:
            score += 1.5
    
    # Knee drive bonus
    left_knee = angles.get("left_knee")
    right_knee = angles.get("right_knee")
    if left_knee and right_knee:
        front_knee = min(left_knee, right_knee)
        if 90 <= front_knee <= 120:
            score += 1
    
    # Unknown phase penalty
    if metrics.phase == SprintPhase.UNKNOWN:
        score -= 1
    
    score = max(0, min(10, score))
    
    if score >= 7:
        rating = "good"
    elif score >= 5:
        rating = "okay"
    else:
        rating = "poor"
    
    summary = ", ".join(notes) if notes else "Keep working on form"
    return score, rating, summary


def get_coaching_cue(metric_name: str, value: float, phase: SprintPhase) -> str:
    """Get actionable coaching cue for a metric."""
    cues = {
        "trunk_lean": {
            "low": "Lean more forward from your ankles, not your waist",
            "high": "You're rising too quickly—stay low longer",
            "good": "Great forward lean angle!"
        },
        "knee_drive": {
            "low": "Drive your knee higher toward your chest",
            "high": "Good knee drive, focus on quick ground contact",
            "good": "Excellent knee drive!"
        },
        "arm_action": {
            "low": "Keep elbows at ~90°, drive arms more aggressively",
            "high": "Relax your arms slightly, maintain 90° bend",
            "good": "Good arm mechanics!"
        }
    }
    
    if value is None or pd.isna(value):
        return "Unable to measure—check video quality"
    
    # Determine if value is low, high, or good
    if metric_name == "trunk_lean":
        value = abs(value)
        if phase in [SprintPhase.SET, SprintPhase.DRIVE]:
            if value < 35:
                return cues["trunk_lean"]["low"]
            elif value > 55:
                return cues["trunk_lean"]["high"]
            return cues["trunk_lean"]["good"]
        else:
            if value > 25:
                return "Transition to more upright posture"
            return cues["trunk_lean"]["good"]
    
    return "Keep practicing this movement pattern"


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Analysis mode
        st.markdown('<div class="sidebar-header">Analysis Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Speed vs Accuracy",
            ["⚡ Fast (10 frames)", "⚖️ Balanced (30 frames)", "🎯 Accurate (100 frames)"],
            index=1,
            label_visibility="collapsed"
        )
        
        mode_map = {"⚡ Fast (10 frames)": 10, "⚖️ Balanced (30 frames)": 30, "🎯 Accurate (100 frames)": 100}
        max_frames = mode_map[mode]
        sample_rate = 10 if max_frames == 10 else 5 if max_frames == 30 else 2
        
        # Display options
        st.markdown('<div class="sidebar-header">Display</div>', unsafe_allow_html=True)
        video_width = st.slider("Video Width %", 50, 80, 65, help="Adjust video size")
        show_skeleton = st.checkbox("Show skeleton", value=True)
        show_angles_on_video = st.checkbox("Show angles on video", value=False, 
            help="Display angle values directly on the video (can be cluttered)")
        
        # Model settings
        with st.expander("🔧 Advanced"):
            model_complexity = st.selectbox("Model Quality", [0, 1, 2], index=1,
                format_func=lambda x: ["Lite", "Full", "Heavy"][x])
            confidence = st.slider("Confidence Threshold", 0.3, 0.9, 0.5)
        
        st.markdown("---")
        
        # Profile
        st.markdown('<div class="sidebar-header">Your Profile</div>', unsafe_allow_html=True)
        st.session_state.user_event = st.selectbox("Event", ["60m", "100m", "200m"], index=1)
        st.session_state.user_level = st.selectbox("Level", 
            ["Beginner", "Intermediate", "Advanced", "Elite"], index=1)
        
        st.markdown("---")
        
        # Help
        with st.expander("❓ Filming Tips"):
            st.markdown("""
            **For best results:**
            - Film from the side (perpendicular)
            - Keep camera stable (tripod recommended)
            - Ensure full body is visible
            - 5-10 seconds is enough
            - Good lighting helps accuracy
            """)
        
        # Limitations
        with st.expander("⚠️ Limitations"):
            st.markdown("""
            **This tool provides training feedback only.**
            
            - Accuracy depends on video quality
            - 2D analysis has depth limitations
            - Camera angle affects measurements
            - Not a substitute for a coach
            
            *Backed by MediaPipe pose estimation*
            """)
    
    return {
        "max_frames": max_frames,
        "sample_rate": sample_rate,
        "video_width": video_width,
        "show_skeleton": show_skeleton,
        "show_angles": show_angles_on_video,
        "model_complexity": model_complexity if 'model_complexity' in dir() else 1,
        "confidence": confidence if 'confidence' in dir() else 0.5,
    }


# =============================================================================
# METRIC CARD COMPONENT
# =============================================================================
def render_metric_card(label: str, value: float | None, unit: str, 
                       optimal: float, target_min: float, target_max: float,
                       coaching_cue: str, icon: str = "📐"):
    """Render a premium metric card with context bar."""
    
    if value is None or pd.isna(value):
        value_display = "—"
        rating, badge_text = "poor", "No Data"
        marker_pos = 50
    else:
        value_display = f"{abs(value):.0f}"
        rating, badge_text = get_rating(abs(value), optimal, 
            good_range=(target_max - target_min) / 3,
            okay_range=(target_max - target_min) / 2)
        # Calculate marker position (0-100%)
        range_size = target_max - target_min
        marker_pos = max(0, min(100, ((abs(value) - target_min + range_size * 0.3) / (range_size * 1.6)) * 100))
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div class="metric-label">{icon} {label}</div>
                <div style="display: flex; align-items: baseline;">
                    <span class="metric-value">{value_display}</span>
                    <span class="metric-unit">{unit}</span>
                </div>
            </div>
            <span class="badge badge-{rating}">{badge_text}</span>
        </div>
        <div class="context-bar">
            <div class="context-marker" style="left: {marker_pos}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; color: #666; font-size: 0.7rem;">
            <span>{target_min}{unit}</span>
            <span>Target: {target_min}–{target_max}{unit}</span>
            <span>{target_max}{unit}</span>
        </div>
        <div class="coaching-cue">{coaching_cue}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FOCUS AREA CARD
# =============================================================================
def render_focus_card(icon: str, title: str, value: str, rating: str, cue: str):
    """Render a focus area card."""
    colors = {"good": "#22c55e", "okay": "#f59e0b", "poor": "#ef4444"}
    color = colors.get(rating, "#a0a0a0")
    
    st.markdown(f"""
    <div class="focus-card">
        <div class="focus-icon">{icon}</div>
        <div class="focus-title">{title}</div>
        <div class="focus-value" style="color: {color};">{value}</div>
        <div class="focus-cue">{cue}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# HERO SCORE
# =============================================================================
def render_hero_score(score: float, rating: str, summary: str):
    """Render the hero score card."""
    st.markdown(f"""
    <div class="hero-score">
        <div class="hero-score-label">Form Score</div>
        <div class="hero-score-value">{score:.1f}<span style="font-size: 2rem;">/10</span></div>
        <span class="badge badge-{rating}" style="margin-top: 1rem; font-size: 0.8rem;">
            {rating.upper()}
        </span>
        <div class="hero-summary">{summary.capitalize()}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN ANALYSIS VIEW
# =============================================================================
def render_analysis_view(settings: dict):
    """Render the main analysis dashboard."""
    
    frames = st.session_state.processed_frames
    metrics_list = st.session_state.frame_metrics
    
    if not frames or not metrics_list:
        st.warning("No analysis data available.")
        return
    
    # Get current frame data
    idx = min(st.session_state.current_frame_idx, len(metrics_list) - 1)
    current_metrics = metrics_list[idx]
    aggregated = st.session_state.aggregated_metrics or aggregate_metrics(metrics_list)
    
    # Calculate scores
    score, rating, summary = calculate_form_score(current_metrics)
    
    # ===== TABS =====
    tab_overview, tab_video, tab_metrics, tab_info = st.tabs([
        "📊 Overview", "🎬 Video & Overlay", "📐 All Metrics", "ℹ️ How It Works"
    ])
    
    # ===== OVERVIEW TAB =====
    with tab_overview:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Hero Score
            render_hero_score(score, rating, summary)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Top 3 Focus Areas
            st.markdown("### 🎯 Focus Areas")
            
            focus_cols = st.columns(3)
            
            # Focus 1: Trunk Lean
            trunk = current_metrics.angles.get("trunk_lean")
            trunk_rating, _ = get_rating(abs(trunk) if trunk else 0, 40, 10, 20)
            with focus_cols[0]:
                render_focus_card(
                    "🏃", "Trunk Lean",
                    f"{abs(trunk):.0f}°" if trunk and not pd.isna(trunk) else "—",
                    trunk_rating,
                    get_coaching_cue("trunk_lean", trunk, current_metrics.phase)
                )
            
            # Focus 2: Knee Drive
            left_knee = current_metrics.angles.get("left_knee")
            right_knee = current_metrics.angles.get("right_knee")
            front_knee = min(left_knee or 180, right_knee or 180) if (left_knee or right_knee) else None
            knee_rating, _ = get_rating(front_knee, 110, 15, 25) if front_knee else ("poor", "—")
            with focus_cols[1]:
                render_focus_card(
                    "🦵", "Knee Drive",
                    f"{front_knee:.0f}°" if front_knee and not pd.isna(front_knee) else "—",
                    knee_rating,
                    "Drive knee high for power" if front_knee and front_knee < 100 else "Good knee lift!"
                )
            
            # Focus 3: Phase
            phase = current_metrics.phase
            phase_rating = "good" if phase != SprintPhase.UNKNOWN else "okay"
            with focus_cols[2]:
                render_focus_card(
                    "⏱️", "Current Phase",
                    phase.display_name,
                    phase_rating,
                    get_phase_description(phase)[:60] + "..."
                )
        
        with col_right:
            # Compact video preview
            st.markdown("### 🎬 Preview")
            current_frame = frames[st.session_state.current_frame_idx]
            st.image(current_frame, use_container_width=True)
            
            # Frame slider
            new_idx = st.slider(
                "Frame", 0, len(frames) - 1, 
                st.session_state.current_frame_idx,
                label_visibility="collapsed"
            )
            if new_idx != st.session_state.current_frame_idx:
                st.session_state.current_frame_idx = new_idx
                st.rerun()
    
    # ===== VIDEO TAB =====
    with tab_video:
        # Navigation
        nav_cols = st.columns([1, 6, 1])
        with nav_cols[0]:
            if st.button("◀ Prev", use_container_width=True):
                if st.session_state.current_frame_idx > 0:
                    st.session_state.current_frame_idx -= 1
                    st.rerun()
        with nav_cols[1]:
            frame_idx = st.slider(
                "Frame", 0, len(frames) - 1,
                st.session_state.current_frame_idx,
                label_visibility="collapsed",
                key="video_slider"
            )
            if frame_idx != st.session_state.current_frame_idx:
                st.session_state.current_frame_idx = frame_idx
                st.rerun()
        with nav_cols[2]:
            if st.button("Next ▶", use_container_width=True):
                if st.session_state.current_frame_idx < len(frames) - 1:
                    st.session_state.current_frame_idx += 1
                    st.rerun()
        
        # Video display
        video_col_ratio = settings["video_width"] / 100
        vid_col, info_col = st.columns([video_col_ratio * 2, (1 - video_col_ratio) * 2])
        
        with vid_col:
            st.image(frames[st.session_state.current_frame_idx], use_container_width=True)
            st.markdown(f"""
            <div style="text-align: center; color: #a0a0a0; font-size: 0.85rem;">
                Frame {current_metrics.frame_index} | {current_metrics.timestamp_sec:.2f}s | 
                {st.session_state.current_frame_idx + 1}/{len(frames)}
            </div>
            """, unsafe_allow_html=True)
        
        with info_col:
            # Phase badge
            phase = current_metrics.phase
            phase_colors = {
                SprintPhase.SET: "#ef4444",
                SprintPhase.DRIVE: "#f59e0b", 
                SprintPhase.ACCELERATION: "#eab308",
                SprintPhase.MAX_VELOCITY: "#22c55e",
                SprintPhase.UNKNOWN: "#a0a0a0",
            }
            st.markdown(f"""
            <div class="phase-badge" style="background: {phase_colors[phase]}20; 
                 border: 2px solid {phase_colors[phase]}; color: {phase_colors[phase]};">
                {phase.display_name}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Quick metrics
            st.markdown("**Quick Stats**")
            angles = current_metrics.angles
            for name, key in [("Trunk Lean", "trunk_lean"), ("Left Knee", "left_knee"), 
                              ("Right Knee", "right_knee")]:
                val = angles.get(key)
                if val and not pd.isna(val):
                    st.markdown(f"- {name}: **{abs(val):.0f}°**")
    
    # ===== METRICS TAB =====
    with tab_metrics:
        st.markdown("### 📐 Detailed Metrics")
        
        col1, col2 = st.columns(2)
        
        angles = current_metrics.angles
        
        with col1:
            render_metric_card(
                "Trunk Lean", angles.get("trunk_lean"), "°",
                optimal=40, target_min=30, target_max=55,
                coaching_cue=get_coaching_cue("trunk_lean", angles.get("trunk_lean"), current_metrics.phase),
                icon="🔄"
            )
            
            render_metric_card(
                "Left Knee", angles.get("left_knee"), "°",
                optimal=110, target_min=90, target_max=140,
                coaching_cue="Knee flexion for power output",
                icon="🦵"
            )
            
            render_metric_card(
                "Left Hip", angles.get("left_hip"), "°",
                optimal=160, target_min=140, target_max=180,
                coaching_cue="Hip extension for stride length",
                icon="🏃"
            )
        
        with col2:
            render_metric_card(
                "Left Elbow", angles.get("left_elbow"), "°",
                optimal=90, target_min=80, target_max=100,
                coaching_cue="Keep elbows at ~90° for efficient arm drive",
                icon="💪"
            )
            
            render_metric_card(
                "Right Knee", angles.get("right_knee"), "°",
                optimal=110, target_min=90, target_max=140,
                coaching_cue="Match left knee drive for symmetry",
                icon="🦵"
            )
            
            render_metric_card(
                "Right Hip", angles.get("right_hip"), "°",
                optimal=160, target_min=140, target_max=180,
                coaching_cue="Full hip extension on each stride",
                icon="🏃"
            )
        
        # Averages section
        with st.expander("📊 Session Averages"):
            avg = aggregated.get("avg_angles", {})
            avg_cols = st.columns(4)
            for i, (key, val) in enumerate(avg.items()):
                with avg_cols[i % 4]:
                    st.metric(key.replace("_", " ").title(), f"{val:.1f}°")
    
    # ===== HOW IT WORKS TAB =====
    with tab_info:
        st.markdown("### ℹ️ How We Calculate Your Score")
        
        st.markdown("""
        <div class="info-box">
        <strong>🔬 Powered by AI Pose Estimation</strong><br><br>
        We use <strong>MediaPipe BlazePose</strong> to detect 33 body landmarks in each frame. 
        From these landmarks, we calculate joint angles (knee, hip, elbow) and body position 
        (trunk lean, hip height) to assess your sprint form.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📐 Angle Calculations")
            st.markdown("""
            - **Trunk Lean**: Angle of torso from vertical (hip-to-shoulder line)
            - **Knee Angle**: Flexion at knee joint (hip-knee-ankle)
            - **Hip Angle**: Extension at hip (shoulder-hip-knee)
            - **Elbow Angle**: Arm bend (shoulder-elbow-wrist)
            """)
        
        with col2:
            st.markdown("#### ⏱️ Phase Detection")
            st.markdown("""
            We classify each frame into sprint phases based on:
            - **Set**: Low hips, strong forward lean (40-55°)
            - **Drive**: Rising hips, maintaining lean (30-50°)
            - **Acceleration**: Transitioning upright (15-35°)
            - **Max Velocity**: Upright running (<15°)
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Important Limitations</strong><br><br>
        • Target ranges are based on coaching literature, not personalized to you<br>
        • 2D video analysis cannot capture depth or rotation accurately<br>
        • Camera angle significantly affects measurements (side view is best)<br>
        • This is training feedback, not a medical or diagnostic tool<br>
        • Always consult a qualified coach for personalized guidance
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# EMPTY STATE / UPLOAD VIEW
# =============================================================================
def render_upload_view(settings: dict):
    """Render the upload interface."""
    
    st.markdown("## 🏃 Sprint Form Analyzer")
    st.markdown("Upload a video of your sprint to get AI-powered form analysis.")
    
    # What you'll get
    with st.container():
        st.markdown("""
        <div class="info-box">
        <strong>📊 What You'll Get:</strong><br>
        • Overall form score (0-10)<br>
        • Top 3 actionable coaching cues<br>
        • Detailed joint angle measurements<br>
        • Phase-by-phase breakdown
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Video quality checklist
    with st.expander("✅ Video Quality Checklist", expanded=True):
        st.markdown("""
        For best results, ensure your video has:
        - ☑️ **Side view** (perpendicular to running direction)
        - ☑️ **Full body visible** in all frames
        - ☑️ **Stable camera** (tripod recommended)
        - ☑️ **Good lighting** (outdoor daylight is ideal)
        - ☑️ **5-10 seconds** of footage
        """)
    
    # Upload
    st.markdown("### 📹 Upload Your Video")
    uploaded = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi"],
        label_visibility="collapsed"
    )
    
    if uploaded:
        st.video(uploaded)
        
        if st.button("🚀 Analyze My Sprint", type="primary", use_container_width=True):
            process_video(uploaded, settings)


# =============================================================================
# VIDEO PROCESSING
# =============================================================================
def process_video(uploaded_file, settings: dict):
    """Process uploaded video."""
    
    progress = st.progress(0)
    status = st.empty()
    
    try:
        status.text("Loading video...")
        video_path = load_video_from_uploaded_file(uploaded_file)
        props = get_video_properties(video_path)
        st.session_state.video_properties = props
        
        target_config = load_target_ranges()
        frames = []
        metrics_list = []
        
        total = min(props["frame_count"] // settings["sample_rate"], settings["max_frames"])
        
        with PoseEstimator(
            static_image_mode=False,
            model_complexity=settings.get("model_complexity", 1),
            min_detection_confidence=settings.get("confidence", 0.5),
            min_tracking_confidence=settings.get("confidence", 0.5),
        ) as estimator:
            
            for i, (frame_idx, frame_rgb) in enumerate(sample_frames(
                video_path, settings["sample_rate"], settings["max_frames"]
            )):
                prog = min((i + 1) / max(total, 1), 1.0)
                progress.progress(prog)
                status.text(f"Analyzing frame {frame_idx}... ({i + 1}/{total})")
                
                result = estimator.process_frame(frame_rgb)
                
                if result:
                    metrics = compute_frame_metrics(
                        frame_idx, props["fps"], result.landmarks,
                        target_config, settings.get("confidence", 0.5)
                    )
                    
                    annotated = annotate_frame(
                        frame_rgb, result.landmarks, metrics.angles, metrics.phase,
                        frame_idx, metrics.timestamp_sec,
                        draw_angles=settings.get("show_angles", False),
                        draw_info=True,
                        visibility_threshold=settings.get("confidence", 0.5)
                    )
                    
                    frames.append(annotated)
                    metrics_list.append(metrics)
                else:
                    frames.append(frame_rgb)
        
        cleanup_temp_file(video_path)
        
        st.session_state.processed_frames = frames
        st.session_state.frame_metrics = metrics_list
        st.session_state.aggregated_metrics = aggregate_metrics(metrics_list)
        st.session_state.processing_complete = True
        st.session_state.current_frame_idx = 0
        
        progress.progress(1.0)
        status.text("✅ Analysis complete!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        status.text("❌ Processing failed")


# =============================================================================
# MAIN
# =============================================================================
def main():
    init_session_state()
    settings = render_sidebar()
    
    if st.session_state.processing_complete:
        render_analysis_view(settings)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📹 Analyze New Video", use_container_width=True):
                st.session_state.processing_complete = False
                st.session_state.processed_frames = []
                st.session_state.frame_metrics = []
                st.session_state.aggregated_metrics = None
                st.rerun()
    else:
        render_upload_view(settings)


if __name__ == "__main__":
    main()
