"""Metrics computation and aggregation.

Handles per-frame metrics calculation and aggregation across
video sequences, including feedback generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .angles import extract_joint_angles, get_hip_height_normalized
from .phases import SprintPhase, detect_sprint_phase

if TYPE_CHECKING:
    from ..pose.mediapipe_pose import Landmark


@dataclass
class FrameMetrics:
    """Metrics computed for a single frame."""
    
    frame_index: int
    timestamp_sec: float
    angles: dict[str, float]
    hip_height: float
    phase: SprintPhase
    feedback: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result = {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "phase": self.phase.value,
            "hip_height": round(self.hip_height, 3) if not math.isnan(self.hip_height) else None,
        }
        
        # Add angles
        for name, value in self.angles.items():
            if not math.isnan(value):
                result[name] = round(value, 1)
            else:
                result[name] = None
        
        return result


def load_target_ranges(config_path: str | Path | None = None) -> dict:
    """
    Load target angle ranges from config file.
    
    Args:
        config_path: Path to targets.yaml, or None to use default
        
    Returns:
        Dictionary of target ranges by phase
    """
    if config_path is None:
        # Use default path relative to this file
        config_path = Path(__file__).parent.parent.parent / "config" / "targets.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Return minimal defaults if no config
        return {
            "phases": {
                "set_position": {"targets": {}},
                "drive_phase": {"targets": {}},
                "acceleration": {"targets": {}},
                "max_velocity": {"targets": {}},
            }
        }
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_feedback(
    angles: dict[str, float],
    phase: SprintPhase,
    target_config: dict,
) -> list[str]:
    """
    Generate feedback by comparing angles to target ranges.
    
    Args:
        angles: Calculated joint angles
        phase: Detected sprint phase
        target_config: Loaded target configuration
        
    Returns:
        List of feedback strings
    """
    feedback = []
    
    # Map phase to config key
    phase_config_map = {
        SprintPhase.SET: "set_position",
        SprintPhase.DRIVE: "drive_phase",
        SprintPhase.ACCELERATION: "acceleration",
        SprintPhase.MAX_VELOCITY: "max_velocity",
    }
    
    phase_key = phase_config_map.get(phase)
    if phase_key is None:
        return feedback
    
    # Get targets for this phase
    phases_config = target_config.get("phases", {})
    phase_config = phases_config.get(phase_key, {})
    targets = phase_config.get("targets", {})
    
    # Check trunk lean
    trunk_lean = angles.get("trunk_lean", float("nan"))
    if not math.isnan(trunk_lean):
        trunk_targets = targets.get("trunk_lean", {})
        if trunk_targets:
            min_val = trunk_targets.get("min", 0)
            max_val = trunk_targets.get("max", 180)
            abs_lean = abs(trunk_lean)
            
            if abs_lean < min_val:
                feedback.append(trunk_targets.get(
                    "feedback_low", 
                    f"Trunk lean ({abs_lean:.0f}°) below target range"
                ))
            elif abs_lean > max_val:
                feedback.append(trunk_targets.get(
                    "feedback_high",
                    f"Trunk lean ({abs_lean:.0f}°) above target range"
                ))
    
    # Check knee angles (use minimum of left/right as "front knee")
    left_knee = angles.get("left_knee", float("nan"))
    right_knee = angles.get("right_knee", float("nan"))
    
    if not math.isnan(left_knee) or not math.isnan(right_knee):
        # In set position, check front knee angle
        if phase == SprintPhase.SET:
            knee_targets = targets.get("front_knee_angle", {})
            if knee_targets:
                # Use the more bent knee as front knee
                front_knee = min(
                    left_knee if not math.isnan(left_knee) else 180,
                    right_knee if not math.isnan(right_knee) else 180
                )
                min_val = knee_targets.get("min", 0)
                max_val = knee_targets.get("max", 180)
                
                if front_knee < min_val:
                    feedback.append(knee_targets.get(
                        "feedback_low",
                        f"Front knee ({front_knee:.0f}°) too closed"
                    ))
                elif front_knee > max_val:
                    feedback.append(knee_targets.get(
                        "feedback_high",
                        f"Front knee ({front_knee:.0f}°) too open"
                    ))
    
    # Check elbow angles
    left_elbow = angles.get("left_elbow", float("nan"))
    right_elbow = angles.get("right_elbow", float("nan"))
    
    arm_targets = targets.get("arm_angle", {})
    if arm_targets:
        for side, elbow_val in [("Left", left_elbow), ("Right", right_elbow)]:
            if not math.isnan(elbow_val):
                min_val = arm_targets.get("min", 0)
                max_val = arm_targets.get("max", 180)
                
                if elbow_val < min_val:
                    feedback.append(f"{side} arm too bent ({elbow_val:.0f}°)")
                elif elbow_val > max_val:
                    feedback.append(f"{side} arm too straight ({elbow_val:.0f}°)")
    
    return feedback


def compute_frame_metrics(
    frame_index: int,
    fps: float,
    landmarks: list["Landmark"],
    target_config: dict | None = None,
    visibility_threshold: float = 0.5,
) -> FrameMetrics:
    """
    Compute all metrics for a single frame.
    
    Args:
        frame_index: Frame number in video
        fps: Video frames per second
        landmarks: Pose landmarks from MediaPipe
        target_config: Target ranges config (loaded via load_target_ranges)
        visibility_threshold: Minimum landmark visibility
        
    Returns:
        FrameMetrics containing angles, phase, and feedback
    """
    # Calculate timestamp
    timestamp_sec = frame_index / fps if fps > 0 else 0.0
    
    # Extract angles
    angles = extract_joint_angles(landmarks, visibility_threshold)
    
    # Get hip height
    hip_height = get_hip_height_normalized(landmarks, visibility_threshold)
    
    # Detect phase
    trunk_lean = angles.get("trunk_lean", float("nan"))
    
    # Get front knee for phase detection
    left_knee = angles.get("left_knee", float("nan"))
    right_knee = angles.get("right_knee", float("nan"))
    front_knee = None
    if not math.isnan(left_knee) or not math.isnan(right_knee):
        front_knee = min(
            left_knee if not math.isnan(left_knee) else 180,
            right_knee if not math.isnan(right_knee) else 180
        )
    
    phase = detect_sprint_phase(
        trunk_lean=trunk_lean,
        hip_height_normalized=hip_height,
        knee_angle_front=front_knee,
    )
    
    # Generate feedback
    if target_config is None:
        target_config = load_target_ranges()
    
    feedback = generate_feedback(angles, phase, target_config)
    
    return FrameMetrics(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        angles=angles,
        hip_height=hip_height,
        phase=phase,
        feedback=feedback,
    )


def aggregate_metrics(
    frame_metrics_list: list[FrameMetrics],
) -> dict[str, Any]:
    """
    Aggregate metrics across all processed frames.
    
    Computes statistics like averages, min/max, and phase distribution.
    
    Args:
        frame_metrics_list: List of FrameMetrics from each processed frame
        
    Returns:
        Dictionary containing:
        - avg_angles: Mean angle values (excluding NaN)
        - min_angles: Minimum angle values
        - max_angles: Maximum angle values
        - phase_distribution: Count of frames in each phase
        - phase_sequence: List of phase transitions
        - overall_feedback: Aggregated feedback points
    """
    if not frame_metrics_list:
        return {
            "avg_angles": {},
            "min_angles": {},
            "max_angles": {},
            "phase_distribution": {},
            "phase_sequence": [],
            "overall_feedback": [],
        }
    
    # Collect angle values
    angle_values: dict[str, list[float]] = {}
    for fm in frame_metrics_list:
        for name, value in fm.angles.items():
            if not math.isnan(value):
                if name not in angle_values:
                    angle_values[name] = []
                angle_values[name].append(value)
    
    # Calculate statistics
    avg_angles = {
        name: sum(values) / len(values)
        for name, values in angle_values.items()
        if values
    }
    min_angles = {
        name: min(values)
        for name, values in angle_values.items()
        if values
    }
    max_angles = {
        name: max(values)
        for name, values in angle_values.items()
        if values
    }
    
    # Phase distribution
    phase_counts: dict[str, int] = {}
    for fm in frame_metrics_list:
        phase_name = fm.phase.value
        phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
    
    # Phase sequence (transitions)
    phase_sequence = []
    prev_phase = None
    for fm in frame_metrics_list:
        if fm.phase != prev_phase:
            phase_sequence.append({
                "phase": fm.phase.value,
                "start_frame": fm.frame_index,
                "timestamp": fm.timestamp_sec,
            })
            prev_phase = fm.phase
    
    # Collect unique feedback
    all_feedback = set()
    for fm in frame_metrics_list:
        all_feedback.update(fm.feedback)
    
    return {
        "avg_angles": {k: round(v, 1) for k, v in avg_angles.items()},
        "min_angles": {k: round(v, 1) for k, v in min_angles.items()},
        "max_angles": {k: round(v, 1) for k, v in max_angles.items()},
        "phase_distribution": phase_counts,
        "phase_sequence": phase_sequence,
        "overall_feedback": sorted(all_feedback),
    }
