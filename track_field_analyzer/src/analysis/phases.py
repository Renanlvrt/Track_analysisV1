"""Sprint phase detection using heuristics.

Classifies frames into sprint phases based on body position:
- SET: In blocks, ready to go
- DRIVE: First 10-30m push phase
- ACCELERATION: Transitioning to upright
- MAX_VELOCITY: Full upright sprinting
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pose.mediapipe_pose import Landmark


class SprintPhase(Enum):
    """Sprint phase classification."""
    
    SET = "set"
    DRIVE = "drive"
    ACCELERATION = "acceleration"
    MAX_VELOCITY = "max_velocity"
    UNKNOWN = "unknown"
    
    @property
    def display_name(self) -> str:
        """Human-readable phase name."""
        names = {
            SprintPhase.SET: "Set Position",
            SprintPhase.DRIVE: "Drive Phase",
            SprintPhase.ACCELERATION: "Acceleration",
            SprintPhase.MAX_VELOCITY: "Max Velocity",
            SprintPhase.UNKNOWN: "Unknown",
        }
        return names[self]
    
    @property
    def color(self) -> tuple[int, int, int]:
        """RGB color for visualization."""
        colors = {
            SprintPhase.SET: (255, 100, 100),       # Red-ish
            SprintPhase.DRIVE: (255, 165, 0),       # Orange
            SprintPhase.ACCELERATION: (255, 255, 0), # Yellow
            SprintPhase.MAX_VELOCITY: (100, 255, 100), # Green
            SprintPhase.UNKNOWN: (150, 150, 150),    # Gray
        }
        return colors[self]


# Default thresholds for phase detection
# These can be overridden with config values
DEFAULT_THRESHOLDS = {
    "set": {
        "max_hip_height": 0.55,  # Hip low (y close to bottom)
        "min_trunk_lean": 40,    # Strong forward lean
    },
    "drive": {
        "min_hip_height": 0.45,
        "max_hip_height": 0.65,
        "min_trunk_lean": 25,
        "max_trunk_lean": 50,
    },
    "acceleration": {
        "min_hip_height": 0.35,
        "max_hip_height": 0.55,
        "min_trunk_lean": 10,
        "max_trunk_lean": 35,
    },
    "max_velocity": {
        "max_hip_height": 0.45,  # Hip high (y close to top)
        "max_trunk_lean": 20,
    },
}


def detect_sprint_phase(
    trunk_lean: float,
    hip_height_normalized: float,
    knee_angle_front: float | None = None,
    thresholds: dict | None = None,
) -> SprintPhase:
    """
    Detect sprint phase using heuristics based on body position.
    
    This is a v1 heuristic approach that classifies based on:
    1. Trunk lean angle (forward = lower phases)
    2. Hip height in frame (lower = earlier phases)
    3. Optional: front knee angle for set position
    
    The classification order matters - we check from most specific
    (set position) to most general (max velocity).
    
    Args:
        trunk_lean: Forward lean angle in degrees (positive = forward)
        hip_height_normalized: Hip y-coordinate [0=top, 1=bottom]
        knee_angle_front: Optional front knee angle for set detection
        thresholds: Optional custom thresholds dict
        
    Returns:
        SprintPhase enum value
        
    Note:
        This heuristic works best when:
        - Camera is positioned at side view
        - Full body is visible in frame
        - Athlete is moving left-to-right or right-to-left
    """
    import math
    
    # Use default thresholds if not provided
    t = thresholds or DEFAULT_THRESHOLDS
    
    # Handle NaN values
    if math.isnan(trunk_lean) or math.isnan(hip_height_normalized):
        return SprintPhase.UNKNOWN
    
    # Convert trunk lean to absolute value for comparison
    # (handles both left-to-right and right-to-left movement)
    abs_trunk_lean = abs(trunk_lean)
    
    # === SET POSITION ===
    # Characterized by: low hip, strong forward lean, bent front knee
    set_thresh = t.get("set", DEFAULT_THRESHOLDS["set"])
    if (hip_height_normalized >= set_thresh.get("max_hip_height", 0.55) and
        abs_trunk_lean >= set_thresh.get("min_trunk_lean", 40)):
        # Additional check: front knee should be bent ~90-110°
        if knee_angle_front is not None:
            if 80 <= knee_angle_front <= 120:
                return SprintPhase.SET
        else:
            return SprintPhase.SET
    
    # === DRIVE PHASE ===
    # Characterized by: hip rising, still strong lean
    drive_thresh = t.get("drive", DEFAULT_THRESHOLDS["drive"])
    if (drive_thresh.get("min_hip_height", 0.45) <= 
            hip_height_normalized <= 
            drive_thresh.get("max_hip_height", 0.65) and
        drive_thresh.get("min_trunk_lean", 25) <= 
            abs_trunk_lean <= 
            drive_thresh.get("max_trunk_lean", 50)):
        return SprintPhase.DRIVE
    
    # === ACCELERATION ===
    # Characterized by: hip at medium height, moderate lean
    accel_thresh = t.get("acceleration", DEFAULT_THRESHOLDS["acceleration"])
    if (accel_thresh.get("min_hip_height", 0.35) <= 
            hip_height_normalized <= 
            accel_thresh.get("max_hip_height", 0.55) and
        accel_thresh.get("min_trunk_lean", 10) <= 
            abs_trunk_lean <= 
            accel_thresh.get("max_trunk_lean", 35)):
        return SprintPhase.ACCELERATION
    
    # === MAX VELOCITY ===
    # Characterized by: hip high, upright posture
    max_vel_thresh = t.get("max_velocity", DEFAULT_THRESHOLDS["max_velocity"])
    if (hip_height_normalized <= max_vel_thresh.get("max_hip_height", 0.45) and
        abs_trunk_lean <= max_vel_thresh.get("max_trunk_lean", 20)):
        return SprintPhase.MAX_VELOCITY
    
    # === UNKNOWN ===
    # Doesn't fit any clear category
    return SprintPhase.UNKNOWN


def get_phase_description(phase: SprintPhase) -> str:
    """Get a description of what characterizes each phase."""
    descriptions = {
        SprintPhase.SET: (
            "Set position in blocks. Body coiled, shoulders over hands, "
            "hips raised, ready to explode forward."
        ),
        SprintPhase.DRIVE: (
            "Drive phase (0-30m). Aggressive push with strong forward lean. "
            "Stay low and drive horizontally."
        ),
        SprintPhase.ACCELERATION: (
            "Acceleration phase. Gradually rising toward upright posture. "
            "Building speed while transitioning body angle."
        ),
        SprintPhase.MAX_VELOCITY: (
            "Maximum velocity phase. Upright running posture. "
            "Focus on turnover and maintaining speed."
        ),
        SprintPhase.UNKNOWN: (
            "Phase could not be determined from current body position."
        ),
    }
    return descriptions.get(phase, "")
