"""Analysis module for biomechanics calculations."""

from .angles import (
    calculate_angle,
    calculate_trunk_lean,
    extract_joint_angles,
)
from .metrics import FrameMetrics, compute_frame_metrics, aggregate_metrics
from .phases import detect_sprint_phase, SprintPhase

__all__ = [
    "calculate_angle",
    "calculate_trunk_lean", 
    "extract_joint_angles",
    "FrameMetrics",
    "compute_frame_metrics",
    "aggregate_metrics",
    "detect_sprint_phase",
    "SprintPhase",
]
