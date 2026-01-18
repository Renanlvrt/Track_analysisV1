"""Joint angle calculations for biomechanics analysis.

Provides functions to calculate angles between body segments
using pose landmark coordinates. All angles are in degrees.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..pose.mediapipe_pose import Landmark


# Import landmark indices
from ..pose.mediapipe_pose import LandmarkIndex


def calculate_angle(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    point_c: tuple[float, float],
) -> float:
    """
    Calculate the angle at point_b formed by the vectors BA and BC.
    
    The angle is measured at the vertex (point_b), between rays
    going to point_a and point_c.
    
    Args:
        point_a: First point (x, y)
        point_b: Vertex point (x, y) - angle is measured here
        point_c: Third point (x, y)
        
    Returns:
        Angle in degrees [0, 180], or NaN if calculation fails
    """
    try:
        # Convert to numpy arrays
        a = np.array(point_a)
        b = np.array(point_b)
        c = np.array(point_c)
        
        # Calculate vectors from vertex
        ba = a - b
        bc = c - b
        
        # Calculate magnitudes
        mag_ba = np.linalg.norm(ba)
        mag_bc = np.linalg.norm(bc)
        
        # Check for zero-length vectors
        if mag_ba < 1e-10 or mag_bc < 1e-10:
            return float("nan")
        
        # Calculate cosine of angle using dot product
        cos_angle = np.dot(ba, bc) / (mag_ba * mag_bc)
        
        # Clamp to [-1, 1] to handle floating point errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return float(angle_deg)
    
    except Exception:
        return float("nan")


def calculate_trunk_lean(
    hip_point: tuple[float, float],
    shoulder_point: tuple[float, float],
) -> float:
    """
    Calculate trunk lean angle from vertical.
    
    Measures the angle between the hip-shoulder vector and 
    the vertical axis. Positive values indicate forward lean.
    
    Note: In image coordinates, Y increases downward, so we
    adjust the calculation accordingly.
    
    Args:
        hip_point: Hip midpoint (x, y) in normalized coordinates
        shoulder_point: Shoulder midpoint (x, y) in normalized coordinates
        
    Returns:
        Angle in degrees from vertical:
        - Positive = forward lean (shoulders ahead of hips)
        - Negative = backward lean
        - 0 = perfectly vertical
        - NaN if calculation fails
    """
    try:
        # Calculate trunk vector (shoulder relative to hip)
        dx = shoulder_point[0] - hip_point[0]
        dy = shoulder_point[1] - hip_point[1]  # Negative when shoulder above hip
        
        # Vertical vector points upward in image coords (negative y)
        # The angle from vertical is the angle of the trunk vector
        # We measure deviation from vertical (straight up)
        
        # Calculate angle from vertical axis
        # atan2 gives angle from positive x-axis, we want from negative y-axis
        trunk_length = math.sqrt(dx * dx + dy * dy)
        
        if trunk_length < 1e-10:
            return float("nan")
        
        # Angle from vertical: positive dx means forward lean
        # Since shoulder is above hip, dy is negative
        # Forward lean means shoulder x > hip x (positive dx)
        angle_rad = math.atan2(dx, -dy)  # -dy because y-axis is inverted
        angle_deg = math.degrees(angle_rad)
        
        return float(angle_deg)
    
    except Exception:
        return float("nan")


def get_midpoint(
    landmark_1: "Landmark",
    landmark_2: "Landmark",
) -> tuple[float, float]:
    """Calculate midpoint between two landmarks."""
    return (
        (landmark_1.x + landmark_2.x) / 2,
        (landmark_1.y + landmark_2.y) / 2,
    )


def extract_joint_angles(
    landmarks: list["Landmark"],
    visibility_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Extract all relevant joint angles from pose landmarks.
    
    Calculates key angles for sprint analysis:
    - Knee angles (flexion/extension)
    - Hip angles (thigh relative to torso)
    - Elbow angles (arm flexion)
    - Trunk lean (forward/backward angle from vertical)
    
    Args:
        landmarks: List of 33 BlazePose landmarks
        visibility_threshold: Minimum visibility to include landmark
        
    Returns:
        Dictionary of angle names to values in degrees.
        Values are NaN if landmarks are not visible enough.
        
        Keys:
        - left_knee, right_knee
        - left_hip, right_hip
        - left_elbow, right_elbow
        - trunk_lean
    """
    
    def get_point(idx: int) -> tuple[float, float] | None:
        """Get point if visible, else None."""
        lm = landmarks[idx]
        if lm.visibility >= visibility_threshold:
            return (lm.x, lm.y)
        return None
    
    angles: dict[str, float] = {}
    
    # === KNEE ANGLES ===
    # Angle at knee between hip-knee-ankle
    
    # Left knee
    left_hip = get_point(LandmarkIndex.LEFT_HIP)
    left_knee = get_point(LandmarkIndex.LEFT_KNEE)
    left_ankle = get_point(LandmarkIndex.LEFT_ANKLE)
    
    if all([left_hip, left_knee, left_ankle]):
        angles["left_knee"] = calculate_angle(left_hip, left_knee, left_ankle)
    else:
        angles["left_knee"] = float("nan")
    
    # Right knee
    right_hip = get_point(LandmarkIndex.RIGHT_HIP)
    right_knee = get_point(LandmarkIndex.RIGHT_KNEE)
    right_ankle = get_point(LandmarkIndex.RIGHT_ANKLE)
    
    if all([right_hip, right_knee, right_ankle]):
        angles["right_knee"] = calculate_angle(right_hip, right_knee, right_ankle)
    else:
        angles["right_knee"] = float("nan")
    
    # === HIP ANGLES ===
    # Angle at hip between shoulder-hip-knee
    
    # Left hip
    left_shoulder = get_point(LandmarkIndex.LEFT_SHOULDER)
    
    if all([left_shoulder, left_hip, left_knee]):
        angles["left_hip"] = calculate_angle(left_shoulder, left_hip, left_knee)
    else:
        angles["left_hip"] = float("nan")
    
    # Right hip
    right_shoulder = get_point(LandmarkIndex.RIGHT_SHOULDER)
    
    if all([right_shoulder, right_hip, right_knee]):
        angles["right_hip"] = calculate_angle(right_shoulder, right_hip, right_knee)
    else:
        angles["right_hip"] = float("nan")
    
    # === ELBOW ANGLES ===
    # Angle at elbow between shoulder-elbow-wrist
    
    # Left elbow
    left_elbow = get_point(LandmarkIndex.LEFT_ELBOW)
    left_wrist = get_point(LandmarkIndex.LEFT_WRIST)
    
    if all([left_shoulder, left_elbow, left_wrist]):
        angles["left_elbow"] = calculate_angle(left_shoulder, left_elbow, left_wrist)
    else:
        angles["left_elbow"] = float("nan")
    
    # Right elbow
    right_elbow = get_point(LandmarkIndex.RIGHT_ELBOW)
    right_wrist = get_point(LandmarkIndex.RIGHT_WRIST)
    
    if all([right_shoulder, right_elbow, right_wrist]):
        angles["right_elbow"] = calculate_angle(right_shoulder, right_elbow, right_wrist)
    else:
        angles["right_elbow"] = float("nan")
    
    # === TRUNK LEAN ===
    # Angle of torso from vertical
    
    if all([left_hip, right_hip, left_shoulder, right_shoulder]):
        hip_mid = get_midpoint(
            landmarks[LandmarkIndex.LEFT_HIP],
            landmarks[LandmarkIndex.RIGHT_HIP],
        )
        shoulder_mid = get_midpoint(
            landmarks[LandmarkIndex.LEFT_SHOULDER],
            landmarks[LandmarkIndex.RIGHT_SHOULDER],
        )
        angles["trunk_lean"] = calculate_trunk_lean(hip_mid, shoulder_mid)
    else:
        angles["trunk_lean"] = float("nan")
    
    return angles


def get_hip_height_normalized(
    landmarks: list["Landmark"],
    visibility_threshold: float = 0.5,
) -> float:
    """
    Get normalized hip height (y-coordinate of hip midpoint).
    
    In image coordinates, 0 = top, 1 = bottom.
    Lower values mean the hip is higher in the frame.
    
    Returns:
        Normalized hip y-coordinate, or NaN if not visible
    """
    left_hip = landmarks[LandmarkIndex.LEFT_HIP]
    right_hip = landmarks[LandmarkIndex.RIGHT_HIP]
    
    if (left_hip.visibility >= visibility_threshold and 
        right_hip.visibility >= visibility_threshold):
        return (left_hip.y + right_hip.y) / 2
    
    return float("nan")
