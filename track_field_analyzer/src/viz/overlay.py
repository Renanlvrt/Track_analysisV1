"""Visualization functions for skeleton and annotation overlay.

Draws pose skeleton, angle annotations, and phase labels on video frames.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..pose.mediapipe_pose import Landmark
    from ..analysis.phases import SprintPhase

from ..pose.mediapipe_pose import POSE_CONNECTIONS, LandmarkIndex


def draw_skeleton(
    frame: np.ndarray,
    landmarks: list["Landmark"],
    visibility_threshold: float = 0.5,
    connection_color: tuple[int, int, int] = (0, 255, 0),
    landmark_color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    circle_radius: int = 5,
) -> np.ndarray:
    """
    Draw pose skeleton on frame.
    
    Args:
        frame: RGB image as numpy array
        landmarks: List of Landmark objects
        visibility_threshold: Min visibility to draw landmark
        connection_color: RGB color for skeleton lines
        landmark_color: RGB color for joint circles
        thickness: Line thickness in pixels
        circle_radius: Joint circle radius in pixels
        
    Returns:
        Frame with skeleton overlay (copy, original unchanged)
    """
    # Make a copy to avoid modifying original
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    
    # Draw connections first (so joints overlay them)
    for start_idx, end_idx in POSE_CONNECTIONS:
        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]
        
        # Check visibility
        if (start_lm.visibility >= visibility_threshold and
            end_lm.visibility >= visibility_threshold):
            
            start_pt = start_lm.to_pixel(width, height)
            end_pt = end_lm.to_pixel(width, height)
            
            cv2.line(annotated, start_pt, end_pt, connection_color, thickness)
    
    # Draw landmarks
    for i, lm in enumerate(landmarks):
        if lm.visibility >= visibility_threshold:
            pt = lm.to_pixel(width, height)
            
            # Use different colors for key joints
            color = landmark_color
            if i in [LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP]:
                color = (255, 165, 0)  # Orange for hips
            elif i in [LandmarkIndex.LEFT_KNEE, LandmarkIndex.RIGHT_KNEE]:
                color = (0, 255, 255)  # Cyan for knees
            elif i in [LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER]:
                color = (255, 0, 255)  # Magenta for shoulders
            
            cv2.circle(annotated, pt, circle_radius, color, -1)
    
    return annotated


def draw_angle_annotations(
    frame: np.ndarray,
    landmarks: list["Landmark"],
    angles: dict[str, float],
    visibility_threshold: float = 0.5,
    font_scale: float = 0.6,
    font_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Draw angle values near corresponding joints.
    
    Args:
        frame: RGB image (should already have skeleton drawn)
        landmarks: List of Landmark objects
        angles: Dictionary of angle names to values
        visibility_threshold: Min visibility to draw
        font_scale: Font size multiplier
        font_color: RGB color for text
        bg_color: RGB color for text background
        
    Returns:
        Frame with angle annotations
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Map angle names to landmark indices for positioning
    angle_positions = {
        "left_knee": LandmarkIndex.LEFT_KNEE,
        "right_knee": LandmarkIndex.RIGHT_KNEE,
        "left_hip": LandmarkIndex.LEFT_HIP,
        "right_hip": LandmarkIndex.RIGHT_HIP,
        "left_elbow": LandmarkIndex.LEFT_ELBOW,
        "right_elbow": LandmarkIndex.RIGHT_ELBOW,
    }
    
    for angle_name, value in angles.items():
        if math.isnan(value):
            continue
            
        if angle_name in angle_positions:
            idx = angle_positions[angle_name]
            lm = landmarks[idx]
            
            if lm.visibility >= visibility_threshold:
                pt = lm.to_pixel(width, height)
                
                # Offset text slightly from joint
                text_pt = (pt[0] + 10, pt[1] - 10)
                text = f"{value:.0f}°"
                
                # Draw background rectangle
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
                cv2.rectangle(
                    annotated,
                    (text_pt[0] - 2, text_pt[1] - text_h - 2),
                    (text_pt[0] + text_w + 2, text_pt[1] + 2),
                    bg_color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated, text, text_pt, font,
                    font_scale, font_color, 1, cv2.LINE_AA
                )
    
    # Draw trunk lean at shoulder midpoint
    if "trunk_lean" in angles and not math.isnan(angles["trunk_lean"]):
        left_shoulder = landmarks[LandmarkIndex.LEFT_SHOULDER]
        right_shoulder = landmarks[LandmarkIndex.RIGHT_SHOULDER]
        
        if (left_shoulder.visibility >= visibility_threshold and
            right_shoulder.visibility >= visibility_threshold):
            
            mid_x = int((left_shoulder.x + right_shoulder.x) / 2 * width)
            mid_y = int((left_shoulder.y + right_shoulder.y) / 2 * height)
            
            text_pt = (mid_x - 30, mid_y - 20)
            text = f"Lean: {abs(angles['trunk_lean']):.0f}°"
            
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
            cv2.rectangle(
                annotated,
                (text_pt[0] - 2, text_pt[1] - text_h - 2),
                (text_pt[0] + text_w + 2, text_pt[1] + 2),
                bg_color,
                -1
            )
            cv2.putText(
                annotated, text, text_pt, font,
                font_scale, (255, 255, 0), 1, cv2.LINE_AA
            )
    
    return annotated


def draw_phase_label(
    frame: np.ndarray,
    phase: "SprintPhase",
    position: str = "top_left",
    font_scale: float = 1.0,
) -> np.ndarray:
    """
    Draw phase label on frame.
    
    Args:
        frame: RGB image
        phase: Current sprint phase
        position: "top_left", "top_right", "bottom_left", "bottom_right"
        font_scale: Font size multiplier
        
    Returns:
        Frame with phase label
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    text = f"Phase: {phase.display_name}"
    color = phase.color
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 2)
    
    # Calculate position
    padding = 10
    if position == "top_left":
        pt = (padding, text_h + padding)
    elif position == "top_right":
        pt = (width - text_w - padding, text_h + padding)
    elif position == "bottom_left":
        pt = (padding, height - padding)
    else:  # bottom_right
        pt = (width - text_w - padding, height - padding)
    
    # Draw background
    cv2.rectangle(
        annotated,
        (pt[0] - 5, pt[1] - text_h - 5),
        (pt[0] + text_w + 5, pt[1] + 5),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(annotated, text, pt, font, font_scale, color, 2, cv2.LINE_AA)
    
    return annotated


def draw_frame_info(
    frame: np.ndarray,
    frame_index: int,
    timestamp: float,
    position: str = "top_right",
) -> np.ndarray:
    """Draw frame number and timestamp on frame."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    
    text = f"Frame: {frame_index} | {timestamp:.2f}s"
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
    
    padding = 10
    if position == "top_right":
        pt = (width - text_w - padding, text_h + padding)
    else:
        pt = (padding, text_h + padding)
    
    cv2.rectangle(
        annotated,
        (pt[0] - 2, pt[1] - text_h - 2),
        (pt[0] + text_w + 2, pt[1] + 2),
        (0, 0, 0),
        -1
    )
    cv2.putText(annotated, text, pt, font, font_scale, (200, 200, 200), 1, cv2.LINE_AA)
    
    return annotated


def annotate_frame(
    frame: np.ndarray,
    landmarks: list["Landmark"],
    angles: dict[str, float],
    phase: "SprintPhase",
    frame_index: int = 0,
    timestamp: float = 0.0,
    draw_angles: bool = True,
    draw_info: bool = True,
    visibility_threshold: float = 0.5,
) -> np.ndarray:
    """
    Full annotation pipeline: skeleton + angles + phase label.
    
    Args:
        frame: RGB image as numpy array
        landmarks: List of Landmark objects
        angles: Dictionary of calculated angles
        phase: Detected sprint phase
        frame_index: Current frame number
        timestamp: Current timestamp in seconds
        draw_angles: Whether to draw angle values
        draw_info: Whether to draw frame info
        visibility_threshold: Min visibility for landmarks
        
    Returns:
        Fully annotated frame
    """
    # Draw skeleton
    annotated = draw_skeleton(
        frame, landmarks,
        visibility_threshold=visibility_threshold
    )
    
    # Draw angles if requested
    if draw_angles:
        annotated = draw_angle_annotations(
            annotated, landmarks, angles,
            visibility_threshold=visibility_threshold
        )
    
    # Draw phase label
    annotated = draw_phase_label(annotated, phase)
    
    # Draw frame info if requested
    if draw_info:
        annotated = draw_frame_info(
            annotated, frame_index, timestamp,
            position="top_right"
        )
    
    return annotated
