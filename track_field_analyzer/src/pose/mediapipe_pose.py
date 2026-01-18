"""MediaPipe BlazePose wrapper for pose estimation.

Provides a clean interface to MediaPipe's pose detection,
returning structured landmark data with visibility scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise ImportError(
        "MediaPipe is required. Install with: pip install mediapipe"
    )


# MediaPipe BlazePose landmark indices
class LandmarkIndex:
    """Constants for BlazePose 33 landmark indices."""
    
    # Face
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    
    # Upper body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    
    # Lower body
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Skeleton connections for drawing
POSE_CONNECTIONS = [
    # Torso
    (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER),
    (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP),
    (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_HIP),
    (LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP),
    
    # Left arm
    (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_ELBOW),
    (LandmarkIndex.LEFT_ELBOW, LandmarkIndex.LEFT_WRIST),
    
    # Right arm
    (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_ELBOW),
    (LandmarkIndex.RIGHT_ELBOW, LandmarkIndex.RIGHT_WRIST),
    
    # Left leg
    (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE),
    (LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE),
    (LandmarkIndex.LEFT_ANKLE, LandmarkIndex.LEFT_HEEL),
    (LandmarkIndex.LEFT_HEEL, LandmarkIndex.LEFT_FOOT_INDEX),
    (LandmarkIndex.LEFT_ANKLE, LandmarkIndex.LEFT_FOOT_INDEX),
    
    # Right leg
    (LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE),
    (LandmarkIndex.RIGHT_KNEE, LandmarkIndex.RIGHT_ANKLE),
    (LandmarkIndex.RIGHT_ANKLE, LandmarkIndex.RIGHT_HEEL),
    (LandmarkIndex.RIGHT_HEEL, LandmarkIndex.RIGHT_FOOT_INDEX),
    (LandmarkIndex.RIGHT_ANKLE, LandmarkIndex.RIGHT_FOOT_INDEX),
]


@dataclass
class Landmark:
    """Single pose landmark with position and visibility."""
    
    x: float  # Normalized x coordinate [0, 1] (left to right)
    y: float  # Normalized y coordinate [0, 1] (top to bottom)
    z: float  # Depth relative to hips (negative = closer to camera)
    visibility: float  # Detection confidence [0, 1]
    
    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))
    
    def is_visible(self, threshold: float = 0.5) -> bool:
        """Check if landmark visibility exceeds threshold."""
        return self.visibility >= threshold


@dataclass
class PoseResult:
    """Container for pose estimation results."""
    
    landmarks: list[Landmark]  # 33 BlazePose landmarks (normalized)
    world_landmarks: list[Landmark] | None  # 3D world coordinates (meters)
    
    def get_landmark(self, index: int) -> Landmark:
        """Get landmark by index."""
        return self.landmarks[index]
    
    def get_visible_landmarks(
        self, 
        threshold: float = 0.5
    ) -> list[tuple[int, Landmark]]:
        """Get all landmarks above visibility threshold."""
        return [
            (i, lm) for i, lm in enumerate(self.landmarks)
            if lm.is_visible(threshold)
        ]


class PoseEstimator:
    """Wrapper for MediaPipe BlazePose estimation.
    
    Usage:
        estimator = PoseEstimator()
        result = estimator.process_frame(frame_rgb)
        if result:
            for landmark in result.landmarks:
                print(landmark.x, landmark.y, landmark.visibility)
        estimator.close()
    
    Or as context manager:
        with PoseEstimator() as estimator:
            result = estimator.process_frame(frame_rgb)
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Pose model.
        
        Args:
            static_image_mode: If True, treats each image independently 
                              (slower but better for unrelated images)
            model_complexity: 0, 1, or 2. Higher = more accurate but slower
            smooth_landmarks: Smooth landmarks across frames (video mode)
            enable_segmentation: Enable body segmentation mask
            min_detection_confidence: Minimum confidence for person detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._is_closed = False
    
    def process_frame(self, frame_rgb: np.ndarray) -> PoseResult | None:
        """
        Run pose estimation on a single frame.
        
        Args:
            frame_rgb: RGB image as numpy array (H, W, 3)
            
        Returns:
            PoseResult with landmarks, or None if no pose detected
        """
        if self._is_closed:
            raise RuntimeError("PoseEstimator has been closed")
        
        # Process frame
        results = self._pose.process(frame_rgb)
        
        # Check if pose was detected
        if results.pose_landmarks is None:
            return None
        
        # Convert to our Landmark format
        landmarks = [
            Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
            )
            for lm in results.pose_landmarks.landmark
        ]
        
        # Convert world landmarks if available
        world_landmarks = None
        if results.pose_world_landmarks is not None:
            world_landmarks = [
                Landmark(
                    x=lm.x,
                    y=lm.y, 
                    z=lm.z,
                    visibility=lm.visibility,
                )
                for lm in results.pose_world_landmarks.landmark
            ]
        
        return PoseResult(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
        )
    
    def close(self) -> None:
        """Release MediaPipe resources."""
        if not self._is_closed:
            self._pose.close()
            self._is_closed = True
    
    def __enter__(self) -> "PoseEstimator":
        """Context manager entry."""
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()
