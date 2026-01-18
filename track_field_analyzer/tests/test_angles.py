"""Unit tests for angle calculation functions."""

import math
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.angles import (
    calculate_angle,
    calculate_trunk_lean,
    extract_joint_angles,
    get_hip_height_normalized,
)
from src.pose.mediapipe_pose import Landmark


class TestCalculateAngle:
    """Tests for calculate_angle function."""
    
    def test_right_angle(self):
        """Test detection of 90-degree angle."""
        # Points forming a right angle at origin
        point_a = (1.0, 0.0)  # Right
        point_b = (0.0, 0.0)  # Vertex at origin
        point_c = (0.0, 1.0)  # Up
        
        angle = calculate_angle(point_a, point_b, point_c)
        assert math.isclose(angle, 90.0, rel_tol=0.01)
    
    def test_straight_line(self):
        """Test detection of 180-degree angle (straight line)."""
        point_a = (-1.0, 0.0)
        point_b = (0.0, 0.0)
        point_c = (1.0, 0.0)
        
        angle = calculate_angle(point_a, point_b, point_c)
        assert math.isclose(angle, 180.0, rel_tol=0.01)
    
    def test_acute_angle(self):
        """Test detection of 60-degree angle."""
        # Equilateral triangle vertex
        point_a = (1.0, 0.0)
        point_b = (0.0, 0.0)
        point_c = (0.5, math.sqrt(3)/2)
        
        angle = calculate_angle(point_a, point_b, point_c)
        assert math.isclose(angle, 60.0, rel_tol=0.01)
    
    def test_zero_angle(self):
        """Test detection of 0-degree angle (same direction)."""
        point_a = (1.0, 0.0)
        point_b = (0.0, 0.0)
        point_c = (2.0, 0.0)  # Same direction as a
        
        angle = calculate_angle(point_a, point_b, point_c)
        assert math.isclose(angle, 0.0, rel_tol=0.01)
    
    def test_obtuse_angle(self):
        """Test detection of 120-degree angle."""
        point_a = (1.0, 0.0)
        point_b = (0.0, 0.0)
        point_c = (-0.5, math.sqrt(3)/2)
        
        angle = calculate_angle(point_a, point_b, point_c)
        assert math.isclose(angle, 120.0, rel_tol=0.01)
    
    def test_coincident_points_returns_nan(self):
        """Test that coincident points return NaN."""
        point_a = (0.0, 0.0)
        point_b = (0.0, 0.0)  # Same as a
        point_c = (1.0, 0.0)
        
        angle = calculate_angle(point_a, point_b, point_c)
        assert math.isnan(angle)
    
    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        point_a = (-1.0, -1.0)
        point_b = (0.0, 0.0)
        point_c = (1.0, -1.0)
        
        angle = calculate_angle(point_a, point_b, point_c)
        assert math.isclose(angle, 90.0, rel_tol=0.01)


class TestTrunkLean:
    """Tests for trunk lean calculation."""
    
    def test_vertical_trunk(self):
        """Test vertical trunk (0 degrees lean)."""
        hip = (0.5, 0.7)    # Lower (higher y value)
        shoulder = (0.5, 0.3)  # Upper (lower y value, directly above)
        
        lean = calculate_trunk_lean(hip, shoulder)
        assert math.isclose(lean, 0.0, abs_tol=0.1)
    
    def test_forward_lean_45_degrees(self):
        """Test 45-degree forward lean."""
        hip = (0.5, 0.7)
        # Moving shoulder forward (higher x) and up
        shoulder = (0.5 + 0.4, 0.7 - 0.4)  # 45 degree angle
        
        lean = calculate_trunk_lean(hip, shoulder)
        assert math.isclose(lean, 45.0, rel_tol=0.05)
    
    def test_backward_lean(self):
        """Test backward lean (negative angle)."""
        hip = (0.5, 0.7)
        shoulder = (0.3, 0.3)  # Shoulder behind hip
        
        lean = calculate_trunk_lean(hip, shoulder)
        assert lean < 0
    
    def test_extreme_forward_lean(self):
        """Test very aggressive forward lean (sprint start)."""
        hip = (0.5, 0.5)
        shoulder = (0.7, 0.4)  # Shoulder ahead and slightly up
        
        lean = calculate_trunk_lean(hip, shoulder)
        # Should be significant positive lean
        assert lean > 30


class TestExtractJointAngles:
    """Tests for joint angle extraction from landmarks."""
    
    def _create_landmark(
        self, x: float, y: float, visibility: float = 1.0
    ) -> Landmark:
        """Helper to create a Landmark."""
        return Landmark(x=x, y=y, z=0.0, visibility=visibility)
    
    def _create_dummy_landmarks(self) -> list[Landmark]:
        """Create a set of dummy landmarks for testing."""
        # Create 33 landmarks (BlazePose format)
        landmarks = [self._create_landmark(0.5, 0.5) for _ in range(33)]
        
        # Set up a simple standing pose
        # Shoulders
        landmarks[11] = self._create_landmark(0.4, 0.3)  # Left shoulder
        landmarks[12] = self._create_landmark(0.6, 0.3)  # Right shoulder
        
        # Elbows
        landmarks[13] = self._create_landmark(0.35, 0.45)  # Left elbow
        landmarks[14] = self._create_landmark(0.65, 0.45)  # Right elbow
        
        # Wrists
        landmarks[15] = self._create_landmark(0.3, 0.6)  # Left wrist
        landmarks[16] = self._create_landmark(0.7, 0.6)  # Right wrist
        
        # Hips
        landmarks[23] = self._create_landmark(0.45, 0.55)  # Left hip
        landmarks[24] = self._create_landmark(0.55, 0.55)  # Right hip
        
        # Knees
        landmarks[25] = self._create_landmark(0.45, 0.75)  # Left knee
        landmarks[26] = self._create_landmark(0.55, 0.75)  # Right knee
        
        # Ankles
        landmarks[27] = self._create_landmark(0.45, 0.95)  # Left ankle
        landmarks[28] = self._create_landmark(0.55, 0.95)  # Right ankle
        
        return landmarks
    
    def test_extracts_all_angles(self):
        """Test that all expected angles are extracted."""
        landmarks = self._create_dummy_landmarks()
        angles = extract_joint_angles(landmarks)
        
        expected_keys = [
            "left_knee", "right_knee",
            "left_hip", "right_hip",
            "left_elbow", "right_elbow",
            "trunk_lean"
        ]
        
        for key in expected_keys:
            assert key in angles
    
    def test_low_visibility_returns_nan(self):
        """Test that low visibility landmarks produce NaN angles."""
        landmarks = self._create_dummy_landmarks()
        
        # Set left knee visibility to low
        landmarks[25] = self._create_landmark(0.45, 0.75, visibility=0.1)
        
        angles = extract_joint_angles(landmarks, visibility_threshold=0.5)
        
        # Left knee angle should be NaN
        assert math.isnan(angles["left_knee"])
        
        # Right knee should still be valid
        assert not math.isnan(angles["right_knee"])
    
    def test_straight_standing_pose(self):
        """Test angles for a straight standing pose."""
        landmarks = self._create_dummy_landmarks()
        angles = extract_joint_angles(landmarks)
        
        # Standing pose should have knee angles close to 180 (straight)
        # Our dummy is not perfectly straight but should be > 150
        assert angles["left_knee"] > 150 or math.isnan(angles["left_knee"])


class TestHipHeight:
    """Tests for hip height extraction."""
    
    def _create_landmark(
        self, x: float, y: float, visibility: float = 1.0
    ) -> Landmark:
        return Landmark(x=x, y=y, z=0.0, visibility=visibility)
    
    def test_hip_height_calculation(self):
        """Test hip height is average of left and right hip y."""
        landmarks = [self._create_landmark(0.5, 0.5) for _ in range(33)]
        landmarks[23] = self._create_landmark(0.4, 0.6)  # Left hip
        landmarks[24] = self._create_landmark(0.6, 0.7)  # Right hip
        
        height = get_hip_height_normalized(landmarks)
        
        expected = (0.6 + 0.7) / 2  # 0.65
        assert math.isclose(height, expected, rel_tol=0.01)
    
    def test_low_visibility_returns_nan(self):
        """Test low visibility hips return NaN."""
        landmarks = [self._create_landmark(0.5, 0.5) for _ in range(33)]
        landmarks[23] = self._create_landmark(0.4, 0.6, visibility=0.1)  # Low vis
        landmarks[24] = self._create_landmark(0.6, 0.7, visibility=1.0)
        
        height = get_hip_height_normalized(landmarks, visibility_threshold=0.5)
        
        assert math.isnan(height)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
