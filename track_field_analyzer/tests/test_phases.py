"""Unit tests for sprint phase detection."""

import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.phases import (
    detect_sprint_phase,
    SprintPhase,
    get_phase_description,
)


class TestDetectSprintPhase:
    """Tests for phase detection heuristics."""
    
    def test_set_position_detection(self):
        """Test set position is detected with low hips and strong lean."""
        # Set position: hips low (high y), strong forward lean
        phase = detect_sprint_phase(
            trunk_lean=50.0,
            hip_height_normalized=0.65,  # Low in frame
            knee_angle_front=95.0,
        )
        
        assert phase == SprintPhase.SET
    
    def test_drive_phase_detection(self):
        """Test drive phase is detected with medium hips and forward lean."""
        # Hip at 0.53 (below SET threshold of 0.55) but with good lean
        phase = detect_sprint_phase(
            trunk_lean=40.0,
            hip_height_normalized=0.53,  # Below SET threshold
            knee_angle_front=None,
        )
        
        assert phase == SprintPhase.DRIVE
    
    def test_acceleration_detection(self):
        """Test acceleration phase detection."""
        phase = detect_sprint_phase(
            trunk_lean=20.0,
            hip_height_normalized=0.45,
            knee_angle_front=None,
        )
        
        assert phase == SprintPhase.ACCELERATION
    
    def test_max_velocity_detection(self):
        """Test max velocity phase with upright posture."""
        # Hip at 0.30 (below ACCELERATION min of 0.35) and low lean
        phase = detect_sprint_phase(
            trunk_lean=10.0,
            hip_height_normalized=0.30,  # Below ACCELERATION threshold
            knee_angle_front=None,
        )
        
        assert phase == SprintPhase.MAX_VELOCITY
    
    def test_nan_trunk_lean_returns_unknown(self):
        """Test NaN trunk lean returns unknown phase."""
        phase = detect_sprint_phase(
            trunk_lean=float("nan"),
            hip_height_normalized=0.5,
        )
        
        assert phase == SprintPhase.UNKNOWN
    
    def test_nan_hip_height_returns_unknown(self):
        """Test NaN hip height returns unknown phase."""
        phase = detect_sprint_phase(
            trunk_lean=30.0,
            hip_height_normalized=float("nan"),
        )
        
        assert phase == SprintPhase.UNKNOWN
    
    def test_negative_trunk_lean_handled(self):
        """Test that negative trunk lean (backward) is handled."""
        # Should use absolute value for comparison
        phase = detect_sprint_phase(
            trunk_lean=-45.0,  # Backward lean (or opposite camera angle)
            hip_height_normalized=0.6,
            knee_angle_front=100.0,
        )
        
        # Should still detect based on absolute lean
        assert phase in [SprintPhase.SET, SprintPhase.DRIVE]
    
    def test_phase_transitions(self):
        """Test realistic phase transition sequence."""
        phases = []
        
        # Simulate frames through sprint start
        test_cases = [
            # Set position
            (50, 0.65, 95),
            (48, 0.62, 100),
            # Drive phase
            (42, 0.55, None),
            (38, 0.50, None),
            # Acceleration
            (25, 0.45, None),
            (20, 0.40, None),
            # Max velocity
            (10, 0.35, None),
            (8, 0.33, None),
        ]
        
        for trunk, hip, knee in test_cases:
            phase = detect_sprint_phase(trunk, hip, knee)
            phases.append(phase)
        
        # Should progress through phases
        assert SprintPhase.SET in phases[:3]
        assert SprintPhase.MAX_VELOCITY in phases[-3:]


class TestSprintPhaseEnum:
    """Tests for SprintPhase enum properties."""
    
    def test_display_names(self):
        """Test all phases have display names."""
        for phase in SprintPhase:
            assert phase.display_name is not None
            assert len(phase.display_name) > 0
    
    def test_colors(self):
        """Test all phases have RGB colors."""
        for phase in SprintPhase:
            color = phase.color
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_value_strings(self):
        """Test phase values are lowercase strings."""
        for phase in SprintPhase:
            assert isinstance(phase.value, str)
            assert phase.value == phase.value.lower()


class TestPhaseDescriptions:
    """Tests for phase description function."""
    
    def test_all_phases_have_descriptions(self):
        """Test all phases return non-empty descriptions."""
        for phase in SprintPhase:
            desc = get_phase_description(phase)
            assert isinstance(desc, str)
            assert len(desc) > 0
    
    def test_set_description_mentions_blocks(self):
        """Test set position description mentions blocks."""
        desc = get_phase_description(SprintPhase.SET)
        assert "block" in desc.lower()
    
    def test_drive_description_mentions_push(self):
        """Test drive phase description mentions forward push."""
        desc = get_phase_description(SprintPhase.DRIVE)
        assert "drive" in desc.lower() or "push" in desc.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
