"""Unit tests for metrics calculation."""

import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.metrics import (
    FrameMetrics,
    compute_frame_metrics,
    aggregate_metrics,
    load_target_ranges,
    generate_feedback,
)
from src.analysis.phases import SprintPhase
from src.pose.mediapipe_pose import Landmark


class TestFrameMetrics:
    """Tests for FrameMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = FrameMetrics(
            frame_index=10,
            timestamp_sec=0.333,
            angles={"left_knee": 95.5, "trunk_lean": 45.0},
            hip_height=0.6,
            phase=SprintPhase.SET,
            feedback=["Keep shoulders forward"],
        )
        
        result = metrics.to_dict()
        
        assert result["frame_index"] == 10
        assert result["timestamp_sec"] == 0.333
        assert result["phase"] == "set"
        assert result["left_knee"] == 95.5
        assert result["trunk_lean"] == 45.0
    
    def test_to_dict_handles_nan(self):
        """Test that NaN values become None in dict."""
        metrics = FrameMetrics(
            frame_index=0,
            timestamp_sec=0.0,
            angles={"left_knee": float("nan")},
            hip_height=float("nan"),
            phase=SprintPhase.UNKNOWN,
        )
        
        result = metrics.to_dict()
        
        assert result["left_knee"] is None
        assert result["hip_height"] is None


class TestAggregateMetrics:
    """Tests for metrics aggregation."""
    
    def _create_frame_metrics(
        self,
        frame_index: int,
        angles: dict[str, float],
        phase: SprintPhase,
    ) -> FrameMetrics:
        return FrameMetrics(
            frame_index=frame_index,
            timestamp_sec=frame_index / 30.0,  # Assume 30 FPS
            angles=angles,
            hip_height=0.5,
            phase=phase,
            feedback=[],
        )
    
    def test_empty_list(self):
        """Test aggregation of empty list."""
        result = aggregate_metrics([])
        
        assert result["avg_angles"] == {}
        assert result["phase_distribution"] == {}
        assert result["phase_sequence"] == []
    
    def test_single_frame(self):
        """Test aggregation of single frame."""
        metrics = [
            self._create_frame_metrics(
                0,
                {"left_knee": 90.0, "trunk_lean": 45.0},
                SprintPhase.SET,
            )
        ]
        
        result = aggregate_metrics(metrics)
        
        assert result["avg_angles"]["left_knee"] == 90.0
        assert result["avg_angles"]["trunk_lean"] == 45.0
        assert result["phase_distribution"]["set"] == 1
    
    def test_average_calculation(self):
        """Test that averages are calculated correctly."""
        metrics = [
            self._create_frame_metrics(0, {"left_knee": 80.0}, SprintPhase.SET),
            self._create_frame_metrics(1, {"left_knee": 100.0}, SprintPhase.SET),
            self._create_frame_metrics(2, {"left_knee": 90.0}, SprintPhase.SET),
        ]
        
        result = aggregate_metrics(metrics)
        
        # Average of 80, 100, 90 = 90
        assert result["avg_angles"]["left_knee"] == 90.0
    
    def test_min_max_calculation(self):
        """Test min/max angle tracking."""
        metrics = [
            self._create_frame_metrics(0, {"left_knee": 80.0}, SprintPhase.SET),
            self._create_frame_metrics(1, {"left_knee": 100.0}, SprintPhase.DRIVE),
            self._create_frame_metrics(2, {"left_knee": 90.0}, SprintPhase.DRIVE),
        ]
        
        result = aggregate_metrics(metrics)
        
        assert result["min_angles"]["left_knee"] == 80.0
        assert result["max_angles"]["left_knee"] == 100.0
    
    def test_phase_distribution(self):
        """Test phase counting."""
        metrics = [
            self._create_frame_metrics(0, {}, SprintPhase.SET),
            self._create_frame_metrics(1, {}, SprintPhase.SET),
            self._create_frame_metrics(2, {}, SprintPhase.DRIVE),
            self._create_frame_metrics(3, {}, SprintPhase.DRIVE),
            self._create_frame_metrics(4, {}, SprintPhase.DRIVE),
            self._create_frame_metrics(5, {}, SprintPhase.ACCELERATION),
        ]
        
        result = aggregate_metrics(metrics)
        
        assert result["phase_distribution"]["set"] == 2
        assert result["phase_distribution"]["drive"] == 3
        assert result["phase_distribution"]["acceleration"] == 1
    
    def test_phase_sequence(self):
        """Test phase transition tracking."""
        metrics = [
            self._create_frame_metrics(0, {}, SprintPhase.SET),
            self._create_frame_metrics(1, {}, SprintPhase.SET),
            self._create_frame_metrics(2, {}, SprintPhase.DRIVE),
            self._create_frame_metrics(3, {}, SprintPhase.ACCELERATION),
        ]
        
        result = aggregate_metrics(metrics)
        
        seq = result["phase_sequence"]
        assert len(seq) == 3  # Three distinct phases
        assert seq[0]["phase"] == "set"
        assert seq[1]["phase"] == "drive"
        assert seq[2]["phase"] == "acceleration"
    
    def test_nan_values_excluded_from_average(self):
        """Test that NaN values are excluded from averaging."""
        metrics = [
            self._create_frame_metrics(0, {"left_knee": 90.0}, SprintPhase.SET),
            self._create_frame_metrics(1, {"left_knee": float("nan")}, SprintPhase.SET),
            self._create_frame_metrics(2, {"left_knee": 100.0}, SprintPhase.SET),
        ]
        
        result = aggregate_metrics(metrics)
        
        # Average of 90 and 100 (excluding NaN) = 95
        assert result["avg_angles"]["left_knee"] == 95.0


class TestLoadTargetRanges:
    """Tests for target range loading."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = load_target_ranges()
        assert isinstance(result, dict)
    
    def test_has_phases_key(self):
        """Test that result has phases configuration."""
        result = load_target_ranges()
        assert "phases" in result
    
    def test_missing_file_returns_defaults(self):
        """Test that missing config file returns defaults."""
        result = load_target_ranges("/nonexistent/path/config.yaml")
        assert isinstance(result, dict)
        assert "phases" in result


class TestGenerateFeedback:
    """Tests for feedback generation."""
    
    def test_empty_targets_no_feedback(self):
        """Test that empty targets produce no feedback."""
        feedback = generate_feedback(
            angles={"trunk_lean": 45.0},
            phase=SprintPhase.SET,
            target_config={"phases": {}},
        )
        
        assert feedback == []
    
    def test_unknown_phase_no_feedback(self):
        """Test that unknown phase produces no feedback."""
        feedback = generate_feedback(
            angles={"trunk_lean": 45.0},
            phase=SprintPhase.UNKNOWN,
            target_config={"phases": {"set_position": {"targets": {}}}},
        )
        
        assert feedback == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
