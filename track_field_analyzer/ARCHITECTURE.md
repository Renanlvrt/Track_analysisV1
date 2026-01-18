# Track & Field Form Analyzer - Architecture Document

> **Version:** 1.0  
> **Last Updated:** 2026-01-18  
> **Status:** PLANNING

---

## 1. Problem Statement & Goals

### Problem
Athletes and coaches lack accessible, real-time tools to analyze sprinting form and technique. Traditional video analysis requires expensive software or manual frame-by-frame review. Key biomechanical metrics (joint angles, trunk lean, phase timing) are difficult to extract without specialized equipment.

### Goals
1. **Democratize form analysis**: Provide a free, easy-to-use web tool for athletes to upload videos and receive instant feedback
2. **Actionable insights**: Calculate key biomechanical metrics and compare against coaching-standard target ranges
3. **Visual feedback**: Overlay skeleton and angle annotations directly on video frames
4. **Sprint-start focus**: Prioritize analysis of sprint start mechanics (set position, drive phase, acceleration)

### Success Criteria
- Process a 10-second video in under 30 seconds
- Accurately detect pose in 90%+ of frames (good lighting conditions)
- Provide clear, understandable feedback for non-technical users

---

## 2. Scope

### In-Scope ✅
- Video upload (MP4, MOV, AVI formats)
- Image upload for single-frame analysis
- MediaPipe BlazePose 33-landmark detection
- Joint angle calculation (knee, hip, elbow - bilateral)
- Trunk lean angle estimation
- Sprint phase detection (set position, drive, acceleration) via heuristics
- Skeleton overlay visualization with angle annotations
- Configurable target ranges for feedback
- Frame sampling for performance optimization
- Streamlit web interface

### Out-of-Scope ❌
- Real-time webcam capture (future enhancement)
- 3D pose reconstruction
- Automatic rep counting
- Multi-person tracking
- Mobile app deployment
- Cloud GPU processing
- Medical/injury diagnosis (explicitly disclaimed)

---

## 3. User Stories

| ID | As a... | I want to... | So that... | Priority |
|----|---------|--------------|------------|----------|
| US-01 | Sprinter | Upload a video of my block start | I can see my joint angles and compare to ideal form | P0 |
| US-02 | Coach | View skeleton overlay on athlete video | I can visually identify form issues | P0 |
| US-03 | Athlete | See specific feedback on my trunk lean | I know if I'm rising too early in drive phase | P0 |
| US-04 | User | Control frame sampling rate | I can balance detail vs. processing speed | P1 |
| US-05 | Coach | Customize target angle ranges | I can set standards appropriate for my athletes | P1 |
| US-06 | Athlete | Export annotated frames | I can share analysis with my coach | P2 |

---

## 4. System Architecture

### 4.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT WEB APP                              │
│                                  (app.py)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Video Upload │  │   Controls   │  │   Metrics    │  │  Visualizer  │    │
│  │    Widget    │  │  (sampling)  │  │    Panel     │  │    Panel     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────▲───────┘  └──────▲───────┘    │
│         │                 │                 │                 │            │
└─────────┼─────────────────┼─────────────────┼─────────────────┼────────────┘
          │                 │                 │                 │
          ▼                 ▼                 │                 │
┌─────────────────────────────────────────────┼─────────────────┼────────────┐
│                    PROCESSING PIPELINE                        │            │
│  ┌──────────────────┐                                         │            │
│  │   src/io/        │                                         │            │
│  │   video.py       │──┐                                      │            │
│  │ - load_video()   │  │                                      │            │
│  │ - sample_frames()│  │                                      │            │
│  └──────────────────┘  │                                      │            │
│                        ▼                                      │            │
│  ┌──────────────────────────────────┐                         │            │
│  │   src/pose/mediapipe_pose.py     │                         │            │
│  │   - PoseEstimator class          │──┐                      │            │
│  │   - process_frame()              │  │                      │            │
│  │   - get_landmarks()              │  │                      │            │
│  └──────────────────────────────────┘  │                      │            │
│                                        ▼                      │            │
│  ┌────────────────────────────────────────────────────────┐   │            │
│  │                 src/analysis/                          │   │            │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │   │            │
│  │  │ angles.py   │ │ metrics.py  │ │ phases.py   │      │───┘            │
│  │  │ -calc_angle │ │ -FrameData  │ │ -detect_    │      │                │
│  │  │ -trunk_lean │ │ -aggregate  │ │  phase()    │      │                │
│  │  └─────────────┘ └─────────────┘ └─────────────┘      │                │
│  └───────────────────────────┬────────────────────────────┘                │
│                              │                                             │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │   src/viz/overlay.py                                         │──────────┘
│  │   - draw_skeleton()                                          │
│  │   - draw_angles()                                            │
│  │   - annotate_frame()                                         │
│  └──────────────────────────────────────────────────────────────┘
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │   config/targets.yaml                                        │          │
│  │   - Target angle ranges for feedback                         │          │
│  └──────────────────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

```
Video File (.mp4/.mov/.avi)
         │
         ▼
┌─────────────────────┐
│ Streamlit Upload    │
│ (UploadedFile obj)  │
└─────────┬───────────┘
          │ Write to temp file
          ▼
┌─────────────────────┐
│ OpenCV VideoCapture │
│ (read frames)       │
└─────────┬───────────┘
          │ Sample every Nth frame
          ▼
┌─────────────────────┐
│ List[np.ndarray]    │◄─── RGB frames
│ (sampled frames)    │
└─────────┬───────────┘
          │ For each frame
          ▼
┌─────────────────────┐
│ MediaPipe Pose      │
│ BlazePose (33 pts)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Landmarks + Vis     │◄─── NormalizedLandmark objects
│ (x, y, z, visibility)
└─────────┬───────────┘
          │
          ├──────────────────────────────┐
          ▼                              ▼
┌─────────────────────┐    ┌─────────────────────┐
│ Angle Calculations  │    │ Skeleton Overlay    │
│ - Knee angles       │    │ - Connection lines  │
│ - Hip angles        │    │ - Joint markers     │
│ - Elbow angles      │    │ - Angle annotations │
│ - Trunk lean        │    └─────────┬───────────┘
└─────────┬───────────┘              │
          │                          │
          ▼                          │
┌─────────────────────┐              │
│ Phase Detection     │              │
│ - Set position      │              │
│ - Drive phase       │              │
│ - Acceleration      │              │
└─────────┬───────────┘              │
          │                          │
          ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│ Metrics DataFrame   │    │ Annotated Frames    │
│ (per-frame data)    │    │ List[np.ndarray]    │
└─────────┬───────────┘    └─────────┬───────────┘
          │                          │
          ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│ Feedback Generation │    │ Display in          │
│ (compare to targets)│    │ Streamlit           │
└─────────┬───────────┘    └─────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Text Feedback +     │
│ Recommendations     │
└─────────────────────┘
```

---

## 5. Folder Structure

```
track_field_analyzer/
├── app.py                      # Streamlit entrypoint
├── ARCHITECTURE.md             # This document
├── README.md                   # Setup & usage instructions
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Optional: modern packaging
│
├── config/
│   └── targets.yaml            # Target angle ranges for feedback
│
├── src/
│   ├── __init__.py
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   └── video.py            # Video loading, frame sampling
│   │
│   ├── pose/
│   │   ├── __init__.py
│   │   └── mediapipe_pose.py   # Pose estimation wrapper
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── angles.py           # Joint angle calculations
│   │   ├── metrics.py          # Per-frame & aggregated metrics
│   │   └── phases.py           # Sprint phase heuristics
│   │
│   └── viz/
│       ├── __init__.py
│       └── overlay.py          # Skeleton & annotation drawing
│
├── tests/
│   ├── __init__.py
│   ├── test_angles.py          # Unit tests for angle math
│   ├── test_metrics.py         # Unit tests for metrics
│   └── test_video.py           # Unit tests for video loading
│
└── assets/                     # Optional: sample videos for testing
    └── sample_sprint.mp4
```

---

## 6. Public Interfaces

### 6.1 `src/io/video.py`

```python
def load_video_from_uploaded_file(uploaded_file: st.UploadedFile) -> str:
    """
    Write Streamlit uploaded file to temp location.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        str: Path to temporary video file
    """
    ...

def get_video_properties(video_path: str) -> dict:
    """
    Get video metadata.
    
    Returns:
        dict: {"fps": float, "frame_count": int, "width": int, "height": int, "duration_sec": float}
    """
    ...

def sample_frames(
    video_path: str, 
    sample_rate: int = 5,
    max_frames: int | None = None
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Generator yielding sampled frames from video.
    
    Args:
        video_path: Path to video file
        sample_rate: Process every Nth frame
        max_frames: Optional cap on total frames to process
        
    Yields:
        tuple: (frame_index, frame_rgb as np.ndarray)
    """
    ...
```

### 6.2 `src/pose/mediapipe_pose.py`

```python
class PoseEstimator:
    """Wrapper for MediaPipe BlazePose."""
    
    def __init__(
        self, 
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """Initialize MediaPipe Pose model."""
        ...
    
    def process_frame(self, frame_rgb: np.ndarray) -> PoseResult | None:
        """
        Run pose estimation on a single frame.
        
        Args:
            frame_rgb: RGB image as numpy array
            
        Returns:
            PoseResult or None if no pose detected
        """
        ...
    
    def close(self) -> None:
        """Release MediaPipe resources."""
        ...

@dataclass
class PoseResult:
    """Container for pose estimation results."""
    landmarks: list[Landmark]  # 33 BlazePose landmarks
    world_landmarks: list[Landmark] | None  # 3D coordinates
    
@dataclass
class Landmark:
    """Single pose landmark."""
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    z: float  # Depth relative to hips
    visibility: float  # Confidence [0, 1]
```

### 6.3 `src/analysis/angles.py`

```python
def calculate_angle(
    point_a: tuple[float, float],
    point_b: tuple[float, float],  # Vertex
    point_c: tuple[float, float]
) -> float:
    """
    Calculate angle at point_b formed by points a-b-c.
    
    Returns:
        float: Angle in degrees [0, 180], or NaN if invalid
    """
    ...

def calculate_trunk_lean(
    hip_mid: tuple[float, float],
    shoulder_mid: tuple[float, float]
) -> float:
    """
    Calculate trunk lean angle from vertical.
    
    Positive = forward lean, Negative = backward lean
    
    Returns:
        float: Angle in degrees, or NaN if invalid
    """
    ...

def extract_joint_angles(
    landmarks: list[Landmark],
    visibility_threshold: float = 0.5
) -> dict[str, float]:
    """
    Extract all relevant joint angles from pose landmarks.
    
    Returns:
        dict: {
            "left_knee": float,
            "right_knee": float,
            "left_hip": float,
            "right_hip": float,
            "left_elbow": float,
            "right_elbow": float,
            "trunk_lean": float
        }
    """
    ...
```

### 6.4 `src/analysis/metrics.py`

```python
@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    frame_index: int
    timestamp_sec: float
    angles: dict[str, float]
    phase: str  # "set", "drive", "acceleration", "unknown"
    feedback: list[str]  # List of feedback strings

def compute_frame_metrics(
    frame_index: int,
    fps: float,
    landmarks: list[Landmark],
    target_ranges: dict,
    visibility_threshold: float = 0.5
) -> FrameMetrics:
    """Compute all metrics for a single frame."""
    ...

def aggregate_metrics(
    frame_metrics_list: list[FrameMetrics]
) -> dict:
    """
    Aggregate metrics across all frames.
    
    Returns:
        dict: {
            "avg_angles": dict[str, float],
            "min_angles": dict[str, float],
            "max_angles": dict[str, float],
            "phase_distribution": dict[str, int],
            "overall_feedback": list[str]
        }
    """
    ...
```

### 6.5 `src/analysis/phases.py`

```python
def detect_sprint_phase(
    trunk_lean: float,
    hip_height_normalized: float,
    knee_angle_front: float
) -> str:
    """
    Detect sprint phase using heuristics.
    
    Heuristic (v1, simplified):
    - SET: Low hip height, trunk lean > 45°, front knee ~90°
    - DRIVE: Hip rising, trunk lean 30-45°, extending knee
    - ACCELERATION: Hip at running height, trunk lean 15-30°
    - UPRIGHT: Trunk lean < 15°
    
    Returns:
        str: One of ["set", "drive", "acceleration", "upright", "unknown"]
    """
    ...
```

### 6.6 `src/viz/overlay.py`

```python
def draw_skeleton(
    frame: np.ndarray,
    landmarks: list[Landmark],
    visibility_threshold: float = 0.5,
    connection_color: tuple[int, int, int] = (0, 255, 0),
    landmark_color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw pose skeleton on frame.
    
    Returns:
        np.ndarray: Frame with skeleton overlay (copy, original unchanged)
    """
    ...

def draw_angle_annotations(
    frame: np.ndarray,
    landmarks: list[Landmark],
    angles: dict[str, float],
    visibility_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw angle values near corresponding joints.
    
    Returns:
        np.ndarray: Frame with angle annotations
    """
    ...

def annotate_frame(
    frame: np.ndarray,
    landmarks: list[Landmark],
    angles: dict[str, float],
    phase: str,
    draw_angles: bool = True
) -> np.ndarray:
    """
    Full annotation pipeline: skeleton + angles + phase label.
    """
    ...
```

---

## 7. Requirements Traceability Matrix

| Requirement | Description | Module(s) | Test(s) |
|-------------|-------------|-----------|---------|
| REQ-01 | Upload video (mp4/mov/avi) | `io/video.py`, `app.py` | `test_video.py::test_load_formats` |
| REQ-02 | Upload image for single-frame | `io/video.py`, `app.py` | `test_video.py::test_load_image` |
| REQ-03 | Pose estimation (33 landmarks) | `pose/mediapipe_pose.py` | Manual integration test |
| REQ-04 | Joint angle calculation | `analysis/angles.py` | `test_angles.py::test_calculate_angle` |
| REQ-05 | Trunk lean calculation | `analysis/angles.py` | `test_angles.py::test_trunk_lean` |
| REQ-06 | Sprint phase detection | `analysis/phases.py` | `test_phases.py::test_phase_detection` |
| REQ-07 | Skeleton overlay | `viz/overlay.py` | Manual visual test |
| REQ-08 | Angle annotations | `viz/overlay.py` | Manual visual test |
| REQ-09 | Frame sampling control | `io/video.py`, `app.py` | `test_video.py::test_sampling` |
| REQ-10 | Configurable target ranges | `config/targets.yaml` | `test_metrics.py::test_feedback` |
| REQ-11 | Metrics display panel | `app.py` | Manual UI test |
| REQ-12 | Handle low visibility landmarks | `analysis/angles.py` | `test_angles.py::test_low_visibility` |
| REQ-13 | Temp file video handling | `io/video.py` | `test_video.py::test_temp_file` |

---

## 8. Configuration: Target Ranges

Reference: Based on common sprint coaching literature. Individual variation is normal.

```yaml
# config/targets.yaml
sprint_start:
  set_position:
    front_knee_angle:
      min: 90
      max: 110
      unit: degrees
      description: "Front knee angle in set position"
    rear_knee_angle:
      min: 120
      max: 135
      unit: degrees
      description: "Rear knee angle in set position"
    hip_angle:
      min: 30
      max: 45
      unit: degrees
      description: "Hip angle (torso to thigh)"
    trunk_lean:
      min: 40
      max: 55
      unit: degrees
      description: "Forward trunk lean from vertical"
      
  drive_phase:
    trunk_lean:
      min: 30
      max: 45
      unit: degrees
    knee_drive:
      min: 80
      max: 100
      unit: degrees
      
  acceleration:
    trunk_lean:
      min: 15
      max: 30
      unit: degrees

disclaimer: |
  These ranges are typical coaching targets based on sprint biomechanics literature.
  Individual optimal ranges vary based on body proportions, flexibility, and 
  personal mechanics. These are NOT medical recommendations. Consult a qualified
  coach for personalized guidance.
```

---

## 9. Build Checklist

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Create ARCHITECTURE.md | ✅ DONE | This document |
| 2 | Set up project structure | ✅ DONE | All folders and `__init__.py` created |
| 3 | Create requirements.txt | ✅ DONE | All dependencies listed |
| 4 | Implement `io/video.py` | ✅ DONE | Temp file handling + frame sampling |
| 5 | Implement `pose/mediapipe_pose.py` | ✅ DONE | BlazePose wrapper with 33 landmarks |
| 6 | Implement `analysis/angles.py` | ✅ DONE | Joint angles + trunk lean |
| 7 | Write `test_angles.py` | ✅ DONE | Comprehensive unit tests |
| 8 | Implement `analysis/metrics.py` | ✅ DONE | Per-frame + aggregated metrics |
| 9 | Implement `analysis/phases.py` | ✅ DONE | Sprint phase heuristics |
| 10 | Implement `viz/overlay.py` | ✅ DONE | Skeleton + annotations |
| 11 | Create `config/targets.yaml` | ✅ DONE | 60m-200m target ranges |
| 12 | Implement `app.py` | ✅ DONE | Full Streamlit app |
| 13 | Write remaining tests | ✅ DONE | test_phases.py, test_metrics.py |
| 14 | Create README.md | ✅ DONE | Full documentation |
| 15 | End-to-end testing | 🔄 PENDING | Requires sample video |
| 16 | Performance optimization | ✅ DONE | Frame sampling + caching |

---

## 10. Technical Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-18 | Use MediaPipe BlazePose over OpenPose | Lighter weight, easier setup, 33 landmarks sufficient |
| 2026-01-18 | Streamlit over Flask | Simpler for data apps, built-in widgets |
| 2026-01-18 | Temp file approach for video | Streamlit UploadedFile has no filesystem path |
| 2026-01-18 | Heuristic-based phase detection | No training data; rule-based v1 is interpretable |
| 2026-01-18 | Generator for frame sampling | Memory efficient for large videos |

---

## Appendix A: MediaPipe BlazePose Landmark Indices

```
0: nose               17: left_pinky
1: left_eye_inner     18: right_pinky
2: left_eye           19: left_index
3: left_eye_outer     20: right_index
4: right_eye_inner    21: left_thumb
5: right_eye          22: right_thumb
6: right_eye_outer    23: left_hip
7: left_ear           24: right_hip
8: right_ear          25: left_knee
9: mouth_left         26: right_knee
10: mouth_right       27: left_ankle
11: left_shoulder     28: right_ankle
12: right_shoulder    29: left_heel
13: left_elbow        30: right_heel
14: right_elbow       31: left_foot_index
15: left_wrist        32: right_foot_index
16: right_wrist
```

**Key landmarks for sprint analysis:**
- Hips: 23, 24 (hip height, phase detection)
- Knees: 25, 26 (knee angles)
- Ankles: 27, 28 (foot strike)
- Shoulders: 11, 12 (trunk lean)
- Elbows: 13, 14 (arm drive)
