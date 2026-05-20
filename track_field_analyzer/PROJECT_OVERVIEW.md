# 🏃 Sprint Form Analyzer

## A Production-Quality Biomechanics Analysis Tool for Track & Field Athletes

**Sprint Form Analyzer** is an AI-powered web application that provides real-time biomechanical feedback for sprinters competing in 60m–200m events. Using state-of-the-art computer vision and pose estimation technology, the app analyzes video recordings of sprint sessions to deliver actionable coaching insights, joint angle measurements, and phase-by-phase breakdowns.

---
2
## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Architecture](#architecture)
5. [Libraries & Dependencies](#libraries--dependencies)
6. [How It Works](#how-it-works)
7. [Sprint Phase Detection](#sprint-phase-detection)
8. [Angle Calculations](#angle-calculations)
9. [Scoring System](#scoring-system)
10. [User Interface](#user-interface)
11. [Project Structure](#project-structure)
12. [Installation](#installation)
13. [Usage](#usage)
14. [Limitations](#limitations)
15. [Future Enhancements](#future-enhancements)

---

## Overview

Sprint Form Analyzer bridges the gap between professional biomechanics labs and everyday athletes. While elite sprinters have access to motion capture systems and sports scientists, most athletes rely on subjective feedback or expensive coaching sessions. This tool democratizes sprint analysis by providing:

- **Objective measurements** of joint angles and body position
- **Phase-specific feedback** tailored to block starts, drive phase, and acceleration
- **Visual overlays** showing skeleton tracking in real-time
- **Actionable coaching cues** that athletes can immediately apply

The application is designed with a "consumer app" feel inspired by [Ochy](https://ochy.com/), prioritizing simplicity, clarity, and trust over raw data dumps.

---

## Key Features

### 🎯 Hero Score Card
A single 0-10 score that answers "How is my form?" at a glance. The score is calculated based on phase-appropriate biomechanical targets and includes a summary of key takeaways.

### 🏃 Top 3 Focus Areas
Instead of overwhelming users with dozens of metrics, the app surfaces the three most actionable coaching cues:
- **Trunk Lean** — Forward angle for power generation
- **Knee Drive** — Height and speed of knee lift
- **Current Phase** — Set, Drive, Acceleration, or Max Velocity

### 📊 Progressive Disclosure
Information is organized in tabs:
1. **Overview** — Hero score + focus areas + quick preview
2. **Video & Overlay** — Full frame-by-frame viewer with skeleton
3. **All Metrics** — Detailed joint angles with target ranges
4. **How It Works** — Methodology + limitations for transparency

### 📐 Context Bars
Each metric includes a visual "context bar" showing where your current value falls within the target range. The bar uses a gradient (red → amber → green → amber → red) making it instantly clear if you're in the optimal zone.

### 🎬 Clean Skeleton Overlay
The pose skeleton is drawn on each frame with color-coded joints:
- **Magenta** — Shoulders
- **Orange** — Hips
- **Cyan** — Knees
- **Green** — Skeleton connections

Angle labels can be toggled on/off to keep the video clean.

### ⚙️ Configurable Settings
All settings are in a collapsible sidebar:
- Analysis mode (Fast / Balanced / Accurate)
- Video width slider (50–80%)
- Skeleton and angle display toggles
- User profile (event type, experience level)

---

## Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **MediaPipe** | 0.10.x | AI pose estimation (BlazePose) |
| **OpenCV** | 4.8+ | Video processing and frame manipulation |
| **Streamlit** | 1.28+ | Web application framework |
| **NumPy** | 1.24+ | Numerical computations |
| **pandas** | 2.0+ | Data handling and aggregation |
| **PyYAML** | 6.0+ | Configuration file parsing |

### Development & Testing

| Library | Purpose |
|---------|---------|
| **pytest** | Unit testing framework |
| **pytest-cov** | Code coverage reporting |
| **mypy** | Static type checking (optional) |

---

## Libraries & Dependencies

### MediaPipe (Pose Estimation)

**MediaPipe** is Google's open-source framework for building multimodal ML pipelines. We use the **BlazePose** model which detects **33 body landmarks** in real-time:

```
NOSE, LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER, RIGHT_EYE_INNER,
RIGHT_EYE, RIGHT_EYE_OUTER, LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT,
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST,
RIGHT_WRIST, LEFT_PINKY, RIGHT_PINKY, LEFT_INDEX, RIGHT_INDEX, LEFT_THUMB,
RIGHT_THUMB, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE,
RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX
```

Each landmark includes:
- `x, y` — Normalized coordinates (0–1)
- `z` — Depth relative to hips
- `visibility` — Detection confidence (0–1)

**Why MediaPipe over other solutions?**
- **Lightweight**: Runs on CPU, no GPU required
- **Accurate**: State-of-the-art pose estimation
- **Easy integration**: Python API with simple interface
- **Free**: No API costs or usage limits

### OpenCV (Computer Vision)

**OpenCV** (Open Source Computer Vision Library) handles all video and image processing:

- **Video capture**: Reading frames from uploaded files
- **Frame manipulation**: RGB/BGR conversion, resizing
- **Drawing**: Skeleton lines, circles, text annotations
- **Temporary file handling**: Writing uploaded bytes to disk for processing

Key OpenCV functions used:
```python
cv2.VideoCapture()      # Open video file
cv2.read()              # Read frames
cv2.line()              # Draw skeleton connections
cv2.circle()            # Draw joint keypoints
cv2.putText()           # Add text overlays
cv2.rectangle()         # Draw text backgrounds
cv2.cvtColor()          # Color space conversion
```

### Streamlit (Web Framework)

**Streamlit** is a Python framework for building data science web apps with minimal code. We use it for:

- **Page layout**: Columns, tabs, containers, expanders
- **Widgets**: Sliders, buttons, file uploaders, checkboxes
- **Session state**: Persisting data across reruns
- **Caching**: Avoiding recomputation of expensive operations
- **Theming**: Custom colors and typography via `config.toml`

Key Streamlit components:
```python
st.set_page_config()    # Page title, layout, icon
st.columns()            # Multi-column layouts
st.tabs()               # Tabbed navigation
st.sidebar              # Collapsible settings panel
st.session_state        # Persistent variables
st.file_uploader()      # Video upload widget
st.image()              # Display processed frames
st.slider()             # Frame navigation
st.progress()           # Processing progress bar
st.expander()           # Collapsible sections
st.markdown()           # Rich HTML/CSS styling
```

### NumPy (Numerical Computing)

**NumPy** is the fundamental package for scientific computing in Python. We use it for:

- **Frame arrays**: Video frames are NumPy arrays (H × W × 3)
- **Angle calculations**: Vector math for computing joint angles
- **Statistics**: Aggregating metrics (mean, min, max)

Key operations:
```python
np.array()              # Create arrays
np.arctan2()            # Compute angles from vectors
np.degrees()            # Convert radians to degrees
np.mean()               # Average angles across frames
np.clip()               # Clamp values to range
np.isnan()              # Handle missing data
```

### pandas (Data Analysis)

**pandas** is used for data manipulation and presentation:

- **DataFrames**: Organizing frame-by-frame metrics
- **Aggregation**: Computing session statistics
- **Display**: Rendering metrics tables in Streamlit

### PyYAML (Configuration)

**PyYAML** parses the `config/targets.yaml` file which contains:
- Target angle ranges for each sprint phase
- Phase detection thresholds
- Coaching feedback messages

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT APP (app.py)                         │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Upload    │  │  Process    │  │   Render    │  │      Sidebar        │ │
│  │   Video     │──│   Frames    │──│   Results   │  │      Settings       │ │
│  └─────────────┘  └──────┬──────┘  └─────────────┘  └─────────────────────┘ │
│                          │                                                   │
└──────────────────────────┼───────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│    src/io     │  │   src/pose    │  │ src/analysis  │
│               │  │               │  │               │
│ • Video load  │  │ • MediaPipe   │  │ • Angles      │
│ • Frame sample│  │ • Landmarks   │  │ • Metrics     │
│ • Temp files  │  │ • Connections │  │ • Phases      │
└───────────────┘  └───────────────┘  └───────────────┘
                           │
                           ▼
                   ┌───────────────┐
                   │    src/viz    │
                   │               │
                   │ • Skeleton    │
                   │ • Annotations │
                   │ • Phase label │
                   └───────────────┘
```

### Module Responsibilities

| Module | Files | Responsibility |
|--------|-------|----------------|
| **io** | `video.py` | Load videos from Streamlit uploads, sample frames, manage temp files |
| **pose** | `mediapipe_pose.py` | Wrap MediaPipe BlazePose, provide Landmark dataclass, define skeleton connections |
| **analysis** | `angles.py`, `metrics.py`, `phases.py` | Calculate joint angles, compute per-frame/aggregate metrics, detect sprint phases |
| **viz** | `overlay.py` | Draw skeleton, angle annotations, phase labels on frames |
| **config** | `targets.yaml` | Define target ranges, thresholds, coaching messages |

---

## How It Works

### Step 1: Video Upload
User uploads a video file (MP4, MOV, or AVI). The file is written to a temporary location because MediaPipe requires a file path, not a byte stream.

### Step 2: Frame Sampling
To balance speed and accuracy, frames are sampled at a configurable rate:
- **Fast mode**: Every 10th frame, max 10 frames
- **Balanced mode**: Every 5th frame, max 30 frames
- **Accurate mode**: Every 2nd frame, max 100 frames

### Step 3: Pose Estimation
Each sampled frame is passed to MediaPipe BlazePose:
```python
with PoseEstimator(model_complexity=1) as estimator:
    result = estimator.process_frame(frame_rgb)
    landmarks = result.landmarks  # 33 Landmark objects
```

### Step 4: Angle Calculation
Joint angles are computed using vector mathematics:
```python
def calculate_angle(a, b, c):
    """Angle at point b between vectors ba and bc."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = sqrt(bc[0]**2 + bc[1]**2)
    
    cos_angle = dot / (mag_ba * mag_bc)
    return degrees(acos(clamp(cos_angle, -1, 1)))
```

### Step 5: Phase Detection
Sprint phase is classified using heuristics based on:
- **Trunk lean** (angle from vertical)
- **Hip height** (normalized y-coordinate)
- **Front knee angle** (for set position detection)

### Step 6: Feedback Generation
Each angle is compared against phase-specific target ranges from `targets.yaml`. If outside the range, a coaching cue is generated.

### Step 7: Visualization
Skeleton and annotations are drawn on each frame using OpenCV drawing primitives.

### Step 8: Aggregation
Session-level statistics are computed:
- Average angles across all frames
- Phase distribution (% of time in each phase)
- Phase sequence (transitions detected)
- Overall coaching feedback

---

## Sprint Phase Detection

### Phase Definitions

| Phase | Trunk Lean | Hip Height | Description |
|-------|------------|------------|-------------|
| **Set** | 40–55° | ≥ 0.55 | In blocks, ready position |
| **Drive** | 30–50° | 0.45–0.55 | First 10–30m, aggressive push |
| **Acceleration** | 15–35° | 0.35–0.45 | Transitioning upright |
| **Max Velocity** | 0–15° | ≤ 0.45 | Full speed, upright posture |

### Detection Algorithm
```python
def detect_sprint_phase(trunk_lean, hip_height, knee_angle_front):
    if hip_height >= 0.55 and trunk_lean >= 40 and knee_angle_front and knee_angle_front < 130:
        return SprintPhase.SET
    elif hip_height >= 0.45 and trunk_lean >= 30:
        return SprintPhase.DRIVE
    elif 0.35 <= hip_height < 0.55 and 10 <= trunk_lean < 35:
        return SprintPhase.ACCELERATION
    elif hip_height <= 0.45 and trunk_lean <= 20:
        return SprintPhase.MAX_VELOCITY
    else:
        return SprintPhase.UNKNOWN
```

---

## Angle Calculations

### Trunk Lean
```
Trunk Lean = angle between (hip→shoulder vector) and (vertical)
```
- Positive = forward lean
- Negative = backward lean

### Knee Angle
```
Knee Angle = angle at knee between (hip→knee) and (knee→ankle)
```
- ~180° = fully extended
- ~90° = right angle flexion

### Hip Angle
```
Hip Angle = angle at hip between (shoulder→hip) and (hip→knee)
```
- Indicates hip extension during stride

### Elbow Angle
```
Elbow Angle = angle at elbow between (shoulder→elbow) and (elbow→wrist)
```
- Ideal for sprinting: ~90°

---

## Scoring System

### Form Score (0–10)

The hero score provides a single number summarizing form quality:

```python
score = 5.0  # Base score

# Phase-appropriate trunk lean
if phase == SET and 40 <= trunk_lean <= 55:
    score += 2.0
elif phase == DRIVE and 30 <= trunk_lean <= 50:
    score += 2.5
elif phase == ACCELERATION and 15 <= trunk_lean <= 35:
    score += 2.0
elif phase == MAX_VELOCITY and trunk_lean <= 15:
    score += 2.5

# Knee drive bonus
if 90 <= front_knee <= 120:
    score += 1.0

# Unknown phase penalty
if phase == UNKNOWN:
    score -= 1.0

return clamp(score, 0, 10)
```

### Rating Badges
| Score | Badge | Color |
|-------|-------|-------|
| 7.0+ | GOOD | Green (#22c55e) |
| 5.0–6.9 | OKAY | Amber (#f59e0b) |
| < 5.0 | NEEDS WORK | Red (#ef4444) |

---

## User Interface

### Color Palette

| Element | Color | Hex |
|---------|-------|-----|
| Background | Deep navy | #0a0a14 |
| Cards | Dark charcoal | #12121f |
| Primary accent | Neon cyan | #00d4ff |
| Success | Neon green | #22c55e |
| Warning | Amber | #f59e0b |
| Error | Bright red | #ef4444 |
| Primary text | Off-white | #e8e8e8 |
| Secondary text | Muted gray | #a0a0a0 |

### Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ⚙️ SIDEBAR                              MAIN CONTENT                    │
│ ┌─────────────────┐  ┌─────────────────────────────────────────────────┐│
│ │ Analysis Mode   │  │ [Overview] [Video] [Metrics] [How It Works]    ││
│ │ ○ Fast          │  │                                                 ││
│ │ ● Balanced      │  │  ┌──────────────────────────────────────────┐  ││
│ │ ○ Accurate      │  │  │           HERO SCORE CARD                │  ││
│ ├─────────────────┤  │  │         Form Score: 7.2/10               │  ││
│ │ Display         │  │  │           [GOOD]                         │  ││
│ │ Video Width: 65%│  │  └──────────────────────────────────────────┘  ││
│ │ ☑ Skeleton      │  │                                                 ││
│ │ ☐ Angles        │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐          ││
│ ├─────────────────┤  │  │ Trunk   │ │ Knee    │ │ Phase   │          ││
│ │ Your Profile    │  │  │ Lean    │ │ Drive   │ │         │          ││
│ │ Event: 100m     │  │  │ 42° ✓   │ │ 105° ⚠️  │ │ Drive   │          ││
│ │ Level: Inter.   │  │  └─────────┘ └─────────┘ └─────────┘          ││
│ ├─────────────────┤  │                                                 ││
│ │ ❓ Filming Tips │  │  [Video Preview]                               ││
│ │ ⚠️ Limitations   │  │  ◀ ═══════●═══════ ▶                          ││
│ └─────────────────┘  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
track_field_analyzer/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Basic setup instructions
├── ARCHITECTURE.md             # System design document
├── UX_SPEC.md                  # UX specification
├── DESIGN_DECISIONS.md         # Design trade-offs log
│
├── .streamlit/
│   └── config.toml             # Theme configuration
│
├── config/
│   └── targets.yaml            # Target angle ranges & coaching messages
│
├── src/
│   ├── __init__.py
│   ├── io/
│   │   ├── __init__.py
│   │   └── video.py            # Video loading & frame sampling
│   ├── pose/
│   │   ├── __init__.py
│   │   └── mediapipe_pose.py   # MediaPipe BlazePose wrapper
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── angles.py           # Joint angle calculations
│   │   ├── metrics.py          # Per-frame & aggregate metrics
│   │   └── phases.py           # Sprint phase detection
│   └── viz/
│       ├── __init__.py
│       └── overlay.py          # Skeleton & annotation drawing
│
└── tests/
    ├── __init__.py
    ├── test_angles.py          # 16 unit tests for angle math
    ├── test_phases.py          # 14 unit tests for phase detection
    └── test_metrics.py         # 14 unit tests for metrics
```

---

## Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)

### Steps

1. **Navigate to the project directory**
   ```bash
   cd track_field_analyzer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   The app will automatically open at `http://localhost:8501`

---

## Usage

### 1. Upload Your Video
- Click "Browse files" or drag and drop
- Supported formats: MP4, MOV, AVI
- Recommended: 5–10 seconds, side view, full body visible

### 2. Adjust Settings (Optional)
- **Analysis Mode**: Fast (quick preview), Balanced (default), Accurate (thorough)
- **Video Width**: Adjust the frame viewer size
- **Display Options**: Toggle skeleton and angle overlays

### 3. Analyze
- Click "🚀 Analyze My Sprint"
- Wait for processing (progress bar shows status)
- Results appear automatically when complete

### 4. Review Results
- **Overview tab**: Hero score + top 3 focus areas
- **Video tab**: Frame-by-frame viewer with skeleton
- **Metrics tab**: Detailed angles with target ranges
- **How It Works tab**: Methodology and limitations

### 5. Take Action
- Note the coaching cues
- Re-film after making adjustments
- Track improvement over time

---

## Limitations

### Technical Limitations
1. **2D Analysis**: Depth and rotation cannot be accurately measured from 2D video
2. **Camera Angle Sensitivity**: Side view is required; oblique angles distort measurements
3. **Occlusion**: Body parts hidden from camera cannot be tracked
4. **Lighting**: Poor lighting reduces pose detection accuracy
5. **Motion Blur**: Very fast movements may cause landmark jitter

### Accuracy Notes
- Target ranges are based on coaching literature, not personalized to individual anatomy
- Heuristic-based phase detection may misclassify ambiguous frames
- Form score is a simplified composite, not a validated biomechanical metric

### Scope
This tool is designed for **training feedback only**. It is not:
- A medical diagnostic tool
- A replacement for qualified coaching
- Suitable for injury prevention assessment

---

## Future Enhancements

### Planned Features
- [ ] **PDF Export**: Download analysis report
- [ ] **Video Comparison**: Side-by-side before/after
- [ ] **Session History**: Track progress over time
- [ ] **Custom Target Ranges**: Personalized thresholds

### Potential Additions
- Real-time webcam analysis
- Multi-angle fusion (front + side cameras)
- Audio coaching cues during recording
- Integration with wearable sensors
- Cloud processing for longer videos

---

## Credits

- **MediaPipe BlazePose** by Google Research
- **OpenCV** by OpenCV.org
- **Streamlit** by Snowflake
- UI design inspired by [Ochy App](https://ochy.com/)

---

## License

MIT License — Free for personal projects, training, and coaching.

---

*Built with ❤️ for sprinters and coaches who want to see what they can't feel.*
