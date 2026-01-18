# 🏃 Track & Field Form Analyzer

AI-powered sprint form analysis for 60m-200m events using computer vision and pose estimation.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)

## 🎯 Features

- **Video & Image Analysis**: Upload sprint videos or single images for analysis
- **AI Pose Estimation**: Uses MediaPipe BlazePose (33 landmarks) for accurate pose detection
- **Joint Angle Calculation**: Measures knee, hip, elbow angles, and trunk lean
- **Sprint Phase Detection**: Automatically identifies Set, Drive, Acceleration, and Max Velocity phases
- **Visual Feedback**: Skeleton overlay with angle annotations on each frame
- **Coaching Feedback**: Compares your form against target ranges with actionable advice

## 📸 Supported Events

- 60m
- 100m
- 200m

Focus areas: Block starts, drive phase, acceleration mechanics

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd track_field_analyzer
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### Video Analysis

1. Click **"📹 Video Analysis"** tab
2. Upload a video file (MP4, MOV, or AVI)
3. Adjust settings in the sidebar:
   - **Sample rate**: Process every Nth frame (higher = faster)
   - **Max frames**: Limit total frames processed
   - **Model complexity**: 0 (fast) to 2 (accurate)
4. Click **"🚀 Analyze Video"**
5. View results:
   - Frame-by-frame viewer with skeleton overlay
   - Phase distribution chart
   - Average angles and feedback

### Image Analysis

1. Click **"📷 Image Analysis"** tab
2. Upload an image (JPG, PNG)
3. Click **"🔍 Analyze Image"**
4. View detected phase, angles, and feedback

## 📊 Metrics Explained

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| **Trunk Lean** | Forward angle of torso from vertical | Set: 40-55° |
| **Front Knee** | Angle at front knee in set position | 90-110° |
| **Rear Knee** | Angle at rear knee in set position | 120-140° |
| **Hip Angle** | Torso-to-thigh angle | Varies by phase |
| **Elbow Angle** | Arm bend at elbow | ~90° (sprint arms) |

## 🏃 Sprint Phases

| Phase | Description | Key Characteristics |
|-------|-------------|---------------------|
| **Set** | In blocks, ready position | Low hips, strong lean, bent knees |
| **Drive** | First 10-30m push | Aggressive forward push, staying low |
| **Acceleration** | 20-40m transition | Gradually rising toward upright |
| **Max Velocity** | Full speed sprinting | Upright posture, high knee drive |

## ⚙️ Configuration

Target angle ranges can be customized in `config/targets.yaml`:

```yaml
phases:
  set_position:
    targets:
      front_knee_angle:
        min: 90
        max: 110
        feedback_low: "Front knee too closed"
        feedback_high: "Front knee too open"
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_angles.py -v
```

## 📁 Project Structure

```
track_field_analyzer/
├── app.py                  # Streamlit entrypoint
├── requirements.txt        # Dependencies
├── config/
│   └── targets.yaml        # Target angle ranges
├── src/
│   ├── io/
│   │   └── video.py        # Video/image loading
│   ├── pose/
│   │   └── mediapipe_pose.py  # Pose estimation
│   ├── analysis/
│   │   ├── angles.py       # Angle calculations
│   │   ├── metrics.py      # Metrics computation
│   │   └── phases.py       # Phase detection
│   └── viz/
│       └── overlay.py      # Visualization
└── tests/                  # Unit tests
```

## 📋 Tech Stack

- **Streamlit**: Web interface
- **MediaPipe**: Pose estimation (BlazePose)
- **OpenCV**: Video/image processing
- **NumPy**: Numerical computations
- **pandas**: Data handling
- **PyYAML**: Configuration

## ⚠️ Disclaimer

Target angle ranges are based on sprint biomechanics literature and common coaching practices. **Individual optimal ranges vary** based on body proportions, flexibility, and personal mechanics. These are NOT medical recommendations. Always consult a qualified coach for personalized guidance.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- More phase detection accuracy
- Additional metrics (step frequency, ground contact time)
- Export functionality
- Real-time webcam support

## 📄 License

MIT License - feel free to use for personal projects, training, or coaching.

---

Built with ❤️ for sprinters and coaches
