# UX Specification: Sprint Form Analyzer

> **Version:** 2.0  
> **Last Updated:** 2026-01-18  
> **Design Philosophy:** Simple, actionable, trustworthy

---

## 1. Target Users & Jobs-to-Be-Done

### Primary Users
1. **Sprinters (60m-200m)** - Self-coached or wanting feedback between sessions
2. **Coaches** - Quick form checks for athletes  
3. **Track enthusiasts** - Learning proper technique

### Top 3 Jobs-to-Be-Done
1. **"Help me see what I can't feel"** → Visual feedback on form
2. **"Tell me 2-3 things to fix"** → Actionable coaching cues
3. **"Am I improving?"** → Track progress over time (future)

---

## 2. Primary Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SETUP     │ → │   UPLOAD    │ → │  ANALYZE    │ → │   RESULTS   │
│  (10 sec)   │    │  (10 sec)   │    │  (30 sec)   │    │  (explore)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
    Event              Video             Processing         Overview
    Camera             Checklist         Progress           Cues
    Level                                                   Details
```

### Step Details
1. **Setup** - Quick questionnaire (event, camera position, experience)
2. **Upload** - Video checklist (stable, full body, ~10s, side view)
3. **Analyze** - Progress bar with phase indicators
4. **Results** - Hero score → 3 cues → expandable details

---

## 3. Information Architecture

### Above the Fold (Results View)
```
┌─────────────────────────────────────────────────────────┐
│  [Video Player - Compact]          [Score Card]         │
│  ┌─────────────────────┐          ┌───────────────┐    │
│  │                     │          │  Overall: 72% │    │
│  │    Skeleton View    │          │    "Good"     │    │
│  │      (compact)      │          └───────────────┘    │
│  │                     │          ┌───────────────┐    │
│  └─────────────────────┘          │ Current Phase │    │
│  [◀] ═══════●═══════ [▶]          │  Drive Phase  │    │
│                                    └───────────────┘    │
├─────────────────────────────────────────────────────────┤
│  🎯 Top 3 Focus Areas                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│  │ Trunk    │ │ Knee     │ │ Arm      │                │
│  │ Lean: ⚠️  │ │ Drive: ✅ │ │ Action: ℹ️│                │
│  │ "Rise..."│ │ "Good.." │ │ "Keep.." │                │
│  └──────────┘ └──────────┘ └──────────┘                │
├─────────────────────────────────────────────────────────┤
│  [📊 All Metrics]  [📈 Timeline]  [ℹ️ How It Works]     │
│  (expandable sections below)                            │
└─────────────────────────────────────────────────────────┘
```

### Sidebar (Settings)
```
┌─────────────────────┐
│ ⚙️ Settings          │
├─────────────────────┤
│ Analysis Mode       │
│ ○ Fast (10 frames)  │
│ ● Balanced (30)     │
│ ○ Accurate (100)    │
├─────────────────────┤
│ Display             │
│ ☑️ Show angles      │
│ ☑️ Show skeleton    │
│ ☐ Show all joints   │
├─────────────────────┤
│ Your Profile        │
│ Event: 100m         │
│ Level: Intermediate │
└─────────────────────┘
```

---

## 4. Component Inventory

| Component | Purpose | Streamlit |
|-----------|---------|-----------|
| Hero Score | Overall form rating | Custom HTML |
| Focus Card | Single actionable cue | st.container |
| Video Player | Compact frame view | st.image |
| Timeline | Frame navigation | st.slider |
| Phase Badge | Current phase label | HTML badge |
| Metric Row | Angle with target | st.columns |
| Tooltip | "How we calculate" | st.help |
| Expander | Progressive disclosure | st.expander |
| Tabs | Section navigation | st.tabs |
| Alert | Warnings/limitations | st.info/warning |

---

## 5. State Model

```python
st.session_state = {
    # User profile
    "user_event": "100m",
    "user_level": "intermediate",
    "camera_position": "side",
    
    # Flow state
    "current_step": 1,  # 1=Setup, 2=Upload, 3=Analyze, 4=Results
    
    # Analysis state
    "video_uploaded": False,
    "processing_complete": False,
    "processed_frames": [],
    "frame_metrics": [],
    "aggregated_metrics": {},
    
    # View state
    "current_frame_idx": 0,
    "show_advanced": False,
}
```

### Caching Strategy
- `@st.cache_resource`: PoseEstimator model
- `@st.cache_data`: Target config, computed metrics

---

## 6. Accessibility Checklist

- [ ] Contrast ratio ≥ 4.5:1 for text
- [ ] All images have alt text
- [ ] Keyboard navigation works
- [ ] Error messages are descriptive
- [ ] Progress communicated (not just visual)
- [ ] No reliance on color alone

---

## 7. Success Criteria

| Metric | Target |
|--------|--------|
| Time to first result | < 45 seconds |
| Steps to upload | ≤ 3 clicks |
| "Above fold" info | Score + 3 cues |
| User confusion | Zero jargon visible |

---

## 8. Trust & Transparency

### Required Elements
1. **"How we calculate" tooltip** on every metric
2. **Limitations callout** explaining accuracy bounds
3. **"Backed by biomechanics"** but no medical claims
4. **Confidence indicators** when pose quality is low

### Copy Guidelines
- Use "suggests" not "shows"
- Use "typical range" not "correct"
- Use "coaching cue" not "diagnosis"
