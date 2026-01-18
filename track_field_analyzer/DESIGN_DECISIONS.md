# Design Decisions Log

## 2026-01-18: Major UX Redesign

### Decision 1: Step-Based Flow vs Single Page
**Choice:** Hybrid - single page with clear step indicators
**Rationale:** Reduces navigation complexity while maintaining guided experience
**Trade-off:** Less isolation between steps, but faster overall flow

### Decision 2: Score Display
**Choice:** Single "Overall Score" rather than multiple scores
**Rationale:** Reduces cognitive load; users want one answer to "how am I doing?"
**Trade-off:** Less granular feedback at first glance, but details available on expand

### Decision 3: "Focus Areas" over "All Metrics"
**Choice:** Show top 3 actionable cues above fold
**Rationale:** Actionable > informative; users can't fix 10 things at once
**Trade-off:** Hides some data, but progressive disclosure addresses this

### Decision 4: Compact Video Player
**Choice:** Smaller video with overlay, not full-width
**Rationale:** Makes room for metrics side-by-side; matches Ochy layout
**Trade-off:** Less video detail, but frame-by-frame still available

### Decision 5: Sidebar for Settings
**Choice:** Move all settings to sidebar, main area for results
**Rationale:** Cleaner main experience; settings are secondary
**Trade-off:** Sidebar collapsed by default on mobile

### Decision 6: Trust Messaging
**Choice:** Explicit "How we calculate" and "Limitations" sections
**Rationale:** Builds trust; manages expectations; avoids overpromising
**Trade-off:** Takes space; could feel defensive if overdone
