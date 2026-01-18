# Design Decisions Log

## 2026-01-18: Premium Sports Dashboard Redesign

### Decision 1: 2:1 Column Ratio for Video/Metrics
**Choice:** Video takes ~65% width, metrics take ~35%
**Rationale:** 
- Video is the primary content users want to see
- Metrics need enough space for context bars and cues
- Matches Ochy's layout proportions
**Trade-off:** On mobile, columns stack vertically

### Decision 2: Sports Color Palette
**Palette:**
- Background: Deep navy (#0a0a14)
- Primary accent: Neon cyan (#00d4ff) - highlights, good states
- Success: Neon green (#22c55e) - optimal ranges
- Warning: Amber (#f59e0b) - needs attention
- Error: Bright red (#ef4444) - poor form
- Text: Off-white (#e8e8e8) / Muted gray (#a0a0a0)

**Rationale:**
- Dark themes reduce eye strain during video analysis
- High contrast meets WCAG AA (4.5:1 minimum)
- Semantic colors match sports analytics conventions
- Neon accents feel premium and modern

### Decision 3: Tabbed Progressive Disclosure
**Tabs:** Overview → Video & Overlay → All Metrics → How It Works
**Rationale:**
- Reduces cognitive overload
- Users see actionable info first (hero score + 3 cues)
- Detailed metrics available for those who want depth
- "How It Works" builds trust without cluttering main view

**Trade-off:** More clicks to see all data, but most users only need Overview

### Decision 4: Context Bars for Metrics
**Choice:** Horizontal gradient bar showing value position vs target range
**Rationale:**
- Instantly shows "am I in the zone?"
- Green center, amber edges, red extremes = intuitive
- More informative than just a number

**Trade-off:** Requires custom CSS (not native Streamlit)

### Decision 5: Cleaner Skeleton Overlay
**Choice:** Remove angle labels from video by default (optional in settings)
**Rationale:**
- Declutters the video view
- Angles are shown cleanly in metrics panel
- Phase badge remains for quick reference

### Decision 6: Hero Score (0-10 Scale)
**Choice:** Single composite score rather than multiple scores
**Rationale:**
- "How did I do?" needs one answer
- Based on phase-appropriate targets
- Summary text explains the score

**Calculation:**
- Base score: 5.0
- Add points for good trunk lean (phase-specific)
- Add points for good knee drive
- Subtract for unknown phase
- Clamp to 0-10

### Decision 7: Sidebar for Settings
**Contents:**
- Analysis mode (Fast/Balanced/Accurate)
- Video width slider
- Display toggles
- User profile (event, level)
- Help section (filming tips)
- Limitations callout

**Rationale:**
- Main area stays focused on results
- Settings are secondary to analysis
- Help/limitations build trust

### Custom CSS Used
Required for:
- Card shadows and borders (native backgrounds too flat)
- Context bar gradients (not possible with st.progress)
- Typography hierarchy (badge pills, small caps)
- Consistent spacing in metric cards

All CSS is inline and documented in app.py.
