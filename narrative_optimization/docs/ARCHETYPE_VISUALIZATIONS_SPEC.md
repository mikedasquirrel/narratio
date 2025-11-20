# Archetype Visualizations Specification

**Date**: November 13, 2025  
**Purpose**: Interactive visualization designs for archetype analysis

---

## 1. Hero's Journey Tracker

### Visual Design

**Circular Journey Diagram**:
```
        Ordinary World (Start)
               |
    12. Return with Elixir
               |
11. Resurrection      2. Call to Adventure
               |
10. Road Back         3. Refusal
               |
 9. Reward            4. Mentor
               |
 8. Ordeal            5. Threshold
               |
 7. Approach          6. Tests
        (Center = Ordeal)
```

### Interactive Features

- **Stage Highlighting**: Click stage → see details
- **Progress Bar**: Visual completion percentage
- **Stage-by-Stage Breakdown**: Confidence scores per stage
- **Comparison Mode**: Compare multiple narratives side-by-side
- **Temporal View**: Show stages on timeline (0-100% of narrative)

### Implementation

**HTML/CSS/JavaScript**:
```html
<div class="journey-tracker">
  <svg class="journey-circle" width="600" height="600">
    <!-- 17 stages as circular nodes -->
    <circle class="stage" data-stage="0" cx="300" cy="50" r="20" />
    <!-- Lines connecting stages -->
    <!-- Color coding by completion -->
  </svg>
  <div class="stage-details">
    <!-- Show stage info on hover/click -->
  </div>
</div>

<script>
// D3.js or vanilla JS for interactivity
// Fetch data from API: /api/archetypes/analyze
// Render journey completion visually
</script>
```

---

## 2. Archetype Space 3D

### Visual Design

**3D Scatter Plot**:
- X-axis: PC1 (Primary Component from PCA)
- Y-axis: PC2 (Secondary Component)
- Z-axis: PC3 (Tertiary Component)
- Points: Narratives colored by domain/genre
- Clusters: Show natural groupings

### Interactive Features

- **Rotate**: 3D manipulation
- **Zoom**: Focus on clusters
- **Hover**: Show narrative title + summary
- **Filter**: By domain, by π range, by archetype
- **Highlight**: Show similar narratives

### Implementation

**Three.js or Plotly**:
```javascript
// archetype_space_3d.js
const data = await fetch('/api/archetypes/space/3d').then(r => r.json());

const scene = new THREE.Scene();
// Create 3D scatter plot
// Color by domain
// Add labels
// Enable rotation
```

---

## 3. Theory Timeline

### Visual Design

**Horizontal Timeline (Ancient → Modern)**:
```
-335 BCE          1928         1949         1957         2005         2025
    |              |            |            |            |            |
 Aristotle → [gap] → Propp → Campbell → Frye → [gap] → Snyder → Our System
 Poetics          Morphology  Hero's    Mythoi      Save Cat   π/λ/θ/ة
                              Journey                          Integration
```

### Interactive Features

- **Click Theory**: See full details
- **Compare Theories**: Show similarities/differences
- **Evolution View**: How theories built on each other
- **Modern Integration**: How we synthesize all

### Implementation

**Timeline.js or custom SVG**:
```html
<div class="theory-timeline">
  <div class="timeline-track">
    <div class="theory-node" data-year="-335" data-theorist="Aristotle">
      <div class="node-circle"></div>
      <div class="node-label">Aristotle</div>
      <div class="node-details">Poetics...</div>
    </div>
    <!-- Repeat for each theory -->
  </div>
</div>
```

---

## 4. θ/λ Phase Space with Frye's Mythoi

### Visual Design

**2D Scatter Plot with Regions**:
```
    λ (Constraints)
    1.0 ┤            TRAGEDY
        │              ● ●
        │             ● ●
    0.75┤            ●
        │          
        │    COMEDY
    0.50┤      ● ●
        │     ● ●
        │              
    0.25┤                    ROMANCE
        │                      ● ● ●
    0.0 ┼────┬────┬────┬────┬────┬────
        0   0.2  0.4  0.6  0.8  1.0
              θ (Awareness)
        
    IRONY: High θ (0.80-0.95), variable λ
```

### Interactive Features

- **Hover**: Show domain/narrative details
- **Regions**: Highlight Frye's four mythoi zones
- **Filter**: By mythos, by domain, by success
- **Predict**: Click point → predict mythos

### Implementation

**D3.js or Chart.js**:
```javascript
// theta_lambda_space.js
const domains = await fetch('/api/archetypes/domains').then(r => r.json());

// Plot each domain at (θ, λ) coordinates
// Color by dominant mythos
// Show Frye's expected regions
```

---

## 5. Archetype Comparison Matrix

### Visual Design

**Heatmap Matrix**:
```
                     Mythology  Film  Literature  Music  Drama
Journey Completion      0.88    0.72    0.70     0.35   0.75
Archetype Clarity       0.92    0.65    0.65     0.40   0.70
Plot Purity             0.80    0.55    0.60     0.30   0.65
Beat Adherence          0.40    0.75    0.45     0.25   0.70
Mythos Clarity          0.75    0.60    0.65     0.40   0.80

Color: Green (high) → Yellow (medium) → Red (low)
```

### Interactive Features

- **Cell Click**: Show detailed breakdown
- **Sort**: By any metric
- **Filter**: Show only specific domains
- **Export**: CSV/JSON download

### Implementation

**Seaborn/Plotly heatmap**:
```python
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
    z=matrix_values,
    x=domains,
    y=metrics,
    colorscale='RdYlGn'
))

fig.write_html('archetype_comparison_matrix.html')
```

---

## 6. Archetype Distribution by Domain

### Visual Design

**Stacked Bar Chart**:
```
Mythology:  [Hero: 40%][Quest: 30%][Divine: 20%][Other: 10%]
Film:       [Hero: 25%][Beat: 30%][Visual: 25%][Other: 20%]
Literature: [Character: 30%][Theme: 25%][Plot: 25%][Other: 20%]
Music:      [Emotional: 50%][Lyrical: 20%][Other: 30%]
Drama:      [Conflict: 35%][Structure: 30%][Other: 35%]
```

### Interactive Features

- **Hover**: Show exact percentages
- **Click**: Drill down into archetype details
- **Compare**: Select multiple domains

---

## 7. Success Prediction Dashboard

### Visual Design

**Interactive Prediction Tool**:
```
Input: Paste narrative text
  ↓
Analyze with all transformers
  ↓
Show:
- Journey completion: 78%
- Dominant archetype: Hero (0.85)
- Plot type: Quest
- Predicted success: 0.82
- Similar narratives: [list]
```

### Features

- **Text input**: Paste any narrative
- **Real-time analysis**: Run transformers
- **Visual feedback**: Show all archetype scores
- **Recommendations**: Suggestions for improvement
- **Similar works**: Find comparable narratives

---

## Implementation Priority

### Phase 1 (Core Visualizations)

1. **Hero's Journey Tracker** - Most requested, clear value
2. **θ/λ Phase Space** - Tests Frye, visually striking
3. **Comparison Matrix** - Cross-domain insights

### Phase 2 (Advanced)

4. **3D Archetype Space** - Technically complex
5. **Theory Timeline** - Educational value
6. **Distribution Charts** - Analysis views

### Phase 3 (Interactive Tools)

7. **Success Prediction Dashboard** - User-facing tool

---

## Technical Stack

### Recommended Libraries

- **D3.js**: Custom, flexible visualizations
- **Plotly**: Interactive scientific plots
- **Chart.js**: Simple, beautiful charts
- **Three.js**: 3D graphics
- **Bootstrap**: UI framework
- **Flask/Jinja2**: Server-side rendering

### Data Flow

```
1. User requests visualization
   ↓
2. Flask route serves HTML template
   ↓
3. JavaScript fetches data from API
   ↓
4. Library renders visualization
   ↓
5. User interacts (hover, click, filter)
```

---

## Visualization Templates Created

```
templates/
├── archetypes_home.html (overview)
├── archetypes_classical.html (theory list)
├── archetypes_domain.html (domain-specific)
├── theory_integration.html (complete framework)
├── archetypes_compare.html (comparison tool)
└── visualizations/
    ├── journey_tracker.html (Hero's Journey)
    ├── archetype_space_3d.html (3D scatter)
    ├── theta_lambda_space.html (Frye mythoi)
    ├── theory_timeline.html (historical)
    └── comparison_matrix.html (heatmap)
```

---

## API Endpoints for Visualizations

```python
# Already implemented:
GET /api/archetypes/all              # Taxonomy
GET /api/archetypes/theory/<name>    # Theory details
GET /api/archetypes/domains          # Domain info
POST /api/archetypes/analyze         # Analyze text

# To add for visualizations:
GET /api/archetypes/space/3d         # 3D coordinates
GET /api/archetypes/matrix           # Comparison matrix data
GET /api/archetypes/timeline         # Theory chronology
```

---

## Quick Implementation Guide

### Step 1: Create Base Template

```html
<!-- templates/visualizations/base_viz.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Archetype Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div id="visualization"></div>
    <script src="/static/js/archetype_viz.js"></script>
</body>
</html>
```

### Step 2: Implement Journey Tracker

```javascript
// static/js/journey_tracker.js
async function renderJourneyTracker(narrativeId) {
    const analysis = await fetch(`/api/archetypes/analyze`, {
        method: 'POST',
        body: JSON.stringify({text: narrativeText, theories: ['campbell']})
    }).then(r => r.json());
    
    // Render circular diagram
    const stages = analysis.campbell.stages;
    renderCircularJourney(stages);
}
```

---

## Estimated Implementation Time

- **Phase 1** (Core 3): 4-5 hours
- **Phase 2** (Advanced 3): 3-4 hours
- **Phase 3** (Interactive 1): 2-3 hours

**Total**: 9-12 hours for complete visualization suite

---

**Specifications complete. Ready for HTML/JS implementation when needed.**

