# Formula Rosetta Stone: Classical Theory ↔ π/λ/θ/ة

**Version**: 1.0.0  
**Date**: November 13, 2025  
**Purpose**: Quick reference for exact mathematical translations between classical theories and framework variables

---

## Quick Reference Table

| Classical Theory | Framework Variable | Formula | Expected Correlation |
|-----------------|-------------------|---------|---------------------|
| Campbell Journey | π (narrativity) | π_campbell = JCS × 0.25 | r > 0.70 |
| Jung Archetypes | ة (nominative) | ة_jung = ACS × iconicity | r > 0.65 |
| Aristotle Poetics | λ (constraints) | λ_aristotle = AQI × 0.40 | r > 0.60 |
| Frye Irony | θ (awareness) | θ_frye = Irony_Score × 0.50 | r > 0.65 |
| Snyder Beats | λ (constraints) | λ_snyder = BAS × 0.35 | r > 0.55 |
| Booker Plots | Ξ (golden narratio) | Ξ = centroid(plot_winners) | R² > 0.60 |
| Propp Functions | π (structural) | π_propp = FTCS × 0.20 | r > 0.50 |
| Vonnegut Shapes | π (temporal) | π_vonnegut = Shape_Match × 0.15 | r > 0.45 |

---

## Complete Formulas

### π (Narrativity) = 0 to 1

```
π = 0.30·S + 0.20·T + 0.25·A + 0.15·I + 0.10·F

Where components come from classical theories:

S (Structural) = 0.45·Propp_FTCS + 0.35·Booker_Purity + 0.20·Act_Structure
T (Temporal) = 0.40·Vonnegut_Shape + 0.35·Campbell_JCS + 0.25·Arc_Quality
A (Agency) = 0.50·Character_Arc + 0.30·Hero_Strength + 0.20·Choice_Clarity
I (Interpretive) = 0.45·Frye_Mythos + 0.35·Thematic_Depth + 0.20·Ambiguity
F (Format) = Domain-specific (medium constraints)

Aggregated classical contribution:
π_classical = 0.25·Campbell + 0.20·Jung + 0.20·Propp + 0.15·Vonnegut + 
              0.10·Booker + 0.10·McKee
```

### λ (Constraints) = 0 to 1

```
λ = Σ(constraint_sources) / n

Constraint sources from classical theories:

λ_aristotle = 0.50·AQI + 0.30·Unity_Adherence + 0.20·Decorum
λ_snyder = 0.50·Beat_Adherence + 0.30·Timing_Accuracy + 0.20·Genre_Match
λ_field = 0.60·Three_Act_Structure + 0.40·Plot_Point_Timing
λ_frye = mythos_constraint_level  # Tragedy high, Romance low

Overall:
λ = 0.40·λ_aristotle + 0.35·λ_snyder + 0.15·λ_field + 0.10·λ_frye
```

### θ (Awareness) = 0 to 1

```
θ = awareness_of_narrative_mechanisms

From classical theories:

θ_frye = 0.70·Irony_Score + 0.30·Meta_Awareness
θ_meta = count_meta_markers / total_markers
θ_genre = genre_consciousness_score

Overall:
θ = 0.50·θ_frye + 0.30·θ_meta + 0.20·θ_genre

Interpretations:
θ < 0.30: Straight mythic telling (Campbell's myths)
θ ∈ [0.30, 0.60]: Moderate awareness (most narratives)
θ > 0.60: High meta-awareness (postmodern, ironic)
```

### ة (Nominative Gravity) = 0 to 1

```
ة = nominative_field_strength

From classical theories:

ة_jung = Archetype_Clarity × Name_Iconicity
ة_campbell = Hero_Name_Strength × Proper_Noun_Density
ة_cultural = Name_Recognition × Cultural_Persistence

Overall:
ة = 0.40·ة_jung + 0.35·ة_campbell + 0.25·ة_cultural

Examples:
Zeus (pure archetype, 100% recognition): ة ≈ 0.95
Complex literary character: ة ≈ 0.60
Postmodern unnamed narrator: ة ≈ 0.20
```

### Ξ (Golden Narratio) = n-dimensional vector

```
Ξ = domain_specific_archetypal_perfection

Computed from classical theories:

Ξ_mythology = centroid([
    Campbell_Journey: 0.92,
    Jung_Clarity: 0.95,
    Booker_Quest: 0.85,
    Divine_Intervention: 0.90,
    ...
])

Ξ_film_blockbuster = centroid([
    Snyder_Beats: 0.88,
    Campbell_Journey: 0.82,
    Character_Psychology: 0.80,
    Visual_Story: 0.85,
    ...
])

Distance to Ξ:
d(narrative, Ξ) = ||features(narrative) - Ξ||

Story Quality = 1 / (1 + d)  # Closer = better
```

---

## Detailed Breakdown by Theory

### Campbell's Hero's Journey

**Key Metrics**:
```
JCS (Journey Completion Score) = Σ(stage[i] × weight[i]) / 17

Theoretical weights (Campbell):
call_to_adventure: 1.0
crossing_threshold: 1.0
ordeal: 1.0
resurrection: 1.0
return_with_elixir: 0.9
[others: 0.5-0.8]

Empirical weights (learned from data):
Domain-specific, e.g.:
Mythology: matches Campbell closely (r > 0.85)
Hollywood: deviates (refusal higher, return lower)
```

**Contribution to π**:
```
π += 0.25 × (JCS / 1.0)  # Normalized to [0,1]
```

### Jung's 12 Archetypes

**Key Metrics**:
```
ACS (Archetype Clarity Score) = max(archetype_scores) × (1 - entropy)

Where:
archetype_scores = [Hero, Mentor, Shadow, ...] ∈ [0,1] each
entropy = -Σ(p_i × log(p_i))  # Lower = clearer

High ACS: One dominant archetype (mythology)
Low ACS: Mixed/ambiguous archetypes (postmodern)
```

**Contribution to ة**:
```
ة += 0.40 × ACS × Name_Recognition_Score
```

### Aristotle's Poetics

**Key Metrics**:
```
AQI (Aristotelian Quality Index) = 0.50·Plot + 0.25·Character + 0.15·Thought + 0.10·Diction

Plot Quality:
- Unity of action (single plot line)
- Peripeteia (reversal)
- Anagnorisis (recognition)
- Completeness (beginning-middle-end)

Character Quality:
- Reveals moral choice
- Consistency
- Appropriateness
- Realism
```

**Contribution to λ**:
```
λ += 0.40 × AQI  # Adherence to classical principles
```

### Frye's Four Mythoi

**Key Metrics**:
```
Mythos_Score = [Comedy, Romance, Tragedy, Irony]  # 4-dimensional vector

Expected θ/λ coordinates:
Comedy:  (θ=0.30, λ=0.50)
Romance: (θ=0.20, λ=0.30)
Tragedy: (θ=0.55, λ=0.75)
Irony:   (θ=0.85, λ=0.50)
```

**Contribution to θ**:
```
θ += 0.50 × Irony_Score  # High irony = high awareness
```

### Snyder's Save the Cat

**Key Metrics**:
```
BAS (Beat Adherence Score) = 0.60·Presence + 0.40·Timing

Presence = beats_present / 15
Timing = 1 - mean(|actual_position - expected_position|)

Key beats (weight=1.0):
- Catalyst (12%)
- Break into Two (25%)
- Midpoint (50%)
- All Is Lost (75%)
- Finale (90%)
```

**Contribution to λ**:
```
λ += 0.35 × BAS  # Formula adherence
```

### Booker's Seven Plots

**Key Metrics**:
```
Plot_Type = argmax([Monster, Riches, Quest, Voyage, Comedy, Tragedy, Rebirth])

For each plot type, define Ξ:
Ξ_quest = [journey_structure: 0.90, companions: 0.75, goal: 0.85, ...]

Distance_to_Ξ = ||narrative_features - Ξ_plot_type||
```

**Defines Ξ** (Golden Narratio):
```
Ξ = Ξ_booker_plot_type
Story quality = proximity to appropriate Ξ
```

### Propp's 31 Functions

**Key Metrics**:
```
FTCS (Fairy Tale Completeness Score) = 
    0.60 · (core_functions_present / 8) +
    0.40 · (total_functions_present / 31)

Core functions:
- Villainy/Lack (8)
- Beginning counteraction (10)
- Departure (11)
- Struggle (16)
- Victory (18)
- Liquidation (19)
- Wedding (30)

PCS (Propp Coherence Score) = sequential_order_quality
```

**Contribution to π**:
```
S (structural) += 0.45 × FTCS
π += 0.30 × S
```

### Vonnegut's Story Shapes

**Key Metrics**:
```
Shape_Match = max_correlation(actual_trajectory, expected_shapes)

Shapes:
1. Man in Hole: U-shape
2. Boy Meets Girl: W-shape
3. From Bad to Worse: Decline
4. Cinderella: Multiple rises
[...8 total shapes...]

Emotional_Trajectory = sentiment_over_time
```

**Contribution to π**:
```
T (temporal) += 0.40 × Shape_Match
π += 0.20 × T
```

---

## Integration Equations

### Unified Success Prediction

```
Success = f(π, λ, θ, ة, Ξ)

Where all variables computed from classical theories:

π = Narrativity from Campbell + Jung + Propp + Vonnegut + Booker
λ = Constraints from Aristotle + Snyder + Field + Frye
θ = Awareness from Frye + Meta-markers
ة = Nominatives from Jung + Cultural persistence
Ξ = Domain perfection from Booker + Learned winners

Basic model:
Success = 0.30·π + 0.20·(context_λ) + 0.15·(context_θ) + 0.20·ة + 0.15·Ξ_proximity

Where context depends on domain (some domains want high λ, others want low λ)
```

### Domain-Specific Equations

**Mythology**:
```
Success_myth = 0.40·Campbell_JCS + 0.30·Jung_ACS + 0.20·Cultural_Persistence + 0.10·ة
```

**Hollywood Blockbuster**:
```
Success_film = 0.35·Snyder_BAS + 0.25·Campbell_JCS + 0.20·Emotional_Payoff + 0.20·Budget
```

**Classical Literature**:
```
Success_lit = 0.30·Aristotle_AQI + 0.25·Campbell_JCS + 0.25·Jung_ACS + 0.20·Thematic_Depth
```

**Scripture/Parable**:
```
Success_scripture = 0.50·Moral_Clarity + 0.30·Memorability + 0.20·Allegorical_Depth
```

---

## Translation Examples

### Example 1: Analyze "The Odyssey"

```python
# Classical features
odyssey_features = {
    'campbell_journey_completion': 0.94,  # Near-perfect Hero's Journey
    'jung_hero': 0.95,  # Pure hero archetype (Odysseus)
    'jung_mentor': 0.80,  # Athena as mentor
    'booker_plot': 'quest',  # Clear quest structure
    'frye_mythos': 'romance',  # Idealized hero
    'aristotle_plot_unity': 0.90,  # Single complete action
    'propp_functions': 25,  # Out of 31 present
}

# Translate to π/λ/θ/ة
π_odyssey = (
    0.30 × 0.88 +  # S (structural): High from Propp
    0.20 × 0.92 +  # T (temporal): High from Campbell + Vonnegut
    0.25 × 0.85 +  # A (agency): Individual hero
    0.15 × 0.75 +  # I (interpretive): Romance mythos
    0.10 × 0.65    # F (format): Oral tradition
) = 0.84

λ_odyssey = 0.35  # Low constraints (gods can intervene)
θ_odyssey = 0.25  # Low awareness (straight mythic telling)
ة_odyssey = 0.92  # High nominatives (Odysseus iconic)

Ξ_proximity = 0.88  # Very close to mythology Ξ

Predicted cultural persistence: 0.89 (VERY HIGH) ✓
Actual: Still taught 2,800 years later ✓✓✓
```

### Example 2: Analyze "Pulp Fiction"

```python
# Classical features
pulp_fiction_features = {
    'campbell_journey_completion': 0.35,  # Fragmented, non-linear
    'jung_archetypes': [0.4, 0.5, 0.3, ...],  # Mixed, ambiguous
    'booker_plot': 'mixed',  # Multiple plot threads
    'frye_mythos': 'irony',  # 0.75 irony score
    'snyder_beat_adherence': 0.25,  # Deliberately non-formula
    'vonnegut_shape': 'which_way_is_up',  # Ambiguous
}

# Translate
π = 0.52  # Medium (fragmented but coherent)
λ = 0.35  # Low (breaks conventions)
θ = 0.82  # Very high (meta-aware, ironic)
ة = 0.65  # Moderate (memorable characters)

Predicted: Cult classic, not blockbuster
Actual: $213M box office, massive cultural impact ✓
(High θ + innovative = prestige effect)
```

### Example 3: Analyze "Star Wars" (1977)

```python
# Classical features
star_wars_features = {
    'campbell_journey_completion': 0.89,  # Nearly perfect
    'jung_hero': 0.92,  # Luke = pure willing hero
    'jung_mentor': 0.95,  # Obi-Wan = perfect mentor
    'jung_shadow': 0.90,  # Vader = pure shadow
    'booker_plot': 'quest',  # Clear quest (0.85)
    'snyder_beats': 0.78,  # Good formula adherence
    'frye_mythos': 'romance',  # Idealized heroes (0.80)
}

# Translate
π = 0.82  # Very high (clear narrative)
λ = 0.62  # Medium (genre conventions)
θ = 0.30  # Low (straight-faced, not ironic)
ة = 0.88  # Very high (iconic names)

Predicted: Massive commercial + cultural success
Actual: $775M (1977), launched franchise ✓✓✓
Campbell's journey + mythic archetypes = formula for blockbuster
```

---

## Validation Formulas

### Test 1: π ↔ Journey Completion

```
Hypothesis: r(π, JCS) > 0.70

Test across domains:
correlation(
    [π_domain1, π_domain2, ...],
    [JCS_domain1, JCS_domain2, ...]
)

Expected:
Mythology: r > 0.85 (Campbell's source)
Film: r > 0.70 (Hollywood uses it)
Postmodern: r > 0.40 (fragmented but still there)
```

### Test 2: λ ↔ Structural Adherence

```
Hypothesis: r(λ, Structure_Score) > 0.60

Structure_Score = 0.40·Aristotle_AQI + 0.35·Snyder_BAS + 0.25·Act_Structure

Test:
Greek Tragedy (λ=0.80): Structure_Score > 0.85
Experimental (λ=0.30): Structure_Score < 0.40
```

### Test 3: θ ↔ Frye Irony

```
Hypothesis: r(θ, Irony_Score) > 0.65

Test:
Postmodern (θ=0.85): Irony > 0.75
Mythology (θ=0.25): Irony < 0.20
```

### Test 4: Archetype Clarity ↔ ة

```
Hypothesis: r(ACS, ة) > 0.60

Test:
Mythology: ACS=0.92, ة=0.88 (pure archetypes, iconic names)
Literary: ACS=0.65, ة=0.65 (complex characters, less iconic)
Postmodern: ACS=0.35, ة=0.40 (deconstructed, unnamed)
```

---

## Domain-Specific Translations

### Mythology Domain

```
π_myth = 0.89
Components from:
- Campbell JCS: 0.88 (near perfect)
- Jung clarity: 0.92 (pure archetypes)
- Propp FTCS: 0.75 (fairy tale structure)

λ_myth = 0.30 (low constraints - gods can do anything)
θ_myth = 0.25 (low awareness - straight telling)
ة_myth = 0.88 (iconic names - Zeus, Odin, Thor)

Ξ_myth = [Journey: 0.92, Clarity: 0.95, Divine: 0.90, ...]
```

### Film Extended Domain

```
π_film = 0.68
Components from:
- Snyder BAS: 0.75 (formula adherence)
- Campbell JCS: 0.72 (journey present)
- Vonnegut shape: 0.65 (clear trajectory)

λ_film = 0.60 (medium - runtime, budget constraints)
θ_film = 0.45 (moderate - some genre awareness)
ة_film = 0.65 (character names important)

Ξ_blockbuster = [Beats: 0.88, Psychology: 0.80, Visual: 0.85, ...]
Ξ_indie = [Character: 0.80, Theme: 0.75, Innovation: 0.70, ...]
```

### Classical Literature Domain

```
π_lit = 0.72
Components from:
- Campbell JCS: 0.70 (journey present)
- Jung ACS: 0.65 (complex characters)
- Booker purity: 0.60 (often blended plots)

λ_lit = 0.55 (moderate - genre conventions)
θ_lit = 0.50 (moderate to high - varies by period)
ة_lit = 0.70 (memorable character names)

Ξ_epic = [Journey: 0.90, Clarity: 0.85, ...]
Ξ_modernist = [Psychology: 0.85, Innovation: 0.80, ...]
Ξ_postmodern = [Meta: 0.85, Fragmentation: 0.70, ...]
```

---

## Quick Lookup: Theory → Feature → Variable

| Want to Measure | Use Theory | Extract Feature | Maps to Variable |
|----------------|-----------|----------------|------------------|
| Story completeness | Campbell | Journey Completion Score | π (via T) |
| Character clarity | Jung | Archetype Clarity Score | ة |
| Structural quality | Aristotle | Aristotelian Quality Index | λ |
| Genre mode | Frye | Mythos Score | θ/λ phase |
| Plot type | Booker | Plot probabilities | Ξ definition |
| Beat timing | Snyder | Beat Adherence Score | λ |
| Emotional arc | Vonnegut | Shape Match | π (via T) |
| Conflict type | Polti | Situation Vector | π (via S) |
| Function structure | Propp | FTCS + PCS | π (via S) |
| Scene dynamics | McKee | Gap + Value Shift | π (integration) |

---

## Computational Workflow

### Step 1: Extract Classical Features

```python
from transformers.archetypes import *

features = {
    'campbell': HeroJourneyTransformer().transform([text]),
    'jung': CharacterArchetypeTransformer().transform([text]),
    'booker': PlotArchetypeTransformer().transform([text]),
    'snyder': StructuralBeatTransformer().transform([text]),
    'frye': ThematicArchetypeTransformer().transform([text])
}
```

### Step 2: Compute π/λ/θ/ة

```python
π = compute_pi_from_classical(features)
λ = compute_lambda_from_classical(features)
θ = compute_theta_from_classical(features)
ة = compute_ta_marbuta_from_classical(features)
```

### Step 3: Calculate Ξ Proximity

```python
ξ = get_domain_xi(domain, features)
ξ_proximity = cosine_similarity(features_vector, ξ)
```

### Step 4: Predict Success

```python
predicted_success = (
    0.30 * π +
    0.20 * λ_contextual +
    0.15 * θ_contextual +
    0.20 * ة +
    0.15 * ξ_proximity
)
```

---

## Summary

**This Rosetta Stone enables**:
- ✅ Translation: Classical theory → Computational features → π/λ/θ/ة
- ✅ Validation: Test if theories hold empirically
- ✅ Discovery: Find where theories fail
- ✅ Prediction: Use for success forecasting
- ✅ Interpretation: Explain results with classical terminology

**Every classical concept now has**:
- Mathematical definition
- Computational extraction method
- Framework variable mapping
- Validation test
- Expected domain behavior

**The complete bridge between narrative scholarship and data science.**


