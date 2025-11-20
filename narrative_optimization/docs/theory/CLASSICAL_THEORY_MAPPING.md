# Classical Narrative Theory → π/λ/θ/ة Mathematical Mapping

**Version**: 1.0.0  
**Date**: November 13, 2025  
**Purpose**: Formal mathematical relationships between classical narrative theories and the narrative optimization framework

---

## Executive Summary

This document establishes precise mathematical mappings between classical narrative theories (Campbell, Jung, Aristotle, etc.) and the π/λ/θ/ة framework. Each classical concept is translated into computable features that can be extracted from text and measured empirically.

---

## Table of Contents

1. [Core Framework Variables](#core-framework-variables)
2. [Campbell's Hero's Journey → π](#campbells-heros-journey--π)
3. [Jung's Archetypes → Character Features](#jungs-archetypes--character-features)
4. [Aristotle's Elements → Structural Features](#aristotles-elements--structural-features)
5. [Frye's Mythoi → θ/λ Interaction](#fryes-mythoi--θλ-interaction)
6. [Propp's Functions → Sequential Features](#propps-functions--sequential-features)
7. [Vonnegut's Shapes → Emotional Trajectory](#vonneguts-shapes--emotional-trajectory)
8. [Blake Snyder's Beats → Temporal Features](#blake-snyders-beats--temporal-features)
9. [Polti's Situations → Conflict Features](#poltis-situations--conflict-features)
10. [Booker's Plots → Ξ Archetypes](#bookers-plots--ξ-archetypes)
11. [Complete Integration Formula](#complete-integration-formula)

---

## Core Framework Variables

### Primary Variables

**π (Narrativity)**: *0 ≤ π ≤ 1*

```
π = 0.30·S + 0.20·T + 0.25·A + 0.15·I + 0.10·F

Where:
S = Structural component (0-1)
T = Temporal component (0-1)
A = Agency component (0-1)
I = Interpretive component (0-1)
F = Format component (0-1)
```

**θ (Awareness)**: *0 ≤ θ ≤ 1*
- Awareness of narrative mechanisms
- Meta-narrative consciousness
- Genre/pattern recognition

**λ (Constraints)**: *0 ≤ λ ≤ 1*
- External limitations on narrative freedom
- Genre conventions, medium restrictions
- Physical/logical constraints

**ة (Nominative Gravity)**: *0 ≤ ة ≤ 1*
- Proper noun density and impact
- Name iconicity and memorability
- Nominative field strength

**Ξ (Golden Narratio)**: *n-dimensional vector*
- Domain-specific archetypal perfection
- Centroid of winner narratives in feature space
- Target toward which quality narratives gravitate

---

## Campbell's Hero's Journey → π

### Stage Detection Mapping

Each of Campbell's 17 stages maps to feature extractors:

#### Stage 1: Ordinary World

```python
def detect_ordinary_world(text):
    """
    Features:
    - Presence of stability markers ("normal", "routine", "always")
    - Character in familiar setting
    - Absence of conflict language
    - Temporal markers indicating "before"
    
    Returns: confidence score (0-1)
    """
    features = {
        'stability_language': count_patterns(['normal', 'usual', 'everyday', 'routine']),
        'comfort_markers': count_patterns(['home', 'family', 'comfortable', 'safe']),
        'pre_conflict': absence_of_conflict_language(),
        'setting_description': presence_of_place_establishment()
    }
    return sigmoid(weighted_sum(features))
```

#### Stage 2: Call to Adventure

```python
def detect_call_to_adventure(text):
    """
    Features:
    - Disruption language ("suddenly", "then", "one day")
    - Introduction of problem/opportunity
    - Messenger/herald presence
    - Change imperative
    
    Returns: confidence score (0-1)
    """
    features = {
        'disruption_markers': count_patterns(['suddenly', 'then', 'but', 'when']),
        'problem_introduction': detect_conflict_onset(),
        'herald_presence': detect_messenger_archetype(),
        'opportunity_language': count_patterns(['could', 'might', 'what if', 'imagine'])
    }
    return sigmoid(weighted_sum(features))
```

#### Stages 3-17: Similar Pattern

Each stage has:
- **Linguistic markers**: Key phrases/patterns
- **Structural position**: Expected location in narrative
- **Archetype presence**: Required character roles
- **Emotional valence**: Expected sentiment shift

### Journey Completion Score

```
JCS = Σ(stage_detected[i] × stage_weight[i]) / total_possible

Where:
stage_detected[i] = 1 if stage i present, 0 otherwise
stage_weight[i] = importance of stage i (sum to 1)

JCS ∈ [0, 1]
```

### Contribution to π

```
π_hero_journey = α · JCS + β · JSC + γ · TDM

Where:
JCS = Journey Completion Score
JSC = Journey Sequential Coherence (stages in order?)
TDM = Transformation Depth Magnitude (character change)

α + β + γ = 1 (typically α=0.50, β=0.30, γ=0.20)
```

### Empirical Relationships

**Hypothesis 1**: High π domains have high JCS
```
Correlation: r(π, JCS) > 0.70 expected
Domains: Mythology (π≈0.92), Epic Literature (π≈0.88), Hero Films (π≈0.80)
```

**Hypothesis 2**: Low π domains have low JCS
```
Correlation: r(π, JCS) still positive but weaker
Domains: Experimental Literature (π≈0.35), Documentary (π≈0.30)
```

**Hypothesis 3**: JCS predicts narrative success within high-π domains
```
Within mythology domain:
R²(cultural_persistence ~ JCS) > 0.60
```

---

## Jung's Archetypes → Character Features

### The 12 Primary Archetypes as Feature Vectors

Each archetype detected via pattern matching:

#### 1. The Hero

```python
HERO_ARCHETYPE = {
    'patterns': [
        'courage', 'brave', 'fight', 'battle', 'overcome',
        'champion', 'warrior', 'duty', 'sacrifice', 'prove'
    ],
    'behavioral_markers': {
        'agency_high': True,  # Takes action
        'risk_taking': True,  # Faces danger
        'goal_oriented': True,  # Clear objective
        'transformative': True  # Changes through ordeal
    },
    'emotional_profile': {
        'determination': 0.8,
        'fear_acknowledged': 0.6,
        'confidence': 0.7
    },
    'narrative_role': 'protagonist',
    'typical_arc': 'reluctant_to_triumphant'
}

def detect_hero_archetype(character_text):
    """
    Returns: (confidence, sub_type)
    confidence ∈ [0, 1]
    sub_type ∈ ['willing_hero', 'unwilling_hero', 'anti_hero', 'tragic_hero']
    """
    pattern_score = match_patterns(character_text, HERO_ARCHETYPE['patterns'])
    behavior_score = match_behaviors(character_text, HERO_ARCHETYPE['behavioral_markers'])
    emotional_score = match_emotional_profile(character_text, HERO_ARCHETYPE['emotional_profile'])
    
    confidence = (pattern_score + behavior_score + emotional_score) / 3
    
    # Determine sub-type
    if match_patterns(character_text, ['reluctant', 'forced', 'must']):
        sub_type = 'unwilling_hero'
    elif match_patterns(character_text, ['flawed', 'morally', 'grey']):
        sub_type = 'anti_hero'
    elif match_patterns(character_text, ['doomed', 'tragic', 'flaw', 'hubris']):
        sub_type = 'tragic_hero'
    else:
        sub_type = 'willing_hero'
    
    return (confidence, sub_type)
```

#### 2-12. Other Archetypes

Similar structures for:
- **Mentor** (wisdom, guidance, teaching patterns)
- **Shadow** (opposition, darkness, repressed traits)
- **Trickster** (chaos, humor, boundary-crossing)
- **Shapeshifter** (ambiguity, transformation, unreliability)
- **Everyman** (relatability, common sense, grounding)
- **Innocent** (purity, faith, optimism, trust)
- **Explorer** (freedom, discovery, autonomy)
- **Rebel** (disruption, revolution, breaking rules)
- **Lover** (passion, intimacy, connection)
- **Creator** (innovation, imagination, self-expression)
- **Ruler** (control, responsibility, order, leadership)

### Archetype Clarity Score

```
ACS = Σ(archetype_confidence[i]² × archetype_weight[i]) / n_characters

Where:
- Square amplifies high-confidence matches
- Penalizes ambiguous characters
- Weight by narrative importance

ACS ∈ [0, 1]
Higher ACS = clearer archetypal definition
```

### Shadow Projection Index

```
SPI = similarity(hero_repressed_traits, shadow_manifest_traits) × shadow_prominence

Where:
- Detect repressed traits in hero (what they avoid/deny)
- Detect manifest traits in shadow (what shadow embodies)
- Measure similarity (cosine similarity in trait space)
- Weight by shadow's narrative prominence

SPI ∈ [0, 1]
High SPI = strong psychological projection
```

### Contribution to π and Ξ

```
π_jungian = δ · ACS + ε · API + ζ · CAC

Where:
ACS = Archetype Clarity Score
API = Archetype Pairing Index (hero-shadow, etc.)
CAC = Character Arc Completeness

δ + ε + ζ = 1 (typically δ=0.40, ε=0.35, ζ=0.25)
```

**Relationship to Ξ**:
- Mythology Ξ: Very high ACS (≥0.90) - pure archetypes
- Literary Fiction Ξ: Medium ACS (0.50-0.70) - complex characters
- Postmodern Ξ: Low ACS (≤0.40) - deconstructed archetypes

---

## Aristotle's Elements → Structural Features

### The Six Elements as Measurable Features

#### 1. Plot (Mythos) - Most Important

```python
def measure_plot_quality(text):
    """
    Aristotle's plot requirements:
    - Unity of action (single complete action)
    - Probability and necessity (logical causation)
    - Beginning, middle, end (complete structure)
    - Peripeteia (reversal)
    - Anagnorisis (recognition)
    """
    features = {
        'unity_of_action': {
            'single_protagonist': detect_protagonist_focus(),
            'single_goal': detect_goal_unity(),
            'causal_chain': measure_causal_coherence()
        },
        'completeness': {
            'has_beginning': detect_exposition(),
            'has_middle': detect_rising_action(),
            'has_end': detect_resolution()
        },
        'complex_plot_elements': {
            'peripeteia': detect_reversal(),
            'anagnorisis': detect_recognition()
        }
    }
    
    # Plot Quality Score
    PQS = (
        0.35 * features['unity_of_action']['causal_chain'] +
        0.25 * features['completeness']['has_middle'] +
        0.20 * features['complex_plot_elements']['peripeteia'] +
        0.20 * features['complex_plot_elements']['anagnorisis']
    )
    
    return PQS  # ∈ [0, 1]
```

#### 2. Character (Ethos)

```python
def measure_character_quality(text):
    """
    Aristotle's character requirements:
    - Good (virtue appropriate to type)
    - Appropriate (fits social role)
    - Realistic (lifelike)
    - Consistent (no contradictions unless explained)
    """
    features = {
        'reveals_choice': detect_moral_decisions(),
        'consistency': measure_character_consistency(),
        'appropriate': match_role_to_behavior(),
        'realistic': measure_psychological_plausibility()
    }
    
    CQS = mean(features.values())  # Character Quality Score
    return CQS  # ∈ [0, 1]
```

#### 3. Thought (Dianoia)

```python
def measure_thought_quality(text):
    """
    Thematic depth and intellectual content
    """
    features = {
        'universal_themes': detect_universal_truths(),
        'argument_quality': measure_logical_coherence(),
        'philosophical_depth': measure_abstraction_level(),
        'moral_framework': detect_ethical_dimensions()
    }
    
    TQS = weighted_mean(features)  # Thought Quality Score
    return TQS  # ∈ [0, 1]
```

#### 4. Diction (Lexis)

```python
def measure_diction_quality(text):
    """
    Language quality and appropriateness
    """
    features = {
        'metaphor_quality': analyze_metaphors(),
        'clarity': measure_readability(),
        'elevation': measure_linguistic_sophistication(),
        'appropriateness': match_style_to_content()
    }
    
    DQS = weighted_mean(features)  # Diction Quality Score
    return DQS  # ∈ [0, 1]
```

#### 5. Song (Melos) & 6. Spectacle (Opsis)

```
# Less relevant for text analysis
# Song: Musical/rhythmic qualities
# Spectacle: Visual elements
```

### Aristotelian Quality Index

```
AQI = 0.50·PQS + 0.25·CQS + 0.15·TQS + 0.10·DQS

Where PQS, CQS, TQS, DQS ∈ [0, 1]

AQI ∈ [0, 1]
Measures adherence to Aristotelian dramatic principles
```

### Tragedy-Specific Features

```python
def measure_tragic_structure(text):
    """
    Specific to tragedy:
    - Hamartia (tragic flaw/error)
    - Hubris (excessive pride)
    - Peripeteia (reversal from good to bad)
    - Anagnorisis (recognition of truth)
    - Catharsis (pity and fear)
    """
    features = {
        'hamartia_present': detect_fatal_flaw(),
        'hubris_markers': count_patterns(['pride', 'arrogance', 'defied', 'gods']),
        'reversal_high_to_low': detect_fortune_reversal(direction='down'),
        'recognition': detect_truth_revelation(),
        'emotional_intensity': measure_pity_and_fear()
    }
    
    TSS = weighted_sum(features)  # Tragic Structure Score
    return TSS  # ∈ [0, 1]
```

### Contribution to λ (Constraints)

Aristotle's principles represent structural constraints:

```
λ_aristotelian = η · AQI + ι · Unity_Adherence + κ · Decorum

Where:
AQI = Aristotelian Quality Index
Unity_Adherence = Following three unities
Decorum = Genre appropriateness

η + ι + κ = 1 (typically η=0.50, ι=0.30, κ=0.20)
```

**Interpretation**:
- High λ domains (Greek tragedy, classical drama): High AQI expected
- Low λ domains (experimental theater): Low AQI, intentional violations

---

## Frye's Mythoi → θ/λ Interaction

### The Four Mythoi as Domain Modes

Frye's four narrative modes map to specific θ/λ combinations:

#### 1. Comedy (Spring) - Integration, Renewal

```
θ_comedy = LOW (0.20-0.40)  # Awareness doesn't interfere
λ_comedy = MEDIUM (0.40-0.60)  # Social conventions matter
π_comedy = MEDIUM-HIGH (0.60-0.75)  # Clear narrative structure

Comedy_Score = presence_of([
    'social_blockage_resolved',
    'confusions_clarified',
    'lovers_united',
    'feast_celebration',
    'new_order_established',
    'outsider_integrated'
])

Emotional_Trajectory: chaotic → harmonious
Ending: Happy, optimistic
```

#### 2. Romance (Summer) - Idealization, Adventure

```
θ_romance = VERY LOW (0.10-0.25)  # Straight-faced idealism
λ_romance = LOW (0.20-0.40)  # Few realistic constraints
π_romance = VERY HIGH (0.80-0.95)  # Clear quest structure

Romance_Score = presence_of([
    'idealized_hero',
    'clear_good_vs_evil',
    'quest_structure',
    'exotic_settings',
    'magical_elements',
    'triumph_of_good'
])

Emotional_Trajectory: adventure → triumph
Ending: Victory, restoration
```

#### 3. Tragedy (Autumn) - Fall, Sacrifice

```
θ_tragedy = MEDIUM (0.45-0.65)  # Some self-awareness (too late)
λ_tragedy = HIGH (0.65-0.85)  # Fate, inevitability
π_tragedy = HIGH (0.70-0.85)  # Structured fall

Tragedy_Score = presence_of([
    'noble_protagonist',
    'fatal_flaw',
    'hubris',
    'reversal_high_to_low',
    'recognition_too_late',
    'death_or_ruin',
    'isolation'
])

Emotional_Trajectory: peak → catastrophe
Ending: Death, loss, learning
```

#### 4. Irony/Satire (Winter) - Realism, Critique

```
θ_irony = VERY HIGH (0.75-0.95)  # Maximum awareness/meta
λ_irony = VARIABLE (0.30-0.70)  # Depends on sub-type
π_irony = LOW-MEDIUM (0.35-0.60)  # Fragmented/ambiguous

Irony_Score = presence_of([
    'unheroic_protagonist',
    'futility_circularity',
    'system_dominance',
    'ambiguous_morality',
    'disillusionment',
    'no_clear_resolution',
    'realism_naturalism'
])

Emotional_Trajectory: ambiguous/cyclical
Ending: Ambiguous, dark, unresolved
```

### Mythoi Detection Algorithm

```python
def detect_mythos(text):
    """
    Classify narrative into one of Frye's four mythoi
    
    Returns: (primary_mythos, confidence, secondary_mythos)
    """
    scores = {
        'comedy': calculate_comedy_score(text),
        'romance': calculate_romance_score(text),
        'tragedy': calculate_tragedy_score(text),
        'irony': calculate_irony_score(text)
    }
    
    primary = max(scores, key=scores.get)
    confidence = scores[primary]
    
    # Secondary mythos (many works blend)
    remaining = {k: v for k, v in scores.items() if k != primary}
    secondary = max(remaining, key=remaining.get) if remaining else None
    
    return (primary, confidence, secondary)
```

### θ/λ Phase Space

```
Frye's mythoi occupy distinct regions in θ/λ space:

      λ (Constraints)
      1.0 ┤                    
          │         
          │     Tragedy (Autumn)
    0.75  ┤         ●
          │        
          │   
    0.50  ┤  Comedy (Spring)    
          │        ●
          │            
    0.25  ┤                Romance (Summer)
          │                    ●
      0.0 ┼────┬────┬────┬────┬────
          0   0.25  0.50  0.75  1.0
                θ (Awareness)
                
      Irony/Satire (Winter): θ=0.80-0.95, λ=variable
```

### Contribution to Domain Classification

```
Mythos_Features = [Comedy_Score, Romance_Score, Tragedy_Score, Irony_Score]

Expected values by domain:
- Epic Literature: Romance dominant (0.85)
- Greek Tragedy: Tragedy dominant (0.95)
- Shakespeare Comedy: Comedy dominant (0.80)
- Postmodern Fiction: Irony dominant (0.75)
```

---

## Propp's Functions → Sequential Features

### 31 Functions as Binary Feature Vector

```python
PROPP_FUNCTIONS = [
    'absentation', 'interdiction', 'violation', 'reconnaissance',
    'delivery', 'trickery', 'complicity', 'villainy', 'lack',
    'mediation', 'beginning_counteraction', 'departure',
    'first_function_donor', 'hero_reaction', 'provision_magical_agent',
    'spatial_transference', 'struggle', 'branding', 'victory',
    'liquidation', 'return', 'pursuit', 'rescue',
    'unrecognized_arrival', 'unfounded_claims', 'difficult_task',
    'solution', 'recognition', 'exposure', 'transfiguration',
    'punishment', 'wedding'
]

def extract_propp_vector(text):
    """
    Returns: 31-dimensional binary vector
    propp_vector[i] = 1 if function i present, 0 otherwise
    """
    vector = np.zeros(31)
    for i, function in enumerate(PROPP_FUNCTIONS):
        vector[i] = detect_function(text, function)
    return vector
```

### Sequential Coherence Score

```python
def calculate_propp_coherence(propp_vector, text_positions):
    """
    Measure how well functions follow Propp's expected order
    
    Args:
        propp_vector: binary vector of function presence
        text_positions: position in text where each function appears
    
    Returns:
        coherence_score ∈ [0, 1]
    """
    present_functions = [(i, pos) for i, pos in enumerate(text_positions) if propp_vector[i] == 1]
    
    # Count order violations
    violations = 0
    for i in range(len(present_functions) - 1):
        func_a, pos_a = present_functions[i]
        func_b, pos_b = present_functions[i + 1]
        
        # If func_a should come after func_b but appears before
        if func_a > func_b and pos_a < pos_b:
            violations += 1
    
    max_violations = len(present_functions) * (len(present_functions) - 1) / 2
    coherence = 1 - (violations / max_violations) if max_violations > 0 else 1.0
    
    return coherence
```

### Fairy Tale Completeness Score

```python
def calculate_fairy_tale_completeness(propp_vector):
    """
    Typical fairy tale has 20-25 functions
    Core functions are weighted higher
    """
    core_functions = [
        7,   # Villainy or
        8,   # Lack
        10,  # Beginning counteraction
        11,  # Departure
        16,  # Struggle
        18,  # Victory
        19,  # Liquidation
        30   # Wedding or equivalent resolution
    ]
    
    core_present = sum([propp_vector[i] for i in core_functions])
    total_present = sum(propp_vector)
    
    # Weighted score
    FTCS = 0.60 * (core_present / len(core_functions)) + 0.40 * (total_present / 31)
    
    return FTCS  # Fairy Tale Completeness Score ∈ [0, 1]
```

### Contribution to π (Structural Component)

```
π_propp = λ · FTCS + μ · PCS + ν · SPC

Where:
FTCS = Fairy Tale Completeness Score
PCS = Propp Coherence Score (sequential order)
SPC = Sphere Presence Count (7 spheres of action)

λ + μ + ν = 1 (typically λ=0.45, μ=0.35, ν=0.20)
```

**Expected Domains**:
- Fairy Tales: High π_propp (0.85+)
- Folk Tales: High π_propp (0.75+)
- Modern Fantasy (fairy tale structure): Medium-High π_propp (0.60-0.75)
- Realistic Fiction: Low π_propp (0.20-0.40)

---

## Vonnegut's Shapes → Emotional Trajectory

### Story Shapes as Mathematical Functions

Each Vonnegut shape is a function V(t) where:
- **t**: Normalized time (0 to 1)
- **V(t)**: Good/bad fortune value (-1 to +1)

#### 1. Man in Hole

```python
def man_in_hole(t):
    """
    Start neutral, descend, ascend back
    V(t) = sin(2πt - π/2)  # or similar
    """
    if t < 0.25:
        return 0  # Neutral start
    elif t < 0.60:
        return -1 * (t - 0.25) / 0.35  # Descend
    else:
        return -1 + (t - 0.60) / 0.40  # Ascend
```

#### 2. Boy Meets Girl

```python
def boy_meets_girl(t):
    """
    Start low, rise (meeting), fall (loss), rise higher (reunion)
    """
    if t < 0.20:
        return -0.5  # Start low
    elif t < 0.40:
        return -0.5 + 1.5 * (t - 0.20) / 0.20  # Meet, rise to +1
    elif t < 0.60:
        return 1.0 - 1.5 * (t - 0.40) / 0.20  # Lose, fall to -0.5
    else:
        return -0.5 + 1.5 * (t - 0.60) / 0.40  # Reunion, rise to +1
```

#### 3-8. Other Shapes

Similar mathematical functions for:
- From Bad to Worse (monotonic decline)
- Which Way Is Up (chaotic/ambiguous)
- Creation Story (exponential rise)
- Old Testament (rise, fall, partial redemption)
- New Testament (low, sacrifice, eternal high)
- Cinderella (low, high, low, higher)

### Shape Detection Algorithm

```python
def detect_story_shape(text):
    """
    Extract emotional trajectory and match to Vonnegut shapes
    
    Returns: (best_match_shape, correlation, emotional_trajectory)
    """
    # 1. Extract sentiment over time
    segments = split_text_into_n_segments(text, n=20)
    sentiments = [analyze_sentiment(seg) for seg in segments]
    
    # Normalize to [-1, +1]
    trajectory = normalize_sentiments(sentiments)
    
    # 2. Compare to each Vonnegut shape
    shapes = [
        ('man_in_hole', man_in_hole),
        ('boy_meets_girl', boy_meets_girl),
        ('bad_to_worse', bad_to_worse),
        # ... all 8 shapes
    ]
    
    correlations = {}
    for name, shape_func in shapes:
        expected = [shape_func(t) for t in np.linspace(0, 1, 20)]
        correlations[name] = pearsonr(trajectory, expected)[0]
    
    best_match = max(correlations, key=correlations.get)
    
    return (best_match, correlations[best_match], trajectory)
```

### Emotional Arc Metrics

```python
def calculate_emotional_metrics(trajectory):
    """
    Quantify emotional journey characteristics
    """
    metrics = {
        'volatility': np.std(trajectory),  # How much variance
        'range': np.max(trajectory) - np.min(trajectory),  # Total range
        'net_change': trajectory[-1] - trajectory[0],  # Start to end
        'lowest_point': np.min(trajectory),  # Darkest moment
        'highest_point': np.max(trajectory),  # Peak happiness
        'reversals': count_direction_changes(trajectory),  # How many turns
        'final_state': trajectory[-1]  # Ending feeling
    }
    return metrics
```

### Contribution to π (Temporal Component)

```
π_vonnegut = ξ · Shape_Match + ο · Arc_Completeness + π · Emotional_Range

Where:
Shape_Match = Correlation with nearest Vonnegut shape
Arc_Completeness = Whether trajectory has clear beginning/middle/end
Emotional_Range = Magnitude of emotional journey

ξ + ο + π = 1 (typically ξ=0.35, ο=0.40, π=0.25)
```

**Interpretation**:
- Stories matching classic shapes have higher temporal coherence
- Contributes to overall π through temporal component (T)
- "Which Way Is Up" (Kafka) has low shape match → low π

---

## Blake Snyder's Beats → Temporal Features

### The 15 Beats as Temporal Markers

```python
SAVE_THE_CAT_BEATS = {
    'opening_image': {'position': 0.00, 'width': 0.02, 'description': 'First impression'},
    'theme_stated': {'position': 0.05, 'width': 0.02, 'description': 'Lesson mentioned'},
    'setup': {'position': 0.05, 'width': 0.08, 'description': 'World establishment'},
    'catalyst': {'position': 0.12, 'width': 0.02, 'description': 'Inciting incident'},
    'debate': {'position': 0.15, 'width': 0.10, 'description': 'Should I go?'},
    'break_into_two': {'position': 0.25, 'width': 0.02, 'description': 'Enter Act II'},
    'b_story': {'position': 0.30, 'width': 0.05, 'description': 'Love story begins'},
    'fun_and_games': {'position': 0.35, 'width': 0.20, 'description': 'Promise of premise'},
    'midpoint': {'position': 0.50, 'width': 0.03, 'description': 'False peak'},
    'bad_guys_close_in': {'position': 0.55, 'width': 0.20, 'description': 'Pressure increases'},
    'all_is_lost': {'position': 0.75, 'width': 0.02, 'description': 'Lowest point'},
    'dark_night': {'position': 0.77, 'width': 0.08, 'description': 'Processing loss'},
    'break_into_three': {'position': 0.85, 'width': 0.02, 'description': 'Aha moment'},
    'finale': {'position': 0.87, 'width': 0.12, 'description': 'Final confrontation'},
    'final_image': {'position': 0.99, 'width': 0.01, 'description': 'Closing mirror'}
}
```

### Beat Detection and Timing Analysis

```python
def analyze_beat_structure(text):
    """
    Detect presence and timing of Save the Cat beats
    
    Returns: beat_adherence_score ∈ [0, 1]
    """
    segments = split_text_into_segments(text, n=100)  # Fine-grained
    
    detected_beats = {}
    for beat_name, beat_info in SAVE_THE_CAT_BEATS.items():
        # Search in expected region ± tolerance
        expected_pos = beat_info['position']
        tolerance = beat_info['width'] * 2  # Allow some variance
        
        search_start = max(0, expected_pos - tolerance)
        search_end = min(1, expected_pos + tolerance)
        
        # Convert to segment indices
        start_idx = int(search_start * len(segments))
        end_idx = int(search_end * len(segments))
        
        # Look for beat markers in this region
        confidence = detect_beat_markers(
            segments[start_idx:end_idx],
            beat_name
        )
        
        if confidence > 0.5:  # Threshold
            actual_position = find_peak_position(segments, start_idx, end_idx, beat_name)
            detected_beats[beat_name] = {
                'present': True,
                'confidence': confidence,
                'expected': expected_pos,
                'actual': actual_position,
                'timing_error': abs(actual_position - expected_pos)
            }
        else:
            detected_beats[beat_name] = {'present': False}
    
    # Calculate beat adherence score
    presence_score = sum([1 for b in detected_beats.values() if b['present']]) / 15
    
    timing_errors = [b['timing_error'] for b in detected_beats.values() if b['present']]
    timing_score = 1 - np.mean(timing_errors) if timing_errors else 0
    
    BAS = 0.60 * presence_score + 0.40 * timing_score  # Beat Adherence Score
    
    return BAS, detected_beats
```

### Pacing Analysis

```python
def analyze_pacing(beat_dict):
    """
    Measure rhythm and pacing based on beat spacing
    """
    key_beats = ['catalyst', 'break_into_two', 'midpoint', 'all_is_lost', 'break_into_three']
    
    if not all([beat_dict[b]['present'] for b in key_beats]):
        return {'pacing_score': 0, 'rhythm': 'undefined'}
    
    # Extract positions
    positions = [beat_dict[b]['actual'] for b in key_beats]
    intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    
    # Expected intervals (ideal pacing)
    expected = [0.13, 0.25, 0.25, 0.10]  # Catalyst→B2, B2→Mid, Mid→AIL, AIL→B3
    
    # Measure deviation
    deviations = [abs(intervals[i] - expected[i]) for i in range(len(intervals))]
    pacing_score = 1 - np.mean(deviations)
    
    # Classify rhythm
    if np.std(intervals) < 0.05:
        rhythm = 'steady'
    elif intervals[0] < intervals[1]:
        rhythm = 'accelerating'
    else:
        rhythm = 'decelerating'
    
    return {'pacing_score': pacing_score, 'rhythm': rhythm, 'intervals': intervals}
```

### Contribution to λ (Structural Constraints)

```
λ_snyder = ρ · BAS + σ · Pacing_Score + τ · Genre_Adherence

Where:
BAS = Beat Adherence Score
Pacing_Score = Rhythm quality
Genre_Adherence = Matching expected genre beats

ρ + σ + τ = 1 (typically ρ=0.50, σ=0.30, τ=0.20)
```

**Hollywood Formula**:
- Commercial films: High λ_snyder (0.75-0.90)
- Indie films: Medium λ_snyder (0.45-0.65)
- Art films: Low λ_snyder (0.10-0.35)

---

## Polti's Situations → Conflict Features

### 36 Dramatic Situations as Feature Space

```python
POLTI_SITUATIONS = {
    'supplication': {
        'roles': ['persecutor', 'suppliant', 'power_in_authority'],
        'markers': ['beg', 'plead', 'mercy', 'help', 'grant'],
        'conflict_type': 'power_imbalance'
    },
    'deliverance': {
        'roles': ['unfortunate', 'threatener', 'rescuer'],
        'markers': ['rescue', 'save', 'deliver', 'free', 'escape'],
        'conflict_type': 'captivity_vs_freedom'
    },
    # ... all 36 situations
}

def detect_polti_situations(text):
    """
    Identify which of Polti's 36 situations are present
    
    Returns: 36-dimensional probability vector
    """
    situation_scores = np.zeros(36)
    
    for i, (name, situation) in enumerate(POLTI_SITUATIONS.items()):
        # Check for role presence
        role_score = sum([detect_role(text, role) for role in situation['roles']]) / len(situation['roles'])
        
        # Check for marker patterns
        marker_score = count_patterns(text, situation['markers']) / len(situation['markers'])
        
        # Check for conflict type
        conflict_score = detect_conflict_type(text, situation['conflict_type'])
        
        # Combined score
        situation_scores[i] = (role_score + marker_score + conflict_score) / 3
    
    return situation_scores
```

### Conflict Complexity Metrics

```python
def analyze_conflict_complexity(polti_vector):
    """
    Measure richness of dramatic conflicts
    """
    metrics = {
        'n_situations': np.sum(polti_vector > 0.5),  # How many situations present
        'primary_situation': np.argmax(polti_vector),  # Dominant situation
        'situation_diversity': entropy(polti_vector),  # How diverse
        'conflict_intensity': np.max(polti_vector),  # Strongest conflict
        'layered_conflicts': np.sum(polti_vector > 0.3)  # Multiple layers
    }
    
    # Conflict Complexity Score
    CCS = (
        0.30 * min(metrics['n_situations'] / 5, 1.0) +  # More situations = complex
        0.25 * metrics['situation_diversity'] +
        0.25 * metrics['conflict_intensity'] +
        0.20 * min(metrics['layered_conflicts'] / 3, 1.0)
    )
    
    return CCS, metrics
```

### Contribution to π (Structural Component)

```
π_polti = υ · CCS + φ · Conflict_Resolution + χ · Character_Agency

Where:
CCS = Conflict Complexity Score
Conflict_Resolution = Whether conflicts are resolved
Character_Agency = Characters' ability to affect conflicts

υ + φ + χ = 1 (typically υ=0.45, φ=0.35, χ=0.20)
```

**Domain Differences**:
- Greek Tragedy: Few situations (1-2) but intense
- Complex Novels: Many situations (5-8) layered
- Action Films: Clear situations (2-3) resolved kinetically
- Experimental: Ambiguous situations, unresolved

---

## Booker's Plots → Ξ Archetypes

### The Seven Basic Plots as Domain Ξ Targets

Each of Booker's plots represents a distinct archetypal perfection:

#### 1. Overcoming the Monster → Ξ_monster

```python
MONSTER_XI = {
    'archetype_requirements': {
        'hero_courage': 0.90,  # Must be brave
        'monster_menace': 0.85,  # Clear threat
        'escalating_danger': 0.80,  # Stakes increase
        'final_confrontation': 0.95,  # Climactic battle
        'victory': 0.90  # Monster defeated
    },
    'story_arc': 'threat → preparation → battles → victory',
    'emotional_trajectory': 'fear → courage → triumph',
    'key_stages': [
        'anticipation_of_monster',
        'dream_stage_preparation',
        'frustration_stage_difficulty',
        'nightmare_stage_near_defeat',
        'miraculous_escape_victory'
    ]
}

def measure_distance_from_monster_xi(text):
    """
    How close is this narrative to ideal "Overcoming Monster" archetype?
    
    Returns: distance ∈ [0, ∞), lower = closer to Ξ_monster
    """
    detected_features = extract_monster_plot_features(text)
    
    # Calculate feature-wise distance
    distances = []
    for feature, required_value in MONSTER_XI['archetype_requirements'].items():
        actual_value = detected_features.get(feature, 0)
        distances.append((required_value - actual_value) ** 2)
    
    # Euclidean distance in archetype space
    distance = np.sqrt(sum(distances))
    
    # Story quality = inverse of distance (closer = better)
    story_quality = 1 / (1 + distance)  # ∈ [0, 1]
    
    return distance, story_quality
```

#### 2. Rags to Riches → Ξ_riches

```python
RICHES_XI = {
    'archetype_requirements': {
        'initial_poverty': 0.85,  # Start low
        'opportunity_appears': 0.75,
        'initial_success': 0.70,
        'crisis_threatens': 0.75,
        'inner_worth_proved': 0.90,  # Key: proves actual worth
        'final_prosperity': 0.85
    },
    'transformation': 'external recognition of internal worth',
    'emotional_trajectory': 'despair → hope → anxiety → confidence → joy'
}
```

#### 3-7. Other Plot Ξ Archetypes

Similar archetypal perfection definitions for:
- **Quest** (Ξ_quest): Journey, companions, trials, goal achievement
- **Voyage and Return** (Ξ_voyage): Strange world, fascination, threat, escape
- **Comedy** (Ξ_comedy): Confusion, complications, revelation, union
- **Tragedy** (Ξ_tragedy): Hubris, nemesis, fall, catharsis
- **Rebirth** (Ξ_rebirth): Spell/curse, imprisonment, redemption, transformation

### Multi-Plot Detection

```python
def detect_booker_plot_type(text):
    """
    Determine which of Booker's 7 plots best fits narrative
    Can detect multiple plot types (many stories blend)
    
    Returns: plot_probabilities (7-dimensional vector summing to 1)
    """
    xi_distances = {
        'monster': measure_distance_from_monster_xi(text)[0],
        'riches': measure_distance_from_riches_xi(text)[0],
        'quest': measure_distance_from_quest_xi(text)[0],
        'voyage': measure_distance_from_voyage_xi(text)[0],
        'comedy': measure_distance_from_comedy_xi(text)[0],
        'tragedy': measure_distance_from_tragedy_xi(text)[0],
        'rebirth': measure_distance_from_rebirth_xi(text)[0]
    }
    
    # Convert distances to probabilities (softmax)
    # Closer distance = higher probability
    inv_distances = {k: 1/(1+v) for k, v in xi_distances.items()}
    total = sum(inv_distances.values())
    probabilities = {k: v/total for k, v in inv_distances.items()}
    
    return probabilities, xi_distances
```

### Story Quality as Ξ Proximity

```
Story_Quality = min(distance_to_nearest_Ξ)

For multi-domain Ξ:
Story_Quality = Σ(w_i · proximity_to_Ξ_i)

Where:
w_i = weight/confidence of plot type i
proximity = 1 / (1 + distance)
```

### Contribution to Overall Framework

```
Ξ_vector = domain_specific_archetype_centroid

For classical literature:
- Epic → Ξ_monster or Ξ_quest (high values)
- Fairy Tales → Ξ_riches or Ξ_rebirth (high values)
- Romance → Ξ_comedy (high values)
- Serious Literature → Ξ_tragedy (high values)

Predicted R²:
R²(success ~ proximity_to_appropriate_Ξ) > 0.65 for classical lit
```

---

## Complete Integration Formula

### Unified Narrative Quality Model

```
Q = f(π, λ, θ, ة, Ξ_proximity)

Where each component is computed from classical theories:

π = Σ w_i · π_i
    i ∈ {campbell, jung, aristotle, propp, vonnegut, snyder, polti, booker}

λ = Σ w_j · λ_j
    j ∈ {aristotle, frye, snyder}

θ = Σ w_k · θ_k
    k ∈ {frye, meta_markers}

ة = Σ w_l · ة_l
    l ∈ {character_iconicity, title_memorability, noun_density}

Ξ_proximity = distance_to_domain_specific_archetype(booker, jung, campbell)
```

### Feature Vector Construction

```python
def construct_classical_feature_vector(text):
    """
    Complete feature extraction from all classical theories
    
    Returns: feature_dict with ~300-400 features
    """
    features = {}
    
    # Campbell (60 features)
    features['campbell'] = {
        'journey_completion': calculate_jcs(text),
        'stage_presence': extract_stage_vector(text),  # 17-dim
        'transformation_depth': measure_transformation(text),
        'mentor_quality': detect_mentor_effectiveness(text),
        'threshold_crossings': count_thresholds(text),
        **extract_detailed_campbell_features(text)
    }
    
    # Jung (40 features)
    features['jung'] = {
        'archetype_vector': detect_all_archetypes(text),  # 12-dim
        'archetype_clarity': calculate_acs(text),
        'shadow_projection': calculate_spi(text),
        'character_arc': measure_character_arc(text),
        **extract_detailed_jung_features(text)
    }
    
    # Aristotle (30 features)
    features['aristotle'] = {
        'plot_quality': measure_plot_quality(text),
        'character_quality': measure_character_quality(text),
        'thought_quality': measure_thought_quality(text),
        'diction_quality': measure_diction_quality(text),
        'peripeteia': detect_reversal(text),
        'anagnorisis': detect_recognition(text),
        **extract_detailed_aristotle_features(text)
    }
    
    # Frye (20 features)
    features['frye'] = {
        'mythos_vector': detect_mythos(text),  # 4-dim
        'mythos_purity': measure_mythos_purity(text),
        **extract_detailed_frye_features(text)
    }
    
    # Propp (35 features)
    features['propp'] = {
        'function_vector': extract_propp_vector(text),  # 31-dim
        'fairy_tale_completeness': calculate_ftcs(text),
        'propp_coherence': calculate_propp_coherence(text),
        **extract_detailed_propp_features(text)
    }
    
    # Vonnegut (25 features)
    features['vonnegut'] = {
        'shape_match': detect_story_shape(text)[1],
        'emotional_trajectory': detect_story_shape(text)[2],
        'emotional_metrics': calculate_emotional_metrics(text),
        **extract_detailed_vonnegut_features(text)
    }
    
    # Snyder (30 features)
    features['snyder'] = {
        'beat_adherence': analyze_beat_structure(text)[0],
        'beat_vector': analyze_beat_structure(text)[1],  # 15-dim
        'pacing_quality': analyze_pacing(text)['pacing_score'],
        **extract_detailed_snyder_features(text)
    }
    
    # Polti (40 features)
    features['polti'] = {
        'situation_vector': detect_polti_situations(text),  # 36-dim
        'conflict_complexity': analyze_conflict_complexity(text)[0],
        **extract_detailed_polti_features(text)
    }
    
    # Booker (30 features)
    features['booker'] = {
        'plot_probabilities': detect_booker_plot_type(text)[0],  # 7-dim
        'xi_distances': detect_booker_plot_type(text)[1],  # 7-dim
        'story_quality': min(detect_booker_plot_type(text)[1].values()),
        **extract_detailed_booker_features(text)
    }
    
    # Flatten to single vector
    flat_features = flatten_nested_dict(features)
    
    return flat_features  # ~300-400 dimensional feature vector
```

### Predictive Model

```python
def predict_narrative_success(text, domain='classical_literature'):
    """
    Use classical theory features to predict success
    
    Returns: predicted_success ∈ [0, 1]
    """
    # Extract features
    features = construct_classical_feature_vector(text)
    
    # Calculate π, λ, θ, ة from features
    π = calculate_pi_from_classical_features(features)
    λ = calculate_lambda_from_classical_features(features)
    θ = calculate_theta_from_classical_features(features)
    ة = calculate_ta_marbuta_from_classical_features(features)
    
    # Get domain-specific Ξ
    ξ = get_domain_xi(domain)
    
    # Measure proximity to Ξ
    feature_vector = features_to_vector(features)
    xi_proximity = cosine_similarity(feature_vector, ξ)
    
    # Integrated model
    success = (
        0.30 * π +
        0.20 * (1 - λ) +  # Lower constraints = more freedom (for high-π domains)
        0.15 * (θ if domain == 'irony' else (1-θ)) +  # Context-dependent
        0.20 * ة +
        0.15 * xi_proximity
    )
    
    return success, {
        'π': π,
        'λ': λ,
        'θ': θ,
        'ة': ة,
        'xi_proximity': xi_proximity
    }
```

---

## Validation Hypotheses

### Hypothesis 1: Campbell's Journey Completion Predicts π

```
H1: r(JCS, π) > 0.70 across all narrative domains

Test across:
- Mythology (expected: r > 0.85)
- Epic Literature (expected: r > 0.75)
- Modern Fiction (expected: r > 0.60)
- Experimental (expected: r > 0.40)
```

### Hypothesis 2: Frye's Mythoi Map to θ/λ Space

```
H2: Comedy, Romance, Tragedy, Irony occupy distinct θ/λ regions

Expected clusters:
- Comedy: θ=0.30±0.10, λ=0.50±0.10
- Romance: θ=0.20±0.10, λ=0.30±0.10
- Tragedy: θ=0.55±0.10, λ=0.75±0.10
- Irony: θ=0.85±0.10, λ=0.50±0.20 (wide variance)

Test: K-means clustering on (θ, λ) should recover four mythoi
```

### Hypothesis 3: Aristotelian Quality Predicts Success in High-λ Domains

```
H3: In high-λ domains (λ > 0.65), AQI predicts success

R²(success ~ AQI | λ > 0.65) > 0.55

Test on:
- Greek Tragedy (λ ≈ 0.80)
- Classical Drama (λ ≈ 0.70)
- vs. Experimental Theater (λ ≈ 0.30) where AQI should NOT predict
```

### Hypothesis 4: Booker's Ξ Proximity Predicts Cultural Persistence

```
H4: Narratives closer to appropriate Ξ have greater cultural persistence

R²(persistence ~ ξ_proximity) > 0.60

Persistence measures:
- Still taught in schools
- Modern adaptations count
- Cultural references per decade
- Name recognition surveys
```

### Hypothesis 5: Jung's Archetype Clarity Correlates with ة

```
H5: r(ACS, ة) > 0.65

Clear archetypes have memorable names
- Mythology: High ACS, high ة
- Literary fiction: Medium ACS, medium ة
- Postmodern: Low ACS, low ة (intentional deconstruction)
```

### Hypothesis 6: Snyder's Beats Define Hollywood Formula

```
H6: BAS predicts box office in commercial films

R²(box_office ~ BAS | commercial_films) > 0.45

But NOT in art films:
R²(box_office ~ BAS | art_films) < 0.15

This demonstrates λ_snyder captures "formula" adherence
```

---

## Summary: Classical Theory Feature Space

### Total Dimensionality

```
Classical Feature Space Dimensions:

Campbell:      ~60 features
Jung:          ~40 features
Aristotle:     ~30 features
Frye:          ~20 features
Propp:         ~35 features
Vonnegut:      ~25 features
Snyder:        ~30 features
Polti:         ~40 features
Booker:        ~30 features
McKee:         ~25 features (to be added)
Vogler:        ~30 features (to be added)
Field:         ~20 features (to be added)

TOTAL:         ~385 features from classical narrative theory
```

### Reduction to π/λ/θ/ة Space

```
Classical_Features[385] → Transformers → π/λ/θ/ة[4] + Ξ_proximity[1]

This 385→5 dimensional reduction enables:
1. Empirical validation of classical theories
2. Cross-domain comparison
3. Predictive modeling
4. Discovery of new patterns
```

### Integration with Existing Framework

```
COMPLETE FEATURE SET:

1. Classical Theory Features (this document): ~385 features
2. Existing Transformers (47 transformers): ~900 features
3. Domain-Specific Features: ~100 features

TOTAL: ~1,385 features available for narrative analysis

Then reduced to:
π, λ, θ, ة, Ξ_proximity → 5-dimensional interpretable space
```

---

## Implementation Roadmap

### Phase 1: Feature Extractors (Week 1-2)
- Implement all detection functions
- Test on small validation set
- Ensure computational efficiency

### Phase 2: Transformer Classes (Week 2-3)
- Build 10 new archetype transformers
- Integrate with existing transformer library
- Add to transformer catalog

### Phase 3: Empirical Validation (Week 3-4)
- Collect classical literature dataset
- Extract features from all texts
- Validate hypotheses 1-6

### Phase 4: Integration (Week 4-5)
- Merge with existing framework
- Update domain configurations
- Document findings

---

**Mathematical Mapping Complete**: All major classical narrative theories now have formal computational definitions compatible with the π/λ/θ/ة framework.

