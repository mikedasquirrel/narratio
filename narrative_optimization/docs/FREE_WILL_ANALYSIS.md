# Free Will vs Determinism Narrative Analysis

A comprehensive system for analyzing narratives to measure free will vs determinism signals through multiple dimensions: semantic fields, temporal dynamics, information theory, causal structure, and network analysis.

## Overview

This system implements a complete pipeline for analyzing how narratives encode free will vs determinism. It extracts features across multiple dimensions and provides composite scores for determinism, agency, and narrative inevitability.

**Key Insight**: Narrativity isn't about actual free will—it's about **perceived agency under uncertainty**. When causes are VISIBLE in narrative → determinism perceived clearly. When causes HIDDEN → characters appear to have free will.

## Quick Start

### Installation

```bash
# Install dependencies
pip install sentence-transformers torch spacy scipy networkx

# Download spaCy model
python -m spacy download en_core_web_sm

# For better accuracy (optional, larger model)
python -m spacy download en_core_web_trf
```

### Basic Usage

```python
from narrative_optimization.src.analysis.free_will_analyzer import NarrativeFreeWillAnalyzer

# Initialize analyzer with default weights
analyzer = NarrativeFreeWillAnalyzer(
    use_sentence_transformers=True,
    use_spacy=True,
    model_name='all-MiniLM-L6-v2',  # Fast model
    spacy_model='en_core_web_sm'
)

# Or with custom weights
analyzer = NarrativeFreeWillAnalyzer(
    temporal_weight=0.25,       # Reduce temporal emphasis
    semantic_weight=0.35,       # Moderate semantic weight
    predictability_weight=0.20, # Lower predictability
    custom_weights={'nominative': 0.20}  # Add nominative component
)

# Prepare narratives
stories = [
    "The prophecy foretold his fate. Destiny was inevitable.",
    "She chose her path freely, knowing she could have chosen differently."
]

# Fit and analyze
analyzer.fit(stories)
results = analyzer.analyze_corpus(stories)

# View results
for i, result in enumerate(results):
    print(f"Story {i+1}:")
    print(f"  Determinism Score: {result['determinism_score']:.3f}")
    print(f"  Agency Score: {result['agency_score']:.3f}")
    print(f"  Free Will Ratio: {result['free_will_ratio']:.3f}")
```

## Architecture

### Components

1. **FreeWillAnalysisTransformer** (`src/transformers/free_will_analysis.py`)
   - Core transformer extracting features across all dimensions
   - Implements sklearn transformer API (fit/transform)
   - Extracts 45+ features per narrative (including nominative features)

2. **NarrativeFreeWillAnalyzer** (`src/analysis/free_will_analyzer.py`)
   - High-level analyzer integrating all components
   - Provides analysis methods and corpus-level insights
   - Handles comparison and clustering operations

## Feature Extraction

### 1. Semantic Field Analysis

Extracts density of fate vs choice language:

- **Fate**: destiny, prophecy, inevitable, predetermined
- **Choice**: choose, decide, option, freedom, agency
- **Causality**: because, caused, therefore, led to
- **Contingency**: might, could, perhaps, uncertain, random
- **Inevitability**: must, forced, compelled, no choice
- **Agency**: chose, decided, intended, deliberately
- **Structure**: pattern, cycle, repetition, order
- **Chaos**: random, accident, unexpected, unpredictable

**Features**: 8 field densities + determinism_balance + agency_ratio = **10 features**

### 2. Temporal Dynamics

Measures future vs past orientation:

- Future-oriented language (will, shall, going to, might, could)
- Past-oriented language (was, were, had, did, before, ago)
- Present-oriented language (is, are, now, currently)
- Future/past ratio

**Hypothesis**: Free will = future-oriented. Determinism = past-bound.

**Features**: 4 temporal features

### 3. Information Theory

Measures predictability and entropy:

- Word-level entropy (Shannon entropy)
- Character-level entropy
- Bigram entropy
- Predictability score (inverse of normalized entropy)
- Redundancy (1 - normalized entropy)

**Hypothesis**: High entropy = unpredictable (free will). Low entropy = predictable (deterministic).

**Features**: 5 information theory features

### 4. Agency Extraction (spaCy)

Extracts who acts vs who is acted upon:

- Number of agents (subjects who act)
- Number of patients (passive subjects)
- Free will score = agents / (agents + patients)
- Agency ratio
- Action verb density

**Hypothesis**: More agents = free will signals. More patients = determinism signals.

**Features**: 5 agency features

### 5. Causal Graph Structure

Builds causal networks and measures structure:

- Path dependency (average shortest path length)
- Branching factor (average out-degree)
- Critical nodes (articulation points)
- Deterministic ratio (causal edges / total edges)

**Hypothesis**: Long paths, low branching = deterministic. Short paths, high branching = free will.

**Features**: 4 graph features

### 6. Sentence Transformer Embeddings

Uses SentenceTransformers for structural analysis:

- Mean embedding value
- Standard deviation
- Min/max values

**Features**: 4 embedding features

### 7. Observability Analysis

Tracks visible vs hidden causality:

- Explicit causality markers (because, therefore, caused)
- Hidden causality markers (somehow, mysteriously, suddenly)
- Omniscient narrator markers (knew, understood, realized)

**Hypothesis**: Visible causality → determinism. Hidden causality → perceived free will.

**Features**: 3 observability features

### 8. Nominative Agency Analysis

Examines how naming patterns encode agency vs determinism:

**Character Naming Patterns**:
- Proper name density (John, Sarah) → agency signals
- Generic label ratio (the man, the victim) → determinism signals
- Title pattern frequency (The Chosen One, The Prophet) → fate markers
- Name consistency score → stable identity vs fluid roles
- Identity assertion patterns (I am, they call me)
- Categorical language density (role labels)
- Agency naming patterns (decides, chooses, acts)
- Deterministic vs agentic naming balance

**Key Insight**: Generic labels and titles signal deterministic narratives, while proper names and consistent identity suggest agency.

**Features**: 8 nominative agency features

### 9. Character Naming Evolution

Tracks how character naming changes through the narrative:

**Evolution Metrics**:
- Names gained (generic → proper) → increasing agency
- Names lost (proper → generic) → decreasing agency
- Title accumulation → increasing determinism
- Identity shift score → character transformation
- Agency evolution → direction of change
- Overall naming stability → narrative consistency

**Hypothesis**: Characters gaining names suggests increasing agency; losing names or accumulating titles suggests determinism taking hold.

**Features**: 6 naming evolution features

### 10. Composite Scores

Weighted combination of components:

- **Determinism Score** (0.0-1.0):
  - Default weights (configurable):
    - 30% temporal (past orientation)
    - 40% semantic (fate language)
    - 30% predictability
  - Weights can be customized via constructor parameters
  
- **Free Will Score**: (1 - determinism) × agency_ratio
  
- **Inevitability Score**: (semantic + predictability) / 2

**Features**: 3 composite scores

**Total Features**: ~45-50 features (varies by configuration, includes 14 nominative features)

## Analysis Methods

### Single Narrative Analysis

```python
analysis = analyzer.analyze(story_text)

# Access scores
determinism = analysis['determinism_score']  # 0.0-1.0
agency = analysis['agency_score']
free_will_ratio = analysis['free_will_ratio']

# Access components
temporal = analysis['temporal_features']
semantic = analysis['semantic_features']
info = analysis['information_theory']
agency_data = analysis['agency_analysis']
causal = analysis['causal_structure']
observability = analysis['observability']
```

### Corpus Analysis

```python
# Analyze all stories
results = analyzer.analyze_corpus(stories)

# Cluster by determinism
clusters = analyzer.cluster_by_determinism(stories)
# Returns: high_determinism, medium_determinism, low_determinism

# Get corpus statistics
stats = analyzer.corpus_stats
```

### Structure-Outcome Prediction

Test if narrative structure predicts outcomes:

```python
beginning = "The hero was born under a dark star..."
ending = "As foretold, the hero brought destruction..."

prediction = analyzer.predict_outcome_from_structure(beginning, ending)

# High similarity (>0.7) = deterministic narrative
# Low similarity (<0.3) = free will/surprise narrative
```

### Fiction vs Reality Comparison

Compare structural similarity between fiction and reality:

```python
fictional = "The wizard cast a spell..."
reality = "The scientist conducted an experiment..."

comparison = analyzer.compare_narrative_to_reality(fictional, reality)

# Returns: semantic_similarity, structural_similarity, 
#          determinism_similarity, maps_to_reality
```

### Nominative Determinism Analysis

Analyze how naming patterns encode agency vs determinism:

```python
# Analyze nominative patterns in a story
nominative_analysis = analyzer.analyze_nominative_determinism(story_text)

# Access results
print(f"Nominative Determinism Score: {nominative_analysis['nominative_determinism_score']:.3f}")
print(f"Character Agency Scores: {nominative_analysis['character_agency_scores']}")
print(f"Naming Evolution: {nominative_analysis['naming_pattern_analysis']}")
print("\nInterpretation:")
print(nominative_analysis['interpretation'])

# Example output:
# Nominative Determinism Score: 0.723
# Character Agency Scores: {
#   'proper_name_score': 0.012,
#   'generic_label_score': 0.045,
#   'title_determinism': 0.023,
#   'identity_strength': 0.008
# }
# Naming Evolution: {
#   'pattern': 'evolving',
#   'direction': 'losing_agency',
#   'title_trend': 'accumulating',
#   'identity_stability': 'shifting'
# }
```

Key insights from nominative analysis:
- **High score (>0.7)**: Characters referred to by titles/roles rather than names
- **Low score (<0.3)**: Characters have consistent proper names, suggesting agency
- **Evolution patterns**: Track if characters gain/lose names through narrative

## Research Questions

### Question 1: Does narrative structure predict outcomes?

**Test**: Compare beginning and ending embeddings. High similarity → deterministic. Low → free will.

```python
prediction = analyzer.predict_outcome_from_structure(beginning, ending)
is_deterministic = prediction['beginning_ending_similarity'] > 0.7
```

### Question 2: What's the free will vs determinism ratio?

**Answer**: Use `determinism_score` and `agency_score` from analysis results.

```python
analysis = analyzer.analyze(story)
determinism_ratio = analysis['determinism_score']
free_will_ratio = analysis['free_will_ratio']
```

### Question 3: How does structure map to reality?

**Test**: Compare causal graph properties and embeddings between fiction and real events.

```python
comparison = analyzer.compare_narrative_to_reality(fiction, reality)
maps_to_reality = comparison['maps_to_reality']  # True if similar
```

### Question 4: Does observability moderate narrative effects?

**Hypothesis**: Visible causality → determinism. Hidden causality → perceived free will.

**Test**: Compare `observability['explicit_ratio']` vs `observability['hidden_ratio']` with determinism scores.

## Configuration Options

### Sentence Transformers

```python
analyzer = NarrativeFreeWillAnalyzer(
    use_sentence_transformers=True,
    model_name='all-MiniLM-L6-v2'  # Fast, good quality
    # Alternative: 'all-mpnet-base-v2'  # Slower, better quality
)
```

### spaCy Models

```python
analyzer = NarrativeFreeWillAnalyzer(
    use_spacy=True,
    spacy_model='en_core_web_sm'  # Fast, smaller
    # Alternative: 'en_core_web_trf'  # Transformer-based, better accuracy
)
```

### Feature Extraction

```python
analyzer = NarrativeFreeWillAnalyzer(
    extract_causal_graphs=True,  # Build causal networks
    track_observability=True      # Track visible vs hidden causality
)
```

## Example: Complete Analysis

See `examples/free_will_analysis_example.py` for a complete demonstration:

```bash
python narrative_optimization/examples/free_will_analysis_example.py
```

This example demonstrates:
- Single narrative analysis
- Corpus-level clustering
- Structure-outcome prediction
- Fiction vs reality comparison

## Integration with Existing System

The `FreeWillAnalysisTransformer` follows the standard transformer pattern:

```python
from narrative_optimization.src.transformers.free_will_analysis import FreeWillAnalysisTransformer

# Use as sklearn transformer
transformer = FreeWillAnalysisTransformer()
features = transformer.fit_transform(stories)

# Get feature names
feature_names = transformer.get_feature_names()

# Get interpretation
interpretation = transformer.get_interpretation()
```

## Performance Considerations

- **Memory**: ~500MB-1GB for SentenceTransformers + spaCy models
- **Speed**: 
  - Single narrative: ~0.1-0.5 seconds
  - 100 narratives: ~10-50 seconds
  - 1000 narratives: ~2-10 minutes
- **Model Loading**: Models loaded lazily on first use

## Dependencies

- `sentence-transformers>=2.2.0` - Semantic embeddings
- `torch>=2.0.0` - PyTorch backend
- `spacy>=3.7.0` - Dependency parsing
- `scipy>=1.11.0` - Statistical functions
- `networkx>=3.1` - Graph analysis
- `numpy>=1.24.0` - Numerical operations
- `scikit-learn>=1.3.0` - Base transformer API

## Citation

If you use this system in research:

```bibtex
@software{free_will_narrative_analysis,
  title = {Free Will vs Determinism Narrative Analysis System},
  author = {Narrative Integration System},
  year = {2025},
  version = {1.0.0}
}
```

## License

See main project LICENSE.

---

**Last Updated**: November 2025  
**Version**: 1.0.0  
**Status**: Production Ready

