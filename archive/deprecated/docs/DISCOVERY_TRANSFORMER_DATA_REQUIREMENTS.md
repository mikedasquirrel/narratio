# Discovery Transformer Data Requirements

**Purpose**: Document what data each discovery transformer needs and how to prepare it.

**Philosophy**: Discovery transformers extract STRUCTURE and GEOMETRY rather than semantics. They discover patterns rather than assume them.

---

## Quick Reference

| Transformer | Input Type | Required Fields | Optional Fields | Outcomes Needed? |
|------------|------------|-----------------|-----------------|------------------|
| Universal Structural Pattern | Text or Feature Sequence | text OR feature_sequence | - | No |
| Relational Topology | Feature Pairs | entity_a_features, entity_b_features | entity_a_text, entity_b_text | No |
| Cross-Domain Embedding | Feature Matrix | genome_features | domain, text | Yes (optional) |
| Temporal Derivative | Temporal Sequence | feature_history, timestamps | current_features | No |
| Meta-Feature Interaction | Feature Matrix | Array or genome_features | feature_names | Yes (optional) |
| Outcome-Conditioned Archetype | Feature Matrix | genome_features | feature_names, domain | **Yes** (required) |
| Anomaly Uniquity | Feature Matrix | Array or genome_features | - | No |

---

## Universal Structural Pattern Transformer

### Purpose
Extracts narrative arc shapes, tension geometry, and archetypal structures geometrically.

### Input Format

**Option 1: Text (Simple)**
```python
X = [
    "Narrative text 1",
    "Narrative text 2",
    ...
]
```

**Option 2: Pre-computed Feature Sequences (Preferred)**
```python
X = [
    {
        'feature_sequence': np.array([0.3, 0.4, 0.5, 0.6, ...]),  # Sequence of feature values
        'text': "Optional narrative text"
    },
    ...
]
```

### Data Preparation

1. **From Text**: Transformer will extract basic sequence automatically (word length variation)
2. **From Features** (recommended): 
   - Extract genome features over time/progression
   - Create sequence of aggregated feature values
   - Normalize to consistent length (transformer handles this)

### Example: Sports Game
```python
# Extract features at different game moments
game_features = []
for quarter in [1, 2, 3, 4]:
    quarter_feat = extract_features(game, quarter)
    game_features.append(np.mean(quarter_feat))  # Aggregate

X = [{
    'feature_sequence': np.array(game_features),
    'text': game['narrative']
}]
```

### Output
45 features describing arc shape, tension, symmetry, complexity, and archetypal fits.

---

## Relational Topology Transformer

### Purpose
Captures structural relationships between entities in matchups WITHOUT assuming which relationships matter.

### Input Format

**Required**: Paired entity features
```python
X = [
    {
        'entity_a_features': np.array([0.7, 0.3, 0.9, ...]),  # Features for entity A
        'entity_b_features': np.array([0.4, 0.8, 0.2, ...]),  # Features for entity B
        'entity_a_text': "Optional text about A",
        'entity_b_text': "Optional text about B"
    },
    ...
]
```

### Data Preparation

1. Extract genome features for both entities separately
2. Ensure features have same length
3. Package as entity pairs

### Example: UFC Fight
```python
X = [
    {
        'entity_a_features': fighter_a_genome,  # From other transformers
        'entity_b_features': fighter_b_genome,
        'entity_a_text': "Fighter A narrative",
        'entity_b_text': "Fighter B narrative"
    }
    for fight in fights
]
```

### Output
35 features describing distance, asymmetry, complementarity, dominance, and interaction topology.

---

## Cross-Domain Embedding Transformer

### Purpose
Projects narratives into universal embedding space where structurally similar narratives cluster regardless of domain.

### Input Format

```python
X = [
    {
        'genome_features': np.array([...]),  # ж from other transformers
        'domain': 'nba',  # Domain label (for transfer learning)
        'text': "Optional narrative text"
    },
    ...
]
```

### Data Preparation

1. **Extract genome features first**: Run other transformers to get ж
2. **Include domain labels**: Enables transfer learning
3. **Mix domains**: Works best with multi-domain data

### Example: Multi-Domain Analysis
```python
# NBA games
nba_data = [
    {'genome_features': nba_game_genome, 'domain': 'nba', 'text': game['narrative']}
    for game in nba_games
]

# NFL games  
nfl_data = [
    {'genome_features': nfl_game_genome, 'domain': 'nfl', 'text': game['narrative']}
    for game in nfl_games
]

X = nba_data + nfl_data  # Mix domains
```

### Output
30 features including cluster membership, embedding coordinates, archetypal distances, and transfer confidence.

---

## Temporal Derivative Transformer

### Purpose
Captures rate-of-change in narrative features WITHOUT assuming which temporal patterns matter.

### Input Format

```python
X = [
    {
        'feature_history': np.array([
            [0.3, 0.4],  # t1 features
            [0.35, 0.45],  # t2 features
            [0.4, 0.5],  # t3 features
            ...
        ]),
        'timestamps': [1, 2, 3, ...],  # or datetime objects
        'current_features': np.array([0.45, 0.55])  # Optional: current state
    },
    ...
]
```

### Data Preparation

1. **Extract features at multiple time points**: For each period/game/moment
2. **Create feature history matrix**: Stack features chronologically
3. **Include timestamps**: Can be integers or datetime objects
4. **Ensure consistent dimensionality**: All time points same feature count

### Example: NBA Season
```python
# For each team's season
season_history = []
timestamps = []

for game_num in range(1, 83):
    game_features = extract_features(team, game_num)
    season_history.append(game_features)
    timestamps.append(game_num)

X = [{
    'feature_history': np.array(season_history),
    'timestamps': timestamps,
    'current_features': season_history[-1]
}]
```

### Output
40 features describing velocity, acceleration, trend consistency, regime shifts, and autocorrelation.

---

## Meta-Feature Interaction Transformer

### Purpose
Generates interaction features automatically WITHOUT assuming which combinations matter.

### Input Format

**Option 1: Simple Array**
```python
X = np.array([
    [0.5, 0.3, 0.8],  # Sample 1 features
    [0.6, 0.4, 0.7],  # Sample 2 features
    ...
])
```

**Option 2: With Feature Names (Preferred)**
```python
X = [
    {
        'genome_features': np.array([0.5, 0.3, 0.8]),
        'feature_names': ['nom_richness', 'arc_quality', 'momentum']
    },
    ...
]
```

### Data Preparation

1. **Extract genome features**: From other transformers
2. **Include feature names**: For interpretable interaction names
3. **Ensure sufficient samples**: Need enough data for variance filtering

### Example: Combined Features
```python
# After running other transformers
genome_features = []
feature_names = []

for transformer in [nominative_t, arc_t, momentum_t]:
    features = transformer.transform(texts)
    genome_features.append(features)
    feature_names.extend(transformer.get_feature_names())

X = [{
    'genome_features': np.hstack(genome_features[i]),
    'feature_names': feature_names
} for i in range(len(texts))]
```

### Output
100+ features (dynamic) including multiplicative, ratio, polynomial, synergy, and antagonism features.

---

## Outcome-Conditioned Archetype Transformer

### Purpose
Discovers Ξ (Golden Narratio) and α (optimal balance) from outcomes. **REQUIRES y**.

### Input Format

```python
X = [
    {
        'genome_features': np.array([...]),
        'feature_names': ['nom_richness', 'arc_quality', ...],
        'domain': 'nba'
    },
    ...
]
y = np.array([1, 0, 1, ...])  # 1=winner, 0=loser
```

### Data Preparation

1. **Extract genome features**: Full ж from other transformers
2. **Prepare outcomes**: Binary (1/0) or continuous
3. **Include domain**: For transfer learning
4. **Include feature names**: For α discovery

### Example: NBA Games with Winners
```python
X = []
y = []

for game in games:
    # Team A (home)
    X.append({
        'genome_features': extract_genome(game, 'home'),
        'feature_names': genome_feature_names,
        'domain': 'nba'
    })
    y.append(1 if game['home_won'] else 0)
    
    # Team B (away)
    X.append({
        'genome_features': extract_genome(game, 'away'),
        'feature_names': genome_feature_names,
        'domain': 'nba'
    })
    y.append(1 if game['away_won'] else 0)

transformer.fit(X, y)  # y is REQUIRED
```

### Output
25 features including distance to Ξ, optimal α, feature balance, sub-archetypes, and transfer confidence.

---

## Anomaly Uniquity Transformer

### Purpose
Measures how unusual/novel a narrative is WITHOUT predefining "unusual".

### Input Format

**Simple**: Just feature matrix
```python
X = np.array([
    [0.5, 0.3],
    [0.6, 0.4],
    [0.1, 0.9],  # This one looks unusual
    ...
])
```

**Or**: With genome features
```python
X = [
    {'genome_features': np.array([...])},
    ...
]
```

### Data Preparation

1. **Extract genome features**: From other transformers
2. **Include diverse samples**: Need distribution to measure anomalies against
3. **Ensure sufficient data**: Anomaly detection needs baseline (~50+ samples)

### Example: Startup Pitches
```python
# Extract features for all pitches
all_genomes = [extract_genome(pitch) for pitch in pitches]

X = np.array(all_genomes)

# Fit learns normal distribution
transformer.fit(X)

# Transform finds anomalies
features = transformer.transform(X)
novelty_scores = features[:, -3]  # Overall novelty score
```

### Output
20 features including statistical outliers, isolation metrics, historical precedent, cluster analysis, and novelty scores.

---

## General Best Practices

### 1. Feature Extraction Pipeline

Recommended order:
```python
# Stage 1: Extract base genome features
base_transformers = [
    NominativeAnalysisTransformer(),
    LinguisticPatternsTransformer(),
    NarrativePotentialTransformer(),
    # ... other base transformers
]

genome_features = extract_genome(texts, base_transformers)

# Stage 2: Apply discovery transformers
structural = UniversalStructuralPatternTransformer()
embedding = CrossDomainEmbeddingTransformer()
interaction = MetaFeatureInteractionTransformer()

structural_feat = structural.fit_transform(texts)
embedding_feat = embedding.fit_transform([{
    'genome_features': gf,
    'domain': domain
} for gf in genome_features])
interaction_feat = interaction.fit_transform(genome_features, y=outcomes)
```

### 2. Data Requirements by Use Case

**Exploratory Analysis** (no outcomes):
- Universal Structural Pattern
- Relational Topology (for matchups)
- Cross-Domain Embedding
- Anomaly Uniquity

**Supervised Learning** (with outcomes):
- Meta-Feature Interaction (optional y)
- Outcome-Conditioned Archetype (requires y)
- Cross-Domain Embedding (optional y)

**Temporal Analysis** (time series data):
- Temporal Derivative
- Universal Structural Pattern (from sequences)

**Cross-Domain Transfer**:
- Cross-Domain Embedding (with domain labels)
- Outcome-Conditioned Archetype (with domain labels)

### 3. Common Issues

**Issue**: "Feature dimensions don't match"
**Solution**: Ensure all entities/samples have same feature dimensionality

**Issue**: "Not enough samples for clustering"
**Solution**: Need minimum 10-20 samples for meaningful clustering

**Issue**: "Temporal sequences vary in length"
**Solution**: Transformers handle normalization automatically

**Issue**: "Missing outcomes for supervised transformers"
**Solution**: Outcome-Conditioned Archetype REQUIRES y; others optional

### 4. Performance Considerations

**Fast**: (<1 sec per sample)
- Universal Structural Pattern (text input)
- Anomaly Uniquity

**Medium**: (1-5 sec per sample)
- Relational Topology
- Temporal Derivative
- Universal Structural Pattern (sequence input)

**Slow**: (>5 sec per sample)
- Cross-Domain Embedding (UMAP fitting)
- Meta-Feature Interaction (many interactions)
- Outcome-Conditioned Archetype (clustering)

**Tip**: Fit once, transform many. Fitting is expensive, transforming is fast.

---

## Complete Example: Multi-Domain Analysis

```python
from narrative_optimization.src.transformers import *

# 1. Prepare data from multiple domains
nba_texts = load_nba_narratives()
nfl_texts = load_nfl_narratives()
nba_outcomes = load_nba_outcomes()
nfl_outcomes = load_nfl_outcomes()

# 2. Extract base genome features
base_pipeline = Pipeline([
    ('nominative', NominativeAnalysisTransformer()),
    ('linguistic', LinguisticPatternsTransformer()),
    ('potential', NarrativePotentialTransformer())
])

nba_genomes = base_pipeline.fit_transform(nba_texts)
nfl_genomes = base_pipeline.transform(nfl_texts)

# 3. Apply discovery transformers

# Structural patterns
structural = UniversalStructuralPatternTransformer()
nba_structural = structural.fit_transform(nba_texts)

# Cross-domain embedding
embedding_data = (
    [{'genome_features': g, 'domain': 'nba'} for g in nba_genomes] +
    [{'genome_features': g, 'domain': 'nfl'} for g in nfl_genomes]
)
embedding = CrossDomainEmbeddingTransformer()
embedding_feat = embedding.fit_transform(embedding_data, 
                                         y=np.concatenate([nba_outcomes, nfl_outcomes]))

# Outcome-conditioned archetypes (discover what winners look like)
archetype = OutcomeConditionedArchetypeTransformer()
archetype_feat = archetype.fit_transform(embedding_data, 
                                         y=np.concatenate([nba_outcomes, nfl_outcomes]))

# 4. Analyze discovered patterns
print(f"Discovered {embedding.metadata['n_clusters']} universal archetypes")
print(f"NBA optimal α: {archetype.optimal_alpha_:.3f}")
print(f"Distance to Ξ predicts outcomes: r={correlation(archetype_feat[:, 0], nba_outcomes):.3f}")
```

---

## Need Help?

- **Data format questions**: See examples above for your use case
- **Integration issues**: Check DISCOVERY_TRANSFORMERS_GUIDE.md
- **Performance problems**: Review performance considerations section
- **Conceptual questions**: See README.md for philosophy

**Remember**: These transformers discover patterns from data, they don't assume them. The better your input data quality, the better the discovered patterns.

