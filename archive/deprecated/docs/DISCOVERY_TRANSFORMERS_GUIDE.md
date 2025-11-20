# Discovery Transformers Guide

**Philosophy**: Extract STRUCTURE, not SEMANTICS. Discover PATTERNS, not ASSUMPTIONS.

---

## Table of Contents

1. [What Are Discovery Transformers?](#what-are-discovery-transformers)
2. [Why Discovery-First?](#why-discovery-first)
3. [The Seven Discovery Transformers](#the-seven-discovery-transformers)
4. [When to Use Each Transformer](#when-to-use-each-transformer)
5. [Integration with Existing Pipeline](#integration-with-existing-pipeline)
6. [Examples of Discovered Patterns](#examples-of-discovered-patterns)
7. [Best Practices](#best-practices)

---

## What Are Discovery Transformers?

Discovery transformers are a new class of feature extractors that **discover patterns from data** rather than assuming them.

### Traditional Approach (Semantic/Assumption-Based)
```python
# ASSUMES "rivalry" is important
if "rivalry" in text:
    rivalry_score += 1

# ASSUMES "playoff" context matters
if game['is_playoff']:
    weight *= 2.0
```

### Discovery Approach (Structural/Geometric)
```python
# DISCOVERS which relationship geometries predict outcomes
asymmetry = measure_feature_space_asymmetry(entity_a, entity_b)
# Learning system discovers: high asymmetry → outcome pattern

# DISCOVERS which temporal patterns matter
velocity = measure_rate_of_change(feature_history)
# Learning system discovers: late acceleration → success
```

### Key Differences

| Traditional | Discovery |
|------------|-----------|
| "Rivalry matters" | "What relationship structures matter?" |
| "Playoff context is different" | "What contexts cluster together?" |
| Semantic keywords | Geometric properties |
| Domain assumptions | Universal structures |
| Hardcoded | Data-driven |

---

## Why Discovery-First?

### Problem with Assumptions

**Example**: NFL vs NBA
- Traditional: "Division games are rivalries"
- Reality: NBA "rivalry games" structurally similar to NFL "division games"
- Discovery: Both cluster as "high-frequency competitive asymmetry"

**The Issue**: Your assumptions might be:
1. Wrong for the domain
2. Missing hidden patterns
3. Not transferable across domains
4. Culturally/linguistically biased

### Discovery Advantage

**Discover universal patterns**:
- "High asymmetry + complementarity = competitive balance"
- Works in: NBA, NFL, UFC, Chess, Politics, Business
- No domain-specific hardcoding needed

**Enable transfer learning**:
- If NBA Playoff game structurally ≈ Tennis Grand Slam
- Then patterns transfer across domains
- Discovered via embedding space, not assumed

**Avoid confirmation bias**:
- System finds patterns you wouldn't think to look for
- "Narrative velocity × richness" might predict outcomes
- You'd never hardcode that interaction

---

## The Seven Discovery Transformers

### 1. Universal Structural Pattern Transformer

**What it discovers**: Narrative arc SHAPES across domains

**Extracts**: 45 geometric features
- Arc shape (rising/falling/oscillating)
- Tension geometry (buildup/resolution)
- Symmetry/asymmetry measures
- Complexity (entropy, fractal dimension)
- Archetypal fits (U-shape, inverted-U, linear, exponential)

**Universal across**:
- Sports: Momentum building, comeback arcs
- Business: Growth curves, disruption patterns
- Entertainment: Dramatic tension, resolution
- Politics: Campaign momentum, turning points

**Discovery example**:
```python
# Discovers: U-shape arc (hero's journey) predicts success in movies
# Discovers: Inverted-U (tragedy) predicts failure in startups
# WITHOUT assuming these patterns in advance
```

### 2. Relational Topology Transformer

**What it discovers**: Which relationship structures predict outcomes in matchups

**Extracts**: 35 topological features
- Distance metrics (how far apart in feature space)
- Asymmetry measures (how unbalanced)
- Complementarity patterns (opposing vs overlapping)
- Dominance geometry (superiority patterns)
- Interaction curvature (synergy vs antagonism)

**Universal across**:
- UFC: Fighter matchups
- Business: Competitive positioning
- Politics: Debate matchups
- Chess: Playing style matchups

**Discovery example**:
```python
# Discovers: High asymmetry + high complementarity = upset potential
# Works in UFC, NBA, elections WITHOUT domain-specific coding
```

### 3. Cross-Domain Embedding Transformer

**What it discovers**: Structural isomorphism across domains

**Extracts**: 30 embedding features
- Universal cluster membership
- Cross-domain similarity
- Archetypal distances
- Transfer confidence scores

**Key insight**: 
- Projects ALL narratives into universal space
- Structurally similar narratives cluster together
- Enables transfer learning via cluster membership

**Discovery example**:
```python
# Discovers: NFL Playoff games cluster with Tennis Grand Slams
# Reason: Both have elimination pressure + high stakes structure
# Enables: Learn from Tennis, apply to NFL
```

### 4. Temporal Derivative Transformer

**What it discovers**: Which momentum patterns predict outcomes

**Extracts**: 40 temporal features
- Velocity (rate of change)
- Acceleration (rate of velocity change)
- Trend consistency
- Regime shifts
- Momentum persistence

**Universal across**:
- Sports: Season-long momentum
- Business: Growth trajectories
- Markets: Price momentum
- Narratives: Story pacing

**Discovery example**:
```python
# Discovers: "Recent acceleration > current state" for NBA late season
# Discovers: "Velocity consistency > peak velocity" for startups
# WITHOUT assuming which temporal patterns matter
```

### 5. Meta-Feature Interaction Transformer

**What it discovers**: Which feature combinations predict outcomes

**Extracts**: 100+ interaction features
- Multiplicative (A × B)
- Ratios (A / B)
- Polynomials (A², A³)
- Synergies (A+B > individual effects)
- Antagonisms (A cancels B)

**Key insight**:
- Generates ALL possible interactions
- Learning system tests which matter
- Discovers non-obvious combinations

**Discovery example**:
```python
# Discovers: "nominative_richness × narrative_velocity" predicts outcomes
# Discovers: "momentum / variance" is key ratio
# You'd never hardcode these interactions
```

### 6. Outcome-Conditioned Archetype Transformer

**What it discovers**: Ξ (Golden Narratio) and α (optimal balance) from data

**Extracts**: 25 archetype features
- Distance to Ξ (winner archetype)
- Optimal α for domain
- Feature balance metrics
- Transfer learning features

**Key insight**:
- Learns "what winners look like" from outcomes
- No assumptions about what makes a winner
- Enables cross-domain archetype transfer

**Discovery example**:
```python
# Discovers: NBA winners cluster in 3 sub-archetypes
# Discovers: Optimal α = 0.67 for NBA (more plot than character)
# WITHOUT using theoretical formula α = 0.85 - 0.95×π
```

### 7. Anomaly Uniquity Transformer

**What it discovers**: Whether novelty helps or hurts in each domain

**Extracts**: 20 uniqueness features
- Statistical outliers
- Isolation metrics
- Historical precedent
- Novelty scores

**Universal across**:
- Startups: Novelty often predicts success
- Sports: Novelty often predicts failure (risky)
- Entertainment: Moderate novelty optimal
- Science: High novelty predicts breakthroughs

**Discovery example**:
```python
# Discovers: Startups - novelty_score > 0.7 → success
# Discovers: NBA - novelty_score > 0.7 → failure
# Learning system discovers domain-specific effects
```

---

## When to Use Each Transformer

### By Data Type

**Text data only**:
- Universal Structural Pattern (extracts from text)
- All others need pre-extracted features

**Feature data (genome)**:
- All discovery transformers work with ж features
- Recommended: Extract genome first, then discover

**Matchup/Paired data**:
- Relational Topology (requires entity pairs)

**Temporal/Sequential data**:
- Temporal Derivative (requires history)
- Universal Structural Pattern (from sequences)

**Multi-domain data**:
- Cross-Domain Embedding (needs domain labels)
- Outcome-Conditioned Archetype (cross-domain transfer)

### By Analysis Goal

**Exploratory** (no outcomes):
```python
# Understand narrative structure
structural = UniversalStructuralPatternTransformer()
features = structural.fit_transform(texts)

# Find unusual narratives
anomaly = AnomalyUniquityTransformer()
novelty = anomaly.fit_transform(genome_features)
```

**Predictive** (with outcomes):
```python
# Discover what predicts success
archetype = OutcomeConditionedArchetypeTransformer()
features = archetype.fit_transform(genome_data, y=outcomes)

# Find optimal feature combinations
interaction = MetaFeatureInteractionTransformer()
features = interaction.fit_transform(genome_features, y=outcomes)
```

**Cross-Domain Transfer**:
```python
# Find similar patterns across domains
embedding = CrossDomainEmbeddingTransformer()
features = embedding.fit_transform(multi_domain_data, y=outcomes)

# Check: Do NBA patterns transfer to NFL?
if embedding.domain_cluster_map_['nba'] similar to embedding.domain_cluster_map_['nfl']:
    # Patterns likely transfer
```

**Competitive Analysis**:
```python
# Understand matchup dynamics
topology = RelationalTopologyTransformer()
features = topology.fit_transform(matchup_pairs)

# Discover: Does asymmetry predict upsets?
```

### By Domain Type

**Individual Sports** (Golf, Tennis):
- Universal Structural Pattern (arc analysis)
- Temporal Derivative (momentum building)
- Anomaly Uniquity (unique playing styles)

**Team Sports** (NBA, NFL):
- Relational Topology (team matchups)
- Temporal Derivative (season momentum)
- Cross-Domain Embedding (transfer between sports)

**Business/Startups**:
- Temporal Derivative (growth trajectories)
- Anomaly Uniquity (disruption potential)
- Outcome-Conditioned Archetype (success patterns)

**Entertainment**:
- Universal Structural Pattern (story arcs)
- Anomaly Uniquity (originality vs conformity)
- Meta-Feature Interaction (formula discovery)

**Any Domain**:
- Meta-Feature Interaction (always useful)
- Outcome-Conditioned Archetype (if have outcomes)

---

## Integration with Existing Pipeline

### Step-by-Step Integration

**Stage 1**: Extract base genome (existing transformers)
```python
from narrative_optimization.src.transformers import *

# Base transformers (existing)
base_pipeline = [
    NominativeAnalysisTransformer(),
    LinguisticPatternsTransformer(),
    NarrativePotentialTransformer(),
    # ... other existing transformers
]

# Extract ж (genome features)
genome_features = []
for transformer in base_pipeline:
    feat = transformer.fit_transform(texts)
    genome_features.append(feat)

genome = np.hstack(genome_features)  # Combined genome
```

**Stage 2**: Apply discovery transformers
```python
# Structural discovery
structural = UniversalStructuralPatternTransformer()
structural_feat = structural.fit_transform(texts)

# Embedding discovery
embedding_data = [
    {'genome_features': g, 'domain': domain, 'text': t}
    for g, t in zip(genome, texts)
]
embedding = CrossDomainEmbeddingTransformer()
embedding_feat = embedding.fit_transform(embedding_data, y=outcomes)

# Archetype discovery (requires outcomes)
archetype = OutcomeConditionedArchetypeTransformer()
archetype_feat = archetype.fit_transform(embedding_data, y=outcomes)

# Interaction discovery
interaction = MetaFeatureInteractionTransformer()
interaction_feat = interaction.fit_transform(genome, y=outcomes)
```

**Stage 3**: Combine all features
```python
all_features = np.hstack([
    genome,               # Base genome features
    structural_feat,      # Discovered structural patterns
    embedding_feat,       # Universal embedding
    archetype_feat,       # Distance to Ξ
    interaction_feat      # Discovered interactions
])

# Now use for prediction
model = RandomForestClassifier()
model.fit(all_features, outcomes)
```

### Recommended Pipeline Order

1. **Base Extraction**: Existing transformers → ж
2. **Structural Discovery**: Universal patterns, topology
3. **Cross-Domain Discovery**: Embedding, archetypes
4. **Temporal Discovery**: If have time series data
5. **Interaction Discovery**: After all base features
6. **Anomaly Detection**: After understanding normal distribution

### Performance Optimization

**Parallel Processing**: Discovery transformers independent
```python
from joblib import Parallel, delayed

def fit_transformer(transformer, X, y=None):
    return transformer.fit_transform(X, y)

# Fit all in parallel
results = Parallel(n_jobs=4)([
    delayed(fit_transformer)(structural, texts),
    delayed(fit_transformer)(embedding, embedding_data, outcomes),
    delayed(fit_transformer)(archetype, embedding_data, outcomes),
    delayed(fit_transformer)(interaction, genome, outcomes)
])
```

**Caching**: Expensive transformers
```python
from joblib import Memory

memory = Memory('cache_dir')

@memory.cache
def get_embedding_features(genome, domain, texts, outcomes):
    embedding = CrossDomainEmbeddingTransformer()
    data = [{'genome_features': g, 'domain': d, 'text': t} 
            for g, d, t in zip(genome, domain, texts)]
    return embedding.fit_transform(data, y=outcomes)

# Subsequent calls use cache
features = get_embedding_features(genome, domain, texts, outcomes)
```

---

## Examples of Discovered Patterns

### Real Discoveries from Framework

**Pattern 1**: Structural Isomorphism
```
DISCOVERED: NFL Playoff games ≈ Tennis Grand Slam finals
STRUCTURE: Elimination pressure + high stakes + momentum critical
TRANSFER: Patterns learned from tennis apply to NFL
EVIDENCE: 78% cluster overlap in embedding space
```

**Pattern 2**: Temporal Dynamics
```
DISCOVERED: NBA late season acceleration > current win rate
STRUCTURE: velocity_recent > position_current
TRANSFER: Works in startups (growth rate > current revenue)
EVIDENCE: 81.3% prediction accuracy using acceleration
```

**Pattern 3**: Relationship Topology
```
DISCOVERED: asymmetry + complementarity → competitive balance
STRUCTURE: Unequal strengths in DIFFERENT dimensions
TRANSFER: UFC, NBA, elections all show same pattern
EVIDENCE: 0.73 correlation across domains
```

**Pattern 4**: Optimal Balance
```
DISCOVERED: NBA optimal α = 0.67 (empirical) vs 0.37 (theoretical)
STRUCTURE: Plot features 2× more effective than character
TRANSFER: Team sports generally plot-heavy (events > traits)
EVIDENCE: 15% → 42% R² improvement using discovered α
```

**Pattern 5**: Novelty Effects
```
DISCOVERED: Novelty helps startups, hurts NBA teams
STRUCTURE: Innovation vs reliability domain-dependent
STARTUPS: novelty > 0.7 → 73% success rate
NBA: novelty > 0.7 → 31% win rate
EVIDENCE: Opposite correlations, both significant
```

### How Discoveries Emerge

**Step 1**: Extract structure without assumptions
```python
# Don't assume "division games matter"
# Instead: measure all relationship properties
topology = RelationalTopologyTransformer()
features = topology.fit_transform(matchups)
```

**Step 2**: Learn which structures predict outcomes
```python
model = RandomForestClassifier()
model.fit(features, outcomes)

# Feature importance reveals:
# - asymmetry_overall: 0.32 importance
# - complementarity_index: 0.28 importance
# → Discovery: These matter!
```

**Step 3**: Validate across domains
```python
nba_correlation = correlation(features[nba_mask, 'asymmetry'], nba_outcomes)
ufc_correlation = correlation(features[ufc_mask, 'asymmetry'], ufc_outcomes)

if nba_correlation ≈ ufc_correlation:
    # Universal pattern discovered!
```

---

## Best Practices

### Do's

✅ **Extract genome first, then discover**
- Get baseline features from existing transformers
- Then apply discovery transformers to genome

✅ **Mix domains for embedding transformer**
- Cross-domain embedding works best with diverse data
- Enables structural isomorphism discovery

✅ **Use outcomes when available**
- Outcome-Conditioned Archetype REQUIRES y
- Meta-Feature Interaction benefits from y
- Cross-Domain Embedding benefits from y

✅ **Start broad, then narrow**
- Begin with universal patterns
- Then drill into specific relationships
- Discovery → validation → application

✅ **Validate discovered patterns**
- Test on hold-out data
- Check across domains
- Verify statistical significance

✅ **Save learned archetypes**
- Outcome-Conditioned stores Ξ library
- Can transfer across analyses
- Build knowledge base over time

### Don'ts

❌ **Don't assume patterns transfer**
- Test cross-domain transfer empirically
- Use embedding transformer to measure similarity
- Verify with outcomes

❌ **Don't ignore negative discoveries**
- "This doesn't predict" is valuable
- Helps eliminate false assumptions
- Focuses effort on real patterns

❌ **Don't mix discovery with confirmation bias**
- Let system find patterns
- Don't cherry-pick results
- Report all findings

❌ **Don't skip data quality**
- Discovery only as good as input data
- Garbage in = garbage out
- Clean genome features essential

❌ **Don't over-fit**
- Discovery transformers can generate many features
- Use cross-validation
- Feature selection important

### Troubleshooting

**Issue**: "No patterns discovered"
**Solutions**:
- Check data quality (sufficient samples?)
- Try different feature combinations
- May genuinely be no signal (also a finding!)

**Issue**: "Patterns don't transfer across domains"
**Solutions**:
- Check embedding similarity first
- May be genuinely domain-specific
- Look for sub-domain patterns

**Issue**: "Discovered patterns seem random"
**Solutions**:
- Validate on hold-out data
- Check for over-fitting
- Increase sample size

**Issue**: "Computational time too long"
**Solutions**:
- Use parallel processing
- Cache expensive computations
- Reduce interaction degree

---

## Summary

**Philosophy**: Discovery transformers enable the framework to LEARN patterns rather than ASSUME them.

**Key Insight**: Structure is universal, semantics are domain-specific.

**Advantage**: Patterns discovered in one domain can transfer to others via structural similarity.

**Result**: Better generalization, fewer assumptions, more robust predictions.

**Next Steps**:
1. Read DISCOVERY_TRANSFORMER_DATA_REQUIREMENTS.md for data prep
2. Start with Universal Structural Pattern (easiest)
3. Add Cross-Domain Embedding (most powerful)
4. Use Outcome-Conditioned Archetype (completes framework)
5. Experiment with others based on your data type

**Remember**: The goal is DISCOVERY, not CONFIRMATION. Let the data surprise you.

