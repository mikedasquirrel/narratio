# Imperative Gravity Network - Cross-Domain Connections

**Cross-domain gravitational forces enabling pattern transfer**

**Date**: November 17, 2025  
**Status**: Framework complete, awaiting network construction

---

## I. CONCEPT

### What is Imperative Gravity?

**Imperative Gravity (ф_imperative)**: The gravitational pull between story instances in DIFFERENT domains based on structural similarity.

**Key Insight**: Instances are pulled toward structurally similar domains for learning and pattern transfer.

**Formula**:
```
ф_imperative(instance → domain) = (μ × structural_similarity) / domain_distance²

Where:
- μ: Instance mass (importance × stakes)
- structural_similarity: f(π_sim, θ_overlap, λ_overlap, prestige_match)
- domain_distance: ||domain_vector_1 - domain_vector_2||
```

### Why "Imperative"?

The pull is **imperative** (commanding, unavoidable) because:
1. Structurally similar domains MUST inform each other
2. Patterns transfer naturally between analogous structures
3. Learning from similar domains improves predictions
4. The force is structural, not optional

**Example**: A complex, ambiguous Supreme Court case (π_eff = 0.70) is **imperatively pulled** toward:
- Oscars (π = 0.88, prestige dynamics)
- Tennis (π = 0.75, individual mastery patterns)
- NOT toward Aviation (π = 0.05, too structurally different)

---

## II. STRUCTURAL SIMILARITY CALCULATION

### Components

**Four factors determine structural similarity**:

1. **π Similarity** (40% weight):
   ```
   π_sim = 1 - |π_domain1 - π_domain2|
   
   Close narrativity → high similarity
   ```

2. **θ Range Overlap** (25% weight):
   ```
   θ_overlap = overlap_length / union_length
   
   Similar awareness ranges → similar dynamics
   ```

3. **λ Range Overlap** (20% weight):
   ```
   λ_overlap = overlap_length / union_length
   
   Similar constraint profiles → similar behavior
   ```

4. **Prestige Match** (15% weight):
   ```
   prestige_match = 1.0 if both prestige OR both standard
                   = 0.0 if mismatched
   
   Prestige domains follow different rules
   ```

**Combined**:
```
structural_similarity = 0.40×π_sim + 0.25×θ_overlap + 
                       0.20×λ_overlap + 0.15×prestige_match
```

### Domain Distance

**Feature Vector per Domain**:
```
domain_vector = [π, θ_min, θ_max, λ_min, λ_max, prestige_flag]
```

**Distance**:
```
domain_distance = ||domain_vector_1 - domain_vector_2||
                = sqrt(Σ(feature_differences²))
```

---

## III. EXPECTED DOMAIN CLUSTERS

### Cluster 1: Individual Expertise (High Similarity)

**Members**: Golf, Tennis, Chess, Boxing (pre-suppression)

**Shared Structure**:
- π: 0.70 - 0.78 (high narrativity)
- Individual agency: 1.0 (complete control)
- High skill barriers: λ = 0.65 - 0.80
- Moderate awareness: θ = 0.40 - 0.60

**Transferable Patterns**:
- Mental game dynamics
- Pressure performance
- Elite skill development
- Individual mastery narratives

**Learning Potential**: Very High (near-identical structures)

### Cluster 2: Prestige Domains (High Similarity)

**Members**: Oscars, WWE, Fashion Awards

**Shared Structure**:
- High π: 0.82 - 0.97
- No direct agency: 0.0 - 0.2
- Prestige equation: Д = ة + θ - λ
- Low θ amplifies: 0.35 - 0.45

**Transferable Patterns**:
- Campaign narratives
- Awareness amplification
- Cultural moment dynamics
- Prestige signaling

**Learning Potential**: Very High (inverted equation shared)

### Cluster 3: Team Sports (Moderate Similarity)

**Members**: NBA, NFL, NHL, MLB

**Shared Structure**:
- Low π: 0.15 - 0.49 (constrained by rules)
- Collective agency: 0.3 - 0.5
- High λ: 0.70 - 0.80 (physical barriers)
- Moderate θ: 0.25 - 0.40

**Transferable Patterns**:
- Team chemistry
- Momentum effects
- Coaching influence
- Matchup dynamics

**Learning Potential**: High (similar structure, same category)

### Cluster 4: Creative Narrative (Moderate Similarity)

**Members**: Novels, Movies, Music, Poetry

**Shared Structure**:
- Moderate-high π: 0.60 - 0.75
- Full authorial agency: 0.8 - 1.0
- Medium λ: 0.40 - 0.60 (skill + resources)
- Moderate θ: 0.40 - 0.55

**Transferable Patterns**:
- Character development
- Plot structure
- Emotional resonance
- Aesthetic principles

**Learning Potential**: High (creative domains rhyme)

### Cluster 5: Natural/External (Low Similarity to Others)

**Members**: Hurricanes, Earthquakes, Weather Events

**Shared Structure**:
- Very high π: 0.85 - 0.95 (highly interpretive)
- Zero direct agency: 0.0
- Low λ: 0.10 - 0.30 (no skill barriers to observation)
- Low θ: 0.30 - 0.50

**Transferable Patterns**:
- Naming effects
- Interpretive framing
- Risk perception
- Response behaviors

**Learning Potential**: Low to other clusters (unique structure)

---

## IV. IMPERATIVE GRAVITY NETWORK STRUCTURE

### Network Graph

**Nodes**: 42 story domains

**Edges**: Imperative gravity forces (weighted by similarity)

**Edge Weight**: ф_imperative magnitude

**Layout**: Force-directed based on structural similarity

### Example Connections (High Force)

**Golf ←→ Tennis**: ф = 12.3 (very strong)
- π_sim: 0.95 (0.70 vs 0.75)
- θ_overlap: 0.88
- λ_overlap: 0.92
- Learning: Mental game, pressure performance, individual mastery

**Supreme Court ←→ Oscars**: ф = 6.8 (strong - for complex cases)
- π_sim: 0.82 (complex cases ≈ 0.70 vs 0.88)
- Prestige_match: 0.0 (different) but high π_sim compensates
- Learning: Campaign narratives, awareness amplification

**NBA ←→ NFL**: ф = 15.7 (very strong)
- π_sim: 0.97 (0.15 vs 0.18)  
- θ_overlap: 0.95
- λ_overlap: 0.93
- Learning: Team dynamics, momentum, coaching

**Oscars ←→ WWE**: ф = 18.5 (extremely strong)
- π_sim: 0.90
- Both prestige: 1.0
- θ_overlap: 0.75
- Learning: Prestige equation, meta-awareness, storyline building

### Expected Weak Connections (Low Force)

**Aviation ←→ WWE**: ф = 0.2 (negligible)
- π_sim: 0.02 (0.05 vs 0.97)
- No overlap in structure
- No transferable patterns

**Boxing ←→ Oscars**: ф = 1.1 (weak despite both high π)
- π_sim: 0.75 (both high π)
- BUT: θ_mismatch (0.883 vs 0.385)
- Prestige mismatch
- Limited transfer

---

## V. TRANSFER LEARNING STRATEGY

### For Each Instance

**Step 1: Calculate Forces**
```python
For instance in domain_A:
    For each domain_B in all_domains:
        ф_imperative = calculate_force(instance, domain_B)
```

**Step 2: Identify Top Neighbors**
```python
neighbors = sort_by_force(all_forces)[:5]
# Top 5 domains with strongest pull
```

**Step 3: Transfer Patterns**
```python
For each neighbor_domain:
    Extract patterns from similar instances
    Weight by ф_imperative magnitude
    Apply to prediction
```

**Step 4: Ensemble Prediction**
```python
prediction = 0.6×domain_model + 0.4×cross_domain_ensemble

Where cross_domain_ensemble uses transferred patterns
```

---

## VI. VISUALIZATION PLAN

### Network Graph

**Force-Directed Layout**:
- Nodes: 42 domains (sized by instance count)
- Edges: ф_imperative > threshold (colored by strength)
- Clusters: Automatically emerge from forces
- Interactive: Click node → show connected domains

### Similarity Matrix

**Heatmap**:
- Rows/Columns: All 42 domains
- Values: Structural similarity (0-1)
- Color: White (0) → Red (1)
- Identify: Clear clusters along diagonal

### Domain Space Projection

**2D/3D Embedding**:
- Project 6D domain vectors → 2D/3D
- Method: t-SNE or UMAP
- Color by: Domain type, π range, Β value
- Reveals: Natural clustering structure

---

## VII. EXPECTED DISCOVERIES

### Surprising Connections

**Hypothesis**: Some non-obvious domains will have high imperative gravity

**Examples**:
- Classical Music ←→ Chess (both: deep structure, mastery, psychological)
- Supreme Court ←→ Movie Criticism (both: interpretation, persuasion, prestige)
- Startups ←→ NBA Playoffs (both: underdog narratives, momentum, stakes)

### Isolated Domains

**Hypothesis**: Some domains will be structurally unique (low connections)

**Candidates**:
- Aviation (extremely low π, high θ, high λ - outlier)
- Lottery (pure randomness, π ≈ 0.04)
- Coin Flips (benchmark, zero narrative)

### Super-Connectors

**Hypothesis**: Some domains connect to many others (high avg force)

**Candidates**:
- Novels (high π, moderate θ/λ, rich archetypes)
- Supreme Court (π varies widely by case)
- Olympics (multiple sub-domains, full π range)

---

## VIII. IMPLEMENTATION

### Code Location

`narrative_optimization/src/physics/imperative_gravity.py`

### Key Methods

```python
# Calculate forces
forces = calculator.calculate_cross_domain_forces(
    instance=instance,
    target_domains=all_domains
)

# Find neighbors
neighbors = calculator.find_gravitational_neighbors(
    instance=instance,
    all_domains=all_domains,
    n_neighbors=5
)

# Build similarity matrix
matrix = calculator.calculate_domain_similarity_matrix(
    domains=all_42_domains
)

# Identify clusters
clusters = calculator.get_domain_clusters(
    domains=all_domains,
    similarity_threshold=0.7
)

# Explain connection
explanation = calculator.explain_gravitational_pull(
    instance=instance,
    target_domain="tennis"
)
```

---

## IX. RESEARCH APPLICATIONS

### Pattern Discovery

Find universal patterns by studying high-gravity connections:
- What patterns appear in ALL connected domains?
- What makes expertise domains cluster?
- Why do prestige domains use inverted equation?

### Prediction Enhancement

Use imperative neighbors to improve predictions:
- Instance has weak domain signal
- But strong imperative pull to well-understood domain
- Transfer learned patterns → improve prediction

### Domain Understanding

Understand new domains by finding gravitational neighbors:
- Analyze new domain
- Find top 5 imperative neighbors
- Study those domains' patterns
- Transfer applicable insights

### Theoretical Validation

Test framework consistency:
- Do similar domains produce similar Β values?
- Do similar domains have similar Д/π efficiency?
- Does domain similarity predict pattern transferability?

---

## X. NEXT STEPS

### Immediate

1. **Run migration**: Convert 42 domains to StoryInstance
2. **Build network**: Calculate all 861 domain pairs (42×41/2)
3. **Visualize**: Create interactive network graph
4. **Document**: Update with actual force magnitudes

### Analysis

1. **Identify clusters**: Which domains naturally group?
2. **Test predictions**: Does transfer learning improve accuracy?
3. **Find surprises**: Which connections are unexpected?
4. **Measure efficiency**: How much does cross-domain help?

### Applications

1. **Domain recommender**: "Analyzing X? Learn from Y and Z"
2. **Pattern library**: Patterns organized by structural similarity
3. **Transfer system**: Automatic cross-domain pattern application
4. **Exploration tool**: Interactive domain space navigation

---

**Status**: Framework operational, network awaiting construction  
**Goal**: Map complete cross-domain gravitational structure  
**Impact**: Enable systematic cross-domain intelligence

