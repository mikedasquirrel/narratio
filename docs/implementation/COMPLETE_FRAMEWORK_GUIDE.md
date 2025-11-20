# Complete Framework Guide - Quick Start

**Everything you need to use the new narrative physics engine**

**Date**: November 17, 2025  
**Framework Version**: 2.0  
**Audience**: Researchers, developers, analysts

---

## I. OVERVIEW

### What's New in Framework 2.0

**Core Enhancements**:
1. **StoryInstance** - Complete data structure for individual narratives
2. **Instance-level π** - Narrativity varies by complexity (revolutionary)
3. **Blind Narratio (Β)** - Equilibrium ratio discovery
4. **Dual Awareness** - Resistance vs amplification effects
5. **Imperative Gravity** - Cross-domain forces and learning
6. **Concurrent Narratives** - Multi-stream integration

**What This Means**:
- More accurate predictions (instance-specific analysis)
- Cross-domain intelligence (learn from similar domains)
- Complete traceability (all instances in repository)
- Theoretical validation (test equilibrium hypotheses)

---

## II. QUICK START - Basic Usage

### 1. Analyze a Single Narrative

```python
from narrative_optimization.src.core.story_instance import StoryInstance
from narrative_optimization.src.config.domain_config import DomainConfig
from narrative_optimization.src.analysis.complexity_scorer import ComplexityScorer

# Create instance
instance = StoryInstance(
    instance_id="tiger_2019_masters",
    domain="golf",
    narrative_text="""
    Tiger Woods' comeback at the 2019 Masters was one of golf's 
    greatest redemption stories. After years of injuries and personal 
    struggles, the 43-year-old showed mental toughness and clutch 
    performance under championship pressure...
    """
)

# Calculate complexity
scorer = ComplexityScorer(domain="golf")
complexity = scorer.calculate_complexity(instance)
print(f"Complexity: {complexity:.3f}")

# Calculate π_effective
config = DomainConfig("golf")
pi_eff = config.calculate_effective_pi(complexity)
instance.pi_effective = pi_eff
print(f"π_effective: {pi_eff:.3f} (base: {config.get_pi():.3f})")

# Show complexity factors
for factor, value in instance.complexity_factors.items():
    print(f"  {factor}: {value:.3f}")
```

**Output**:
```
Complexity: 0.42
π_effective: 0.784 (base: 0.700)
  evidence_ambiguity: 0.15
  precedent_clarity_inverse: 0.35
  instance_novelty: 0.65
  factual_disputes: 0.10
  outcome_variance: 0.45
```

### 2. Calculate Blind Narratio for Domain

```python
from narrative_optimization.src.analysis.blind_narratio_calculator import BlindNarratioCalculator

# Load all instances for domain
instances = []  # ... load your instances

# Calculate domain Β
calculator = BlindNarratioCalculator()
result = calculator.calculate_domain_blind_narratio(
    instances=instances,
    domain_name="golf"
)

print(f"Domain Β: {result['Β']:.3f}")
print(f"Stability: {result['stability']:.3f}")
print(f"Instances analyzed: {result['n_instances']}")

# Check variance by complexity
if result['variance_by_context']:
    for complexity_level, data in result['variance_by_context'].items():
        print(f"{complexity_level}: Β={data['beta_mean']:.3f} (n={data['n']})")
```

**Output**:
```
Domain Β: 0.731
Stability: 0.882
Instances analyzed: 150
low_complexity: Β=0.654 (n=50)
mid_complexity: Β=0.728 (n=50)
high_complexity: Β=0.811 (n=50)
```

### 3. Find Cross-Domain Gravitational Neighbors

```python
from narrative_optimization.src.physics.imperative_gravity import ImperativeGravityCalculator

# Initialize with all domain configs
all_configs = {
    'golf': DomainConfig('golf'),
    'tennis': DomainConfig('tennis'),
    'oscars': DomainConfig('oscars'),
    # ... load all 42
}

gravity_calc = ImperativeGravityCalculator(all_configs)

# Find neighbors
neighbors = gravity_calc.find_gravitational_neighbors(
    instance=instance,
    all_domains=list(all_configs.keys()),
    n_neighbors=5,
    exclude_same_domain=True
)

print("Top 5 Imperative Neighbors:")
for domain, force in neighbors:
    explanation = gravity_calc.explain_gravitational_pull(instance, domain)
    print(f"  {domain}: force={force:.3f}")
    print(f"    {explanation['learning_potential']}")
```

**Output**:
```
Top 5 Imperative Neighbors:
  tennis: force=12.34
    High - domains are highly analogous, direct pattern transfer likely effective
  chess: force=8.67
    Good - substantial overlap, many transferable patterns
  oscars: force=3.21
    Moderate - some transferable concepts but requires adaptation
  boxing: force=2.45
    Moderate - some transferable concepts but requires adaptation
  supreme_court: force=1.89
    Low - limited transferability, mainly conceptual insights
```

### 4. Use the Instance Repository

```python
from narrative_optimization.src.data.instance_repository import InstanceRepository

# Create/load repository
repo = InstanceRepository()

# Add instance
repo.add_instance(instance)

# Query by structure
similar_instances = repo.query_by_structure(
    pi_range=(0.65, 0.85),  # Similar narrativity
    theta_range=(0.40, 0.60),  # Similar awareness
    exclude_domain="golf"  # From other domains
)

print(f"Found {len(similar_instances)} structurally similar instances")

# Find imperative neighbors
neighbors = repo.calculate_imperative_neighbors(
    instance=instance,
    n_neighbors=10
)

for neighbor, similarity in neighbors[:5]:
    print(f"  {neighbor.domain}::{neighbor.instance_id} (sim={similarity:.3f})")

# Get domain statistics
stats = repo.get_domain_statistics("golf")
print(f"\nGolf domain: {stats['count']} instances")
print(f"  π mean: {stats['pi_mean']:.3f}")
print(f"  Β mean: {stats['beta_mean']:.3f}")
print(f"  Success rate: {stats['success_rate']:.1%}")

# Save to disk
repo.save_to_disk()
```

---

## III. ADVANCED USAGE

### Analyze π Variance Within Domain

```python
from narrative_optimization.src.analysis.dynamic_narrativity import DynamicNarrativityAnalyzer

# Analyze domain
analyzer = DynamicNarrativityAnalyzer(domain_config=config)
result = analyzer.analyze_pi_variance(
    instances=instances,
    domain_name="supreme_court"
)

# Check if variance is significant
if result['pi_variance_significant']:
    print(f"✓ Significant π variance detected")
    print(f"  π range: [{result['pi_range'][0]:.3f}, {result['pi_range'][1]:.3f}]")
    print(f"  Complexity-π correlation: {result['complexity_pi_correlation']:.3f}")
    
    # Examine by complexity tertile
    for tertile, data in result['complexity_pi_relationship'].items():
        print(f"  {tertile}: π={data['pi_mean']:.3f} (n={data['n_instances']})")

# Generate report
print(analyzer.generate_report())

# Visualize
analyzer.visualize_pi_distribution(
    instances=instances,
    domain_name="supreme_court",
    output_path="supreme_court_pi_variance.png"
)
```

### Extract Awareness Amplification Features

```python
from narrative_optimization.src.transformers.awareness_amplification import AwarenessAmplificationTransformer

# Create transformer
awareness_trans = AwarenessAmplificationTransformer()

# Fit on corpus
awareness_trans.fit(all_narratives)

# Extract features
features = awareness_trans.transform([instance.narrative_text])

# Get feature breakdown
feature_names = awareness_trans.get_feature_names()
for name, value in zip(feature_names, features[0]):
    if value > 0.01:  # Only show non-zero
        print(f"  {name}: {value:.3f}")

# Calculate amplification effect
amplification = awareness_trans.calculate_amplification_effect(
    amplification_score=features[0][-1],  # Last feature is aggregate
    potential_energy=0.75,  # From NarrativePotentialTransformer
    consciousness=1.0  # Explicitly recognized
)

print(f"\nAmplification multiplier: {amplification:.2f}x")
```

### Build Full Genome with All Components

```python
from narrative_optimization.src.config.genome_structure import CompleteGenomeExtractor
from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer
from narrative_optimization.src.transformers.archetypes.archetypal_distance import ArchetypalDistanceTransformer

# Create extractors
nom_trans = NominativeAnalysisTransformer()
arch_trans = ArchetypalDistanceTransformer(domain_config=config)

# Create complete extractor
extractor = CompleteGenomeExtractor(
    nominative_transformer=nom_trans,
    archetypal_transformer=arch_trans,
    domain_config=config,
    complexity_scorer=complexity_scorer
)

# Fit on historical data
extractor.fit(
    texts=historical_narratives,
    outcomes=historical_outcomes,
    timestamps=historical_timestamps
)

# Extract for new instance
genome, metadata = extractor.transform(
    texts=[instance.narrative_text],
    return_metadata=True
)

# Store on instance
instance.genome_full = genome[0]
instance.pi_effective = metadata['pi_effective'][0]
instance.complexity_factors = metadata['complexity_factors'][0]

# Get feature names
feature_names = extractor.get_feature_names()
print(f"Genome dimensions: {genome.shape[1]}")
print(f"  Nominative: {len(feature_names['nominative'])}")
print(f"  Archetypal: {len(feature_names['archetypal'])}")
print(f"  Historial: {len(feature_names['historial'])}")
print(f"  Uniquity: {len(feature_names['uniquity'])}")
```

---

## IV. DOMAIN MIGRATION

### Migrate Existing Domain to New Framework

```python
from narrative_optimization.scripts.migrate_domains_to_story_instance import DomainMigrator

# Initialize migrator
domains_dir = Path('narrative_optimization/domains')
migrator = DomainMigrator(domains_dir)

# Migrate single domain
result = migrator.migrate_domain(
    domain_name="golf",
    verbose=True
)

print(f"Migrated: {result['instances_migrated']} instances")
print(f"Domain Β: {result['blind_narratio']:.3f}")

# Access migrated instances from repository
golf_instances = migrator.repository.get_instances_by_domain("golf")
```

### Migrate All Domains

```python
# Migrate test domains first
test_domains = ['golf', 'supreme_court', 'boxing', 'tennis', 'oscars']
results = migrator.migrate_all_domains(
    domain_list=test_domains,
    verbose=True
)

# Then migrate all 42 domains
# results = migrator.migrate_all_domains(verbose=True)

# Export report
migrator.export_migration_report('migration_report.json')
```

---

## V. WORKING WITH STORY INSTANCES

### Create from Scratch

```python
instance = StoryInstance(
    instance_id="unique_id",
    domain="domain_name",
    narrative_text="The narrative...",
    outcome=1.0,  # Success
    timestamp=datetime.now(),
    context={
        'stakes': 'championship',
        'complexity': 'high',
        'type': 'redemption'
    }
)

# Set genome components
instance.set_genome_components(
    nominative=nom_features,
    archetypal=arch_features,
    historial=hist_features,
    uniquity=uniq_features
)

# Calculate properties
instance.importance_score = 1.5  # Above average importance
instance.stakes_multiplier = 2.5  # Championship level
instance.calculate_mass()  # μ = 1.5 × 2.5 = 3.75

# Set dynamic properties
instance.pi_effective = 0.75
instance.pi_domain_base = 0.70
instance.blind_narratio = 0.82

# Set awareness
instance.theta_resistance = 0.45
instance.theta_amplification = 0.65
```

### Save and Load

```python
# Save to JSON
instance.save('instances/golf/tiger_2019_masters.json')

# Load from JSON
loaded = StoryInstance.load('instances/golf/tiger_2019_masters.json')

print(loaded)  # StoryInstance(id='tiger_2019_masters', domain='golf', ю=0.94, ❊=1.0, π_eff=0.75)
```

### Query Repository

```python
# Load repository
repo = InstanceRepository()

# Get all instances from domain
golf_instances = repo.get_instances_by_domain("golf")

# Query by π range
high_pi_instances = repo.query_by_structure(
    pi_range=(0.7, 1.0),
    exclude_domain=None
)

# Get specific instance
instance = repo.get_instance("tiger_2019_masters")

# Get statistics
all_stats = repo.get_repository_statistics()
golf_stats = repo.get_domain_statistics("golf")
```

---

## VI. ANALYSIS WORKFLOWS

### Workflow 1: Complete Single Instance Analysis

```python
# 1. Create instance
instance = StoryInstance(instance_id="case_id", domain="supreme_court", ...)

# 2. Calculate complexity
complexity = complexity_scorer.calculate_complexity(instance)

# 3. Calculate π_effective
pi_eff = config.calculate_effective_pi(complexity)
instance.pi_effective = pi_eff

# 4. Extract genome (full pipeline)
genome, metadata = genome_extractor.transform([text], return_metadata=True)
instance.genome_full = genome[0]

# 5. Calculate story quality
quality = story_quality_calc.compute_ю_with_dynamic_pi(
    genome, feature_names, metadata['pi_effective']
)
instance.story_quality = quality[0]

# 6. Calculate Blind Narratio
beta = beta_calc.calculate_instance_blind_narratio(instance)

# 7. Find imperative neighbors
neighbors = imperative_calc.find_gravitational_neighbors(instance, all_domains)
for domain, force in neighbors[:3]:
    instance.add_imperative_gravity(domain, "aggregate", force)

# 8. Store in repository
repo.add_instance(instance)

# 9. Save
repo.save_to_disk()
```

### Workflow 2: Domain-Level Analysis

```python
# 1. Load/create all instances for domain
instances = [...] # Your instances

# 2. Analyze π variance
pi_analysis = dynamic_pi_analyzer.analyze_pi_variance(
    instances=instances,
    domain_name="domain_name"
)

# 3. Calculate domain Β
beta_result = beta_calculator.calculate_domain_blind_narratio(
    instances=instances,
    domain_name="domain_name"
)

# 4. Build domain similarity connections
similarity_matrix = imperative_calc.calculate_domain_similarity_matrix(
    domains=all_domain_names
)

# 5. Identify clusters
clusters = imperative_calc.get_domain_clusters(
    domains=all_domain_names,
    similarity_threshold=0.7
)

# 6. Generate reports
print(pi_analyzer.generate_report())
print(beta_calculator.summarize_results())

# 7. Export results
pi_analyzer.export_results('domain_pi_analysis.json')
beta_calculator.export_results('domain_beta_results.json')
```

### Workflow 3: Cross-Domain Analysis

```python
# 1. Load instance from one domain
instance = repo.get_instance("complex_supreme_court_case")

# 2. Find structurally similar domains
neighbors = imperative_calc.find_gravitational_neighbors(
    instance=instance,
    all_domains=all_42_domains,
    n_neighbors=10
)

# 3. For each neighbor, find similar instances
for neighbor_domain, force in neighbors[:5]:
    similar = repo.query_by_structure(
        pi_range=(instance.pi_effective - 0.1, instance.pi_effective + 0.1),
        theta_range=(instance.theta_resistance - 0.1, instance.theta_resistance + 0.1),
        exclude_domain=instance.domain
    )
    
    # Filter to neighbor domain
    neighbor_instances = [s for s in similar if s.domain == neighbor_domain]
    
    print(f"\n{neighbor_domain} (force={force:.2f}):")
    print(f"  {len(neighbor_instances)} similar instances found")
    
    # Analyze patterns in neighbor domain
    if neighbor_instances:
        avg_quality = np.mean([s.story_quality for s in neighbor_instances])
        avg_outcome = np.mean([s.outcome for s in neighbor_instances])
        print(f"  Avg quality: {avg_quality:.3f}")
        print(f"  Avg outcome: {avg_outcome:.3f}")

# 4. Transfer learning (future enhancement)
# Use patterns from neighbors to improve prediction
```

---

## VII. TROUBLESHOOTING

### Common Issues

**Issue 1: "π_effective is None"**

```python
# Solution: Calculate complexity first
complexity = complexity_scorer.calculate_complexity(instance)
instance.pi_effective = config.calculate_effective_pi(complexity)
```

**Issue 2: "Genome has wrong dimensions"**

```python
# Solution: Use CompleteGenomeExtractor, not individual transformers
extractor = CompleteGenomeExtractor(...)
genome, metadata = extractor.transform(texts, return_metadata=True)
```

**Issue 3: "Repository queries return nothing"**

```python
# Solution: Rebuild indices after bulk add
repo.add_instances_bulk(instances)  # This rebuilds automatically
# OR
repo._rebuild_indices()  # Manual rebuild
```

**Issue 4: "Blind Narratio is infinity"**

```python
# This means free_will_forces ≈ 0
# Check: θ_resistance and agency values
# Some domains may have no resistance (pure determinism)
```

---

## VIII. BEST PRACTICES

### Instance Creation

1. **Use descriptive IDs**: "tiger_2019_masters" not "instance_1"
2. **Store context**: Include stakes, type, circumstances
3. **Set timestamps**: Enables temporal analysis
4. **Calculate mass**: Important for gravity calculations

### Complexity Scoring

1. **Use domain-specific scorers**: Different domains weight factors differently
2. **Check complexity_factors**: Understand what makes instance complex
3. **Validate range**: complexity should be [0, 1]

### Repository Management

1. **Bulk operations**: Use `add_instances_bulk()` for efficiency
2. **Save regularly**: Call `save_to_disk()` after major updates
3. **Index awareness**: Indices auto-rebuild, but can manually trigger
4. **Cache hits**: Similarity calculations cached automatically

### Force Calculations

1. **Set mass first**: Required for meaningful gravity calculations
2. **Use domain configs**: Provides structural similarity data
3. **Check cache**: Similar pairs cached for speed
4. **Interpret magnitudes**: Force > 5.0 = strong, > 10.0 = very strong

---

## IX. PERFORMANCE TIPS

### For Large Datasets

```python
# 1. Use batch operations
repo.add_instances_bulk(instances)  # Not one at a time

# 2. Disable verbose in loops
for instance in instances:
    migrator.migrate_domain(domain, verbose=False)

# 3. Cache embeddings
# Embeddings automatically cached by EmbeddingManager

# 4. Save periodically
if i % 1000 == 0:
    repo.save_to_disk()
```

### Memory Management

```python
# For very large genomes (>100MB sparse):
# CompleteGenomeExtractor handles this automatically
# Uses sparse matrices when appropriate

# For repository with 10,000+ instances:
# Similarity cache may grow large
# Clear periodically:
repo.similarity_cache.clear()
```

---

## X. VALIDATION CHECKLIST

### Before Analysis

- [ ] Domain config loaded
- [ ] Complexity scorer initialized
- [ ] All transformers fitted
- [ ] π_base confirmed
- [ ] β (sensitivity) set appropriately

### After Instance Creation

- [ ] instance.pi_effective calculated
- [ ] instance.complexity_factors populated
- [ ] instance.genome_full exists
- [ ] instance.mass calculated
- [ ] instance.story_quality computed

### Before Using Repository

- [ ] Repository initialized
- [ ] Instances added
- [ ] Indices built
- [ ] Saved to disk
- [ ] Statistics validated

---

## XI. NEXT STEPS

### For Researchers

1. **Test hypotheses**: Run analyses on your domains
2. **Calculate Β**: Discover equilibrium ratios
3. **Measure π variance**: Test instance-level variation
4. **Build networks**: Map imperative gravity connections
5. **Publish findings**: Document discoveries

### For Developers

1. **Integrate**: Add StoryInstance to your pipelines
2. **Extend**: Build on the framework
3. **Visualize**: Create tools for exploration
4. **Optimize**: Improve performance for scale
5. **Document**: Share your enhancements

### For Analysts

1. **Migrate**: Convert existing analyses
2. **Re-run**: Test with new framework
3. **Compare**: Measure improvements
4. **Explore**: Use cross-domain intelligence
5. **Report**: Generate insights

---

## XII. RESOURCES

### Documentation

- `THEORETICAL_FRAMEWORK.md` - Complete formal system
- `FORMAL_VARIABLE_SYSTEM.md` - Variable definitions
- `NARRATIVE_CATALOG.md` - Universal patterns
- `BLIND_NARRATIO_RESULTS.md` - Β analysis
- `IMPERATIVE_GRAVITY_NETWORK.md` - Cross-domain forces

### Code Locations

- Core: `narrative_optimization/src/core/`
- Analysis: `narrative_optimization/src/analysis/`
- Transformers: `narrative_optimization/src/transformers/`
- Physics: `narrative_optimization/src/physics/`
- Scripts: `narrative_optimization/scripts/`

### Example Domains

- Golf: `narrative_optimization/domains/golf/`
- Supreme Court: `narrative_optimization/domains/supreme_court/`
- Tennis: `narrative_optimization/domains/tennis/`

---

**Status**: Framework operational and ready for use  
**Support**: See documentation or examine example analyses  
**Questions**: Refer to THEORETICAL_FRAMEWORK.md for complete formal system

