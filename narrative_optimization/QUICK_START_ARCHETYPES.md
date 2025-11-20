# Quick Start: Domain-Specific Archetypes

Get started with the new domain-specific Ξ architecture in 5 minutes.

---

## 1. Basic Usage (Existing Domain)

```python
from src.analysis import DomainSpecificAnalyzer

# Analyze golf narratives
analyzer = DomainSpecificAnalyzer('golf')

results = analyzer.analyze_complete(
    texts=['Tiger Woods showing incredible mental toughness...',
           'Rookie displays world-class ball striking...'],
    outcomes=[1, 15]  # Finish positions
)

print(f"R²: {results['r_squared']:.1%}")
print(f"Д: {results['delta']:.3f}")
print(f"Story quality: {results['story_quality'].mean():.3f}")
```

---

## 2. Discover New Archetypes

```python
from src.config import ArchetypeDiscovery

discovery = ArchetypeDiscovery()

# Discover from winner texts
archetypes = discovery.discover_archetypes(
    winner_texts=[...],  # List of successful narratives
    n_archetypes=5
)

# View discovered patterns
for name, data in archetypes.items():
    print(f"\n{name}:")
    print(f"  Patterns: {data['patterns'][:5]}")
    print(f"  Coherence: {data['coherence']:.2f}")
```

---

## 3. Add New Domain

**Step 1**: Add to `src/config/domain_archetypes.py`:

```python
DOMAIN_ARCHETYPES['soccer'] = {
    'archetype_patterns': {
        'tactical_awareness': ['positioning', 'formation', 'strategy'],
        'individual_skill': ['dribbling', 'passing', 'finishing'],
        'team_cohesion': ['chemistry', 'communication', 'unity'],
        'momentum': ['confidence', 'flowing', 'rhythm']
    },
    'archetype_weights': {
        'tactical_awareness': 0.30,
        'individual_skill': 0.30,
        'team_cohesion': 0.25,
        'momentum': 0.15
    },
    'nominative_richness_requirement': 25,
    'pi': 0.68
}
```

**Step 2**: Use immediately:

```python
analyzer = DomainSpecificAnalyzer('soccer')
results = analyzer.analyze_complete(texts, outcomes)
```

---

## 4. Extract Complete Genome (with Historial + Uniquity)

```python
from src.config import CompleteGenomeExtractor
from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.archetypes import GolfArchetypeTransformer

# Create extractor
extractor = CompleteGenomeExtractor(
    nominative_transformer=NominativeAnalysisTransformer(),
    archetypal_transformer=GolfArchetypeTransformer()
)

# Fit on historical data
extractor.fit(
    texts=historical_narratives,
    outcomes=historical_outcomes,
    timestamps=dates  # Optional
)

# Extract complete genomes
genomes = extractor.transform(new_narratives)

# Access components
genome_struct = extractor.genome_structure
nominative = genome_struct.get_nominative(genomes[0])
archetypal = genome_struct.get_archetypal(genomes[0])
historial = genome_struct.get_historial(genomes[0])  # Historical weight
uniquity = genome_struct.get_uniquity(genomes[0])    # Rarity/novelty
```

---

## 5. Discover and Validate New Patterns

```python
from src.config import ArchetypeValidator

# Test if patterns are predictive
validator = ArchetypeValidator()

results = validator.validate_archetype(
    archetype_patterns=['clutch', 'pressure', 'composure'],
    texts=all_texts,
    outcomes=all_outcomes
)

if results['is_significant']:
    print(f"✓ Valid archetype! (r={results['correlation']:.3f}, p={results['p_value']:.4f})")
else:
    print("✗ Not significant - refine patterns")
```

---

## 6. Discover Contextual Boosters

```python
from src.config import ArchetypeDiscovery

discovery = ArchetypeDiscovery()

# Find what contexts boost an archetype
boosters = discovery.discover_contextual_boosters(
    texts=all_texts,
    outcomes=all_outcomes,
    archetype_patterns=['mental', 'game', 'composure']
)

# Results: {'championship': 1.3, 'major': 1.25, 'playoff': 1.2, ...}
for context, multiplier in boosters.items():
    boost_pct = (multiplier - 1) * 100
    print(f"'{context}' boosts by {boost_pct:.1f}%")
```

---

## 7. Register and Use Custom Archetypes

```python
from src.config import ArchetypeRegistry

registry = ArchetypeRegistry()

# Register new archetype
registry.register_archetype(
    name='clutch_performance',
    patterns=['pressure', 'big moment', 'crunch time', 'delivered'],
    domain='basketball',
    weight=0.35,
    description='Performs under pressure in critical moments'
)

# Register sub-archetype
registry.register_sub_archetype(
    parent='clutch_performance',
    name='fourth_quarter_clutch',
    patterns=['fourth quarter', 'final minutes', 'game on line'],
    domain='basketball'
)

# Add contextual booster
registry.add_contextual_booster(
    archetype='clutch_performance',
    context='playoffs',
    boost_multiplier=1.4,
    domain='basketball'
)

# Export registry
import json
with open('my_archetypes.json', 'w') as f:
    json.dump(registry.export_to_dict(), f, indent=2)
```

---

## 8. Command-Line Discovery

```bash
# Discover main archetypes
python tools/discover_archetypes.py \
    --discover \
    --file data/winners.txt \
    --n-archetypes 5 \
    --domain basketball

# Discover sub-archetypes
python tools/discover_archetypes.py \
    --discover-sub clutch_performance \
    --file data/all_games.txt

# Test new patterns
python tools/discover_archetypes.py \
    --test "pressure" "big moment" "delivered" \
    --file data/games_with_outcomes.json \
    --name clutch_test

# Register interactively
python tools/discover_archetypes.py --register
```

---

## 9. Compare Domains

```python
# Analyze multiple domains
domains = ['golf', 'tennis', 'boxing', 'nba', 'wwe']

for domain in domains:
    analyzer = DomainSpecificAnalyzer(domain)
    results = analyzer.analyze_complete(texts[domain], outcomes[domain])
    
    print(f"\n{domain.upper()}:")
    print(f"  R²: {results['r_squared']:.1%}")
    print(f"  Д: {results['delta']:.3f}")
    print(f"  Efficiency: {results['efficiency']:.3f}")
    print(f"  Passes threshold: {results['passes_threshold']}")
```

---

## 10. Access Historial and Uniquity Features

```python
# After running analysis
results = analyzer.analyze_complete(texts, outcomes, timestamps=dates)

# Historial features (historical positioning)
historial = results['historial_features']
print("Historical positioning:")
print(f"  Distance to winners: {historial[:, 0].mean():.3f}")
print(f"  Historical gravity: {historial[:, 6].mean():.3f}")
print(f"  Pattern recency: {historial[:, 7].mean():.3f}")

# Uniquity features (rarity/novelty)
uniquity = results['uniquity_features']
print("\nUniquity (rarity):")
print(f"  Rarity score: {uniquity[:, 0].mean():.3f}")
print(f"  Novelty gradient: {uniquity[:, 2].mean():.3f}")
print(f"  Uniquity constant: {uniquity[:, 4].mean():.3f}")
```

---

## What's Different?

**OLD WAY**: Generic features, same patterns for all domains
```python
# Applied same 33 transformers everywhere
# TF-IDF often won
# No domain-specific knowledge
```

**NEW WAY**: Domain-specific Ξ, learned from winners
```python
# Each domain has its own perfect story archetype
# Story quality = distance from THAT domain's Ξ
# Includes historical + uniquity dimensions
# Easily discover and add new patterns
```

---

## Troubleshooting

**Q: "My domain isn't improving"**  
A: Try discovering archetypes from YOUR winner data:
```python
discovery = ArchetypeDiscovery()
archetypes = discovery.discover_archetypes(winner_texts, n_archetypes=5)
# Use these patterns instead of guessing
```

**Q: "How do I know if patterns are good?"**  
A: Validate them:
```python
validator = ArchetypeValidator()
results = validator.validate_archetype(patterns, texts, outcomes)
print(f"Significant: {results['is_significant']}")
```

**Q: "Can I have sub-sub-archetypes?"**  
A: Yes! Register hierarchically:
```python
registry.register_sub_archetype(parent='mental_game', name='pressure_performance', ...)
registry.register_sub_archetype(parent='pressure_performance', name='championship_pressure', ...)
```

---

## Next Steps

1. **Try existing domains**: `analyzer = DomainSpecificAnalyzer('golf')`
2. **Discover patterns**: Use `ArchetypeDiscovery` on your data
3. **Add your domain**: Edit `domain_archetypes.py`
4. **Validate**: Run `validate_domain_archetypes.py`

**Full documentation**: See `DOMAIN_ARCHETYPE_SYSTEM.md`

