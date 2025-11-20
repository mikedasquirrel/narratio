# Narrative Optimization Framework

**A holistic learning system for discovering and analyzing narrative patterns across domains**

---

## What This System Does

The Narrative Optimization Framework discovers **what makes stories work** in any domain - from sports to startups, from weather events to entertainment. It learns patterns that predict outcomes, identifies universal archetypes, and measures how narrative quality influences real-world results.

### Core Question

> **"What are the stories of this domain, checking for familiar stories from structurally similar domains? Do they unfold at predicted frequency (accounting for observation bias)? Does a pattern emerge among trends towards story realization?"**

---

## Quick Start

### Adding a New Domain

```bash
python MASTER_INTEGRATION.py chess data/domains/chess_games.json --pi 0.78 --type individual_expertise
```

This will:
1. Check for universal stories (underdog, comeback, rivalry, etc.)
2. Find structurally similar domains
3. Transfer relevant patterns
4. Learn domain-specific patterns
5. Analyze story frequency vs predictions
6. Identify emerging trends

### Running Analysis

```python
from src.learning import LearningPipeline
from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer

# Initialize learning pipeline
pipeline = LearningPipeline()

# Ingest data
pipeline.ingest_domain('golf', golf_texts, golf_outcomes)
pipeline.ingest_domain('tennis', tennis_texts, tennis_outcomes)

# Learn patterns
metrics = pipeline.learn_cycle(
    domains=['golf', 'tennis'],
    learn_universal=True,
    learn_domain_specific=True
)

# Analyze specific domain
analyzer = DomainSpecificAnalyzer('golf')
results = analyzer.analyze_complete(texts, outcomes)

print(f"R²: {results['r_squared']:.1%}")
print(f"Story Quality (ю): {results['story_quality'].mean():.3f}")
```

---

## System Architecture

### Three-Layer Learning System

```
┌─────────────────────────────────────────────────────────────┐
│                   UNIVERSAL LAYER                            │
│  Cross-domain patterns (underdog, comeback, pressure, etc.) │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   DOMAIN LAYER                               │
│  Domain-specific patterns (e.g., golf: course mastery)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   CONTEXT LAYER                              │
│  Situational patterns (high-stakes, regular season, etc.)   │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **Learning Pipeline** (`src/learning/`): Continuous learning from data
- **Domain Analyzers** (`src/analysis/`): Domain-specific analysis
- **Transformers** (`src/transformers/`): Feature extraction (56 transformers)
- **Archetypes** (`src/config/`): Pattern definitions and discovery
- **Visualization** (`src/visualization/`): Pattern visualization tools

---

## Core Concepts

### The Genome (ж)

Every narrative instance has a **genome** - its complete feature vector:

```
ж = [nominative, archetypal, historial, uniquity]
```

- **Nominative**: Named entities (players, venues, events)
- **Archetypal**: Pattern matching (universal + domain-specific)
- **Historial**: Narrative lineage and precedent
- **Uniquity**: Rarity and novelty

### Story Quality (ю)

Computed from the genome, measures how well a narrative matches domain archetypes:

```
ю = distance_from_Ξ(ж)
```

Where Ξ is the domain's "golden narratio" - the ideal archetype.

### Narrative Agency (Д)

The bridge between story quality and outcomes:

```
Д = п × |r| × κ    (regular domains)
Д = ة + θ - λ      (prestige domains)
```

Where:
- п (pi): Narrativity (how open domain is to narrative)
- ة (taa): Nominative gravity
- θ (theta): Awareness resistance
- λ (lambda): Skill/constraint floor

---

## Adding a New Domain

### Step 1: Prepare Data

Create a JSON file with:

```json
{
  "texts": ["narrative 1", "narrative 2", ...],
  "outcomes": [1, 0, 1, ...],
  "names": ["entity1", "entity2", ...],
  "timestamps": [1234567890, ...]
}
```

### Step 2: Run Integration

```bash
python MASTER_INTEGRATION.py your_domain data/domains/your_data.json
```

### Step 3: Review Results

Check `narrative_optimization/domains/your_domain/`:
- `integration_results.json`: Full analysis results
- `ANALYSIS_REPORT.md`: Human-readable report

### Step 4: Refine (Optional)

1. Add domain-specific archetype transformer in `src/transformers/archetypes/`
2. Add domain config in `src/config/domain_archetypes.py`
3. Re-run analysis

---

## Project Structure

```
narrative_optimization/
├── src/
│   ├── learning/           # Learning system (14 modules)
│   ├── analysis/           # Analysis tools
│   ├── transformers/       # 56 narrative transformers
│   ├── config/             # Domain configurations
│   ├── visualization/      # Visualization tools
│   ├── optimization/       # Caching & profiling
│   └── data/              # Data processing
├── domains/               # Domain-specific analyses
├── examples/              # Usage examples
├── tests/                 # Test suite
└── MASTER_INTEGRATION.py  # Main integration script
```

---

## Learning System Features

### Continuous Learning
- Patterns improve automatically from new data
- No retraining needed (incremental)
- Statistical validation of all patterns

### Multi-Level Patterns
- **Universal**: Work across all domains
- **Domain-specific**: Unique to each domain
- **Sub-patterns**: Fine-grained hierarchies

### Meta-Learning
- Transfer patterns between similar domains
- Few-shot learning for new domains
- Zero-shot prediction using universal patterns

### Advanced Capabilities
- **Active Learning**: Focus on uncertain patterns
- **Ensemble Methods**: Multiple hypotheses
- **Online Learning**: Real-time updates
- **Causal Discovery**: Identify causal patterns
- **Context-Aware**: Situational adaptation

---

## Current Domains

**12 Integrated Domains**:
- Golf (R²=97.7%)
- Tennis (R²=93%)
- Boxing
- NBA
- WWE
- Chess
- Oscars
- Crypto
- Mental Health
- Startups
- Hurricanes
- Housing

**30+ Additional Domains** with existing data ready for integration

---

## Performance

### Learning Metrics
- Pattern discovery: < 1 minute per domain
- Validation: Statistical significance testing
- Improvement tracking: Version control & A/B testing

### Analysis Speed
- Single prediction: < 1ms
- Full domain analysis: < 10 seconds
- Caching enabled for repeated queries

---

## Documentation

- **`DOMAIN_ARCHETYPE_SYSTEM.md`**: Technical details on archetype system
- **`QUICK_START_ARCHETYPES.md`**: Guide to archetype usage
- **`docs/architecture.md`**: System architecture
- **`docs/findings.md`**: Research findings
- **`docs/SUPERVISED_TRANSFORMER_HANDOFF.md`**: Instructions for wiring supervised / genome-dependent transformers via a dedicated pipeline

### Transformer Registry Sync

- Whenever you add a new transformer module under `src/transformers/`, you **must** register it in both `transformers/__init__.py` (for lazy loading) and `src/transformers/transformer_selector.py`.
- The selector now auto-checks coverage at import time; a warning means a transformer exists but is not part of any selection pool.
- This safeguard prevents downstream jobs (like the NBA rebuild) from silently missing the latest narrative features.
- Some advanced transformers require supervised labels or special “genome” inputs; the feature-extraction pipeline automatically skips them with a clear log (`⊘ transformer: reason`). Provide the required inputs (or move the transformer to a supervised pipeline) before enabling them.

---

## Examples

### Example 1: Discover Universal Patterns

```python
from src.learning import UniversalArchetypeLearner

learner = UniversalArchetypeLearner()
patterns = learner.discover_patterns(all_texts, all_outcomes)

for name, data in patterns.items():
    print(f"{name}: {data['description']}")
    print(f"  Frequency: {data['frequency']:.1%}")
    print(f"  Win rate: {data['win_rate']:.1%}")
```

### Example 2: Transfer Learning

```python
from src.learning import MetaLearner

meta = MetaLearner()

# Find similar domains
similar = meta.find_similar_domains('chess', n_similar=3)
# Returns: [('tennis', 0.72), ('golf', 0.68), ...]

# Transfer patterns
transferred = meta.transfer_patterns('golf', 'chess', min_transferability=0.6)
```

### Example 3: Visualize Patterns

```python
from src.visualization import PatternVisualizer

viz = PatternVisualizer()

# Visualize pattern space
viz.visualize_pattern_space(patterns, method='tsne')

# Plot learning history
viz.plot_learning_history(pipeline.learning_history)

# Show causal graph
viz.plot_causal_graph(causal_graph)
```

---

## Requirements

```
Python 3.8+
numpy
scipy
scikit-learn
networkx
matplotlib
seaborn
```

Install:
```bash
pip install -r requirements.txt
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_learning_pipeline.py

# Run demo
python examples/learning_pipeline_demo.py
```

---

## Contributing

When adding new features:
1. Follow existing code structure
2. Add tests for new functionality
3. Update documentation
4. Ensure backward compatibility

---

## License

[Your License Here]

---

## Contact

[Your Contact Info]

---

**Built with the principle: "Narrative patterns are discoverable, measurable, and predictive."**
