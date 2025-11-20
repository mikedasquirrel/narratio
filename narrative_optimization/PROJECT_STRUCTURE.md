# Project Structure

**Clean, organized, ready for future development**

---

## Directory Layout

```
narrative_optimization/
│
├── src/                          # Core source code
│   ├── learning/                 # Learning system (14 modules)
│   │   ├── learning_pipeline.py         # Main orchestrator
│   │   ├── universal_learner.py         # Cross-domain patterns
│   │   ├── domain_learner.py            # Domain-specific patterns
│   │   ├── validation_engine.py         # Statistical validation
│   │   ├── registry_versioned.py        # Version control
│   │   ├── explanation_generator.py     # Human explanations
│   │   ├── hierarchical_learner.py      # Multi-level hierarchies
│   │   ├── active_learner.py            # Active learning
│   │   ├── meta_learner.py              # Transfer learning
│   │   ├── ensemble_learner.py          # Ensemble methods
│   │   ├── online_learner.py            # Streaming updates
│   │   ├── causal_discovery.py          # Causal inference
│   │   ├── pattern_refiner.py           # Pattern refinement
│   │   └── context_aware_learner.py     # Contextual patterns
│   │
│   ├── analysis/                # Analysis tools
│   │   ├── domain_specific_analyzer.py  # Main analyzer
│   │   ├── story_quality.py             # ю calculation
│   │   ├── bridge_calculator.py         # Д calculation
│   │   ├── multi_modal_analyzer.py      # Multi-modal patterns
│   │   └── uncertainty_quantifier.py    # Uncertainty quantification
│   │
│   ├── transformers/            # 56 narrative transformers
│   │   ├── domain_archetype.py          # Base domain transformer
│   │   ├── domain_adaptive_base.py      # Adaptive base class
│   │   └── archetypes/                  # Domain-specific (12 domains)
│   │       ├── golf_archetype.py
│   │       ├── tennis_archetype.py
│   │       ├── chess_archetype.py
│   │       ├── oscars_archetype.py
│   │       ├── crypto_archetype.py
│   │       ├── mental_health_archetype.py
│   │       ├── startups_archetype.py
│   │       ├── hurricanes_archetype.py
│   │       ├── housing_archetype.py
│   │       ├── boxing_archetype.py
│   │       ├── nba_archetype.py
│   │       └── wwe_archetype.py
│   │
│   ├── config/                  # Configuration
│   │   ├── domain_archetypes.py         # Archetype registry
│   │   ├── domain_config.py             # Domain configuration
│   │   ├── genome_structure.py          # Genome definition
│   │   ├── archetype_discovery.py       # Discovery system
│   │   ├── advanced_archetype_discovery.py  # Advanced discovery
│   │   └── temporal_decay.py            # Temporal weighting
│   │
│   ├── visualization/           # Visualization tools
│   │   └── pattern_visualizer.py        # Pattern visualization
│   │
│   ├── optimization/            # Performance optimization
│   │   ├── cache_manager.py             # Caching system
│   │   └── performance_profiler.py      # Profiling tools
│   │
│   ├── data/                    # Data processing
│   │   ├── data_loader.py               # Unified data loading
│   │   └── streaming_processor.py       # Stream processing
│   │
│   └── pipeline_config.py       # Pipeline configuration
│
├── domains/                     # Domain analyses (30+ domains)
│   ├── golf/
│   ├── tennis/
│   ├── chess/
│   └── ...
│
├── integration/                 # Integration tools
│   ├── migrate_domain_analysis.py
│   ├── real_data_validator.py
│   └── backward_compatible_analyzer.py
│
├── tools/                       # Utility scripts
│   ├── discover_archetypes.py
│   ├── discover_domain_archetypes.py
│   └── make_transformers_adaptive.py
│
├── examples/                    # Usage examples
│   └── learning_pipeline_demo.py
│
├── tests/                       # Test suite
│   └── test_integration_system.py
│
├── docs/                        # Documentation
│   ├── architecture.md
│   ├── findings.md
│   └── FREE_WILL_ANALYSIS.md
│
├── MASTER_INTEGRATION.py        # ★ Main integration script
├── RUN_COMPLETE_SYSTEM.py       # ★ Complete demonstration
│
├── README.md                    # ★ Main documentation
├── QUICK_START.md               # ★ Quick start guide
├── DOMAIN_ADDITION_TEMPLATE.md  # ★ Template for new domains
├── DOMAIN_ARCHETYPE_SYSTEM.md   # Technical details
└── QUICK_START_ARCHETYPES.md    # Archetype guide
```

---

## Key Files for Future Development

### For Adding New Domains
1. **`MASTER_INTEGRATION.py`** - Run this to add domains
2. **`DOMAIN_ADDITION_TEMPLATE.md`** - Step-by-step guide
3. **`src/config/domain_archetypes.py`** - Add archetype definitions

### For Analysis
4. **`src/analysis/domain_specific_analyzer.py`** - Main analyzer
5. **`src/learning/learning_pipeline.py`** - Learning orchestrator
6. **`RUN_COMPLETE_SYSTEM.py`** - Full system demo

### For Understanding
7. **`README.md`** - Complete documentation
8. **`QUICK_START.md`** - Get started quickly
9. **`DOMAIN_ARCHETYPE_SYSTEM.md`** - Technical details

---

## Development Workflow

### Adding a New Domain

```bash
# 1. Prepare data (JSON or CSV)
# 2. Run integration
python MASTER_INTEGRATION.py new_domain data/domains/new_domain.json --pi 0.7

# 3. Review results in domains/new_domain/
# 4. (Optional) Add custom transformer
# 5. (Optional) Update domain config
```

### Improving Existing Domain

```bash
# 1. Load more data
python MASTER_INTEGRATION.py golf data/domains/golf_additional.json

# 2. Run learning cycle
python -c "
from src.learning import LearningPipeline
pipeline = LearningPipeline()
# Load state, ingest new data, learn, save
"
```

### Running Tests

```bash
pytest tests/ -v
```

### Profiling Performance

```python
from src.optimization import get_global_profiler

profiler = get_global_profiler()
# ... run analyses ...
profiler.print_report()
```

---

## What's Clean

- ✅ **34 redundant MD files removed**
- ✅ **Clear entry points** (MASTER_INTEGRATION.py, RUN_COMPLETE_SYSTEM.py)
- ✅ **Organized src/ structure** (learning, analysis, transformers, config)
- ✅ **Comprehensive documentation** (README, templates, guides)
- ✅ **Integrated learning system** (universal + domain-specific)
- ✅ **Seamless data loading** (multiple formats handled)
- ✅ **Future-proof architecture** (extensible, modular)

---

## What Future Developers Need to Know

### Core Philosophy

> **"What are the stories of this domain? Do they match familiar stories from structurally similar domains? Do they unfold at predicted frequency? What patterns emerge?"**

### Three-Layer System

1. **Universal Layer**: Patterns that work everywhere (underdog, comeback, pressure)
2. **Domain Layer**: Domain-specific patterns (golf: course mastery, chess: endgame technique)
3. **Context Layer**: Situational patterns (championship vs regular season)

### The Genome (ж)

Every story has a genome:
```
ж = [nominative, archetypal, historial, uniquity]
```

This is extracted automatically and used to compute **story quality (ю)**.

### Learning Loop

```
Data → Discover → Validate → Measure → Update → Apply
  ↑__________________________________________________|
```

Continuous improvement from feedback.

---

## Quick Reference Commands

```bash
# Add domain
python MASTER_INTEGRATION.py DOMAIN data/domains/DOMAIN.json

# Run complete system
python RUN_COMPLETE_SYSTEM.py

# Run tests
pytest tests/

# Make transformers adaptive
python tools/make_transformers_adaptive.py

# Discover patterns
python tools/discover_domain_archetypes.py --domain DOMAIN

# Run demo
python examples/learning_pipeline_demo.py
```

---

## Dependencies

All in `requirements.txt`:
- numpy, scipy (numerical computing)
- scikit-learn (machine learning)
- networkx (graph operations)
- matplotlib, seaborn (visualization)
- pandas (data handling)

---

**The system is clean, organized, and ready for continuous development.**

