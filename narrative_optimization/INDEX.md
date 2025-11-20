# Narrative Optimization Framework - Navigation Index

**Your guide to the complete system**

---

## 🚀 Start Here

### New to the Project?
1. **Read**: [`README.md`](README.md) - Complete overview
2. **Quick Start**: [`QUICK_START.md`](QUICK_START.md) - Get running in 5 minutes
3. **Add Domain**: [`DOMAIN_ADDITION_TEMPLATE.md`](DOMAIN_ADDITION_TEMPLATE.md) - Step-by-step guide

### Want to Run Something?
- **Add New Domain**: `python MASTER_INTEGRATION.py DOMAIN data/domains/DOMAIN.json`
- **Run Complete System**: `python RUN_COMPLETE_SYSTEM.py`
- **Run Demo**: `python examples/learning_pipeline_demo.py`
- **Run Tests**: `pytest tests/ -v`

---

## 📚 Documentation

### Core Documentation
- [`README.md`](README.md) - Main documentation
- [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) - Directory layout
- [`DOMAIN_ARCHETYPE_SYSTEM.md`](DOMAIN_ARCHETYPE_SYSTEM.md) - Archetype system details
- [`QUICK_START_ARCHETYPES.md`](QUICK_START_ARCHETYPES.md) - Archetype guide

### Technical Docs
- [`docs/architecture.md`](docs/architecture.md) - System architecture
- [`docs/findings.md`](docs/findings.md) - Research findings
- [`docs/FREE_WILL_ANALYSIS.md`](docs/FREE_WILL_ANALYSIS.md) - Free will analysis

### Templates
- [`DOMAIN_ADDITION_TEMPLATE.md`](DOMAIN_ADDITION_TEMPLATE.md) - Add new domains

---

## 💻 Code Organization

### Core Systems
```
src/
├── learning/         # 14 learning modules
│   ├── learning_pipeline.py          ★ Main orchestrator
│   ├── universal_learner.py           Cross-domain patterns
│   ├── domain_learner.py              Domain-specific patterns
│   ├── validation_engine.py           Statistical validation
│   ├── registry_versioned.py          Version control
│   ├── hierarchical_learner.py        Hierarchies
│   ├── meta_learner.py                Transfer learning
│   ├── ensemble_learner.py            Ensemble methods
│   ├── online_learner.py              Streaming
│   ├── causal_discovery.py            Causal inference
│   └── ...
│
├── analysis/         # Analysis tools
│   ├── domain_specific_analyzer.py    ★ Main analyzer
│   ├── story_quality.py               ю calculation
│   ├── bridge_calculator.py           Д calculation
│   └── multi_modal_analyzer.py        Multi-modal
│
├── transformers/     # 56+ transformers
│   ├── archetypes/                    12 domain transformers
│   └── ...                            44+ feature transformers
│
├── config/           # Configuration
│   ├── domain_archetypes.py           ★ Archetype definitions
│   ├── genome_structure.py            Genome (ж) structure
│   └── ...
│
├── data/             # Data processing
│   └── data_loader.py                 ★ Unified data loading
│
├── visualization/    # Visualization
│   └── pattern_visualizer.py          Pattern viz
│
└── optimization/     # Performance
    ├── cache_manager.py               Caching
    └── performance_profiler.py        Profiling
```

---

## 🎯 Common Tasks

### Adding a New Domain

```bash
# Prepare data/domains/YOUR_DOMAIN.json
python MASTER_INTEGRATION.py YOUR_DOMAIN data/domains/YOUR_DOMAIN.json --pi 0.7
```

See: [`DOMAIN_ADDITION_TEMPLATE.md`](DOMAIN_ADDITION_TEMPLATE.md)

### Analyzing Existing Domain

```python
from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer

analyzer = DomainSpecificAnalyzer('golf')
results = analyzer.analyze_complete(texts, outcomes)
```

### Learning from Multiple Domains

```python
from src.learning import LearningPipeline

pipeline = LearningPipeline()
pipeline.ingest_domain('golf', golf_texts, golf_outcomes)
pipeline.ingest_domain('tennis', tennis_texts, tennis_outcomes)
metrics = pipeline.learn_cycle(learn_universal=True, learn_domain_specific=True)
```

### Discovering Patterns

```bash
python tools/discover_domain_archetypes.py --domain golf
```

### Visualizing Results

```python
from src.visualization import PatternVisualizer

viz = PatternVisualizer()
viz.visualize_pattern_space(patterns)
viz.plot_learning_history(history)
```

---

## 🔍 Finding Things

### "Where is...?"

**Learning system**: `src/learning/`  
**Analysis code**: `src/analysis/`  
**Transformers**: `src/transformers/`  
**Domain configs**: `src/config/domain_archetypes.py`  
**Domain data**: `data/domains/`  
**Domain results**: `narrative_optimization/domains/`  
**Examples**: `examples/`  
**Tests**: `tests/`

### "How do I...?"

**Add a domain**: See [`DOMAIN_ADDITION_TEMPLATE.md`](DOMAIN_ADDITION_TEMPLATE.md)  
**Run analysis**: See [`QUICK_START.md`](QUICK_START.md)  
**Understand archetypes**: See [`DOMAIN_ARCHETYPE_SYSTEM.md`](DOMAIN_ARCHETYPE_SYSTEM.md)  
**Use transformers**: See [`QUICK_START_ARCHETYPES.md`](QUICK_START_ARCHETYPES.md)

---

## 📊 Current Domains

**12 Integrated Domains**:
- Golf, Tennis, Boxing, NBA, WWE
- Chess, Oscars, Crypto, Mental Health
- Startups, Hurricanes, Housing

**30+ Additional Domains** with data ready:
- MLB, NFL, UFC, Music, Movies, Oscars
- And many more in `domains/`

Each domain in: `narrative_optimization/domains/DOMAIN_NAME/`

---

## 🧪 Testing & Validation

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_complete_integration.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🛠 Tools

- `MASTER_INTEGRATION.py` - Main integration script
- `RUN_COMPLETE_SYSTEM.py` - Complete demonstration
- `tools/discover_domain_archetypes.py` - Discover patterns
- `tools/make_transformers_adaptive.py` - Convert transformers
- `integration/real_data_validator.py` - Validate data
- `integration/migrate_domain_analysis.py` - Migrate analyses

---

## 📈 Performance

### Caching
```python
from src.optimization import get_global_cache

cache = get_global_cache()
cache.get_stats()
```

### Profiling
```python
from src.optimization import get_global_profiler

profiler = get_global_profiler()
profiler.print_report()
```

---

## 🤝 Contributing

1. Follow existing code structure
2. Add tests for new functionality
3. Update relevant documentation
4. Ensure backward compatibility
5. Run test suite before committing

---

## Key Concepts (Quick Reference)

**Genome (ж)**: Complete feature vector [nominative, archetypal, historial, uniquity]  
**Story Quality (ю)**: Distance from domain's golden narratio (Ξ)  
**Narrative Agency (Д)**: Bridge between story and outcomes  
**Narrativity (п)**: How open domain is to narrative influence  
**Ξ (Xi)**: Domain's ideal archetype pattern  

---

## System Verification

```bash
# Verify everything is working
python VERIFY_SYSTEM.py

# Initialize if needed
python INITIALIZE_SYSTEM.py

# Run complete demonstration
python DEMO_COMPLETE_SYSTEM.py
```

---

## Quick Links

**Essential**:
- [Main Documentation](README.md)
- [Quick Start](QUICK_START.md)
- [Setup Guide](SETUP_GUIDE.md)
- [System Overview](SYSTEM_OVERVIEW.md)

**Guides**:
- [Add Domain Template](DOMAIN_ADDITION_TEMPLATE.md)
- [Developer Guide](DEVELOPER_GUIDE.md)
- [Project Structure](PROJECT_STRUCTURE.md)

**Technical**:
- [Archetype System](DOMAIN_ARCHETYPE_SYSTEM.md)
- [Archetype Quick Start](QUICK_START_ARCHETYPES.md)

---

**Navigate efficiently. Build confidently. Learn continuously.**

