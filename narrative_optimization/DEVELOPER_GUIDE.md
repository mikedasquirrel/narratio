# Developer Guide

**For future developers and AI agents working on this system**

---

## System Philosophy

The Narrative Optimization Framework operates on a core principle:

> Stories are discoverable patterns that can be learned from data, validated statistically, and used to predict outcomes. The system learns both universal stories (that work everywhere) and domain-specific stories (unique to each context).

---

## How The System Works

### 1. Multi-Layer Learning

**Universal Layer** (Cross-Domain):
- Patterns like "underdog", "comeback", "rivalry" appear in all domains
- Learned from aggregated data across domains
- Transfer automatically to new domains

**Domain Layer** (Domain-Specific):
- Patterns unique to each domain (golf: "course mastery", chess: "endgame technique")
- Learned from domain-specific winner data
- Don't transfer but inform similar domain analysis

**Context Layer** (Situational):
- Patterns that activate in specific situations (championship, playoff, final)
- Conditional on context features

### 2. The Genome (ж)

Every narrative has a complete feature vector:

```
ж = [nominative, archetypal, historial, uniquity]
```

- **Nominative**: Named entities (proper nouns, key terms)
- **Archetypal**: Distance from domain's ideal pattern (Ξ)
- **Historial**: Narrative lineage (precedent, momentum)
- **Uniquity**: Rarity and novelty (elusive patterns)

### 3. Story Quality (ю)

Computed as distance from domain's "golden narratio" (Ξ):

```python
ю = story_quality_function(ж, Ξ_domain)
```

The closer a narrative to Ξ, the higher its quality for that domain.

### 4. Learning Cycle

```
1. Ingest Data
   └→ DataLoader handles all formats

2. Discover Patterns
   ├→ Universal: UniversalArchetypeLearner
   └→ Domain: DomainSpecificLearner

3. Validate Patterns
   └→ ValidationEngine (correlation, effect size, predictive power)

4. Measure Performance
   └→ Compare R² before vs after

5. Update Archetypes
   └→ VersionedArchetypeRegistry (version control, A/B tests)

6. Apply to Analysis
   └→ DomainSpecificAnalyzer uses learned patterns

Loop back to step 1 with new data
```

---

## Adding Features

### Adding a New Domain

```python
# Method 1: Automated
python MASTER_INTEGRATION.py new_domain data/domains/new_domain.json --pi 0.7

# Method 2: Programmatic
from MASTER_INTEGRATION import MasterDomainIntegration

integration = MasterDomainIntegration()
results = integration.analyze_new_domain(
    'new_domain',
    texts,
    outcomes,
    {'pi': 0.7, 'type': 'expertise'}
)
```

### Adding a Transformer

Create `src/transformers/my_transformer.py`:

```python
from .base import NarrativeTransformer

class MyTransformer(NarrativeTransformer):
    def __init__(self, domain_config=None):
        super().__init__(
            narrative_id="my_transformer",
            description="What this measures"
        )
        self.domain_config = domain_config
    
    def fit(self, X, y=None):
        # Learn from training data
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        # Extract features
        return np.array([self._extract(text) for text in X])
    
    def _extract(self, text: str) -> np.ndarray:
        # Your feature extraction logic
        return np.zeros(10)  # 10 features
```

### Adding Domain-Specific Archetype

Create `src/transformers/archetypes/my_domain_archetype.py`:

```python
from ..domain_archetype import DomainArchetypeTransformer
from ...config.domain_config import DomainConfig

class MyDomainArchetypeTransformer(DomainArchetypeTransformer):
    def __init__(self):
        config = DomainConfig('my_domain')
        super().__init__(config)
        
        self.key_events = ['championship', 'final']
    
    def _extract_archetype_features(self, X):
        base = super()._extract_archetype_features(X)
        
        # Add domain-specific boosts
        enhanced = []
        for i, text in enumerate(X):
            boost = 1.3 if any(e in text.lower() for e in self.key_events) else 1.0
            enhanced.append(base[i] * boost)
        
        return np.array(enhanced)
```

Add to `src/config/domain_archetypes.py`:

```python
'my_domain': {
    'archetype_patterns': {
        'key_pattern': ['keyword1', 'keyword2'],
        # ...
    },
    'nominative_richness_requirement': 20,
    'archetype_weights': {
        'key_pattern': 0.40,
        # ... weights sum to 1.0
    },
    'pi': 0.70,
    'theta_range': (0.40, 0.50),
    'lambda_range': (0.60, 0.70)
}
```

### Adding a Learning Strategy

Create `src/learning/my_learner.py`:

```python
class MyLearner:
    def discover_patterns(self, texts, outcomes):
        # Your discovery logic
        patterns = {}
        return patterns
    
    def validate_patterns(self, patterns, texts, outcomes):
        # Your validation logic
        validated = {}
        return validated
```

Integrate into `LearningPipeline`:

```python
# In src/learning/learning_pipeline.py
from .my_learner import MyLearner

class LearningPipeline:
    def __init__(self, ...):
        self.my_learner = MyLearner()
```

---

## Code Organization Principles

### 1. Separation of Concerns
- **Learning** (`src/learning/`): Pattern discovery and validation
- **Analysis** (`src/analysis/`): Domain analysis and prediction
- **Transformers** (`src/transformers/`): Feature extraction
- **Config** (`src/config/`): Domain definitions
- **Data** (`src/data/`): Data loading and processing

### 2. Domain Awareness
All transformers accept optional `domain_config`:

```python
def __init__(self, domain_config=None):
    self.domain_config = domain_config
    # Load domain-specific patterns if available
```

### 3. Learning Integration
Components communicate through:
- `LearningPipeline`: Central orchestrator
- `VersionedArchetypeRegistry`: Shared pattern storage
- `DomainRegistry`: Domain metadata

### 4. Data Flow

```
DataLoader → LearningPipeline → ValidationEngine
                ↓
         DomainAnalyzer ← TransformerFactory
                ↓
          Results/Registry
```

---

## Testing Strategy

### Unit Tests
```python
# Test individual components
pytest tests/test_integration_system.py
```

### Integration Tests
```python
# Test component interactions
pytest tests/test_complete_integration.py
```

### End-to-End Tests
```python
# Test complete workflows
pytest tests/test_end_to_end.py
```

### Regression Tests
```python
# Ensure changes don't break existing functionality
python scripts/regression_test.py
```

---

## Performance Optimization

### Caching
```python
from src.optimization import cached, get_global_cache

@cached(get_global_cache(), ttl=3600)
def expensive_function(x):
    # Expensive computation
    return result
```

### Profiling
```python
from src.optimization import profile

@profile("function_name")
def my_function():
    # Your code
    pass

# View results
from src.optimization import get_global_profiler
get_global_profiler().print_report()
```

### Batch Processing
```bash
# Process multiple domains in parallel
python tools/batch_analyze_domains.py --parallel
```

---

## Common Patterns

### Loading and Analyzing a Domain

```python
from src.data import DataLoader
from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer

# Load
loader = DataLoader()
data = loader.load_domain('golf')

# Analyze
analyzer = DomainSpecificAnalyzer('golf')
results = analyzer.analyze_complete(data['texts'], data['outcomes'])

print(f"R²: {results['r_squared']:.1%}")
print(f"Д: {results['delta']:.3f}")
```

### Learning from Multiple Domains

```python
from src.learning import LearningPipeline

pipeline = LearningPipeline()

# Ingest
pipeline.ingest_domain('golf', golf_texts, golf_outcomes)
pipeline.ingest_domain('tennis', tennis_texts, tennis_outcomes)

# Learn
metrics = pipeline.learn_cycle(
    domains=['golf', 'tennis'],
    learn_universal=True,
    learn_domain_specific=True
)

# Get learned patterns
golf_patterns = pipeline.get_archetypes('golf')
```

### Transfer Learning

```python
from src.learning import MetaLearner

meta = MetaLearner()

# Find similar domains
similar = meta.find_similar_domains('chess', n_similar=3)

# Transfer patterns
transferred = meta.transfer_patterns('golf', 'chess', min_transferability=0.6)
```

---

## Extending the System

### Adding New Learning Algorithms

1. Create learner in `src/learning/`
2. Add to `LearningPipeline` if needed
3. Test with `pytest tests/`
4. Document in this guide

### Adding New Analysis Methods

1. Create analyzer in `src/analysis/`
2. Integrate with existing analyzers
3. Add API endpoint if needed
4. Add tests

### Adding New Domains

1. Prepare data file
2. Run `MASTER_INTEGRATION.py`
3. Review and refine
4. Add archetype transformer (optional)
5. Update domain config (optional)

---

## Debugging

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check System Health

```bash
python scripts/health_check.py
```

### Profile Performance

```bash
python scripts/benchmark_performance.py
```

### Inspect Cache

```python
from src.optimization import get_global_cache
cache = get_global_cache()
print(cache.get_stats())
```

---

## Best Practices

1. **Always validate data** before analysis
2. **Use domain_config** for domain-specific behavior
3. **Cache expensive computations**
4. **Profile before optimizing**
5. **Test additions** with pytest
6. **Document new features** in relevant files
7. **Run regression tests** before merging
8. **Use type hints** for clarity
9. **Handle errors gracefully**
10. **Keep backwards compatibility**

---

##Production Deployment

### Docker (Recommended)

```bash
# Build and deploy
make docker

# Or manually
docker-compose up -d
```

### Manual

```bash
# Setup
bash deploy/production_setup.sh

# Start services
python api/api_server.py &
python workflows/continuous_learning_workflow.py --continuous &
python monitoring/dashboard.py
```

---

## Maintenance

**Daily**: `make health` + `make monitor`  
**Weekly**: `make validate` + `make report`  
**Monthly**: `make discover` + `make analyze` + `make docs`

---

**System designed for continuous improvement and seamless extension.**

