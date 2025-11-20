# Developer Guide: Narrative Optimization Framework

## Architecture Overview

The Narrative Optimization Framework is a unified system for analyzing narrative quality across diverse domains (sports, entertainment, business, etc.).

### Core Components

1. **Domain Configuration System** (`src/pipelines/domain_config.py`)
   - YAML-based domain definitions
   - Narrativity (п) calculation
   - Data schema specification

2. **Pipeline Composer** (`src/pipelines/pipeline_composer.py`)
   - Assembles data loader → transformers → analyzer → validator
   - Handles caching and optimization
   - Integrates multi-perspective/multi-method analysis

3. **Transformer System** (`src/transformers/`)
   - 100+ modular transformers (auto-discovered by registry)
   - Registry-based discovery and validation
   - п-based intelligent selection
   - Domain-type aware augmentation
   - **See**: `/docs/TRANSFORMERS_AND_PIPELINES.md` for complete guide

4. **Domain Types** (`src/pipelines/domain_types/`)
   - Base abstraction for domain-specific logic
   - Templates: Sports, Entertainment, Nominative, Business, Medical
   - Perspective preferences and validation metrics

5. **Analysis System** (`src/analysis/`)
   - Multi-perspective quality calculator
   - Multiple calculation methods (weighted_mean, ensemble, temporal, etc.)
   - Multi-scale analysis (nano, micro, meso, macro)
   - Quality aggregator

6. **Testing Suite** (`tests/domain_tests/`)
   - Pipeline composition tests
   - Transformer selection tests
   - Cross-domain comparison tests
   - Regression test framework

---

## Key Concepts

### Narrativity (п)

**Definition:** Scalar value [0, 1] representing how open or constrained a narrative space is.

**Calculation:**
```
п = 0.30×п_structural + 0.20×п_temporal + 
    0.25×п_agency + 0.15×п_interpretive + 
    0.10×п_format
```

**Interpretation:**
- **Low п (<0.3)**: Plot-driven, constrained (aviation, chess)
- **High п (>0.7)**: Character-driven, open (WWE, creative writing)
- **Mid п (0.3-0.7)**: Balanced, discover optimal mix

### Story Quality (ю)

**Definition:** Narrative quality score computed from genome features (ж).

**Formula:**
```
ю = Σ w_k × ж_k
```

Where `w_k` are п-based weights.

**Multi-Perspective:**
- ю_director: Creator's intent
- ю_audience: Viewer engagement
- ю_critic: Critical standards
- ю_character: Character development
- ю_cultural: Cultural resonance
- ю_meta: Self-awareness

**Multiple Methods:**
- weighted_mean: Standard п-based weights
- ensemble: Multiple weight schemes vote
- temporal: ю(t) over time
- context_dependent: ю(context)
- multi_scale: Aggregate across scales

### Narrative Agency (Д)

**Definition:** Bridge between narrative quality and outcomes.

**Formula:**
```
Д = п × r × κ
```

Where:
- п: Narrativity
- r: Correlation between ю and outcomes
- κ: Coupling factor (default 0.5)

**Efficiency Test:**
```
Д/п > 0.5 → Narrative laws apply
Д/п ≤ 0.5 → Reality constraints dominate
```

---

## Working with Transformers

**For complete transformer documentation, see**: `/docs/TRANSFORMERS_AND_PIPELINES.md`

### Finding Available Transformers

```bash
# List all transformers
python -m narrative_optimization.tools.list_transformers

# Search for specific transformers
python -m narrative_optimization.tools.list_transformers --filter nominative

# Validate transformer names
python -m narrative_optimization.tools.list_transformers --check Alpha ContextPattern
```

### Using Transformers

```python
# Import from registry (lazy loading)
from narrative_optimization.src.transformers import NominativeAnalysisTransformer

# Or use factory
from narrative_optimization.src.transformers.transformer_factory import TransformerFactory
factory = TransformerFactory()
transformer = factory.create_transformer('NominativeAnalysis')
```

## Adding a New Transformer

### Step 1: Create Transformer Class

```python
# narrative_optimization/src/transformers/my_transformer.py
from .base_transformer import NarrativeTransformer
import numpy as np

class MyTransformer(NarrativeTransformer):
    """My custom transformer"""
    
    def __init__(self):
        super().__init__(
            narrative_id='my_transformer',
            description='My custom narrative features'
        )
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Extract features
        features = np.array([self._extract_features(text) for text in X])
        return features
    
    def _extract_features(self, text):
        # Feature extraction logic
        return np.array([feature1, feature2, ...])
    
    def get_feature_names(self):
        return ['my_feature_1', 'my_feature_2', ...]
```

### Step 2: Register in Transformer Library

```python
# src/transformers/transformer_library.py
from transformers.my_transformer import MyTransformer

class TransformerLibrary:
    def __init__(self):
        self.transformers = {
            # ... existing transformers
            'my_transformer': {
                'class': MyTransformer,
                'category': 'narrative',  # or 'statistical', 'linguistic', etc.
                'features': 10,  # Number of features
                'description': 'My custom transformer'
            }
        }
```

### Step 3: Add п-Based Selection Rules

```python
# In TransformerLibrary.select_for_pi()
if п > 0.7:
    # High п: Emphasize narrative features
    if 'my_transformer' in self.transformers:
        selected.append('my_transformer')
```

### Step 4: Test

```python
# tests/transformers/test_my_transformer.py
def test_my_transformer():
    transformer = MyTransformer()
    texts = ['sample text 1', 'sample text 2']
    features = transformer.fit_transform(texts)
    assert features.shape[1] == 10  # Expected feature count
```

---

## Adding a New Domain Type

### Step 1: Create Domain Type Class

```python
# src/pipelines/domain_types/my_domain.py
from .base import BaseDomainType
from typing import List

class MyDomainType(BaseDomainType):
    """Template for my domain type"""
    
    def get_perspective_preferences(self) -> List[str]:
        """Which perspectives matter for this domain type"""
        return ['director', 'audience', 'critic']
    
    def get_default_transformers(self, п: float) -> List[str]:
        """Additional transformers beyond core"""
        if п < 0.3:
            return ['statistical', 'quantitative']
        elif п > 0.7:
            return ['nominative', 'self_perception']
        else:
            return ['ensemble', 'relational']
    
    def get_validation_metrics(self) -> List[str]:
        """Domain-specific validation metrics"""
        return ['custom_metric_1', 'custom_metric_2']
```

### Step 2: Register in Domain Types

```python
# src/pipelines/domain_types/__init__.py
from .my_domain import MyDomainType
from pipelines.domain_config import DomainType

def get_domain_type_class(domain_type: DomainType) -> type:
    type_map = {
        # ... existing mappings
        DomainType.MY_TYPE: MyDomainType,
    }
    return type_map.get(domain_type)
```

### Step 3: Add to DomainType Enum

```python
# src/pipelines/domain_config.py
class DomainType(str, Enum):
    # ... existing types
    MY_TYPE = "my_type"
```

---

## Extending Multi-Perspective System

### Adding a New Perspective

```python
# src/analysis/perspective_weights.py
class NarrativePerspective(str, Enum):
    # ... existing perspectives
    MY_PERSPECTIVE = "my_perspective"

class PerspectiveWeightSchemas:
    @staticmethod
    def get_my_perspective_weights(п: float) -> Dict[str, float]:
        """Weight schema for my perspective"""
        if п < 0.3:
            return {'statistical': 0.3, 'linguistic': 0.2, ...}
        elif п > 0.7:
            return {'nominative': 0.3, 'narrative_potential': 0.2, ...}
        else:
            return {'ensemble': 0.25, 'relational': 0.2, ...}
```

---

## Adding a New Quality Method

```python
# src/analysis/quality_methods.py
class MyMethod(QualityMethod):
    """My custom ю calculation method"""
    
    def compute_ю(
        self,
        genome: np.ndarray,
        feature_names: List[str],
        п: float,
        **kwargs
    ) -> np.ndarray:
        # Custom calculation logic
        ю = ...  # Your calculation
        return ю
    
    def get_name(self) -> str:
        return "my_method"

# Register in QualityMethodRegistry
class QualityMethodRegistry:
    def __init__(self):
        self.methods = {
            # ... existing methods
            'my_method': MyMethod(),
        }
```

---

## Testing Guidelines

### Unit Tests

Test individual components:

```python
def test_transformer():
    transformer = MyTransformer()
    result = transformer.fit_transform(['text'])
    assert result.shape[0] == 1

def test_perspective_calculation():
    calculator = MultiPerspectiveQualityCalculator(п=0.6)
    ю = calculator.compute_single_perspective(
        NarrativePerspective.DIRECTOR, genome, feature_names
    )
    assert len(ю) == n_organisms
```

### Integration Tests

Test pipeline composition:

```python
def test_pipeline_composition():
    config = DomainConfig.from_yaml('domains/test/config.yaml')
    composer = PipelineComposer()
    pipeline_info = composer.compose_pipeline(config)
    assert pipeline_info['feature_count'] > 0
```

### Regression Tests

Compare to baselines:

```python
def test_regression():
    baseline_mgr = RegressionBaseline()
    new_results = run_analysis()
    comparison = baseline_mgr.compare_results('domain', new_results)
    assert comparison['all_passed']
```

---

## Performance Optimization

### Caching

Pipeline composer uses joblib caching:

```python
# Enable caching
results = composer.run_pipeline(config, use_cache=True)

# Cache location
cache_dir = project_root / 'narrative_optimization' / '.cache'
```

### Feature Selection

Limit feature count:

```python
# Target feature count
pipeline_info = composer.compose_pipeline(
    config, target_feature_count=300
)
```

### Parallel Processing

Transformers can be parallelized:

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(transformer.transform)(texts) 
    for transformer in transformers
)
```

---

## Debugging

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Pipeline Info

```python
pipeline_info = composer.compose_pipeline(config)
print(pipeline_info['rationales'])  # Why transformers selected
print(pipeline_info['transformers'])  # Which transformers
```

### Validate Config

```python
config = DomainConfig.from_yaml('config.yaml')
config.validate()  # Raises if invalid
```

---

## Code Style

- **Type hints**: Use for all function signatures
- **Docstrings**: Google-style docstrings
- **Tests**: pytest, one test file per module
- **Naming**: snake_case for functions, PascalCase for classes

---

## Resources

- **Framework Quick Reference**: `FRAMEWORK_QUICKREF.md`
- **Formal Variable System**: `FORMAL_VARIABLE_SYSTEM.md`
- **Onboarding Handbook**: `ONBOARDING_HANDBOOK.md`
- **Migration Guide**: `MIGRATION_GUIDE.md`

