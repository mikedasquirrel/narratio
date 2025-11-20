# Migration Map: Old Patterns → New Patterns

**Last Updated**: November 20, 2025

This guide maps deprecated patterns to current best practices.

---

## Pattern 1: Finding Available Transformers

### Old Pattern (Deprecated)

Looking through code files or hardcoded lists in documentation:

```markdown
## Available Transformers (55 total)

1. StatisticalTransformer
2. NominativeAnalysisTransformer
3. SelfPerceptionTransformer
...
```

**Problems**:
- Lists get out of sync with code
- No way to validate names
- Manual maintenance required
- Typos cause cryptic errors

### New Pattern (Current)

Use the registry CLI tool:

```bash
# List all transformers
python -m narrative_optimization.tools.list_transformers

# Search for specific transformers
python -m narrative_optimization.tools.list_transformers --filter nominative

# Validate transformer names
python -m narrative_optimization.tools.list_transformers --check Alpha ContextPattern

# Get category summary
python -m narrative_optimization.tools.list_transformers --summary

# Export to JSON
python -m narrative_optimization.tools.list_transformers --format json
```

**Benefits**:
- Always accurate (auto-discovers from code)
- Fuzzy search for typos
- Validates names before runtime
- Groups by category

---

## Pattern 2: Importing Transformers

### Old Pattern (Deprecated)

Manual imports from specific files:

```python
from narrative_optimization.src.transformers.statistical import StatisticalTransformer
from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer
from narrative_optimization.src.transformers.self_perception import SelfPerceptionTransformer

transformers = [
    StatisticalTransformer(),
    NominativeAnalysisTransformer(),
    SelfPerceptionTransformer()
]
```

**Problems**:
- Imports all dependencies immediately (TensorFlow deadlocks)
- Need to know exact file paths
- Typos in class names cause import errors
- No suggestions for fixes

### New Pattern (Current)

Use lazy imports via registry:

```python
# Lazy import (no deadlocks)
from narrative_optimization.src.transformers import (
    StatisticalTransformer,
    NominativeAnalysisTransformer,
    SelfPerceptionTransformer
)

# Or use factory
from narrative_optimization.src.transformers.transformer_factory import TransformerFactory

factory = TransformerFactory()
transformers = [
    factory.create_transformer('Statistical'),
    factory.create_transformer('NominativeAnalysis'),
    factory.create_transformer('SelfPerception')
]
```

**Benefits**:
- Lazy loading (no mutex deadlocks)
- Automatic typo suggestions
- Cleaner code
- Works with any name format (CamelCase, snake_case, partial)

---

## Pattern 3: Selecting Transformers

### Old Pattern (Deprecated)

Hardcoded lists per domain:

```python
# NBA transformers (hardcoded)
nba_transformers = [
    'StatisticalTransformer',
    'NominativeAnalysisTransformer',
    'SelfPerceptionTransformer',
    'LinguisticPatternsTransformer',
    'EnsembleNarrativeTransformer',
    'ConflictTensionTransformer',
    'TemporalEvolutionTransformer',
    'InformationTheoryTransformer'
]

# Golf transformers (different hardcoded list)
golf_transformers = [
    'StatisticalTransformer',
    'NominativeAnalysisTransformer',
    'PhoneticTransformer',
    'SocialStatusTransformer',
    ...
]
```

**Problems**:
- Manual selection for each domain
- No principled selection logic
- Lists get out of sync
- No explanation for choices

### New Pattern (Current)

Use narrativity-based automatic selection:

```python
from narrative_optimization.src.transformers.transformer_library import TransformerLibrary

library = TransformerLibrary()

# Automatic selection based on narrativity
selected, feature_count = library.get_for_narrativity(
    π=0.49,  # NBA narrativity
    target_feature_count=300,
    include_statistical=True
)

# Prints selection rationale:
# π-GUIDED TRANSFORMER SELECTION (π=0.49)
# Selected 8 transformers (312 features):
#   • statistical (200 feat) - Plot/content features essential
#   • nominative (51 feat) - Identity foundation
#   • linguistic (36 feat) - Voice patterns
#   ...
```

**Benefits**:
- Principled selection based on theory
- Automatic rationale generation
- Consistent across domains
- Adapts to domain narrativity

---

## Pattern 4: Building Feature Extraction Pipelines

### Old Pattern (Deprecated)

Manual pipeline construction:

```python
# Manual feature extraction
features_list = []
feature_names_list = []

for transformer in transformers:
    try:
        transformer.fit(X_train)
        features = transformer.transform(X_test)
        features_list.append(features)
        feature_names_list.extend(transformer.get_feature_names())
    except Exception as e:
        print(f"Error with {transformer}: {e}")
        continue

# Manual concatenation
if features_list:
    combined_features = np.hstack(features_list)
else:
    combined_features = np.array([])
```

**Problems**:
- Boilerplate code repeated everywhere
- No caching
- Manual error handling
- No provenance tracking
- No feature name management

### New Pattern (Current)

Use pipeline classes:

```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

# Unsupervised pipeline
pipeline = FeatureExtractionPipeline(
    transformer_names=['Statistical', 'Nominative', 'Linguistic'],
    domain_name='nba',
    cache_dir='narrative_optimization/cache/features',
    enable_caching=True,
    verbose=True
)

# Extract features
features = pipeline.extract_features(texts)
feature_names = pipeline.get_feature_names()
provenance = pipeline.get_provenance()
status = pipeline.transformer_status
```

**Benefits**:
- Automatic caching
- Graceful error handling
- Feature provenance tracking
- Progress reporting
- Reusable across domains

---

## Pattern 5: Supervised Feature Extraction

### Old Pattern (Deprecated)

Manually separating supervised/unsupervised transformers:

```python
# Manual separation
unsupervised = ['Statistical', 'Nominative', 'Linguistic']
supervised = ['Alpha', 'ContextPattern']

# Run unsupervised first
unsupervised_pipeline = build_pipeline(unsupervised)
unsupervised_features = unsupervised_pipeline.extract(texts)

# Then supervised
supervised_pipeline = build_pipeline(supervised)
supervised_features = supervised_pipeline.extract(unsupervised_features, y)

# Manual concatenation
all_features = np.hstack([unsupervised_features, supervised_features])
```

**Problems**:
- Manual separation logic
- Need to know which transformers need labels
- Complex concatenation
- No unified caching

### New Pattern (Current)

Use supervised pipeline:

```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline_supervised import SupervisedFeatureExtractionPipeline

# Unified pipeline handles separation automatically
pipeline = SupervisedFeatureExtractionPipeline(
    transformer_names=['Statistical', 'Nominative', 'Alpha', 'ContextPattern'],
    domain_name='nba',
    enable_caching=True
)

# Single call
features = pipeline.extract_features(texts, outcomes=y)
```

**Benefits**:
- Automatic separation of supervised/unsupervised
- Unified caching with mode metadata
- Single API call
- Handles genome feature generation

---

## Pattern 6: Domain-Specific Scripts

### Old Pattern (Deprecated)

Separate scripts for each domain:

```
scripts/
  run_nba_transformers.py
  run_nfl_transformers.py
  run_golf_transformers.py
  run_mlb_transformers.py
  ...
```

Each with hardcoded transformer lists and domain-specific logic.

**Problems**:
- Code duplication
- Inconsistent patterns
- Hard to maintain
- No cross-domain learning

### New Pattern (Current)

Universal pipeline with domain config:

```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline
from narrative_optimization.src.pipelines.domain_config import DomainConfig
from narrative_optimization.src.transformers.transformer_library import TransformerLibrary

# Load domain config
config = DomainConfig('nba')  # or 'nfl', 'golf', etc.

# Auto-select transformers
library = TransformerLibrary()
transformer_names, rationales, _ = library.select_for_config(
    config=config,
    target_feature_count=300
)

# Create pipeline
pipeline = FeatureExtractionPipeline(
    transformer_names=transformer_names,
    domain_name=config.domain,
    enable_caching=True
)

# Extract features
features = pipeline.extract_features(texts)
```

**Benefits**:
- Single script for all domains
- Domain config in YAML files
- Automatic transformer selection
- Consistent patterns
- Easy to add new domains

---

## Pattern 7: Validating Transformer Names

### Old Pattern (Deprecated)

Runtime errors with no suggestions:

```python
# Typo in transformer name
transformers = ['Statistical', 'NominativeAnalysis', 'NarrativePotentail']  # typo

# Runtime error with no help
pipeline = FeatureExtractionPipeline(transformer_names=transformers, domain_name='test')
# AttributeError: module 'narrative_optimization.src.transformers' has no attribute 'NarrativePotentail'
```

**Problems**:
- No validation before runtime
- Cryptic error messages
- No suggestions for fixes
- Wastes time debugging

### New Pattern (Current)

Pre-validate with CLI or registry:

```bash
# Validate before runtime
python -m narrative_optimization.tools.list_transformers --check Statistical NominativeAnalysis NarrativePotentail

# Output:
# Missing transformers detected:
#   - NarrativePotentail Suggestions: NarrativePotentialTransformer, NarrativePotentialV2Transformer
```

Or in code:

```python
from narrative_optimization.src.transformers.registry import get_transformer_registry

registry = get_transformer_registry()
transformer_names = ['Statistical', 'NominativeAnalysis', 'NarrativePotentail']

missing = registry.describe_missing(transformer_names)
if missing:
    for name, suggestions in missing.items():
        print(f"Unknown: {name}")
        print(f"Did you mean: {', '.join(suggestions)}")
```

**Benefits**:
- Catch errors before runtime
- Fuzzy suggestions for typos
- Fast validation
- Clear error messages

---

## Pattern 8: Documentation References

### Old Pattern (Deprecated)

Referencing hardcoded transformer lists in docs:

```markdown
## Available Transformers

We have 55 transformers:

1. StatisticalTransformer - TF-IDF baseline
2. NominativeAnalysisTransformer - Name analysis
...
55. CrossDomainEmbeddingTransformer - Cross-domain features
```

**Problems**:
- Gets out of sync immediately
- Manual maintenance
- No single source of truth
- Confuses new developers

### New Pattern (Current)

Reference the registry and CLI tool:

```markdown
## Available Transformers

To see all available transformers:

```bash
python -m narrative_optimization.tools.list_transformers
```

For detailed information, see `/docs/TRANSFORMERS_AND_PIPELINES.md`.
```

**Benefits**:
- Always accurate
- Single source of truth
- Self-documenting
- Easy to maintain

---

## Quick Reference

| Old Pattern | New Pattern | Documentation |
|------------|-------------|---------------|
| Hardcoded transformer lists | Registry CLI tool | `/docs/TRANSFORMER_CATALOG.md` |
| Manual imports | Lazy imports via registry | `/docs/TRANSFORMERS_AND_PIPELINES.md` |
| Domain-specific selection | Narrativity-based selection | `/docs/TRANSFORMERS_AND_PIPELINES.md` |
| Manual pipeline construction | Pipeline classes | `/docs/TRANSFORMERS_AND_PIPELINES.md` |
| Separate supervised logic | Supervised pipeline | `/docs/SUPERVISED_TRANSFORMER_HANDOFF.md` |
| Domain-specific scripts | Universal pipeline + config | `/docs/ONBOARDING_HANDBOOK.md` |
| Runtime validation | Pre-validation with CLI | `/docs/TRANSFORMER_CATALOG.md` |
| Hardcoded docs | Reference registry | `/docs/BOT_ONBOARDING.md` |

---

## Getting Started with New Patterns

1. **Read**: `/docs/BOT_ONBOARDING.md` (2-3 minutes)
2. **Explore**: `python -m narrative_optimization.tools.list_transformers`
3. **Learn**: `/docs/TRANSFORMERS_AND_PIPELINES.md` (complete guide)
4. **Build**: Use `FeatureExtractionPipeline` for your domain

---

**Remember**: If you're following a pattern from `/archive/deprecated/`, stop and check this migration map for the current approach.

