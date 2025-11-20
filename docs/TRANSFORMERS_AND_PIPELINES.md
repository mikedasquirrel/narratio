# Transformers and Pipelines: Complete Guide

**Last Updated**: November 20, 2025  
**Status**: Production System

This is the single source of truth for understanding transformers and pipelines in the Narrative Optimization Framework.

---

## Quick Start for New Bots

### Find Available Transformers (30 seconds)

```bash
# List all transformers
python -m narrative_optimization.tools.list_transformers

# Search for specific transformers
python -m narrative_optimization.tools.list_transformers --filter nominative

# Validate transformer names
python -m narrative_optimization.tools.list_transformers --check NarrativeMass SomeTransformer

# Get category summary
python -m narrative_optimization.tools.list_transformers --summary
```

### Use Transformers in Code

```python
# Import from registry (lazy loading, no deadlocks)
from narrative_optimization.src.transformers import (
    NominativeAnalysisTransformer,
    StatisticalTransformer,
    AVAILABLE_TRANSFORMERS  # tuple of all transformer names
)

# Or use the factory
from narrative_optimization.src.transformers.transformer_factory import TransformerFactory
factory = TransformerFactory()
transformer = factory.create_transformer('NominativeAnalysis')
```

### Run Feature Extraction

```python
# Unsupervised pipeline (most common)
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

pipeline = FeatureExtractionPipeline(
    transformer_names=['Statistical', 'NominativeAnalysis', 'SelfPerception'],
    domain_name='nba',
    enable_caching=True
)
features = pipeline.extract_features(texts)

# Supervised pipeline (when you have labels)
from narrative_optimization.src.pipelines.feature_extraction_pipeline_supervised import SupervisedFeatureExtractionPipeline

pipeline = SupervisedFeatureExtractionPipeline(
    transformer_names=['Statistical', 'Alpha', 'ContextPattern'],
    domain_name='nba',
    enable_caching=True
)
features = pipeline.extract_features(texts, outcomes=y)
```

---

## Transformer Registry System

### What It Is

The registry automatically discovers all transformer classes in `narrative_optimization/src/transformers/` using Python's AST. No manual maintenance required.

**Key Features**:
- Discovers 100+ transformers automatically
- Resolves class names, snake_case slugs, and partial names
- Provides fuzzy suggestions for typos
- Groups transformers by category
- Zero import overhead (lazy loading)

### How It Works

```python
from narrative_optimization.src.transformers.registry import get_transformer_registry

registry = get_transformer_registry()

# Get all transformer names
all_transformers = registry.class_names()

# Resolve any name format
metadata = registry.resolve('narrative_potential')  # snake_case
metadata = registry.resolve('NarrativePotential')   # without suffix
metadata = registry.resolve('NarrativePotentialTransformer')  # full name

# Get suggestions for typos
suggestions = registry.suggest('NarativePotentail')
# Returns: ['NarrativePotentialTransformer', 'NarrativePotentialV2Transformer', ...]

# List by category
semantic_transformers = registry.list_metadata(category='semantic')
```

### Transformer Categories

The registry automatically categorizes transformers based on their location:

- **core**: Foundational transformers (top-level files)
- **semantic**: Embedding-based understanding
- **temporal**: Time-based features
- **narrative**: Story structure features
- **meta**: Self-awareness and interference
- **cognitive**: Mental models and fluency
- **cultural**: Cultural context and zeitgeist
- **relational**: Relationship dynamics
- **sports**: Sports-specific features
- **geographic**: Location-based narrative
- **legal**: Legal argument structure
- **media**: Broadcast and media narrative
- **archetype**: Domain archetypes
- **infrastructure**: Caching and utilities

---

## Pipeline Architecture

### Two Pipeline Modes

#### 1. Unsupervised Pipeline (Default)

**Use when**: You only have text/narratives, no outcome labels yet

**What it does**:
- Applies transformers that work on text alone
- Skips transformers that require labels (Alpha, ContextPattern, etc.)
- Caches results for fast re-runs
- Returns feature matrix + provenance metadata

**Example**:
```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

pipeline = FeatureExtractionPipeline(
    transformer_names=['Statistical', 'NominativeAnalysis', 'Linguistic'],
    domain_name='movies',
    enable_caching=True
)

# Just texts, no labels
features = pipeline.extract_features(movie_plots)
```

**Automatically skips**:
- `AlphaTransformer` (needs labels)
- `GoldenNarratioTransformer` (needs labels)
- `ContextPatternTransformer` (needs labels)
- `MetaFeatureInteractionTransformer` (needs labels)
- `EnsembleMetaTransformer` (needs labels)
- `CrossDomainEmbeddingTransformer` (needs genome features)

#### 2. Supervised Pipeline

**Use when**: You have outcome labels and want label-aware features

**What it does**:
1. Runs unsupervised pipeline first to build canonical feature matrix
2. Builds genome payloads (nominative + archetypal + historial + uniquity)
3. Applies supervised transformers with labels
4. Returns combined feature matrix

**Example**:
```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline_supervised import SupervisedFeatureExtractionPipeline

pipeline = SupervisedFeatureExtractionPipeline(
    transformer_names=['Statistical', 'NominativeAnalysis', 'Alpha', 'ContextPattern'],
    domain_name='nba',
    enable_caching=True
)

# Texts + outcome labels
features = pipeline.extract_features(game_narratives, outcomes=win_loss)
```

**Supervised transformers**:
- `AlphaTransformer`: Measures narrative-outcome coupling (α)
- `GoldenNarratioTransformer`: Discovers golden ratio patterns (Ξ)
- `ContextPatternTransformer`: Finds context-dependent patterns
- `MetaFeatureInteractionTransformer`: Discovers feature interactions
- `EnsembleMetaTransformer`: Stacked ensemble features
- `CrossDomainEmbeddingTransformer`: Cross-domain genome embeddings

### Pipeline Caching

Both pipelines cache results to avoid recomputation:

```python
# Cache key includes:
# - Domain name
# - Transformer names (sorted)
# - Input data hash
# - Pipeline mode (supervised/unsupervised)
# - Label hash (supervised only)

# Cache location
cache_dir = 'narrative_optimization/cache/features/{domain_name}/'

# Clear cache if needed
pipeline.clear_cache()
```

---

## Transformer Selection

### Manual Selection

```python
transformer_names = [
    'StatisticalTransformer',
    'NominativeAnalysisTransformer',
    'SelfPerceptionTransformer',
    'LinguisticPatternsTransformer'
]
```

### Automatic Selection (Recommended)

Use the `TransformerLibrary` for intelligent selection based on domain narrativity (π):

```python
from narrative_optimization.src.transformers.transformer_library import TransformerLibrary

library = TransformerLibrary()

# Select based on narrativity
selected, feature_count = library.get_for_narrativity(
    π=0.49,  # NBA narrativity
    target_feature_count=300,
    include_statistical=True
)

# Generates report:
# π-GUIDED TRANSFORMER SELECTION (π=0.49)
# Selected 8 transformers (312 features):
#   • statistical (200 feat) - Plot/content features essential
#   • nominative (51 feat) - Identity foundation
#   • linguistic (36 feat) - Voice patterns
#   ...
```

**Selection logic**:
- **Low π (<0.3)**: Plot-driven domains → Statistical, Quantitative, Conflict, Suspense
- **High π (>0.7)**: Character-driven domains → Nominative, SelfPerception, NarrativePotential, Phonetic
- **Mid π (0.3-0.7)**: Balanced → Mix of both approaches

### Domain-Aware Selection

```python
from narrative_optimization.src.pipelines.domain_config import DomainConfig

config = DomainConfig('nba')
selected, rationales, feature_count = library.select_for_config(
    config=config,
    target_feature_count=300,
    require_core=True
)

# Adds domain-type specific transformers:
# - SPORTS → Ensemble, Conflict
# - ENTERTAINMENT → Conflict, Suspense, Framing
# - NOMINATIVE → Nominative, Phonetic, SocialStatus
# - BUSINESS → Authenticity, Expertise, NarrativePotential
```

---

## Common Patterns

### Pattern 1: Quick Feature Extraction

```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

# Minimal setup
pipeline = FeatureExtractionPipeline(
    transformer_names=['Statistical', 'Nominative'],
    domain_name='test'
)
features = pipeline.extract_features(texts)
```

### Pattern 2: Production Pipeline with Caching

```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline
from narrative_optimization.src.transformers.transformer_library import TransformerLibrary

# Select transformers
library = TransformerLibrary()
transformer_names, _ = library.get_for_narrativity(π=0.49, target_feature_count=300)

# Create pipeline
pipeline = FeatureExtractionPipeline(
    transformer_names=transformer_names,
    domain_name='nba',
    cache_dir='narrative_optimization/cache/features',
    enable_caching=True,
    verbose=True
)

# Extract with caching
features = pipeline.extract_features(texts)
feature_names = pipeline.get_feature_names()
provenance = pipeline.get_provenance()
```

### Pattern 3: Supervised Training Pipeline

```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline_supervised import SupervisedFeatureExtractionPipeline

# Include supervised transformers
transformer_names = [
    'Statistical', 'Nominative', 'Linguistic',
    'Alpha', 'ContextPattern'  # Supervised
]

pipeline = SupervisedFeatureExtractionPipeline(
    transformer_names=transformer_names,
    domain_name='nba',
    enable_caching=True
)

# Extract with labels
features = pipeline.extract_features(texts, outcomes=y)
```

### Pattern 4: Validate Transformer Names

```python
from narrative_optimization.src.transformers.registry import get_transformer_registry

registry = get_transformer_registry()

# Check if transformers exist
transformer_names = ['Statistical', 'NominativeAnalysis', 'FakeTransformer']
missing = registry.describe_missing(transformer_names)

if missing:
    for name, suggestions in missing.items():
        print(f"Missing: {name}")
        print(f"Did you mean: {', '.join(suggestions)}")
```

---

## Error Handling

### Class Not Found Errors

**Old behavior** (before registry):
```
AttributeError: module 'narrative_optimization.src.transformers' has no attribute 'NarrativePotentail'
```

**New behavior** (with registry):
```
AttributeError: module 'narrative_optimization.src.transformers' has no transformer 'NarrativePotentail'.
Did you mean: NarrativePotentialTransformer, NarrativePotentialV2Transformer?
Run 'python -m narrative_optimization.tools.list_transformers' to inspect the catalog.
```

### Pipeline Errors

The pipeline handles transformer errors gracefully:

```python
pipeline = FeatureExtractionPipeline(
    transformer_names=['Statistical', 'BrokenTransformer', 'Nominative'],
    domain_name='test'
)

# Pipeline continues, logs errors
features = pipeline.extract_features(texts)

# Check status
status = pipeline.transformer_status
# {
#   'StatisticalTransformer': 'success',
#   'BrokenTransformer': 'error: ...',
#   'NominativeAnalysisTransformer': 'success'
# }
```

---

## Migration from Old Patterns

### Old: Hardcoded Transformer Lists

```python
# DON'T DO THIS (outdated)
from narrative_optimization.src.transformers.statistical import StatisticalTransformer
from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer

transformers = [
    StatisticalTransformer(),
    NominativeAnalysisTransformer()
]
```

### New: Use Registry + Factory

```python
# DO THIS (current)
from narrative_optimization.src.transformers.transformer_factory import TransformerFactory

factory = TransformerFactory()
transformer_names = ['Statistical', 'NominativeAnalysis']
transformers = [factory.create_transformer(name) for name in transformer_names]
```

### Old: Manual Pipeline Construction

```python
# DON'T DO THIS (outdated)
features_list = []
for transformer in transformers:
    transformer.fit(X_train)
    features = transformer.transform(X_test)
    features_list.append(features)
combined = np.hstack(features_list)
```

### New: Use Pipeline

```python
# DO THIS (current)
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

pipeline = FeatureExtractionPipeline(
    transformer_names=['Statistical', 'NominativeAnalysis'],
    domain_name='test'
)
features = pipeline.extract_features(texts)
```

---

## Reference

### Key Files

- **Registry**: `narrative_optimization/src/transformers/registry.py`
- **CLI Tool**: `narrative_optimization/tools/list_transformers.py`
- **Unsupervised Pipeline**: `narrative_optimization/src/pipelines/feature_extraction_pipeline.py`
- **Supervised Pipeline**: `narrative_optimization/src/pipelines/feature_extraction_pipeline_supervised.py`
- **Transformer Factory**: `narrative_optimization/src/transformers/transformer_factory.py`
- **Transformer Library**: `narrative_optimization/src/transformers/transformer_library.py`
- **Base Classes**: `narrative_optimization/src/transformers/base_transformer.py`

### Documentation

- **This Guide**: `/docs/TRANSFORMERS_AND_PIPELINES.md` (you are here)
- **Transformer Catalog**: `/docs/TRANSFORMER_CATALOG.md` (CLI usage)
- **Supervised Handoff**: `/docs/SUPERVISED_TRANSFORMER_HANDOFF.md` (supervised details)
- **Developer Guide**: `/docs/DEVELOPER_GUIDE.md` (architecture overview)
- **Onboarding**: `/docs/ONBOARDING_HANDBOOK.md` (domain onboarding)

### Deprecated Documentation

**DO NOT USE** - These files are outdated and archived in `/archive/deprecated/`:
- `docs/reference/UPDATED_CANONICAL_TRANSFORMERS.md` (hardcoded list)
- `docs/reference/TRANSFORMER_EFFECTIVENESS_ANALYSIS.md` (old analysis)
- `narrative_optimization/docs/DISCOVERY_TRANSFORMERS_GUIDE.md` (outdated patterns)
- See `/archive/deprecated/WARNING.md` for full list

---

## FAQ

### How do I find all available transformers?

```bash
python -m narrative_optimization.tools.list_transformers
```

### How do I know which transformers to use?

Use the `TransformerLibrary` with your domain's narrativity (π):

```python
from narrative_optimization.src.transformers.transformer_library import TransformerLibrary
library = TransformerLibrary()
selected, _ = library.get_for_narrativity(π=0.49, target_feature_count=300)
```

### What's the difference between unsupervised and supervised pipelines?

- **Unsupervised**: Text-only features, no outcome labels needed
- **Supervised**: Includes label-aware features (Alpha, ContextPattern, etc.)

### Why am I getting "class not found" errors?

1. Check the transformer name with the CLI:
   ```bash
   python -m narrative_optimization.tools.list_transformers --check YourTransformerName
   ```
2. The registry will suggest correct names if there's a typo
3. Make sure you're not using old transformer names from archived docs

### How do I clear the cache?

```python
pipeline.clear_cache()
```

Or manually delete: `narrative_optimization/cache/features/{domain_name}/`

### Can I create custom transformers?

Yes. Inherit from `NarrativeTransformer` and place in `narrative_optimization/src/transformers/`. The registry will auto-discover it.

```python
from narrative_optimization.src.transformers.base_transformer import NarrativeTransformer

class MyCustomTransformer(NarrativeTransformer):
    def __init__(self):
        super().__init__(
            narrative_id='my_custom',
            description='My custom narrative features'
        )
    
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        # Your feature extraction logic
        return features
```

---

## Getting Help

1. **Check the CLI**: `python -m narrative_optimization.tools.list_transformers --help`
2. **Read this guide**: You're in the right place
3. **Check test files**: `narrative_optimization/tests/test_transformer_registry.py`
4. **Avoid archived docs**: Anything in `/archive/deprecated/` is outdated

---

**Last Updated**: November 20, 2025  
**Maintained by**: Narrative Optimization Framework  
**Questions**: Check `/docs/BOT_ONBOARDING.md` for quick answers

