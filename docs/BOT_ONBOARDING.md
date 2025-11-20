# Bot Onboarding: Quick Start Guide

**For**: New AI assistants joining a conversation  
**Time**: 2-3 minutes to get oriented  
**Goal**: Understand current system architecture without getting lost in archived materials

---

## 1. System Overview (30 seconds)

**What this is**: A framework that tests whether narrative structure predicts outcomes across different domains (sports, entertainment, business, etc.)

**Current status**: Production system with 2 validated sports betting systems (NHL 69.4% win rate, NFL 66.7% win rate)

**Core formula**: `Д = π × |r| × κ` where:
- `π` (pi) = narrativity (how open/constrained the domain is)
- `r` = correlation between story quality and outcomes
- `κ` (kappa) = coupling strength

---

## 2. Transformers (1 minute)

### What You Need to Know

**Transformers** = Feature extractors that convert narratives into numerical features

**How many?** 100+ transformers automatically discovered by registry system

**How to find them?**
```bash
python -m narrative_optimization.tools.list_transformers
```

**How to use them?**
```python
from narrative_optimization.src.transformers import NominativeAnalysisTransformer
# Or use the factory
from narrative_optimization.src.transformers.transformer_factory import TransformerFactory
factory = TransformerFactory()
transformer = factory.create_transformer('NominativeAnalysis')
```

### Key Transformer Categories

- **Core**: Foundational (Nominative, SelfPerception, NarrativePotential, Linguistic, Ensemble, Relational)
- **Semantic**: Embedding-based understanding
- **Temporal**: Time-based features
- **Structural**: Plot structure (Conflict, Suspense, Framing)
- **Supervised**: Require outcome labels (Alpha, ContextPattern, GoldenNarratio)

### Registry System (NEW - Nov 2025)

The registry automatically discovers all transformers. No manual lists to maintain.

**Key features**:
- Resolves class names, snake_case, partial names
- Provides fuzzy suggestions for typos
- Groups by category
- Zero import overhead (lazy loading)

**If you see "class not found" errors**, the registry will suggest correct names:
```
AttributeError: module has no transformer 'NarrativePotentail'.
Did you mean: NarrativePotentialTransformer, NarrativePotentialV2Transformer?
Run 'python -m narrative_optimization.tools.list_transformers' to inspect the catalog.
```

---

## 3. Pipelines (1 minute)

### Two Pipeline Modes

#### Unsupervised (Default)
**Use when**: You only have text, no outcome labels

```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

pipeline = FeatureExtractionPipeline(
    transformer_names=['Statistical', 'Nominative', 'Linguistic'],
    domain_name='nba',
    enable_caching=True
)
features = pipeline.extract_features(texts)
```

**Automatically skips** transformers that need labels (Alpha, ContextPattern, etc.)

#### Supervised
**Use when**: You have outcome labels and want label-aware features

```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline_supervised import SupervisedFeatureExtractionPipeline

pipeline = SupervisedFeatureExtractionPipeline(
    transformer_names=['Statistical', 'Nominative', 'Alpha', 'ContextPattern'],
    domain_name='nba',
    enable_caching=True
)
features = pipeline.extract_features(texts, outcomes=y)
```

---

## 4. Documentation Structure (30 seconds)

### Current Documentation (USE THESE)

- **`/docs/TRANSFORMERS_AND_PIPELINES.md`** ← **START HERE** for transformers/pipelines
- **`/docs/TRANSFORMER_CATALOG.md`** ← CLI tool usage
- **`/docs/SUPERVISED_TRANSFORMER_HANDOFF.md`** ← Supervised pipeline details
- **`/docs/DEVELOPER_GUIDE.md`** ← Architecture overview
- **`/docs/ONBOARDING_HANDBOOK.md`** ← Domain onboarding process
- **`/README.md`** ← Project overview

### Archived Documentation (DON'T USE)

**`/archive/deprecated/`** contains outdated materials with hardcoded transformer lists and old patterns.

**Warning signs of outdated docs**:
- References to "55 transformers" or "48 transformers" (hardcoded lists)
- Manual transformer imports from specific files
- Pre-November 2024 dates
- Domain-specific transformer guides (MLB_TRANSFORMER_*, NHL_TRANSFORMER_*)

See `/archive/deprecated/WARNING.md` for full list of deprecated files.

---

## 5. Common Tasks

### Task: Find Available Transformers
```bash
python -m narrative_optimization.tools.list_transformers
python -m narrative_optimization.tools.list_transformers --filter nominative
python -m narrative_optimization.tools.list_transformers --summary
```

### Task: Extract Features from Text
```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

pipeline = FeatureExtractionPipeline(
    transformer_names=['Statistical', 'Nominative'],
    domain_name='test'
)
features = pipeline.extract_features(texts)
```

### Task: Select Transformers Automatically
```python
from narrative_optimization.src.transformers.transformer_library import TransformerLibrary

library = TransformerLibrary()
selected, feature_count = library.get_for_narrativity(
    π=0.49,  # Domain narrativity
    target_feature_count=300
)
```

### Task: Validate Transformer Names
```bash
python -m narrative_optimization.tools.list_transformers --check Alpha ContextPattern FakeTransformer
```

### Task: Run Production Pipeline
```python
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline
from narrative_optimization.src.transformers.transformer_library import TransformerLibrary

# Auto-select transformers
library = TransformerLibrary()
transformer_names, _ = library.get_for_narrativity(π=0.49, target_feature_count=300)

# Create pipeline with caching
pipeline = FeatureExtractionPipeline(
    transformer_names=transformer_names,
    domain_name='nba',
    cache_dir='narrative_optimization/cache/features',
    enable_caching=True,
    verbose=True
)

# Extract features
features = pipeline.extract_features(texts)
```

---

## 6. Key Files to Know

### Transformer System
- **Registry**: `narrative_optimization/src/transformers/registry.py`
- **CLI Tool**: `narrative_optimization/tools/list_transformers.py`
- **Factory**: `narrative_optimization/src/transformers/transformer_factory.py`
- **Library**: `narrative_optimization/src/transformers/transformer_library.py`
- **Base Classes**: `narrative_optimization/src/transformers/base_transformer.py`
- **All Transformers**: `narrative_optimization/src/transformers/` (100+ files)

### Pipeline System
- **Unsupervised**: `narrative_optimization/src/pipelines/feature_extraction_pipeline.py`
- **Supervised**: `narrative_optimization/src/pipelines/feature_extraction_pipeline_supervised.py`
- **Domain Config**: `narrative_optimization/src/pipelines/domain_config.py`

### Web Application
- **Main App**: `app.py` (Flask application)
- **Routes**: `routes/` (58 route files)
- **Templates**: `templates/` (63 HTML files)

---

## 7. What NOT to Do

### ❌ Don't Use Hardcoded Transformer Lists
```python
# DON'T DO THIS
from narrative_optimization.src.transformers.statistical import StatisticalTransformer
from narrative_optimization.src.transformers.nominative import NominativeAnalysisTransformer
transformers = [StatisticalTransformer(), NominativeAnalysisTransformer()]
```

### ✅ Use Registry + Factory
```python
# DO THIS
from narrative_optimization.src.transformers.transformer_factory import TransformerFactory
factory = TransformerFactory()
transformers = [factory.create_transformer(name) for name in ['Statistical', 'NominativeAnalysis']]
```

### ❌ Don't Reference Archived Docs
- Anything in `/archive/deprecated/`
- Docs with "55 transformers" or "48 transformers"
- Domain-specific transformer guides (pre-Nov 2024)

### ✅ Use Current Docs
- `/docs/TRANSFORMERS_AND_PIPELINES.md`
- `/docs/TRANSFORMER_CATALOG.md`
- Registry CLI tool

### ❌ Don't Manually Construct Pipelines
```python
# DON'T DO THIS
features_list = []
for transformer in transformers:
    transformer.fit(X_train)
    features = transformer.transform(X_test)
    features_list.append(features)
combined = np.hstack(features_list)
```

### ✅ Use Pipeline Classes
```python
# DO THIS
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline
pipeline = FeatureExtractionPipeline(transformer_names=['Statistical', 'Nominative'], domain_name='test')
features = pipeline.extract_features(texts)
```

---

## 8. Quick Troubleshooting

### "Class not found" errors
1. Run: `python -m narrative_optimization.tools.list_transformers --check YourTransformerName`
2. Registry will suggest correct names
3. Check you're not using old names from archived docs

### Pipeline errors
- Check `pipeline.transformer_status` for error details
- Pipeline continues even if some transformers fail
- Enable `verbose=True` for detailed logging

### Import deadlocks (macOS)
- Always run: `source scripts/env_setup.sh` before starting Python
- Sets environment variables to prevent TensorFlow/PyTorch mutex deadlocks

### Cache issues
- Clear cache: `pipeline.clear_cache()`
- Or delete: `narrative_optimization/cache/features/{domain_name}/`

---

## 9. Next Steps

After orientation, dive deeper:

1. **Read full transformer guide**: `/docs/TRANSFORMERS_AND_PIPELINES.md`
2. **Understand architecture**: `/docs/DEVELOPER_GUIDE.md`
3. **Learn domain onboarding**: `/docs/ONBOARDING_HANDBOOK.md`
4. **Check production systems**: `/docs/betting_systems/` for NHL/NFL examples

---

## 10. Quick Reference Card

```bash
# List transformers
python -m narrative_optimization.tools.list_transformers

# Search transformers
python -m narrative_optimization.tools.list_transformers --filter semantic

# Validate names
python -m narrative_optimization.tools.list_transformers --check Alpha ContextPattern

# Run Flask app (after env setup)
source scripts/env_setup.sh
python3 app.py

# Extract features (Python)
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline
pipeline = FeatureExtractionPipeline(transformer_names=['Statistical', 'Nominative'], domain_name='test')
features = pipeline.extract_features(texts)
```

---

**Welcome to the Narrative Optimization Framework!**

Start with `/docs/TRANSFORMERS_AND_PIPELINES.md` and you'll be productive in minutes.

**Last Updated**: November 20, 2025

