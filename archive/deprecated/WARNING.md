# ⚠️ DEPRECATED MATERIALS - DO NOT USE

**Last Updated**: November 20, 2025

---

## Important Notice

**All files in this directory are OUTDATED and should NOT be used for new development.**

These materials have been archived because they:
- Contain hardcoded transformer lists (replaced by registry system)
- Reference old pipeline patterns (pre-November 2024)
- Use manual transformer instantiation (replaced by factory pattern)
- Provide domain-specific guidance that's been superseded by universal patterns

---

## What to Use Instead

### For Transformers

**OLD (Don't Use)**:
- Hardcoded lists like "55 transformers" or "48 transformers"
- Manual imports from specific transformer files
- Domain-specific transformer guides

**NEW (Use This)**:
```bash
# Find available transformers
python -m narrative_optimization.tools.list_transformers

# Validate transformer names
python -m narrative_optimization.tools.list_transformers --check TransformerName
```

**Documentation**: `/docs/TRANSFORMERS_AND_PIPELINES.md`

### For Pipelines

**OLD (Don't Use)**:
- Manual pipeline construction with loops
- Domain-specific pipeline scripts
- Hardcoded transformer instantiation

**NEW (Use This)**:
```python
# Unsupervised pipeline
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

pipeline = FeatureExtractionPipeline(
    transformer_names=['Statistical', 'Nominative'],
    domain_name='test',
    enable_caching=True
)
features = pipeline.extract_features(texts)

# Supervised pipeline
from narrative_optimization.src.pipelines.feature_extraction_pipeline_supervised import SupervisedFeatureExtractionPipeline

pipeline = SupervisedFeatureExtractionPipeline(
    transformer_names=['Statistical', 'Alpha', 'ContextPattern'],
    domain_name='test',
    enable_caching=True
)
features = pipeline.extract_features(texts, outcomes=y)
```

**Documentation**: `/docs/TRANSFORMERS_AND_PIPELINES.md`

### For Transformer Selection

**OLD (Don't Use)**:
- Manual transformer lists
- Domain-specific selection logic

**NEW (Use This)**:
```python
from narrative_optimization.src.transformers.transformer_library import TransformerLibrary

library = TransformerLibrary()
selected, feature_count = library.get_for_narrativity(
    π=0.49,  # Domain narrativity
    target_feature_count=300
)
```

**Documentation**: `/docs/TRANSFORMERS_AND_PIPELINES.md`

---

## Archived Contents

### Documentation (9 files)

These docs contain outdated transformer lists and patterns:

1. `UPDATED_CANONICAL_TRANSFORMERS.md` - Hardcoded list of 45 transformers (replaced by registry)
2. `README_TRANSFORMER_ANALYSIS.md` - Old analysis patterns
3. `TRANSFORMER_EFFECTIVENESS_ANALYSIS.md` - Superseded by recent backtests
4. `TRANSFORMER_SPEED_OPTIMIZATION_GUIDE.md` - Outdated optimization advice
5. `MLB_TRANSFORMER_OPTIMIZATION_COMPLETE.md` - Domain-specific, old patterns
6. `NHL_TRANSFORMER_DISCOVERY.md` - Domain-specific, old patterns
7. `DISCOVERY_TRANSFORMERS_GUIDE.md` - Good concepts but outdated implementation
8. `DISCOVERY_TRANSFORMER_DATA_REQUIREMENTS.md` - Outdated data specs
9. `CONTEXT_PATTERN_TRANSFORMER_GUIDE.md` - Single transformer guide, too specific

### Scripts (13 files)

These scripts use hardcoded transformer lists and old patterns:

1. `test_ALL_55_transformers_NBA_COMPREHENSIVE.py` - Hardcoded list
2. `test_ALL_55_transformers_GOLF.py` - Hardcoded list
3. `test_ALL_transformers_comprehensive.py` - Hardcoded list
4. `run_all_transformers_movies.py` - Hardcoded list
5. `cleanup_transformers_UPDATED.py` - One-off cleanup script
6. `cleanup_transformers.py` - Old cleanup script
7. `fix_transformer_input_shapes.py` - One-off fix
8. `analyze_transformer_performance_simple.py` - Outdated analysis
9. `analyze_transformer_performance.py` - Outdated analysis
10. `phase4_narrative_transformers.py` - Domain-specific old pattern
11. `phase2_nfl_transformer.py` - Domain-specific old pattern
12. `run_all_transformers.py` - Hardcoded list
13. `fix_all_transformers.py` - One-off fix

---

## Migration Guide

See `/archive/deprecated/MIGRATION_MAP.md` for detailed migration instructions from old patterns to new patterns.

---

## Current Documentation

**Start here for new development**:

1. **Bot Onboarding**: `/docs/BOT_ONBOARDING.md` - Quick start (2-3 minutes)
2. **Transformers & Pipelines**: `/docs/TRANSFORMERS_AND_PIPELINES.md` - Complete guide
3. **Transformer Catalog**: `/docs/TRANSFORMER_CATALOG.md` - CLI tool usage
4. **Developer Guide**: `/docs/DEVELOPER_GUIDE.md` - Architecture overview
5. **Onboarding Handbook**: `/docs/ONBOARDING_HANDBOOK.md` - Domain onboarding

---

## Questions?

If you're unsure whether to use something from this directory:

**The answer is NO. Use the current documentation instead.**

1. Check `/docs/BOT_ONBOARDING.md` for quick answers
2. Read `/docs/TRANSFORMERS_AND_PIPELINES.md` for detailed guidance
3. Run `python -m narrative_optimization.tools.list_transformers` to see current transformers

---

**Remember**: If it's in `/archive/deprecated/`, it's outdated. Period.

Use the current documentation in `/docs/` instead.

