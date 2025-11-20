# Migration Guide: Legacy Domains to Unified Pipeline System

## Overview

This guide helps migrate existing domains (NBA, Movies, Housing, etc.) from the legacy fragmented pipeline code to the new unified pipeline system.

**Benefits of Migration:**
- Unified codebase (no duplication across 17+ domains)
- Automated transformer selection (п-based + domain-type logic)
- Multi-perspective/multi-method analysis
- Standardized validation and testing
- Faster onboarding (4 weeks → 2-3 days)

---

## Migration Checklist

### Phase 1: Assessment

- [ ] Identify domain type (sports, entertainment, nominative, etc.)
- [ ] Locate existing data files
- [ ] Document current pipeline steps
- [ ] Note any custom transformers/validators
- [ ] Record current results (r-values, Д, efficiency)

### Phase 2: Configuration

- [ ] Create `config.yaml` using onboarding wizard
- [ ] Assess narrativity components (п_structural, п_temporal, etc.)
- [ ] Define data schema (text fields, outcome field, context fields)
- [ ] Select domain type (auto-detected from existing code)

### Phase 3: Data Migration

- [ ] Ensure data files are in standard format (JSON or CSV)
- [ ] Verify data schema matches config
- [ ] Test data loading with new system

### Phase 4: Pipeline Migration

- [ ] Run onboarding wizard to generate files
- [ ] Review auto-selected transformers
- [ ] Add any custom transformers to `transformer_augmentation`
- [ ] Run initial analysis

### Phase 5: Validation

- [ ] Compare results to legacy system
- [ ] Verify r-values match (within tolerance)
- [ ] Check that Д/п efficiency is consistent
- [ ] Run regression tests

### Phase 6: Cleanup

- [ ] Archive old domain-specific scripts
- [ ] Update Flask routes to use new system
- [ ] Update documentation
- [ ] Remove duplicate code

---

## Step-by-Step Migration

### Step 1: Run Onboarding Wizard

```bash
cd narrative_optimization
python scripts/onboard_domain.py
```

The wizard will:
1. Ask for domain information
2. Guide narrativity assessment
3. Define data schema
4. Auto-select transformers
5. Generate all files

### Step 2: Review Generated Config

Check `domains/{domain_name}/config.yaml`:

```yaml
domain: nba
type: sports_team
narrativity:
  structural: 0.75
  temporal: 0.90
  agency: 0.75
  interpretive: 0.50
  format: 0.40
data:
  text_fields: ['description', 'summary']
  outcome_field: 'win'
  context_fields: ['season', 'home_away']
  name_field: 'team_name'
outcome_type: binary
perspectives: ['audience', 'authority', 'star', 'collective']
quality_methods: ['weighted_mean', 'ensemble']
scales: ['micro', 'meso', 'macro']
```

### Step 3: Customize if Needed

**Add custom transformers:**
```yaml
transformer_augmentation:
  - custom_transformer_1
  - custom_transformer_2
```

**Override perspective preferences:**
```yaml
perspectives: ['director', 'audience', 'critic']  # Override defaults
```

### Step 4: Test Pipeline Composition

```python
from src.pipelines.domain_config import DomainConfig
from src.pipelines.pipeline_composer import PipelineComposer

config = DomainConfig.from_yaml('domains/nba/config.yaml')
composer = PipelineComposer()
pipeline_info = composer.compose_pipeline(config, target_feature_count=300)

print(f"Selected {len(pipeline_info['transformers'])} transformers")
print(f"Total features: {pipeline_info['feature_count']}")
```

### Step 5: Run Analysis

```python
results = composer.run_pipeline(
    config,
    data_path='data/nba_data.json',
    target_feature_count=300,
    use_cache=True
)

print(f"r_narrative: {results['analysis']['r_narrative']:.3f}")
print(f"Д: {results['analysis']['Д']:.3f}")
print(f"Efficiency: {results['analysis']['efficiency']:.3f}")
```

### Step 6: Compare to Legacy Results

```python
# Legacy results
legacy_r = 0.75
legacy_D = 0.45
legacy_efficiency = 0.60

# New results
new_r = results['analysis']['r_narrative']
new_D = results['analysis']['Д']
new_efficiency = results['analysis']['efficiency']

# Check tolerance
tolerance = 0.05
assert abs(new_r - legacy_r) < tolerance, f"r-value changed: {new_r} vs {legacy_r}"
assert abs(new_D - legacy_D) < tolerance, f"Д changed: {new_D} vs {legacy_D}"
```

### Step 7: Save Baseline

```python
from tests.domain_tests.test_regression import RegressionBaseline

baseline_mgr = RegressionBaseline()
baseline_mgr.save_baseline('nba', results['analysis'])
```

---

## Common Migration Issues

### Issue 1: Data Format Mismatch

**Problem:** Legacy data format doesn't match new schema.

**Solution:** Create data adapter:

```python
def adapt_legacy_data(legacy_data):
    """Adapt legacy data format to new schema"""
    adapted = []
    for record in legacy_data:
        adapted.append({
            'text': record['old_text_field'],
            'outcome': record['old_outcome_field'],
            'name': record.get('old_name_field'),
            'context': {
                'season': record.get('season'),
                # ... other context fields
            }
        })
    return adapted
```

### Issue 2: Custom Transformers

**Problem:** Domain uses custom transformers not in library.

**Solution:** 
1. Add transformer to `transformer_library.py`
2. Add to `transformer_augmentation` in config
3. Or create domain-specific transformer class

### Issue 3: Different п Calculation

**Problem:** Legacy system used different п value.

**Solution:** 
1. Review narrativity components
2. Adjust if needed (but document why)
3. Compare results with both п values

### Issue 4: Missing Context Features

**Problem:** Legacy system didn't use context features.

**Solution:**
1. Set `context_fields: []` in config
2. Or extract context features from existing data
3. Context features are optional

---

## Example: Migrating NBA Domain

### Before (Legacy)

```python
# analyze_nba_complete.py
import pandas as pd
from transformers.nominative import NominativeTransformer
from transformers.ensemble import EnsembleTransformer
# ... 200+ lines of domain-specific code

def analyze_nba():
    data = pd.read_csv('nba_data.csv')
    # ... custom processing
    transformers = [NominativeTransformer(), EnsembleTransformer(), ...]
    # ... custom analysis
    return results
```

### After (Unified System)

```yaml
# domains/nba/config.yaml
domain: nba
type: sports_team
narrativity:
  structural: 0.75
  temporal: 0.90
  agency: 0.75
  interpretive: 0.50
  format: 0.40
data:
  text_fields: ['description']
  outcome_field: 'win'
outcome_type: binary
```

```python
# Run analysis
from src.pipelines.pipeline_composer import PipelineComposer
from src.pipelines.domain_config import DomainConfig

config = DomainConfig.from_yaml('domains/nba/config.yaml')
composer = PipelineComposer()
results = composer.run_pipeline(config, data_path='data/nba_data.csv')
```

**Result:** 200+ lines → 10 lines, with better features.

---

## Validation After Migration

### 1. Results Match

```python
# Compare key metrics
legacy_metrics = {'r': 0.75, 'Д': 0.45, 'efficiency': 0.60}
new_metrics = results['analysis']

for metric, legacy_val in legacy_metrics.items():
    new_val = new_metrics.get(metric, 0)
    assert abs(new_val - legacy_val) < 0.05, f"{metric} mismatch"
```

### 2. Features Similar

```python
# Compare feature importance
legacy_top_features = ['nominative_density', 'ensemble_coherence', ...]
new_top_features = [f['name'] for f in results['top_features'][:10]]

# Should have significant overlap
overlap = len(set(legacy_top_features) & set(new_top_features))
assert overlap >= 5, "Feature importance mismatch"
```

### 3. Performance Acceptable

```python
# Check analysis time
analysis_time = results['metadata']['analysis_time_seconds']
assert analysis_time < 300, "Analysis too slow"  # 5 minutes max
```

---

## Post-Migration

### 1. Archive Legacy Code

```bash
mkdir -p docs/archive/legacy_domains
mv analyze_nba_complete.py docs/archive/legacy_domains/
```

### 2. Update Flask Routes

Replace old route with new:

```python
# Old
@nba_bp.route('/nba')
def nba_dashboard():
    # ... custom code
    return render_template('nba_dashboard.html', results=results)

# New
@nba_bp.route('/nba')
def nba_dashboard():
    return render_template('nba_dashboard.html')  # Uses unified system
```

### 3. Update Documentation

- Update domain-specific README
- Note migration date
- Document any customizations

---

## Support

For migration issues:
1. Check `docs/DEVELOPER_GUIDE.md` for system details
2. Review `docs/ONBOARDING_HANDBOOK.md` for onboarding process
3. Run tests: `pytest tests/domain_tests/`
4. Check continuous validation dashboard

---

**Migration Complete When:**
- ✅ Config file created and validated
- ✅ Pipeline runs successfully
- ✅ Results match legacy (within tolerance)
- ✅ Tests pass
- ✅ Legacy code archived
- ✅ Documentation updated

