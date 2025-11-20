# Domain Onboarding Handbook

**Version 2.0 - Unified Pipeline System**

## Overview

This handbook guides you through onboarding a new domain into the Narrative Optimization Framework. The new unified system reduces onboarding time from **4 weeks to 2-3 days**.

---

## Quick Start

### Automated Onboarding (Recommended)

```bash
cd narrative_optimization
python scripts/onboard_domain.py
```

The wizard will guide you through:
1. Domain information
2. Narrativity assessment
3. Data schema definition
4. Transformer selection (automated)
5. File generation
6. Data loading and validation
7. Initial analysis

**Time:** 30-60 minutes

---

## Manual Onboarding

If you prefer manual setup or need more control:

### Step 1: Create Domain Directory

```bash
mkdir -p narrative_optimization/domains/{domain_name}
cd narrative_optimization/domains/{domain_name}
```

### Step 2: Create Configuration File

Create `config.yaml`:

```yaml
domain: {domain_name}
type: {domain_type}  # sports, entertainment, nominative, business, medical, hybrid
narrativity:
  structural: 0.5  # How many narrative paths possible? [0, 1]
  temporal: 0.5    # Does it unfold over time? [0, 1]
  agency: 0.5      # Do actors have choice? [0, 1]
  interpretive: 0.5  # Is judgment subjective? [0, 1]
  format: 0.5      # How flexible is the medium? [0, 1]
data:
  text_fields: ['text']  # Fields containing narrative text
  outcome_field: 'outcome'  # Field containing outcomes
  context_fields: []  # Optional context features
  name_field: null  # Optional organism names
outcome_type: binary  # binary, continuous, or ranked
perspectives: ['director', 'audience', 'critic']  # Auto-set by domain type
quality_methods: ['weighted_mean', 'ensemble']  # Auto-set by domain type
scales: ['micro', 'meso', 'macro']  # Auto-set by domain type
```

### Step 3: Assess Narrativity

Rate each component from 0.0 to 1.0:

**п_structural**: How many narrative paths possible?
- 0.0: Single deterministic path (chess moves)
- 1.0: Infinite narrative possibilities (creative writing)

**п_temporal**: Does it unfold over time?
- 0.0: Static snapshot (photo)
- 1.0: Dynamic temporal sequence (movie, game)

**п_agency**: Do actors have choice?
- 0.0: No agency (natural disaster)
- 1.0: Full agency (character decisions)

**п_interpretive**: Is judgment subjective?
- 0.0: Objective (scientific measurement)
- 1.0: Highly subjective (artistic critique)

**п_format**: How flexible is the medium?
- 0.0: Rigid format (sports scoreboard)
- 1.0: Flexible format (novel)

**Example: NBA**
- п_structural: 0.75 (many game outcomes possible)
- п_temporal: 0.90 (games unfold over time)
- п_agency: 0.75 (players make choices)
- п_interpretive: 0.50 (some subjectivity in calls)
- п_format: 0.40 (structured game format)

**Calculated п**: 0.70 (high narrativity)

### Step 4: Define Data Schema

**Text Fields**: Which fields contain narrative text?
- Examples: `['description']`, `['plot', 'summary']`, `['text']`

**Outcome Field**: Which field contains outcomes?
- Examples: `'win'`, `'rating'`, `'price'`, `'success'`

**Context Fields** (optional): Additional features
- Examples: `['genre', 'budget']`, `['season', 'home_away']`

**Name Field** (optional): Organism names
- Examples: `'name'`, `'title'`, `'team_name'`

### Step 5: Test Configuration

```python
from src.pipelines.domain_config import DomainConfig

config = DomainConfig.from_yaml('config.yaml')
print(f"Domain: {config.domain}")
print(f"п: {config.pi:.3f}")
print(f"Type: {config.type.value}")
```

### Step 6: Select Transformers and Run Pipeline

**For complete transformer documentation, see**: `/docs/TRANSFORMERS_AND_PIPELINES.md`

```python
# Auto-select transformers based on narrativity
from narrative_optimization.src.transformers.transformer_library import TransformerLibrary

library = TransformerLibrary()
transformer_names, feature_count = library.get_for_narrativity(
    π=config.pi,
    target_feature_count=300
)

# Run feature extraction pipeline
from narrative_optimization.src.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

pipeline = FeatureExtractionPipeline(
    transformer_names=transformer_names,
    domain_name=config.domain,
    enable_caching=True
)

features = pipeline.extract_features(texts)
```

**CLI Tool**: List available transformers anytime:
```bash
python -m narrative_optimization.tools.list_transformers
```

### Step 7: Validate Results

```python
analysis = results['analysis']
print(f"r_narrative: {analysis['r_narrative']:.3f}")
print(f"Д: {analysis['Д']:.3f}")
print(f"Efficiency: {analysis['efficiency']:.3f}")

# Efficiency test
if analysis['efficiency'] > 0.5:
    print("✓ Narrative laws apply")
else:
    print("⚠ Reality constraints dominate")
```

---

## Domain Type Selection

### Sports Domains

**Type**: `sports`, `sports_individual`, or `sports_team`

**Characteristics**:
- Competitive outcomes (win/loss)
- Temporal structure (games, seasons)
- Agency (players/teams make choices)
- Medium narrativity (п ≈ 0.5-0.8)

**Examples**: NBA, Tennis, Golf, Soccer

**Perspectives**: audience, authority (coach), star (players), collective

### Entertainment Domains

**Type**: `entertainment`

**Characteristics**:
- Creative content (movies, music, WWE)
- High narrativity (п ≈ 0.7-0.9)
- Subjective evaluation
- Cultural context matters

**Examples**: Movies, Music, WWE, TV Shows

**Perspectives**: director, audience, critic, cultural, meta

### Nominative Domains

**Type**: `nominative`

**Characteristics**:
- Name-based analysis (housing, hurricanes)
- Nominative features emphasized
- Variable narrativity (п ≈ 0.3-0.7)

**Examples**: Housing, Hurricanes, Real Estate

**Perspectives**: character, cultural, meta

### Business Domains

**Type**: `business`

**Characteristics**:
- Market outcomes (success/failure, valuation)
- Strategic narratives
- Medium narrativity (п ≈ 0.4-0.6)

**Examples**: Startups, Crypto, Companies

**Perspectives**: authority (CEO), audience (market), cultural, meta

### Medical Domains

**Type**: `medical`

**Characteristics**:
- Health outcomes
- Patient narratives
- Medium narrativity (п ≈ 0.5-0.7)

**Examples**: Mental Health, Clinical Trials

**Perspectives**: authority (clinician), character (patient), cultural

---

## Data Format

### JSON Format

```json
[
  {
    "text": "Narrative text here...",
    "outcome": 1,
    "name": "Organism Name",
    "context": {
      "field1": "value1",
      "field2": "value2"
    }
  },
  ...
]
```

### CSV Format

```csv
text,outcome,name,context_field1,context_field2
"Narrative text...",1,"Name","value1","value2"
...
```

### Required Fields

- **Text field(s)**: At least one field containing narrative text
- **Outcome field**: Field containing outcomes (binary, continuous, or ranked)

### Optional Fields

- **Name field**: Organism names (for nominative analysis)
- **Context fields**: Additional features (for context-dependent analysis)

---

## Transformer Selection

Transformers are **automatically selected** based on:

1. **п value**: Low п → statistical/plot features, High п → narrative/character features
2. **Domain type**: Sports → ensemble/conflict, Entertainment → suspense/framing
3. **Core transformers**: Always included (nominative, self_perception, etc.)

### Custom Transformers

Add custom transformers to config:

```yaml
transformer_augmentation:
  - custom_transformer_1
  - custom_transformer_2
```

---

## Validation Process

### Presumption

**Hypothesis**: Narrative laws should apply to this domain

**Test**: Д/п > 0.5

### Expected Results

**If Д/п > 0.5**:
- ✓ Narrative quality influences outcomes
- ✓ Better stories predict success
- ✓ Framework validated for this domain

**If Д/п ≤ 0.5**:
- ⚠ Reality constraints dominate
- ⚠ External factors matter more
- ⚠ Narrative has limited causal impact

### Validation Report

After analysis, review `VALIDATION_REPORT.md`:

```markdown
# Domain - Validation Report

## Results
- Narrativity (п): 0.70
- Correlation (r): 0.75
- Bridge (Д): 0.45
- Efficiency: 0.64

## Validation Result
✓ PASS (Д/п = 0.64 > 0.5)
```

---

## Testing

### Run Domain Tests

```bash
pytest narrative_optimization/domains/{domain_name}/tests/
```

### Run All Domain Tests

```bash
pytest narrative_optimization/tests/domain_tests/
```

### Continuous Validation

```bash
python narrative_optimization/tests/continuous/validation_monitor.py
```

---

## Troubleshooting

### Issue: Config Validation Fails

**Solution**: Check narrativity components are in [0, 1]

```python
config.narrativity.validate()  # Raises if invalid
```

### Issue: Data Loading Fails

**Solution**: Verify data format matches schema

```python
# Check text fields exist
assert 'text' in data[0]  # or your text field name

# Check outcome field exists
assert 'outcome' in data[0]  # or your outcome field name
```

### Issue: No Transformers Selected

**Solution**: Check п value and domain type

```python
# Low п domains need statistical features
# High п domains need narrative features
# Check transformer selection rationales
print(pipeline_info['rationales'])
```

### Issue: Results Don't Match Expectations

**Solution**: 
1. Review narrativity assessment
2. Check data quality
3. Verify outcome field is correct
4. Compare to similar domains

---

## Next Steps After Onboarding

1. **Review Results**: Check validation report and analysis results
2. **Run Tests**: Ensure all tests pass
3. **Save Baseline**: Save results for regression testing
4. **Update Dashboard**: Add Flask route and template
5. **Documentation**: Update domain-specific documentation

---

## Resources

- **Migration Guide**: `docs/MIGRATION_GUIDE.md` (for migrating existing domains)
- **Developer Guide**: `docs/DEVELOPER_GUIDE.md` (for extending the framework)
- **Framework Quick Reference**: `FRAMEWORK_QUICKREF.md`
- **Formal Variable System**: `FORMAL_VARIABLE_SYSTEM.md`

---

## Support

For onboarding issues:
1. Check this handbook
2. Review example domains in `domains/`
3. Run onboarding wizard for guidance
4. Check test files for examples

**Onboarding Complete When:**
- ✅ Config file created and validated
- ✅ Data loaded successfully
- ✅ Pipeline runs without errors
- ✅ Validation report generated
- ✅ Tests pass
- ✅ Results saved

