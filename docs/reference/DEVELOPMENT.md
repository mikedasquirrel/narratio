# Development Guide
## Working with the Narrative Optimization Framework

**Last Updated**: November 16, 2025  
**Status**: Active Development

---

## Quick Start for Developers

### Project Structure

```
novelization/
├── README.md                          # Main project overview
├── DOMAIN_STATUS.md                   # Current domain progress
├── DOMAIN_DEVELOPMENT_STAGES.md       # 10-stage framework
├── NARRATIVE_CATALOG.md               # Universal narrative patterns
├── FORMAL_VARIABLE_SYSTEM.md          # Technical variable reference
├── DOMAIN_SPECTRUM_ANALYSIS.md        # Cross-domain findings
│
├── app.py                             # Flask web application
├── requirements.txt                   # Python dependencies
│
├── routes/                            # Web routes (48 domain routes)
├── templates/                         # HTML templates
├── static/                            # CSS, JS, assets
│
├── data/                              # Real data only
│   └── domains/                       # Domain-specific datasets
│
├── narrative_optimization/            # Core framework
│   ├── src/                           # Source code
│   │   ├── transformers/              # Feature extraction
│   │   ├── analysis/                  # Analysis tools
│   │   └── learning/                  # ML pipelines
│   │
│   ├── domains/                       # Domain-specific work
│   │   └── {domain}/                  # Per-domain analysis
│   │       ├── config.yaml            # Domain configuration
│   │       ├── data/                  # Domain data
│   │       └── models/                # Production models
│   │
│   ├── experiments/                   # Research experiments
│   ├── scripts/                       # Utility scripts
│   └── utils/                         # Helper functions
│
├── docs/                              # Extended documentation
│   ├── theory/                        # Theoretical frameworks
│   ├── guides/                        # How-to guides
│   ├── domains/                       # Domain-specific docs
│   └── technical/                     # Technical reference
│
├── scripts/                           # Top-level utility scripts
└── utils/                             # Top-level utilities
```

---

## Development Workflow

### Adding a New Domain

**Complete guide**: See `ADD_NEW_DOMAIN_TEMPLATE.md`

**Quick version**:

1. **Create domain config** (`narrative_optimization/domains/{domain}/config.yaml`)
2. **Collect data** (use scripts in `narrative_optimization/scripts/`)
3. **Apply transformers** (universal + domain-specific)
4. **Calculate formula** (п, Д, r, κ)
5. **Build optimization** (if applicable)
6. **Create web route** (`routes/{domain}.py`)
7. **Update DOMAIN_STATUS.md**

**Stages**: Follow the 10-stage framework in `DOMAIN_DEVELOPMENT_STAGES.md`

---

### Working with Transformers

**Location**: `narrative_optimization/src/transformers/`

**Universal Transformers** (47 total):
- Apply to ALL domains
- Extract universal narrative patterns
- Examples: `NominativeAnalysisTransformer`, `TemporalMomentumTransformer`, `CharacterComplexityTransformer`

**Domain-Specific Transformers**:
- Contextualize universal patterns for specific domains
- Location: `narrative_optimization/src/transformers/{category}/`
- Examples: `NFLPerformanceTransformer`, `NBAPerformanceTransformer`, `TennisPerformanceTransformer`

**Creating a New Transformer**:

```python
from narrative_optimization.src.transformers.base import BaseTransformer

class MyDomainTransformer(BaseTransformer):
    def __init__(self):
        super().__init__(name="my_domain")
    
    def transform(self, X):
        # Extract domain-specific features
        features = self._extract_features(X)
        return features
    
    def _extract_features(self, X):
        # Your feature extraction logic
        pass
```

---

### Running the Web Interface

```bash
# Start the Flask application
python3 app.py

# Access at http://127.0.0.1:5738/
```

**Key routes**:
- `/` - Home page
- `/domains/compare` - Compare all domains
- `/nfl` - NFL domain page
- `/nba` - NBA domain page
- `/{domain}` - Any domain page

---

### Data Management

**Real Data Location**: `data/domains/`

**Domain Data Structure**:
```
data/domains/
├── nfl_real_games.json           # NFL games with odds
├── nba_real_games.json           # NBA games
├── tennis_matches.json           # Tennis matches
└── {domain}_data.json            # Domain-specific data
```

**Caching**:
- Feature caches stored in `narrative_optimization/domains/{domain}/`
- Production models in `narrative_optimization/domains/{domain}/models/`
- Regenerable caches can be deleted safely

---

## Domain Formula Workflow

### 1. Calculate Narrativity (п)

Five components:
- **Subjectivity** (0-1): How much subjective interpretation matters
- **Agency** (0-1): How much narrator control affects outcomes
- **Observability** (0-1): How visible/observable the outcomes are
- **Generativity** (0-1): How much new narrative content is created
- **Constraint** (0-1): How constrained by external forces (inverted)

**Formula**: `п = (subjectivity + agency + observability + generativity + constraint) / 5`

### 2. Extract Features (ж)

- Apply universal transformers (47 features)
- Apply domain-specific transformers (10-30 features)
- Combine into feature matrix (ж)

### 3. Calculate Story Quality (ю)

- Aggregate features into single story quality score (0-1)
- Weight by domain-specific importance (α)

### 4. Measure Correlation (r)

- Correlate story quality (ю) with outcomes (❊)
- Positive r = better stories win
- Negative r = worse stories win (anti-narrative)

### 5. Determine Coupling (κ)

- **κ = 1.0**: Narrator and narrated are same (self-perception)
- **κ = 0.5-0.9**: Partial coupling (sports, entertainment)
- **κ = 0.0-0.4**: Weak coupling (physics, math)

### 6. Calculate Narrative Agency (Д)

**Formula**: `Д = п × |r| × κ`

**Interpretation**:
- **Д/п > 0.5**: Narrative matters (passes threshold)
- **Д/п < 0.5**: Narrative doesn't control outcomes (fails)

### 7. Document Results

Update `DOMAIN_STATUS.md` with:
- Domain name
- Stage (1-10)
- Formula values (п, Д, r, κ)
- Verdict (pass/fail)
- Notes

---

## Optimization Workflow

**When**: After domain formula is complete (Stage 6+)

**Goal**: Find practical utility even if formula fails threshold

### Process

1. **Feature Selection**:
   - Identify most predictive features
   - Test different feature combinations
   - Use domain knowledge to guide selection

2. **Model Training**:
   - Train on historical data
   - Use temporal validation (not random split)
   - Test for overfitting

3. **Pattern Discovery**:
   - Look for conditional edges (when does signal strengthen?)
   - Test timing effects (late season, playoffs, etc.)
   - Identify market inefficiencies

4. **Validation**:
   - Test with REAL odds/outcomes
   - Calculate ROI and accuracy
   - Risk assessment

5. **Production Deployment**:
   - Save model to `narrative_optimization/domains/{domain}/models/`
   - Create betting system route (if applicable)
   - Monitor performance

---

## Best Practices

### Code Organization

1. **Keep root directory clean**: Only essential files
2. **Use domain folders**: All domain work in `narrative_optimization/domains/{domain}/`
3. **Delete experimental files**: Don't commit intermediate results
4. **Document decisions**: Update relevant markdown files

### Data Management

1. **Real data only**: No synthetic/simulated data in production
2. **Version control data**: Track data changes in DOMAIN_STATUS.md
3. **Cache wisely**: Cache expensive computations, but document what can be regenerated
4. **Validate data**: Always check data quality before analysis

### Documentation

1. **Single source of truth**: One canonical doc per concept
2. **Update on changes**: Keep DOMAIN_STATUS.md current
3. **Be honest**: Report failures as prominently as successes
4. **Version docs**: Date stamp and version all major docs

### Testing

1. **Temporal validation**: Use time-based splits, not random
2. **Real validation**: Test with actual odds/outcomes
3. **Out-of-sample**: Never test on training data
4. **Document failures**: Failed experiments are valuable

---

## Common Tasks

### Adding a Domain Route

1. Create `routes/{domain}.py`:

```python
from flask import Blueprint, render_template

{domain}_bp = Blueprint('{domain}', __name__)

@{domain}_bp.route('/{domain}')
def {domain}_page():
    # Load domain data and results
    return render_template('{domain}.html', data=data)
```

2. Register in `app.py`:

```python
from routes.{domain} import {domain}_bp
app.register_blueprint({domain}_bp)
```

3. Create template `templates/{domain}.html`

### Running an Analysis

```bash
# From domain directory
cd narrative_optimization/domains/{domain}

# Run analysis script
python3 analyze_{domain}_complete.py

# Results saved to:
# - {domain}_results.json
# - {domain}_analysis.json
```

### Updating Transformers

```bash
# Location
cd narrative_optimization/src/transformers/

# Test changes
python3 -m pytest tests/

# Regenerate features (if needed)
cd narrative_optimization/domains/{domain}
rm -rf cache/  # Clear cache
python3 extract_features.py
```

---

## Troubleshooting

### "Import Error: No module named..."

```bash
# Install dependencies
pip install -r requirements.txt

# Or specific package
pip install {package_name}
```

### "File Not Found" errors

- Check paths are correct (absolute vs relative)
- Verify data files exist in `data/domains/`
- Check domain config in `narrative_optimization/domains/{domain}/config.yaml`

### Web interface not loading

```bash
# Check Flask is running
ps aux | grep python

# Restart Flask
pkill -f app.py
python3 app.py
```

### Transformer errors

- Verify transformer is registered in `__init__.py`
- Check feature names are unique
- Ensure transform() returns correct shape

---

## Production Checklist

Before deploying a domain to production:

- [ ] Data collection complete (Stage 3)
- [ ] Domain formula calculated (Stage 4-6)
- [ ] Optimization complete (Stage 7-8)
- [ ] Real validation done (Stage 9)
- [ ] Web route created and tested
- [ ] DOMAIN_STATUS.md updated
- [ ] Production model saved
- [ ] Documentation complete
- [ ] Risk assessment done
- [ ] Performance monitoring enabled

---

## Key Files Reference

### Core Documentation
- `README.md` - Project overview
- `DOMAIN_STATUS.md` - Domain progress tracker
- `DOMAIN_DEVELOPMENT_STAGES.md` - 10-stage framework
- `NARRATIVE_CATALOG.md` - Universal patterns
- `FORMAL_VARIABLE_SYSTEM.md` - Variable definitions

### Development Tools
- `ADD_NEW_DOMAIN_TEMPLATE.md` - Domain addition guide
- `WEBSITE_ACCESS_GUIDE.md` - Web interface guide
- `EASY_COMMANDS.md` - Quick command reference

### Configuration
- `requirements.txt` - Python dependencies
- `app.py` - Flask configuration
- `Dockerfile` - Container configuration

---

## Contact & Support

For questions or issues:
1. Check existing documentation first
2. Review DOMAIN_STATUS.md for current state
3. Look at similar domains for examples
4. Consult theory docs in `docs/theory/`

---

**Remember**: 
- Honest testing over impressive results
- Real validation over simulated performance
- Clear documentation over clever code
- Production quality over quick hacks

**This framework values scientific rigor and practical utility in equal measure.**

