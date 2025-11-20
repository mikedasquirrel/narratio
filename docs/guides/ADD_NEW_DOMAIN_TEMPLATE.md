# How to Add a New Domain (Takes 2 Minutes)

Adding a new domain to the analysis framework is **trivial**. Just follow this template.

---

## Step 1: Prepare Your Data (5 minutes)

Your data file should be JSON with narratives and outcomes:

```json
[
  {
    "narrative_field_name": "Long narrative text here...",
    "outcome_field_name": 0.75,
    "any_other_metadata": "..."
  },
  ...
]
```

**Requirements**:
- JSON format (list of dicts)
- Narrative field (text, > 50 characters)
- Outcome field (number or binary)
- At least 100 samples

**Save to**: `data/domains/your_domain_name.json`

---

## Step 2: Register Domain (2 minutes)

Open: `narrative_optimization/domain_registry.py`

Add to `DOMAINS` dict:

```python
'your_domain': DomainConfig(
    name='your_domain',
    data_path='data/domains/your_domain_name.json',
    narrative_field='narrative_field_name',  # or ['field1', 'field2'] to try multiple
    outcome_field='outcome_field_name',
    estimated_pi=0.60,  # Your estimate of narrativity (0-1)
    description='Brief description of domain',
    outcome_type='continuous',  # or 'binary'
    min_narrative_length=50  # Minimum characters
),
```

**That's it!** Domain is now registered.

---

## Step 3: Process Domain (1 command)

```bash
python3 -c "
from narrative_optimization.universal_domain_processor import UniversalDomainProcessor
processor = UniversalDomainProcessor()
processor.process_domain('your_domain', sample_size=1000)
"
```

Or use the convenience script:

```bash
python3 process_domain.py your_domain --sample 1000
```

**Results automatically saved** to: `narrative_optimization/results/domains/your_domain/`

---

## Step 4: Review Results (2 minutes)

Check the JSON output:

```python
import json
with open('narrative_optimization/results/domains/your_domain/n1000_analysis.json') as f:
    results = json.load(f)

print(f"Patterns discovered: {results['n_patterns']}")
print(f"Significant predictors: {len(results['significant_correlations'])}")
```

Done! Your domain is analyzed with same framework as all others.

---

## Examples

### Example 1: Add Video Games

```python
# In domain_registry.py, add:

'video_games': DomainConfig(
    name='video_games',
    data_path='data/domains/games.json',
    narrative_field='plot_summary',
    outcome_field='metacritic_score',
    estimated_pi=0.75,
    description='Video game narratives',
    outcome_type='continuous'
),
```

### Example 2: Add Historical Events

```python
'historical_events': DomainConfig(
    name='historical_events',
    data_path='data/domains/wikipedia_events.json',
    narrative_field=['full_description', 'summary'],  # Try multiple fields
    outcome_field='historical_impact',
    estimated_pi=0.60,
    description='Historical event narratives',
    outcome_type='continuous',
    min_narrative_length=200  # Longer narratives
),
```

### Example 3: Add Scientific Papers

```python
'scientific_papers': DomainConfig(
    name='scientific_papers',
    data_path='data/domains/arxiv_abstracts.json',
    narrative_field='abstract',
    outcome_field='citation_count',
    estimated_pi=0.50,
    description='Scientific paper abstracts',
    outcome_type='continuous'
),
```

---

## Expanding Existing Domains

### Add More Data to Existing Domain

1. **Collect more data** → Save to same file or new file
2. **If new file**: Update `data_path` in registry
3. **Run again** with larger `sample_size`
4. **Compare**: Does pattern count change? Should be stable.

### Re-analyze with Different Parameters

```python
processor.process_domain(
    'movies',
    sample_size=4000,  # Larger sample
    min_cluster_size=100  # Larger clusters (fewer, more robust patterns)
)
```

---

## Batch Processing New Domains

Add multiple domains at once:

```python
# Register all new domains in domain_registry.py
# Then:

from narrative_optimization.universal_domain_processor import UniversalDomainProcessor

processor = UniversalDomainProcessor()

# Process batch
processor.process_batch(
    domain_names=['domain1', 'domain2', 'domain3'],
    sample_size_per_domain=1000
)

# Or process all available
processor.process_all_available(max_per_domain=2000)
```

---

## Custom Extraction (Advanced)

If your data format is non-standard, provide custom extractor:

```python
def custom_chess_extractor(data):
    """Extract from custom chess data format."""
    narratives = []
    outcomes = []
    
    for game in data:
        # Your custom logic
        pgn_narrative = game['moves_with_commentary']
        result = 1 if game['white_won'] else 0
        
        narratives.append(pgn_narrative)
        outcomes.append(result)
    
    return narratives, np.array(outcomes), len(data)

# Register with custom extractor
'chess': DomainConfig(
    name='chess',
    data_path='data/domains/chess_games.json',
    narrative_field='moves',  # Ignored when custom_extractor provided
    outcome_field='result',
    estimated_pi=0.40,
    custom_extractor=custom_chess_extractor
),
```

---

## Validation Checklist

After adding domain, verify:

- [ ] Data loads without errors
- [ ] Narratives extracted (> 100 samples)
- [ ] Outcomes have variance (not all same)
- [ ] Pattern count reasonable (5-40)
- [ ] Some patterns significant (20-80% of patterns)
- [ ] Effect sizes realistic (r=0.05-0.50)
- [ ] Matches π prediction (±10 patterns)

---

## Domain Categories

Organize domains by type for easier management:

```python
DOMAIN_CATEGORIES = {
    'entertainment': ['movies', 'oscars', 'music'],
    'sports': ['nba', 'nfl', 'mlb', 'tennis', 'ufc', 'golf', 'boxing'],
    'business': ['startups', 'crypto'],
    'natural': ['hurricanes', 'dinosaurs'],
    'cultural': ['mythology', 'poker'],
}

# Process all sports domains
sports_domains = DOMAIN_CATEGORIES['sports']
processor.process_batch(sports_domains, sample_size_per_domain=1000)
```

---

## Complete Workflow Example

```bash
# 1. Add domain to registry (edit domain_registry.py)
# 2. Test loading
python3 -c "
from narrative_optimization.domain_registry import load_domain_safe
narratives, outcomes, config = load_domain_safe('new_domain')
print(f'Loaded: {len(narratives)} narratives')
"

# 3. Process domain
python3 -c "
from narrative_optimization.universal_domain_processor import UniversalDomainProcessor
processor = UniversalDomainProcessor()
processor.process_domain('new_domain', sample_size=1000)
"

# 4. Review results
python3 -c "
import json
with open('narrative_optimization/results/domains/new_domain/n1000_analysis.json') as f:
    r = json.load(f)
print(f'Patterns: {r[\"n_patterns\"]}')
print(f'Significant: {len(r[\"significant_correlations\"])}')
"
```

**Total time**: ~10 minutes including data prep

---

## Summary

**To add domain**:
1. Prepare JSON data (5 min)
2. Add 5-line config to registry (2 min)
3. Run `process_domain()` command (1 min + processing time)
4. Review results (2 min)

**Total**: ~10 minutes + processing time

**To expand existing domain**:
1. Add more data to file OR
2. Re-run with larger `sample_size`
3. Compare to previous run
4. That's it!

**The system is designed for easy expansion.** Add 10 new domains in an hour if you have the data files ready.

