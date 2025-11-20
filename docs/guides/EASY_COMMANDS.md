# Easy Commands Reference

**Your framework is now plug-and-play.** Here are the commands you'll use most:

---

## Quick Start: Process Any Domain

### List What's Available

```bash
python3 process_domain.py --list
```

Shows all 15 registered domains with π values.

### Process Single Domain

```bash
# Movies
python3 process_domain.py movies --sample 2000

# NBA  
python3 process_domain.py nba --sample 1000

# Tennis
python3 process_domain.py tennis --sample 5000

# Any registered domain
python3 process_domain.py DOMAIN_NAME --sample N
```

### Process All Domains

```bash
python3 process_domain.py all --sample 1000
```

Runs unsupervised discovery on ALL 15 available domains (1K each).

---

## Add New Domain (2 Minutes)

### 1. Add to Registry

Edit `narrative_optimization/domain_registry.py`, add entry:

```python
'new_domain': DomainConfig(
    name='new_domain',
    data_path='data/domains/new_domain.json',
    narrative_field='text_field',
    outcome_field='outcome_field',
    estimated_pi=0.60
),
```

### 2. Process It

```bash
python3 process_domain.py new_domain --sample 1000
```

Done! Results in `narrative_optimization/results/domains/new_domain/`

---

## Systematic Scaling (Safeguarded)

### Scale with Checkpoints

```python
from narrative_optimization.universal_domain_processor import UniversalDomainProcessor

processor = UniversalDomainProcessor()

# Scale incrementally with validation
processor.process_domain('movies', sample_size=2000)  # Checkpoint 1
processor.process_domain('movies', sample_size=4000)  # Checkpoint 2  
processor.process_domain('movies', sample_size=6000)  # Checkpoint 3

# Review stability between checkpoints
```

### Batch Process Multiple Domains

```python
processor.process_batch(
    domain_names=['movies', 'nba', 'nfl', 'ufc'],
    sample_size_per_domain=1000
)
```

---

## View Results

### Check Specific Domain

```python
import json

with open('narrative_optimization/results/domains/movies/n2000_analysis.json') as f:
    results = json.load(f)

print(f"Patterns: {results['n_patterns']}")
print(f"Significant: {len(results['significant_correlations'])}")

# See strongest predictor
sig = results['significant_correlations']
strongest = max(sig, key=lambda x: abs(x['effect_size']))
print(f"Strongest: {strongest['pattern_id']} r={strongest['effect_size']:.3f}")
```

### Compare Domains

```bash
python3 -c "
import json
from pathlib import Path

for result_file in Path('narrative_optimization/results/domains').rglob('*.json'):
    with open(result_file) as f:
        r = json.load(f)
    print(f\"{r['domain']:<15s}: {r['n_patterns']} patterns, π={r['domain_info']['estimated_pi']:.2f}\")
"
```

---

## Current Analysis Status

### What's Been Analyzed

✓ **Movies**: 2,000 analyzed → 20 patterns, 12 significant  
✓ **NBA**: 1,000 analyzed → 16 patterns, 11 significant  
⏳ **Remaining**: 13 domains ready (194K narratives)

### What's Ready to Process

**Available right now** (just run command):
- Tennis: 74,906 matches
- Music: 50,012 lyrics
- MLB: 23,264 games
- Golf, UFC, NFL, Startups, Poker, Boxing, etc.

### Quick Commands for Each

```bash
# Process each domain with reasonable sample
python3 process_domain.py tennis --sample 5000   # Largest
python3 process_domain.py mlb --sample 2000
python3 process_domain.py music --sample 2000
python3 process_domain.py ufc --sample 1000
python3 process_domain.py golf --sample 1000
python3 process_domain.py nfl --sample 1000
python3 process_domain.py startups --sample 474  # All of them
python3 process_domain.py poker --sample 2000
python3 process_domain.py hurricanes --sample 1000
python3 process_domain.py dinosaurs --sample 952  # All of them

# Or just:
python3 process_domain.py all --sample 1000
```

**Estimated time**: ~5-10 minutes per domain (1K samples)  
**Total for all 13 remaining**: 1-2 hours

---

## Expanding to Literary Corpus

### When Gutenberg Texts Downloaded

```python
# Add to domain_registry.py:

'novels': DomainConfig(
    name='novels',
    data_path='data/literary_corpus/gutenberg/novels_complete.json',
    narrative_field='full_text',
    outcome_field='citation_count',  # or download_count
    estimated_pi=0.85,
    description='Complete novels from Gutenberg',
    outcome_type='continuous',
    min_narrative_length=10000  # Novels are long
),
```

Then:
```bash
python3 process_domain.py novels --sample 500
```

### When Biographies Collected

```python
'biographies': DomainConfig(
    name='biographies',
    data_path='data/literary_corpus/biographies/wikipedia_biographies.json',
    narrative_field='text',
    outcome_field='historical_impact',
    estimated_pi=0.75,
    description='Biography narratives',
    outcome_type='continuous'
),
```

---

## Systematic Expansion Strategy

### Week 1: Process Existing (Easy)

```bash
# Process what we have with 1K samples each
python3 process_domain.py all --sample 1000
```

Gets patterns from all 15 available domains in ~2 hours.

### Week 2: Scale Up (Validate)

```bash
# Scale key domains to larger samples
python3 process_domain.py movies --sample 4000
python3 process_domain.py nba --sample 2000
python3 process_domain.py tennis --sample 5000
```

Validates pattern stability.

### Week 3: Add New Domains (Expand)

Download Gutenberg texts, collect biographies, etc.  
Add to registry (5 lines each).  
Process (1 command each).

### Week 4: Meta-Analysis (Synthesize)

Run cross-domain analysis on all discovered patterns.

---

## Domain Extensibility Features

### ✓ Central Registry

All domains in one place (`domain_registry.py`)  
Easy to see what exists  
Easy to add new entries

### ✓ Universal Processor

Same processor for ALL domains  
No domain-specific code needed  
Automatic validation and safeguards

### ✓ Flexible Extraction

Try multiple narrative fields automatically  
Custom extractors for complex formats  
Handles lists, dicts, nested structures

### ✓ Standardized Output

All domains save to same format  
Easy to compare across domains  
Consistent file naming

### ✓ Batch Operations

Process multiple domains at once  
Process by π range  
Process by category

### ✓ Easy Expansion

New domain = 5 lines + 1 command  
No code changes needed  
Framework handles everything

---

## Summary

**Adding domain**: Edit 1 file (5 lines) + Run 1 command  
**Processing domain**: 1 command  
**Expanding domain**: Re-run with larger sample  
**Batch processing**: Process multiple domains with 1 command  
**View results**: Standardized JSON format

**Your 197K narratives across 15 domains are now**:
- Registered ✓
- Ready to process ✓
- One command away ✓

**Adding 10 more domains**: ~1 hour if data files ready

**The system is built for systematic scaling AND easy expansion simultaneously.**

