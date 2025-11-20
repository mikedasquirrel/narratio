# Quick Start Guide

Get up and running with domain processing in 5 minutes.

## Before You Start (macOS Only)

Always source the environment helper to avoid the TensorFlow Metal mutex hang:

```bash
source scripts/env_setup.sh
python3 app.py
```

Or run a single command:

```bash
scripts/env_setup.sh python3 narrative_optimization/tests/run_import_check.py
```

If you forget this step, Python may appear to “hang with no output” because TensorFlow is blocked before our logging starts.

---

## Installation

### 1. Start Flask Application

```bash
cd /path/to/novelization
python app.py
```

Application starts on `http://localhost:5000`

### 2. Access Domain Processor

Open browser to: `http://localhost:5000/process/domain-processor`

---

## Process Your First Domain

### Via Web Interface (Easiest)

1. **Select Domain**: Choose "tennis" from dropdown
2. **Configure**: Set sample size to 100 (for quick test)
3. **Click "Process"**: Watch real-time progress
4. **Wait ~2 minutes**: Small sample processes quickly
5. **View Results**: See discovered patterns

### Via Python

```python
from narrative_optimization.universal_domain_processor import UniversalDomainProcessor

# Create processor
processor = UniversalDomainProcessor()

# Process domain
results = processor.process_domain(
    domain_name='tennis',
    sample_size=100
)

print(f"Patterns discovered: {results['n_patterns']}")
```

### Via API

```bash
# Start processing
curl -X POST http://localhost:5000/process/api/process-domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "tennis", "sample_size": 100}'

# Response includes job_id
# {"job_id": "abc123", ...}

# Check status
curl http://localhost:5000/process/api/job-history?limit=1
```

---

## Common Tasks

### List Available Domains

**Web**: Visit domain processor page - domains in dropdown

**Python**:
```python
from narrative_optimization.domain_registry import DOMAINS

for name, config in DOMAINS.items():
    print(f"{name}: π={config.estimated_pi:.2f}")
```

**API**:
```bash
curl http://localhost:5000/process/api/domain-list
```

### Monitor Progress

**Web**: Real-time progress bar and logs on page

**Python with Callback**:
```python
def progress_update(step, percent, message):
    print(f"[{percent*100:.0f}%] {step}: {message}")

results = processor.process_domain(
    'tennis',
    sample_size=1000,
    progress_callback=progress_update
)
```

**API (SSE)**:
```javascript
const eventSource = new EventSource(`/process/api/process-status/${jobId}`);
eventSource.onmessage = (event) => {
    const status = JSON.parse(event.data);
    console.log(`${status.progress*100}%`);
};
```

### View Results

**Web**: Results panel appears when complete

**File System**:
```bash
ls narrative_optimization/results/domains/tennis/
# n1000_analysis.json - Complete results
# patterns.json - Discovered patterns
```

**API**:
```bash
curl http://localhost:5000/process/api/results/tennis
```

---

## Add a New Domain

### Method 1: Basic (2 minutes)

Edit `narrative_optimization/domain_registry.py`:

```python
'my_domain': DomainConfig(
    name='my_domain',
    data_path='data/domains/my_domain.json',
    narrative_field='text',
    outcome_field='success',
    estimated_pi=0.6,
    description='My custom domain'
)
```

Then process:
```bash
# Via web interface
http://localhost:5000/process/domain-processor

# Or Python
processor.process_domain('my_domain', sample_size=500)
```

### Method 2: With Config (10 minutes)

1. Create config file: `narrative_optimization/domains/my_domain/config.yaml`

```yaml
domain: my_domain
type: business
narrativity:
  structural: 0.7
  temporal: 0.6
  agency: 0.8
  interpretive: 0.5
  format: 0.4

data:
  text_fields: [description, narrative]
  outcome_field: success_score

outcome_type: continuous
```

2. Register in `domain_registry.py` (points to config)

3. Process as normal

---

## Troubleshooting

### "Domain not found"

**Problem**: Domain not registered

**Fix**: Add to `domain_registry.py`

### "No data file"

**Problem**: Data file missing

**Fix**: 
```bash
# Verify file exists
ls data/domains/your_domain.json

# Check path in domain config matches actual file
```

### Processing Stuck

**Situation**: Normal for large samples (>2000)

**Action**: 
- Check logs - should show progress
- Wait patiently - feature extraction is slow
- Try smaller sample size first

### Out of Memory

**Problem**: System RAM exhausted

**Fix**:
- Reduce sample size
- Close other applications
- Process one domain at a time

---

## Examples

### Example 1: Quick Test

```python
# Test with tiny sample
processor = UniversalDomainProcessor()
results = processor.process_domain('tennis', sample_size=50)
print(f"Success! Found {results['n_patterns']} patterns")
```

### Example 2: Production Run

```python
# Full processing with monitoring
import logging
logging.basicConfig(level=logging.INFO)

def progress(step, pct, msg):
    print(f"[{pct*100:3.0f}%] {msg}")

results = processor.process_domain(
    domain_name='tennis',
    sample_size=5000,
    progress_callback=progress
)

# Save results summary
with open('tennis_summary.txt', 'w') as f:
    f.write(f"Patterns: {results['n_patterns']}\n")
    f.write(f"Significant: {len(results['significant_correlations'])}\n")
```

### Example 3: Batch Processing

```python
domains = ['tennis', 'golf', 'ufc', 'poker']

for domain in domains:
    print(f"\n=== Processing {domain} ===")
    
    results = processor.process_domain(
        domain_name=domain,
        sample_size=1000
    )
    
    print(f"  Patterns: {results['n_patterns']}")
    print(f"  Status: {results.get('status', 'unknown')}")
```

### Example 4: API with Python SDK

```python
import requests
import time

# Start job
response = requests.post(
    'http://localhost:5000/process/api/process-domain',
    json={'domain': 'tennis', 'sample_size': 1000}
)
job_id = response.json()['job_id']

# Poll for completion
while True:
    response = requests.get(
        f'http://localhost:5000/process/api/job-history?limit=100'
    )
    jobs = response.json()['jobs']
    
    job = next((j for j in jobs if j['job_id'] == job_id), None)
    
    if job['status'] in ['completed', 'failed']:
        print(f"Done! Status: {job['status']}")
        break
    
    print(f"Progress: {job.get('progress', 0)*100:.0f}%")
    time.sleep(5)
```

---

## Configuration Tips

### π (Narrativity) Estimation

Quick reference:

| Domain Type | Typical π | Examples |
|-------------|-----------|----------|
| Random/Physical | 0.0 - 0.3 | Lottery, coin flips, avalanches |
| Skill-based Sport | 0.4 - 0.6 | NBA, NFL, MLB |
| Individual Sport | 0.6 - 0.8 | Tennis, golf, UFC |
| Entertainment | 0.6 - 0.8 | Movies, music, novels |
| Business/Prestige | 0.7 - 0.9 | Startups, crypto, Oscars |
| Pure Nominative | 0.85 - 1.0 | Housing, ships, character |

### Sample Sizes

| Purpose | Recommended Size | Processing Time |
|---------|------------------|-----------------|
| Quick Test | 50-100 | 1-2 minutes |
| Development | 500-1000 | 5-10 minutes |
| Validation | 1000-2000 | 10-20 minutes |
| Production | 2000-5000 | 20-60 minutes |
| Full Analysis | 5000+ | 1-3 hours |

### Timeout Recommendations

System auto-calculates, but override if needed:

```python
# Small sample
processor.process_domain('tennis', sample_size=100)  # ~5min timeout

# Medium sample  
processor.process_domain('tennis', sample_size=1000)  # ~15min timeout

# Large sample
processor.process_domain('tennis', sample_size=5000)  # ~45min timeout
```

---

## Next Steps

### For Beginners
1. ✅ Process tennis (100 samples) - verify system works
2. ✅ Try another domain (movies, nba) - see variety
3. ✅ Scale up sample size gradually

### For Researchers
1. ✅ Process full datasets (5000+ samples)
2. ✅ Compare patterns across domains
3. ✅ Validate discoveries against theory

### For Developers
1. ✅ Add custom domain with your data
2. ✅ Create domain-specific transformers
3. ✅ Build on top of API

---

## Documentation

- **Web Interface Guide**: `docs/DOMAIN_PROCESSING_GUIDE.md`
- **Configuration Guide**: `docs/DOMAIN_CONFIGURATION_GUIDE.md`
- **API Reference**: `docs/API_REFERENCE.md`

---

## Getting Help

### Check Logs

**Web Interface**: Live log viewer shows all activity

**File Logs**: `logs/processing/{domain}_{timestamp}.log`

**Console**: Run `python app.py` to see console output

### Common Patterns

**Normal**:
- Feature extraction takes longest (50% of time)
- Warnings about timeout at 80% are expected
- Some transformer failures OK (90%+ should work)

**Investigate**:
- Processing stuck >30min on 1000 samples
- Many transformer failures (>20%)
- Out of memory errors

### Report Issues

Include:
- Domain name
- Sample size
- Error message
- Log excerpt (last 20 lines)
- Job ID (from web interface)

---

**Last Updated**: November 2025  
**Version**: 2.0
