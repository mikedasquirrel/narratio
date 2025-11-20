# Domain Processing Guide

Complete guide to processing story domains through the web interface.

## Table of Contents

1. [Overview](#overview)
2. [Accessing the Interface](#accessing-the-interface)
3. [Processing a Domain](#processing-a-domain)
4. [Monitoring Progress](#monitoring-progress)
5. [Understanding Results](#understanding-results)
6. [Troubleshooting](#troubleshooting)

---

## Critical First Step (macOS)

TensorFlow will deadlock on macOS unless the CPU-only environment variables are set
**before** Python starts. Always run commands through the helper:

```bash
source scripts/env_setup.sh
python3 app.py
```

You can also run single commands:

```bash
scripts/env_setup.sh python3 narrative_optimization/test_imports.py
```

If you skip this step, the server can appear to “hang with no output” because the
mutex deadlock happens before Flask prints anything.

---

## Overview

The Domain Processing Pipeline provides a web-based interface for running transformer-based analysis on any registered story domain. Features include:

- **Real-time progress tracking** - See exactly what's happening
- **Live logging** - Watch the system work
- **Error handling** - Graceful failure with detailed reports
- **Timeout monitoring** - Automatic warnings and graceful shutdown
- **Job history** - Track all processing runs

---

## Accessing the Interface

### 1. Start the Flask Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 2. Navigate to Domain Processor

Go to: `http://localhost:5000/process/domain-processor`

Or click **"Domain Processor"** in the navigation menu.

---

## Processing a Domain

### Step 1: Select Domain

1. Click the **"Select Domain"** dropdown
2. Browse available domains (sorted by π narrativity value)
3. Select a domain (e.g., `tennis`, `movies`, `nba`)

**Domain Info Display:**
- **π (Narrativity)**: Complexity indicator (0-1)
- **Type**: Domain category (sports, entertainment, etc.)
- **Est. Time**: Expected processing duration

### Step 2: Configure Processing

**Sample Size:**
- Number of narratives to process
- Default: 1000
- Range: 10 - 10,000
- **Tip**: Start with 100-500 for testing

**Timeout (minutes):**
- Maximum processing time
- Default: Auto-calculated based on sample size and π
- **Recommendation**: Leave blank for auto

**Fail Fast:**
- ☑ **Checked**: Stop immediately on first error
- ☐ **Unchecked**: Continue processing, collect all errors (recommended)

### Step 3: Start Processing

1. Click the **"Process {domain}"** button
2. The progress panel will appear
3. Processing runs in the background
4. You can navigate away - processing continues

---

## Monitoring Progress

### Progress Bar

Shows overall completion percentage (0% - 100%)

### Processing Steps

The system shows 5 main steps:

1. **Loading** (0-15%)
   - Loading domain data
   - Validating configuration
   
2. **Feature Extraction** (15-50%)
   - Applying 40-50 transformers
   - Extracting 1000-2000 features
   - **Slowest step** - be patient!
   
3. **Pattern Discovery** (50-80%)
   - Finding narrative patterns
   - Clustering similar narratives
   
4. **Validation** (80-95%)
   - Statistical significance testing
   - Correlation analysis
   
5. **Complete** (100%)
   - Saving results
   - Finalizing reports

### Live Logs

**Log Viewer Features:**
- Auto-scrolling (toggle with button)
- Color-coded by level:
  - 🟢 **INFO**: Normal progress
  - 🟡 **WARNING**: Issues but continuing
  - 🔴 **ERROR**: Problems detected
  - ⚫ **DEBUG**: Detailed diagnostics

**Tips:**
- Turn off auto-scroll to read previous logs
- Errors appear in red - watch for these
- Warnings at 80% timeout are normal

### Timing Information

- **Elapsed**: Time since processing started
- **Remaining**: Estimated time to completion (after 10% progress)
- **Progress**: Current completion percentage

---

## Understanding Results

### Completion Status

**✓ Completed** (Green)
- Processing finished successfully
- Results saved and ready to view

**✗ Failed** (Red)
- Critical error occurred
- Check error panel for details
- Review logs for context

**⊗ Cancelled** (Orange)
- User cancelled processing
- Partial results may be saved

**⏱ Timeout** (Orange)
- Processing exceeded time limit
- Partial results preserved
- Consider increasing timeout or reducing sample size

### Results Summary

When complete, you'll see:

1. **Number of patterns discovered**
2. **Significant correlations** (patterns that predict outcomes)
3. **Error count** (transformers that failed)
4. **Warning count** (issues encountered)

### Viewing Detailed Results

Results are saved to:
```
narrative_optimization/results/domains/{domain_name}/
```

Files include:
- `n{sample_size}_analysis.json` - Complete results
- `patterns.json` - Discovered patterns
- `significant_correlations.csv` - Predictive patterns

**Access via API:**
```
GET /process/api/results/{domain}
```

---

## Troubleshooting

### Common Issues

#### 1. Domain Shows "No Data"

**Problem**: Domain data file doesn't exist

**Solution**:
- Check `data/domains/{domain_name}.json` exists
- Download/collect data for domain
- See `DOMAIN_CONFIGURATION_GUIDE.md` for adding new domains

#### 2. Processing Stuck at Feature Extraction

**Problem**: Transformer taking very long

**Situation**: This is **normal** for:
- Large sample sizes (>2000)
- High π domains (>0.7)
- First run (model loading)

**Action**: Wait patiently - check logs for progress

#### 3. Timeout Warnings

**Problem**: Approaching time limit

**Solutions**:
- **Option 1**: Wait - partial results will be saved
- **Option 2**: Cancel and restart with:
  - Smaller sample size
  - Longer timeout
  - Fast mode (future feature)

#### 4. Memory Errors

**Problem**: System running out of RAM

**Solutions**:
- Reduce sample size
- Close other applications
- Process domains one at a time

#### 5. Errors in Specific Transformers

**Problem**: Some transformers failing

**Situation**: **Not critical** if fail-fast is OFF

**Action**:
- Review error logs
- Processing continues with remaining transformers
- Results still valid with 90%+ transformers working

### Error Messages Explained

| Error | Meaning | Solution |
|-------|---------|----------|
| `Domain not found` | Domain not registered | Check spelling, or register domain |
| `Failed to load domain` | Data file missing/corrupt | Verify data file exists and is valid JSON |
| `Timeout` | Processing took too long | Increase timeout or reduce sample size |
| `Transformer {name} failed` | Single transformer error | Continue - not critical if others work |
| `Out of memory` | System RAM exhausted | Reduce sample size significantly |

---

## Best Practices

### For Testing
1. Start with small sample (100-500)
2. Verify results look reasonable
3. Scale up gradually (1000 → 2000 → 5000)

### For Production
1. Use auto timeout (system calculates optimal)
2. Leave fail-fast **unchecked** (graceful error handling)
3. Process during off-hours for large jobs
4. Monitor logs for warnings

### For Experiments
1. Document configuration used
2. Save job IDs for reproducibility
3. Compare results across sample sizes
4. Validate patterns make domain sense

---

## Advanced Features

### Cancelling Jobs

Click **"Cancel"** button during processing:
- Graceful shutdown
- Partial results may be saved
- Job marked as cancelled in history

### Job History

View all previous processing runs:
- Domain name
- Sample size
- Duration
- Status
- Timestamp

**Click any job** to see detailed information

### API Access

For programmatic access, see `API_REFERENCE.md`

---

## Getting Help

### Check Logs

1. **Web logs**: Live log viewer in interface
2. **File logs**: `logs/processing/{domain}_{timestamp}.log`
3. **System logs**: Console output

### Report Issues

Include:
- Domain name
- Sample size
- Error message
- Relevant log excerpt
- Job ID (if available)

---

## Next Steps

- **Configure new domains**: See `DOMAIN_CONFIGURATION_GUIDE.md`
- **Understand API**: See `API_REFERENCE.md`
- **Quick commands**: See `QUICK_START.md`

---

**Last Updated**: November 2025  
**Version**: 2.0

