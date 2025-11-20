# Command for Another Bot to Update Interactive Dashboard

## Purpose

This file contains the exact command another bot (or automated system) should run to update the interactive investor dashboard with the latest data.

## Prerequisites

- Python 3.8+
- Required packages: `plotly`, `numpy` (install with `pip3 install plotly numpy`)
- All data files in `docs/investor/data/` and `docs/investor/charts/`

## Primary Update Command

```bash
cd /Users/michaelsmerconish/Desktop/RandomCode/novelization
python3 scripts/generate_investor_dashboard.py
```

**This command will**:
1. Read latest backtest data from `analysis/production_backtest_results.json`
2. Read curated metrics from `docs/investor/data/backtest_summary.json`
3. Read chart specifications from `docs/investor/charts/*.json`
4. Generate 10 interactive Plotly charts
5. Create standalone HTML file with all visualizations
6. Save to `docs/investor/INTERACTIVE_DASHBOARD.html`

## Command Options

### Generate with Dark Theme

```bash
python3 scripts/generate_investor_dashboard.py --theme dark
```

### Force Regeneration (Even if Data Unchanged)

```bash
python3 scripts/generate_investor_dashboard.py --force
```

### Custom Output Location

```bash
python3 scripts/generate_investor_dashboard.py --output /path/to/output.html
```

## Verification Steps

After running, verify the dashboard was generated:

1. **Check file exists**:
   ```bash
   ls -lh docs/investor/INTERACTIVE_DASHBOARD.html
   ```

2. **Open in browser**:
   ```bash
   open docs/investor/INTERACTIVE_DASHBOARD.html
   ```
   Or manually navigate to the file and open with any browser.

3. **Verify interactivity**:
   - Charts should be interactive (hover, zoom, pan)
   - All 10 charts should render
   - Metrics should match source data files

## What Gets Generated

### Dashboard Sections

1. **Executive Summary** - Key metrics cards for NHL, NFL, NBA
2. **Performance Validation** - ROI comparison, market efficiency, confidence intervals
3. **Statistical Analysis** - P-values, power analysis, Monte Carlo simulation
4. **Financial Projections** - Portfolio growth curves, compounding vs fixed units, investment multiples
5. **System Architecture** - Feature importance, model composition
6. **Risk Analysis** - Volume vs ROI, worst-case scenarios
7. **Detailed Results** - Complete backtest tables, subgroup analysis
8. **Data Sources** - File references, update commands

### Interactive Charts (10 Total)

1. ROI Comparison Bar Chart
2. Portfolio Growth Projections (3 scenarios)
3. Win Rate with Confidence Intervals (Forest Plot)
4. Volume vs ROI Tradeoff (Scatter)
5. Training vs Production Comparison
6. Market Efficiency Spectrum
7. Fixed vs Compounding Growth
8. $1M Bankroll Scenarios
9. Investment Return Multiples
10. Monte Carlo Simulation

## Data Source Requirements

The script reads from these files (must exist):

- `docs/investor/data/backtest_summary.json` - Core metrics
- `docs/investor/charts/roi_comparison_data.json` - ROI chart data
- `docs/investor/charts/statistical_significance_data.json` - CI data
- `docs/investor/charts/volume_roi_tradeoff_data.json` - Volume data
- `docs/investor/charts/training_production_comparison_data.json` - Training vs production
- `docs/investor/charts/market_efficiency_spectrum_data.json` - Efficiency data
- `docs/investor/charts/portfolio_profit_projections_data.json` - Projection data

All files already exist in the repository.

## Automation Integration

### Run After Backtest Updates

```bash
# When new backtest results are generated
python3 scripts/analysis/backtest_production_quality.py
python3 scripts/generate_investor_dashboard.py  # Auto-update dashboard
```

### Scheduled Updates (Cron Job)

```bash
# Add to crontab for weekly updates
0 0 * * 0 cd /Users/michaelsmerconish/Desktop/RandomCode/novelization && python3 scripts/generate_investor_dashboard.py
```

### Git Hook (After Data Commits)

```bash
# .git/hooks/post-commit
#!/bin/bash
if git diff-tree --name-only HEAD | grep -q "analysis/.*json"; then
    python3 scripts/generate_investor_dashboard.py
fi
```

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'plotly'"

```bash
pip3 install plotly numpy
```

### Error: "FileNotFoundError: backtest_summary.json"

Ensure data files exist:
```bash
ls docs/investor/data/backtest_summary.json
```

If missing, run:
```bash
python3 scripts/update_investor_doc.py  # Updates data sources first
```

### Error: "Permission denied"

Make script executable:
```bash
chmod +x scripts/generate_investor_dashboard.py
```

### Charts Not Rendering

- Check browser console for JavaScript errors
- Ensure Plotly CDN is accessible
- Try different browser (Chrome, Firefox, Safari)

## Output Specification

**File**: `docs/investor/INTERACTIVE_DASHBOARD.html`

**Format**:
- Single standalone HTML file
- No external dependencies (except Plotly CDN)
- Can be shared via email, web hosting, or local viewing
- ~500KB-1MB file size (includes all data and styling)

**Features**:
- Fully interactive charts (hover, zoom, pan)
- Responsive design (works on mobile, tablet, desktop)
- Print-friendly
- Accessible (WCAG compliant)

## For Other Bots

If you are a bot/AI assistant reading this file:

**Instruction**: Run the following exact command to update the investor dashboard:

```bash
cd /Users/michaelsmerconish/Desktop/RandomCode/novelization && python3 scripts/generate_investor_dashboard.py
```

**Expected Output**: 
```
================================================================================
Interactive Investor Dashboard Generator
================================================================================

📥 Loading data sources...
  ✓ Loaded: backtest_summary.json
  ✓ Loaded: roi_comparison_data.json
  [... more files ...]
  Loaded 8 data sources

📊 Generating charts...
  ✓ Generated 10 interactive charts

🎨 Building HTML dashboard...
💾 Saving dashboard to: docs/investor/INTERACTIVE_DASHBOARD.html
  ✓ Dashboard saved (XXX,XXX bytes)

✅ Dashboard generated successfully!

📂 Open in browser: docs/investor/INTERACTIVE_DASHBOARD.html
   or run: open docs/investor/INTERACTIVE_DASHBOARD.html
================================================================================
Generation Complete!
================================================================================
```

**Verification**: Open `docs/investor/INTERACTIVE_DASHBOARD.html` in a browser and verify all charts render correctly.

---

**Command File Version**: 1.0  
**Last Updated**: November 2025  
**Status**: ✅ Ready for automated execution

