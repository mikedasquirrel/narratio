# Interactive Investor Dashboard

## Overview

`INTERACTIVE_DASHBOARD.html` is a beautiful, single-page HTML presentation with interactive Plotly visualizations that showcases all validation data, statistical analysis, and financial projections for the Narrative Optimization betting systems.

## Features

- **11 Interactive Charts**: ROI comparison, portfolio growth, confidence intervals, Monte Carlo simulation, and more
- **Complete Statistical Analysis**: P-values, confidence intervals, power analysis, multiple testing corrections
- **Financial Projections**: Conservative/Moderate/Aggressive scenarios with $1M bankroll
- **Data-Driven**: Automatically generated from JSON sources
- **Standalone**: Single HTML file, no server needed, works offline
- **Responsive**: Mobile, tablet, and desktop compatible
- **Updateable**: Regenerates from data sources as research continues

## Quick Start

### View the Dashboard

```bash
open docs/investor/INTERACTIVE_DASHBOARD.html
```

Or double-click the file to open in your default browser.

### Update the Dashboard

When new backtest results or validation data are available:

```bash
python3 scripts/generate_investor_dashboard.py
```

## Command for Another Bot

**Simple Command**:
```bash
cd /Users/michaelsmerconish/Desktop/RandomCode/novelization && python3 scripts/generate_investor_dashboard.py
```

**Full Update Flow**:
```bash
# 1. Run new backtest (if needed)
python3 scripts/analysis/backtest_production_quality.py

# 2. Update investor data summary (if structure changed)
# Manually update docs/investor/data/backtest_summary.json

# 3. Regenerate dashboard
python3 scripts/generate_investor_dashboard.py
```

## Dashboard Sections

1. **Executive Summary** - Key metrics cards (NHL 69.4%, NFL 66.7%, NBA 54.5%)
2. **Performance Validation** - ROI charts, market efficiency, confidence intervals
3. **Statistical Analysis** - P-values, power analysis, Monte Carlo simulation (10,000 trials)
4. **Financial Projections** - Portfolio growth curves, investment multiples, compounding effects
5. **System Architecture** - Feature importance, model composition, 79-feature breakdown
6. **Risk Analysis** - Volume vs ROI, worst-case scenarios, stress testing
7. **Detailed Results** - Complete backtest tables, subgroup analysis, temporal stability
8. **Data Sources** - File references, update commands, verification instructions

## Interactive Features

- **Hover** over charts to see detailed data points
- **Zoom** into regions of interest
- **Pan** across large datasets
- **Toggle** scenarios and comparisons
- **Download** charts as PNG images
- **Responsive** design adapts to screen size

## Charts Included

1. ROI Comparison (NHL, NFL, NBA systems)
2. Portfolio Growth Projections (3-year curves)
3. Win Rate Confidence Intervals (Forest plot)
4. Volume vs ROI Tradeoff (Scatter)
5. Training vs Production Performance (Overfitting check)
6. Market Efficiency Spectrum (NHL > NFL > NBA)
7. Fixed Units vs Kelly Compounding
8. $1M Bankroll Scenarios (Year 1 and 3-year)
9. Investment Return Multiples (1.34x to 62.57x)
10. Monte Carlo Simulation (Probability distribution)
11. Feature Importance (Performance vs Nominative)

## Data Sources

The dashboard pulls from these JSON files:

- `docs/investor/data/backtest_summary.json` - Core metrics
- `docs/investor/charts/roi_comparison_data.json`
- `docs/investor/charts/statistical_significance_data.json`
- `docs/investor/charts/portfolio_profit_projections_data.json`
- `docs/investor/charts/training_production_comparison_data.json`
- `docs/investor/charts/market_efficiency_spectrum_data.json`
- `docs/investor/charts/volume_roi_tradeoff_data.json`

## Keeping Updated

### Manual Update

When you have new backtest results:

1. Update source JSON files in `docs/investor/data/`
2. Run: `python3 scripts/generate_investor_dashboard.py`
3. Verify: Open `INTERACTIVE_DASHBOARD.html` in browser

### Automated Update

Add to your workflow:

```bash
# After any backtest
python3 scripts/generate_investor_dashboard.py --force
```

### For Other Bots/AI Assistants

If you're another bot updating the dashboard, run:

```bash
cd /Users/michaelsmerconish/Desktop/RandomCode/novelization
python3 scripts/generate_investor_dashboard.py
```

See `UPDATE_DASHBOARD_COMMAND.md` for detailed instructions.

## Customization

### Dark Theme

```bash
python3 scripts/generate_investor_dashboard.py --theme dark
```

### Custom Output Location

```bash
python3 scripts/generate_investor_dashboard.py --output /path/to/custom_dashboard.html
```

## File Size

- Typical size: ~200-250 KB
- Contains: Full HTML, CSS, embedded Plotly data
- Self-contained: No external dependencies (except Plotly CDN)

## Browser Compatibility

- Chrome/Edge: Full support
- Firefox: Full support  
- Safari: Full support
- Mobile browsers: Responsive design

## Sharing

The HTML file can be:
- Emailed as attachment
- Hosted on web server
- Shared via file sharing services
- Opened locally (no server needed)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 2025 | Initial creation with 11 charts |

## Questions?

- Technical: See `UPDATE_DASHBOARD_COMMAND.md`
- Data sources: See `DATA_SOURCES.md`
- Update procedure: See `UPDATE_PROCEDURE.md`

---

**Dashboard Status**: ✅ Operational  
**Last Generated**: Check file timestamp  
**Command**: `python3 scripts/generate_investor_dashboard.py`

