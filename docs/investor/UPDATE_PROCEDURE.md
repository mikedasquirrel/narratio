# Investor Document Update Procedure

## Overview

This document describes how to keep `INVESTOR_PRESENTATION.md` synchronized with ongoing investigations, backtest results, and data analysis.

## Data Sources

The investor document pulls from these authoritative sources:

### Primary Sources

1. **Backtest Results**:
   - `analysis/production_backtest_results.json` - Production backtest data
   - `analysis/EXECUTIVE_SUMMARY_BACKTEST.md` - Human-readable summary
   - `analysis/RECENT_SEASON_BACKTEST_REPORT.md` - Detailed backtest report

2. **Validation Data**:
   - `analysis/multi_market_validation_status.json` - Multi-sport validation metrics
   - `docs/investor/data/backtest_summary.json` - Curated summary for investor doc

3. **System Status**:
   - `app.py` - Validated systems (see comments around lines 83-95)
   - `VALIDATED_DOMAINS.md` - Domain validation status

4. **Performance Tracking**:
   - `scripts/nhl_performance_tracker.py` - Live performance data
   - `logs/betting/` - Betting logs and results

### Secondary Sources

- Domain-specific results: `narrative_optimization/domains/*/results/`
- Experiment results: `narrative_optimization/experiments/*/results/`
- Model performance: `narrative_optimization/domains/*/models/`

## Update Checklist

Run this checklist whenever:
- New backtest results are generated
- Validation metrics are updated
- New systems are validated
- Performance data changes significantly

### Before Updating

- [ ] Review latest backtest results in `analysis/`
- [ ] Check for new validation reports
- [ ] Verify system status in `app.py`
- [ ] Review any new domain validations

### Update Steps

1. **Run Update Script**:
   ```bash
   python scripts/update_investor_doc.py
   ```

2. **Manual Review**:
   - [ ] Check updated metrics match source data
   - [ ] Verify financial projections are recalculated
   - [ ] Review statistical significance sections
   - [ ] Update date stamps

3. **Version Control**:
   - [ ] Commit changes with descriptive message
   - [ ] Tag version if major update
   - [ ] Update changelog

### After Updating

- [ ] Verify document renders correctly
- [ ] Check all links and references
- [ ] Review for consistency
- [ ] Update data source references if needed

## Automated Updates

The `scripts/update_investor_doc.py` script can automatically:

1. **Pull Metrics** from JSON source files
2. **Recalculate Projections** based on latest data
3. **Update Tables** with new results
4. **Generate Charts** from updated data
5. **Validate Consistency** between sections

## Manual Updates Required

Some sections require manual review/updates:

- **Executive Summary**: May need narrative updates
- **Competitive Advantages**: Requires strategic thinking
- **Expansion Opportunities**: Based on research priorities
- **Implementation Roadmap**: Reflects current priorities

## Version Tracking

Document versions are tracked in:
- File header: `Document Version: X.Y`
- Git commits: Descriptive commit messages
- Changelog: `docs/investor/CHANGELOG.md` (to be created)

## Data Source Mapping

| Investor Doc Section | Source File(s) | Update Frequency |
|---------------------|----------------|------------------|
| Key Metrics Table | `backtest_summary.json` | After each backtest |
| NHL Results | `production_backtest_results.json` | After NHL backtest |
| NFL Results | `production_backtest_results.json` | After NFL backtest |
| NBA Results | `production_backtest_results.json` | After NBA backtest |
| Statistical Tests | Calculated from backtest data | After each backtest |
| Financial Projections | Calculated from ROI metrics | After each backtest |
| Training vs Production | `EXECUTIVE_SUMMARY_BACKTEST.md` | After each backtest |
| System Status | `app.py` (comments) | When systems validated |

## Regular Maintenance Schedule

- **Weekly**: Review for new backtest results
- **Monthly**: Full update with latest data
- **Quarterly**: Comprehensive review and refresh
- **After Major Validation**: Immediate update

## Troubleshooting

**Issue**: Metrics don't match source data
- **Solution**: Check data source file timestamps, verify JSON structure

**Issue**: Financial projections seem off
- **Solution**: Verify bankroll size ($1M), check Kelly fraction (1%), recalculate

**Issue**: Missing new system data
- **Solution**: Check if system added to `app.py` validated systems list

**Issue**: Outdated validation dates
- **Solution**: Update date stamps in document header and sections

## Questions?

Contact: [To be filled]
Last Updated: November 2025

