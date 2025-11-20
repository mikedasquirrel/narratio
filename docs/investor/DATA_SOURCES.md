# Investor Document Data Sources

This document tracks all data sources used in `INVESTOR_PRESENTATION.md` and their update status.

## Primary Data Sources

### Backtest Results

**File**: `analysis/production_backtest_results.json`  
**Last Updated**: [Check file timestamp]  
**Used For**: NHL, NFL, NBA performance metrics  
**Update Frequency**: After each production backtest  
**Status**: ✅ Active

**File**: `analysis/EXECUTIVE_SUMMARY_BACKTEST.md`  
**Last Updated**: November 17, 2025  
**Used For**: Training vs production comparisons, key insights  
**Update Frequency**: After major backtest runs  
**Status**: ✅ Active

**File**: `analysis/RECENT_SEASON_BACKTEST_REPORT.md`  
**Last Updated**: November 17, 2025  
**Used For**: Detailed backtest methodology and results  
**Update Frequency**: After each season backtest  
**Status**: ✅ Active

### Curated Summary

**File**: `docs/investor/data/backtest_summary.json`  
**Last Updated**: November 2025  
**Used For**: Key metrics table, financial projections  
**Update Frequency**: After each backtest (manually curated)  
**Status**: ✅ Active

### Validation Status

**File**: `analysis/multi_market_validation_status.json`  
**Last Updated**: [Check file timestamp]  
**Used For**: Cross-sport validation metrics  
**Update Frequency**: After validation runs  
**Status**: ✅ Active

**File**: `app.py` (lines 83-95)  
**Last Updated**: [Check git history]  
**Used For**: System validation status  
**Update Frequency**: When systems are validated/invalidated  
**Status**: ✅ Active

## Secondary Data Sources

### Domain-Specific Results

- `narrative_optimization/domains/nhl/models/` - NHL model files
- `narrative_optimization/domains/nfl/` - NFL analysis results
- `narrative_optimization/domains/nba/` - NBA analysis results

### Performance Tracking

- `scripts/nhl_performance_tracker.py` - Live NHL performance
- `logs/betting/` - Betting logs and results
- `data/domains/` - Domain data files

## Data Flow

```
Source Data Files
    ↓
[Update Script] (scripts/update_investor_doc.py)
    ↓
Curated Summary (docs/investor/data/backtest_summary.json)
    ↓
Investor Document (docs/investor/INVESTOR_PRESENTATION.md)
```

## Update Procedure

1. **Run Backtest**: Generate new results in `analysis/`
2. **Update Summary**: Manually curate `backtest_summary.json` if needed
3. **Run Script**: `python scripts/update_investor_doc.py`
4. **Review**: Check updated metrics match source data
5. **Commit**: Version control changes

## Data Validation

Before updating investor document, verify:

- [ ] Source files exist and are readable
- [ ] JSON files are valid
- [ ] Metrics match between source files
- [ ] Date stamps are current
- [ ] Financial calculations are correct

## Version History

| Date | Update | Source Files Changed |
|------|--------|---------------------|
| Nov 2025 | Initial creation | `production_backtest_results.json`, `EXECUTIVE_SUMMARY_BACKTEST.md` |
| [Future] | [To be filled] | [To be filled] |

## Notes

- Financial projections assume $1M bankroll, 1% Kelly sizing
- All ROI percentages are per-bet ROI, not annualized
- Win rates are from holdout testing, not training data
- Statistical significance calculated from binomial tests

## Questions?

See `UPDATE_PROCEDURE.md` for detailed update instructions.

