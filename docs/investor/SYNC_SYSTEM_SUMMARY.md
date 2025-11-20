# Investor Document Synchronization System

## Overview

A complete system has been set up to keep `INVESTOR_PRESENTATION.md` synchronized with ongoing investigations, backtest results, and data analysis.

## Components Created

### 1. Update Script (`scripts/update_investor_doc.py`)

**Purpose**: Automatically updates investor document from source data files

**Features**:
- Pulls metrics from JSON source files
- Updates key metrics table
- Recalculates financial projections ($1M bankroll)
- Updates date stamps
- Checks data freshness
- Dry-run mode for testing

**Usage**:
```bash
# Check data freshness only
python3 scripts/update_investor_doc.py --check-only

# Dry run (see what would change)
python3 scripts/update_investor_doc.py --dry-run

# Actually update the document
python3 scripts/update_investor_doc.py
```

### 2. Update Procedure (`docs/investor/UPDATE_PROCEDURE.md`)

**Purpose**: Step-by-step guide for keeping document updated

**Contents**:
- Data source mapping
- Update checklist
- Manual vs automated updates
- Version tracking
- Troubleshooting guide

### 3. Data Sources Tracking (`docs/investor/DATA_SOURCES.md`)

**Purpose**: Tracks all data sources and their status

**Contents**:
- Primary data sources (backtest results, validation files)
- Secondary sources (domain results, performance tracking)
- Data flow diagram
- Update frequency guidelines
- Version history

### 4. Investor README (`docs/investor/README.md`)

**Purpose**: Quick reference for the investor documentation directory

## Data Flow

```
Source Files (analysis/, etc.)
    ↓
[Update Script] (scripts/update_investor_doc.py)
    ↓
Curated Summary (docs/investor/data/backtest_summary.json)
    ↓
Investor Document (docs/investor/INVESTOR_PRESENTATION.md)
```

## Key Data Sources

### Primary Sources
- `analysis/production_backtest_results.json` - Production backtest data
- `analysis/EXECUTIVE_SUMMARY_BACKTEST.md` - Human-readable summary
- `analysis/RECENT_SEASON_BACKTEST_REPORT.md` - Detailed report
- `docs/investor/data/backtest_summary.json` - Curated summary

### System Status
- `app.py` (lines 83-95) - Validated systems list
- `VALIDATED_DOMAINS.md` - Domain validation status

## Update Workflow

### When New Backtest Results Are Generated

1. **Run Backtest**: Results saved to `analysis/production_backtest_results.json`
2. **Update Summary**: Manually curate `docs/investor/data/backtest_summary.json` if structure changed
3. **Run Update Script**: `python3 scripts/update_investor_doc.py`
4. **Review Changes**: Check updated metrics match source data
5. **Commit**: Version control changes

### When New Systems Are Validated

1. **Add to Validated List**: Update `app.py` comments (lines 83-95)
2. **Run Backtest**: Generate results for new system
3. **Update Summary**: Add new system to `backtest_summary.json`
4. **Run Update Script**: Update investor document
5. **Manual Review**: Add narrative sections if needed

### Regular Maintenance

- **Weekly**: Check for new backtest results (`--check-only`)
- **Monthly**: Run full update
- **After Major Validation**: Immediate update required

## Features

### Automated Updates
- ✅ Metrics table updates from JSON
- ✅ Financial projections recalculated
- ✅ Date stamps updated
- ✅ Data freshness checking

### Manual Review Required
- Executive summary narrative
- Competitive advantages section
- Expansion opportunities
- Implementation roadmap

## Testing

The update script has been tested and works correctly:
```bash
$ python3 scripts/update_investor_doc.py --check-only
✅ backtest_summary: 0 days old
✅ production_backtest: 3 days old
✅ executive_summary: 3 days old
```

## Benefits

1. **Consistency**: Document always reflects latest validated data
2. **Efficiency**: Automated updates save time
3. **Accuracy**: Reduces manual calculation errors
4. **Traceability**: Clear data source mapping
5. **Maintainability**: Well-documented update procedures

## Next Steps

1. **Set Reminders**: Schedule regular updates (weekly/monthly)
2. **Monitor Sources**: Watch for new backtest results
3. **Review Regularly**: Check document accuracy periodically
4. **Improve Script**: Add more automated sections as needed

## Questions?

See:
- `UPDATE_PROCEDURE.md` - Detailed update instructions
- `DATA_SOURCES.md` - Complete data source list
- `README.md` - Quick reference guide

---

**System Created**: November 2025  
**Status**: ✅ Operational

