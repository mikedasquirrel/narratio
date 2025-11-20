# Investor Documentation

This directory contains investor-facing documentation and presentations.

## Files

- **`INVESTOR_PRESENTATION.md`** - Main investor presentation document
- **`UPDATE_PROCEDURE.md`** - How to keep the document up-to-date
- **`DATA_SOURCES.md`** - Tracking of all data sources
- **`data/`** - Curated data files for the presentation
- **`charts/`** - Chart specifications and data

## Quick Start

### Viewing the Document

The main investor presentation is in `INVESTOR_PRESENTATION.md`. It can be:
- Viewed directly in markdown
- Converted to PDF (using pandoc or similar)
- Exported to HTML

### Updating the Document

**Automatic Update**:
```bash
python scripts/update_investor_doc.py
```

**Check Data Freshness**:
```bash
python scripts/update_investor_doc.py --check-only
```

**Dry Run** (see what would change):
```bash
python scripts/update_investor_doc.py --dry-run
```

### Manual Update

See `UPDATE_PROCEDURE.md` for detailed manual update instructions.

## Data Sources

All metrics come from authoritative source files:
- `analysis/production_backtest_results.json`
- `analysis/EXECUTIVE_SUMMARY_BACKTEST.md`
- `docs/investor/data/backtest_summary.json`

See `DATA_SOURCES.md` for complete list.

## Maintenance

- **Weekly**: Check for new backtest results
- **Monthly**: Run update script
- **After Major Validation**: Immediate update required

## Version Control

Document versions are tracked in:
- File header: `Document Version: X.Y`
- Git commits
- Update timestamps

## Questions?

See `UPDATE_PROCEDURE.md` for detailed procedures.

