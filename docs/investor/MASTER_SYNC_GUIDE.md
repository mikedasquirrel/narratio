# Master Synchronization Guide
## Keeping Analysis, Data, Investor Docs, and Website in Perfect Sync

**Last Updated**: November 2025  
**Status**: ✅ Complete System Operational

---

## The Problem We Solved

You're running continuous analysis and backtests. Without a system, we risk:
- Outdated metrics on investor pages
- Mismatched numbers between documents
- Overwriting prior work
- Repeating calculations
- Investor docs showing old data

**Solution**: Single source of truth with automated regeneration pipeline.

---

## The Complete Workflow

### 1. Source of Truth: JSON Data Layer

**Canonical Data Files** (docs/investor/data/ and docs/investor/charts/):
```
docs/investor/data/
└── backtest_summary.json          ← SINGLE SOURCE OF TRUTH for metrics

docs/investor/charts/
├── roi_comparison_data.json       ← Chart specifications
├── statistical_significance_data.json
├── portfolio_profit_projections_data.json
├── training_production_comparison_data.json
├── market_efficiency_spectrum_data.json
└── volume_roi_tradeoff_data.json
```

**Rules**:
1. Never hardcode metrics in HTML/Markdown—always pull from JSON
2. Only edit JSON files to change metrics
3. All documents regenerate from these sources
4. Version control the JSON files (git tracks changes)

---

### 2. Update Pipeline (Always Follow This Order)

```
Step 1: Run Analysis/Backtest
    ↓
Step 2: Results saved to analysis/*.json
    ↓
Step 3: Update docs/investor/data/backtest_summary.json (curate metrics)
    ↓
Step 4: Regenerate dashboard (python3 scripts/generate_investor_dashboard.py)
    ↓
Step 5: Update markdown docs (python3 scripts/update_investor_doc.py)
    ↓
Step 6: Verify website (/investor route shows correct data)
    ↓
Step 7: Commit to git (version control)
```

---

### 3. Single Command for Bots

**Copy this to another bot** (zero context needed):

```
cd /Users/michaelsmerconish/Desktop/RandomCode/novelization && python3 scripts/generate_investor_dashboard.py && echo "Dashboard updated. Open at: http://127.0.0.1:5738/investor/dashboard"
```

This regenerates the interactive dashboard and makes it immediately available on the website.

---

### 4. Website Integration (Flask Routes)

**Investor Pages** (all live on website):

| Route | Description | Source |
|-------|-------------|--------|
| `/investor` | Landing page with navigation | `templates/investor_landing.html` |
| `/investor/dashboard` | Interactive HTML with 11 Plotly charts | `docs/investor/INTERACTIVE_DASHBOARD.html` |
| `/investor/proposal` | One-page proposal (multiples format) | `docs/investor/INVESTMENT_PROPOSAL.md` |
| `/investor/presentation` | Comprehensive 100+ page doc | `docs/investor/INVESTOR_PRESENTATION.md` |
| `/investor/validation` | Technical validation report | `docs/investor/TECHNICAL_VALIDATION_REPORT.md` |
| `/investor/api/metrics` | JSON API endpoint | `backtest_summary.json` |

**Betting System Pages**:
| Route | Win Rate | ROI | Status |
|-------|----------|-----|--------|
| `/nhl` | 69.4% | 32.5% | ✅ Primary |
| `/nfl` | 66.7% | 27.3% | ✅ Secondary |
| `/nba` | 54.5% | 7.6% | ⚠ Marginal |

**Home Page**: Updated with prominent investor section at top (green CTA)

---

### 5. How to Prevent Conflicts

**Rule**: Only one person/bot edits JSON files at a time

**Protocol**:
1. **Before starting analysis**: Check if JSON files are being edited (git status)
2. **After analysis**: Update JSON, regenerate, commit
3. **Never manually edit**: Generated files (INTERACTIVE_DASHBOARD.html)
4. **Always regenerate**: Don't copy-paste metrics into HTML/Markdown

**Verification**:
```bash
# Check what changed
git status

# See JSON diffs
git diff docs/investor/data/

# Verify dashboard regenerated
ls -lh docs/investor/INTERACTIVE_DASHBOARD.html
```

---

### 6. Data Freshness Checks

**Check if data is current**:
```bash
python3 scripts/update_investor_doc.py --check-only
```

**Output**:
```
📊 Checking data freshness...
  ✅ backtest_summary: 0 days old
  ✅ production_backtest: 3 days old
  ✅ executive_summary: 3 days old
```

**If data is stale** (>30 days):
- Review source files in `analysis/`
- Run new backtest if needed
- Update `backtest_summary.json`
- Regenerate all documents

---

### 7. Complete Update Checklist

**When New Backtest Results Arrive**:

- [ ] New backtest saved: `analysis/production_backtest_results.json`
- [ ] Review numbers: Check win rates, ROI, sample sizes
- [ ] Update curated summary: `docs/investor/data/backtest_summary.json`
- [ ] Update chart data (if structure changed): `docs/investor/charts/*.json`
- [ ] Regenerate dashboard: `python3 scripts/generate_investor_dashboard.py`
- [ ] Update markdown docs: `python3 scripts/update_investor_doc.py`
- [ ] Verify website: Visit `http://127.0.0.1:5738/investor`
- [ ] Check all routes work: /investor, /investor/dashboard, /nhl, /nfl, /nba
- [ ] Verify charts render: Open INTERACTIVE_DASHBOARD.html in browser
- [ ] Commit changes: `git add docs/investor/ && git commit -m "Update investor docs with latest backtest"`

---

### 8. What Gets Updated Automatically

**INTERACTIVE_DASHBOARD.html** (generated by script):
- ✅ All 11 Plotly charts
- ✅ Key metrics cards
- ✅ Statistical tables
- ✅ Financial projections
- ✅ Data source timestamps
- **DO NOT EDIT MANUALLY**

**INVESTOR_PRESENTATION.md** (partially automated):
- ✅ Metrics tables (via update script)
- ✅ Date stamps
- ⚠ Narrative sections (manual review)
- ⚠ New discoveries (manual addition)

**Website Routes** (/investor/*):
- ✅ Dashboard serves latest HTML
- ✅ API endpoint returns latest JSON
- ✅ Landing page pulls from JSON
- **Routes automatically show current data**

---

### 9. What Requires Manual Review

**After regeneration, manually check**:
- Executive summary narratives (did findings change?)
- Competitive advantages (any new edges discovered?)
- Risk analysis (any new risks identified?)
- Expansion opportunities (any new domains validated?)

**If substantive changes**: Update the prose in markdown documents manually.

---

### 10. Verification Commands

**After updating, verify everything is in sync**:

```bash
# 1. Check dashboard generated
ls -lh docs/investor/INTERACTIVE_DASHBOARD.html

# 2. Verify metrics match in all files
grep "69.4" docs/investor/*.md  # Should find NHL win rate
grep "32.5" docs/investor/*.md  # Should find NHL ROI

# 3. Check website serves latest
curl http://127.0.0.1:5738/investor/api/metrics | jq '.systems.nhl.results.meta_ensemble_65.win_rate'
# Should output: 0.694

# 4. Open dashboard in browser
open docs/investor/INTERACTIVE_DASHBOARD.html
# Verify all 11 charts render

# 5. Test website routes
open http://127.0.0.1:5738/investor
open http://127.0.0.1:5738/investor/dashboard
```

---

### 11. Git Workflow

**Commit Strategy**:
```bash
# After updating data
git add docs/investor/data/*.json
git add docs/investor/charts/*.json
git commit -m "Update backtest metrics: NHL 69.4%, NFL 66.7%"

# After regenerating documents
git add docs/investor/INTERACTIVE_DASHBOARD.html
git add docs/investor/INVESTOR_PRESENTATION.md
git commit -m "Regenerate investor documents from latest data"

# Never commit separately - keeps consistency
git add docs/investor/ && git commit -m "Complete investor package update"
```

**This ensures**: All files update together, no partial updates, version history clear.

---

### 12. For Continuous Analysis

**As you continue research**:

1. **New Domain Validated**: 
   - Add to `docs/investor/data/backtest_summary.json`
   - Regenerate dashboard
   - Update markdown docs

2. **Better Model Trained**:
   - Run holdout test
   - Update metrics in JSON
   - Regenerate all docs

3. **New Statistical Test**:
   - Add to `technical_validation_report` markdown manually
   - Reference in dashboard HTML

4. **New Context Discovered**:
   - Add to backtest summary
   - Update NFL/NBA sections
   - Regenerate

**The system handles incremental updates gracefully.**

---

### 13. Current Status (As of Now)

**Data Layer**: ✅ Complete
- 8 JSON files with all metrics and chart specs
- Source data in `analysis/`
- All files version controlled

**Documents**: ✅ Complete
- INVESTOR_PRESENTATION.md (57KB, comprehensive)
- INVESTMENT_PROPOSAL.md (6KB, one-page multiples)
- TECHNICAL_VALIDATION_REPORT.md (37KB, skeptic-proof)
- INTERACTIVE_DASHBOARD.html (227KB, 11 charts)
- INVESTMENT_TERMS_TEMPLATE.md (6KB, negotiation)

**Website**: ✅ Integrated
- `/investor` - Beautiful landing page
- `/investor/dashboard` - Serves interactive HTML
- `/nhl`, `/nfl`, `/nba` - Betting system pages
- Home page has prominent investor CTA (green)

**Scripts**: ✅ Operational
- `generate_investor_dashboard.py` - Dashboard generator (tested)
- `update_investor_doc.py` - Markdown updater
- Both scripts working, tested, documented

**Verification**: ✅ Tested
- Dashboard generated: 227KB, 11 charts
- Website routes work
- All data synchronized
- No linter errors

---

## Summary: How It All Stays in Sync

1. **Single source**: JSON files in `docs/investor/data/` and `docs/investor/charts/`
2. **One command**: `python3 scripts/generate_investor_dashboard.py` regenerates everything
3. **Website integration**: Flask routes serve latest files automatically
4. **No manual editing**: Generated files always come from scripts
5. **Version control**: Git tracks all JSON changes
6. **Verification**: Scripts check data freshness

**Result**: Analysis → JSON → Regenerate → Website shows latest data. Always in sync.

---

## Quick Reference Card

**Update Command**:
```bash
python3 scripts/generate_investor_dashboard.py
```

**What It Updates**:
- docs/investor/INTERACTIVE_DASHBOARD.html (11 charts)
- Website serves latest at /investor/dashboard
- Takes 3-5 seconds

**When To Run**:
- After new backtest results
- After validation changes
- Weekly/monthly maintenance
- When metrics update

**How To Verify**:
```bash
open docs/investor/INTERACTIVE_DASHBOARD.html
# Check: charts render, metrics match, timestamp current
```

**That's It**: One command, everything stays synchronized.

---

**System Status**: ✅ Complete and Operational  
**Command**: `python3 scripts/generate_investor_dashboard.py`  
**Output**: Synchronized investor package ready for confidential investors

