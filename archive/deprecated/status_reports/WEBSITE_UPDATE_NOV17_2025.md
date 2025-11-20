# Website Update - November 17, 2025
## Validation-Based Page Visibility

**Date**: November 17, 2025  
**Change Type**: Major restructuring - disabled non-validated pages  
**Reason**: Show only systems validated through most recent production pipeline

---

## Summary

Disabled all website pages that have not been validated through the November 2025 production backtest pipeline. Only the three sports systems with validated holdout performance (NHL, NFL, NBA) remain active, along with core framework tools.

---

## What Changed

### ✅ KEPT ACTIVE (Validated Systems)

**Validated Sports (Nov 17, 2025 Production Backtest):**
1. **NHL** - 69.4% win rate, 32.5% ROI ✅
   - Routes: `/nhl`, `/nhl-results`, NHL betting
   - Status: PRODUCTION READY

2. **NFL** - 66.7% win rate, 27.3% ROI ✅
   - Routes: `/nfl`, `/nfl-results`, NFL betting
   - Status: PRODUCTION READY

3. **NBA** - 54.5% win rate, 7.6% ROI ✅
   - Routes: `/nba`, `/nba-results`, NBA betting
   - Status: VALIDATED (low priority due to ROI)

**Core Tools (Always Available):**
- Home page (`/`)
- Analysis tools (`/analyze`)
- Narrative Analyzer
- Domain Processor (`/process`)
- Framework Story (`/framework-story`)
- Project Overview (`/project-overview`)
- Variables page (`/variables`)
- Narrativity spectrum page
- Betting infrastructure (dashboard, API)
- Cross-domain tools
- Transformer analysis (`/transformers/analysis`)
- Simple findings pages

---

### ❌ TEMPORARILY DISABLED (Awaiting Re-validation)

**Sports Not in Recent Validation:**
- Tennis (older analysis, not in Nov 2025 backtest)
- Golf (older analysis, not in Nov 2025 backtest)
- UFC (older analysis, not in Nov 2025 backtest)
- MLB (has data but not validated in recent backtest)
- Tennis betting
- Golf betting
- UFC betting

**Older Domain Analyses (Pre-Nov 2025):**
- Mental Health (`/mental-health`)
- Movies/IMDB (`/imdb`, `/movie-results`)
- Oscars (`/oscar-results`)
- Crypto (`/crypto`)
- Startups (`/startups`)
- Housing (`/housing`)
- Music (`/music`)
- Supreme Court (`/supreme-court`)
- Free Will (theory page)

**Domain Explorer Pages:**
- Domain Index (`/domains`)
- Domain Explorer (`/domains/explorer`)
- Domain Detail (`/domains/<name>`) - now returns 404
- Domain Comparison (`/domains/compare`)

**Archetype System (Not Validated):**
- All archetype routes (`/archetypes/*`)
- Archetype API endpoints (`/api/archetypes/*`)
- Betting archetype opportunities page
- Theory integration pages

**Experimental/Incomplete (Already Disabled):**
- Bible, Conspiracies, Dinosaurs, Hurricanes, Poker, Novels, WWE, Temporal Linguistics

**Utility/Meta Pages:**
- Meta Evaluation
- Insights Dashboard
- Prediction AI
- Checkpoint Feeds

---

## Files Modified

### Primary Changes

**File**: `app.py`

**Changes Made:**
1. Commented out blueprint imports for non-validated systems
2. Commented out blueprint registrations for non-validated routes
3. Disabled standalone route functions for non-validated result pages
4. Disabled domain explorer routes
5. Disabled all archetype-related routes and APIs
6. Updated startup messages to reflect validated-only status
7. Added clear documentation comments explaining what's disabled and why

**Line Changes:**
- Imports section: Lines 66-135 - Reorganized with validated systems only
- Registration section: Lines 145-238 - Only register validated blueprints
- Results pages: Lines 278-350 - Disabled non-validated domain results
- Domain explorer: Lines 368-400 - Disabled domain detail/comparison
- Archetypes: Lines 984-997 - Removed archetype system entirely
- Startup: Lines 999-1017 - Updated to show validated systems only

---

## Validation Criteria

**"Validated" means:**
- System was tested on Nov 17, 2025 production backtest
- Used actual trained models with complete feature extraction
- Tested on most recent season holdout data (2024-25 or 2024)
- Results documented in `analysis/EXECUTIVE_SUMMARY_BACKTEST.md`
- Win rate and ROI calculated on real unseen data

**Systems that passed:**
- ✅ NHL: 69.4% win, 32.5% ROI (85 bets, ≥65% confidence)
- ✅ NFL: 66.7% win, 27.3% ROI (9 bets, QB Edge + Home Dog pattern)
- ✅ NBA: 54.5% win, 7.6% ROI (44 bets, Elite Team + Close Game pattern)

---

## Re-Enabling Disabled Pages

To re-enable a disabled system after validation:

1. **Run through current production pipeline** (as documented in validation report)
2. **Test on holdout data** from most recent season
3. **Document results** in analysis/ directory
4. **Uncomment routes** in app.py:
   - Remove `#` from blueprint import
   - Remove `#` from blueprint registration
   - Remove `#` from standalone routes
5. **Update startup messages** to include the validated system
6. **Update this file** with new validation status

---

## Impact

### Before Update
- ~40 routes active (many with outdated/unvalidated data)
- Mix of validated and non-validated systems
- User confusion about what's current

### After Update
- ~15-20 routes active (all validated or core tools)
- Only 3 sports systems shown (all validated Nov 2025)
- Clear focus on production-ready systems
- Users see only current, reliable information

---

## Benefits

1. **Data Integrity**: Only show validated, current results
2. **User Trust**: No outdated or unverified claims
3. **Clear Status**: Easy to see what's production-ready vs in-progress
4. **Maintainability**: Clear separation between validated and unvalidated
5. **Professional**: Clean, focused interface showing only quality results

---

## Next Steps

### Priority 1: Maintain Validated Systems
- Monitor NHL, NFL, NBA performance in live betting
- Update results weekly
- Maintain models and pipelines

### Priority 2: Re-validate High-Value Domains
Re-run through current pipeline in this order:
1. **Tennis** - Previously showed 93.1% R², 127% ROI
2. **Golf** - Previously showed 97.7% R² with nominative enrichment
3. **UFC** - Performance-dominated but high narrativity
4. **MLB** - Has data, not yet backtested

### Priority 3: Re-validate Analysis Domains
After sports validation complete:
- Mental Health
- Crypto
- Startups
- Movies/IMDB/Oscars

### Priority 4: Archetype System
- Complete archetype validation framework
- Test on validated sports first
- Re-enable if adds value to existing systems

---

## Technical Notes

### Code Organization

**Disabled sections clearly marked:**
```python
# ============================================================================
# DISABLED: REASON (Not validated in Nov 2025 pipeline)
# ============================================================================
```

**Easy to find and re-enable:**
- All disabled imports in one section (lines 96-135)
- All disabled registrations in one section (lines 191-238)
- All disabled routes grouped by type
- Clear comments explaining original functionality

### No Breaking Changes

- All disabled code preserved in comments
- Can be re-enabled by removing `#` comments
- No data files deleted
- No templates deleted
- Routes return 404 instead of errors

---

## Validation Reference

**See full validation details:**
- `analysis/EXECUTIVE_SUMMARY_BACKTEST.md` - Executive summary
- `analysis/RECENT_SEASON_BACKTEST_REPORT.md` - Detailed technical report (323 lines)
- `analysis/production_backtest_results.json` - Raw numerical results

**Validation methodology:**
- Production models loaded from pickle files
- Complete 79-feature extraction for NHL (50 performance + 29 nominative)
- Proper temporal splits (training on historical, testing on 2024-25)
- Real predictions on unseen data
- Conservative confidence thresholds
- Multiple model comparison (Meta-Ensemble, GBM)

---

**Status**: Complete  
**Approved By**: Production validation results  
**Next Review**: After next batch of domain validations

