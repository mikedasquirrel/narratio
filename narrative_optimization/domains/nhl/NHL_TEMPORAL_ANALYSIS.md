# NHL Temporal Feature Analysis & Results

**Date:** November 19, 2025  
**Purpose:** Document temporal framework implementation and results

---

## Implementation Summary

### What Was Built
1. **Three-Scale Temporal Framework** (50 features)
   - Macro-temporal: 18 features (season-long narratives)
   - Meso-temporal: 22 features (recent form)
   - Micro-temporal: 10 features (in-game dynamics)

2. **Data Enrichment Pipeline**
   - Enriched 15,927 training games
   - Added real data for scoring trends, venue splits, comeback patterns
   - Final dataset: 956 features (900 baseline + 56 temporal/real)

3. **Model Training**
   - Trained Logistic, Gradient Boosting, Random Forest, Meta-Ensemble
   - Temporal holdout validation (last 3,186 games)

---

## Results

### Temporal Models (V2 - With Real Data)

| Confidence Tier | Threshold | Bets | Win Rate | ROI | Record |
|-----------------|-----------|------|----------|-----|--------|
| High-Confidence | ≥60% | 703 (22.1%) | 65.4% | +25.0% | 460-243 |
| Ultra-Confident | ≥65% | 278 (8.7%) | 68.3% | +30.5% | 190-88 |
| Elite | ≥70% | 91 (2.9%) | 71.4% | +36.4% | 65-26 |

### Baseline Models (900 features, Nov 17)

| Confidence Tier | Win Rate | ROI | Record |
|-----------------|----------|-----|--------|
| Ultra-Confident (≥65%) | 69.4% | +32.5% | 59-26 |

---

## Analysis: Why Temporal Features Underperformed

### Features That Worked (High Correlation with Outcome)
1. **expectation_differential** (r=0.154) - Teams exceeding/missing preseason projections
2. **l20_win_differential** (r=0.121) - 20-game form differential
3. **away_vs_expectation** (r=0.110) - Away team momentum vs expectations
4. **l10_win_differential** (r=0.093) - 10-game form (already in baseline as ctx_form_differential)
5. **playoff_push_differential** (r=0.063) - Desperation/urgency

### Features That Didn't Work (Zero Variance or No Data)
1. **Coach change features** - No coach change database (all returned 100.0)
2. **Divisional records** - No division assignments (all returned 0.5)
3. **Power play stats** - No PP/PK data (all returned 0.2)
4. **Period-by-period stats** - No checkpoint data in training set
5. **Overtime tendencies** - No OT-specific tracking

### Key Finding: Redundancy with Baseline
Many temporal features (L5/L10/L20 wins, goals for/against) are **already captured** in the baseline 900 features through:
- `ctx_home_l10_wins`, `ctx_away_l10_wins`
- Narrative transformer features (temporal momentum, scoring patterns)

Adding them again as explicit features creates **multicollinearity** without adding new information.

---

## What Actually Adds Value

### 1. Expectation Differential (r=0.154)
**What it is:** Difference between actual wins and preseason projections  
**Why it matters:** Markets are slow to adjust to teams exceeding/missing expectations  
**Implementation:** Need preseason win total odds for each team

### 2. Playoff Push Intensity (r=0.063)
**What it is:** Desperation index for teams on playoff bubble  
**Why it matters:** Teams play harder when season is on the line  
**Implementation:** Need real-time playoff standings

### 3. Post-Trade Deadline Effects
**What it is:** Performance change after roster moves  
**Why it matters:** Integration lag creates temporary mispricing  
**Implementation:** Need trade deadline transaction database

### 4. Venue-Specific Momentum
**What it is:** Home/away splits beyond simple win %  
**Why it matters:** Some teams are dramatically better at home  
**Implementation:** Already calculated in v2 dataset

---

## Recommendations

### Option A: Focus on High-Value Temporal Features Only
Instead of 50 features (many redundant/placeholder), add only the 8-10 that provide unique signal:
1. Expectation differential (vs preseason projections)
2. Playoff push intensity (real-time standings)
3. Post-trade deadline indicator
4. Venue-specific momentum (home/away L10 splits)
5. Multi-window form (L5 vs L20 divergence)
6. Desperation index (bubble teams, games remaining)

**Expected improvement:** 69.4% → 71-72% win rate

### Option B: Build Live Betting System (Micro-Temporal Focus)
The real value of temporal features is in **live betting** where you can:
- Track period-by-period momentum
- Identify comeback patterns in real-time
- Exploit lead protection inefficiencies
- Bet on momentum shifts

**Expected value:** New market (currently not offered in daily pipeline)

### Option C: Collect Missing Data Sources
To make all 50 temporal features work, we need:
- Preseason win total odds (from sportsbooks)
- Real-time playoff standings API
- Trade deadline transaction database
- Period-by-period scoring data
- Power play/penalty kill stats
- Coach change tracking

**Timeline:** 1-2 weeks of data collection

---

## Recommendation: Option A + B

**Immediate (Today):**
1. Implement focused temporal features (8-10 high-value only)
2. Retrain on streamlined feature set
3. Validate improvement over baseline

**Short-Term (This Week):**
4. Build live betting module using micro-temporal framework
5. Test on tonight's games with period-by-period updates

**Medium-Term (Next Week):**
6. Collect missing data sources (preseason odds, playoff standings, trades)
7. Expand to full 50-feature temporal framework

---

## Current Status

✅ Framework designed and coded  
✅ 15,927 games enriched with temporal features  
✅ Models trained and validated  
⚠️ Performance not yet better than baseline (need focused features)  
⏳ Ready for focused feature implementation

---

**Next Action:** Implement Option A (focused temporal features) to achieve 71-72% win rate target.

