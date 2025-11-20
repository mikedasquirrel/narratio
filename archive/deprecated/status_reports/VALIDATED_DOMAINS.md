# Validated Domains - Production Status
## Updated: November 17, 2025

Track which domains have been validated through the current universal pipeline and are ready for website display.

---

## VALIDATED & LIVE

### Betting Systems (Production Ready)

**NHL** ✅
- **Validated**: Nov 17, 2025
- **Win Rate**: 69.4%
- **ROI**: 32.5%
- **Sample**: 2,779 games (2024-25 season)
- **Status**: PRODUCTION READY
- **Routes**: `/nhl`, `/nhl-results`, NHL betting
- **File**: `analysis/production_backtest_results.json`

**NFL** ✅  
- **Validated**: Nov 17, 2025
- **Win Rate**: 66.7%
- **ROI**: 27.3%
- **Sample**: 285 games (2024 season)
- **Pattern**: QB Edge + Home Dog
- **Status**: PRODUCTION READY
- **Routes**: `/nfl`, `/nfl-results`, NFL betting
- **File**: `analysis/production_backtest_results.json`

**NBA** ✅
- **Validated**: Nov 17, 2025
- **Win Rate**: 54.5%
- **ROI**: 7.6%
- **Sample**: 1,230 games (2023-24 season)
- **Pattern**: Elite Team + Close Game
- **Status**: VALIDATED (low priority - marginal ROI)
- **Routes**: `/nba`, `/nba-results`, NBA betting
- **File**: `analysis/production_backtest_results.json`

### Research Domains (Validated Theory)

**Hurricanes** ✅
- **Validated**: Nov 17, 2025
- **Dual π**: Storm 0.30 / Response 0.68
- **Sample**: 819 hurricanes (complete Atlantic dataset)
- **Finding**: Name effects on evacuation confirmed
  - Gender effects detected
  - Phonetic hardness: 0.20-0.26
  - Evacuation intent variation: 34%-56%
  - Nominative bias in fatality modeling validated
- **Status**: RESEARCH DOMAIN (validates nominative determinism)
- **Routes**: `/hurricanes`, `/hurricanes-results`
- **File**: `narrative_optimization/results/domains/hurricanes/n819_analysis.json`

### Individual Sports (Validated)

**Golf** ✅
- **Validated**: Nov 17, 2025
- **Narrativity (π)**: 0.70 (individual sport, mental game awareness)
- **Sample**: 5,000 tournaments (from 7,700 available)
- **Patterns**: 20 distinct narrative patterns
- **Effect Sizes**:
  - Strongest: +0.117 to -0.107 (11% range)
  - Median: 0.07 (7% effect - moderate)
- **Statistical Significance**: 100% (high calibration, monitor for overfitting)
- **Finding**: Moderate narrative effects in tournament play
  - Individual agency (1.00) - personal performance matters
  - Mental game awareness shows in patterns
  - Nominative enrichment potential (not yet tested in this run)
- **Status**: SPORTS ANALYSIS (moderate predictive power, not betting-tested yet)
- **Routes**: `/golf`, `/golf-results`
- **File**: `narrative_optimization/results/domains/golf/n5000_analysis.json`
- **Note**: Previous analysis showed 97.7% R² with nominative enrichment - current results more conservative at ~7% median effect. Different methodologies.

### Business Domains (Validated)

**Startups** ⚠️
- **Validated**: Nov 17, 2025
- **Narrativity (π)**: 0.76 (business/speculation domain)
- **Sample**: 258 companies (complete available dataset)
- **Patterns**: 4 discovered (expected 21 based on π)
- **Effect Sizes**: ~0.13 (13% effect - moderate)
- **Statistical Significance**: 0% cluster patterns, only latent dimensions significant
- **Finding**: Some correlation exists but pattern structure is weak
  - Latent dimension effects: ±0.13
  - Too few patterns discovered
  - Small sample size limits robustness
- **Status**: ⚠️ MARGINAL - Validated but with warnings
- **Issues**:
  - Sample too small (258 vs thousands needed)
  - Pattern count low (4 vs expected 21)
  - Low significance rate
  - Needs larger dataset for robust conclusions
- **Routes**: `/startups`, `/startups-results`
- **File**: `narrative_optimization/results/domains/startups/n258_analysis.json`
- **Previous Claim**: r=0.980 NOT replicated - that appears to have been overfit or different methodology
- **Recommendation**: Collect more startup data (target 1000+) before making production claims

### Legal Domains (Validated)

**Supreme Court** ✅
- **Validated**: Nov 17, 2025
- **Narrativity (π)**: 0.52 (moderate - adversarial, evidence-based)
- **Agency (Δ)**: 0.306 (moderate leverage of narrative in legal system)
- **Sample**: 26 opinions (from 26 available - small sample)
- **Correlation**: r = 0.785 (narrative quality → citations)
- **R²**: 61.6% (strong predictive power)
- **Predictive Model**: RandomForest R² = 75.8% (train on 20, test on 6)
- **Finding**: Narrative quality strongly predicts forward citations
  - Better-written opinions get cited more
  - Works in evidence-weighted adversarial system
  - π variance not confirmed (unanimous vs split cases similar)
- **Status**: RESEARCH DOMAIN (legal analysis, citation prediction)
- **Routes**: `/supreme-court`, Supreme Court pages
- **File**: `narrative_optimization/results/domains/supreme_court_results.json`
- **Limitations**: Small sample (26 opinions) - need more cases for robust validation

### Entertainment Domains (Validated)

**Movies / IMDB** ✅
- **Validated**: Nov 17, 2025
- **Narrativity (π)**: 0.65 (entertainment domain)
- **Sample**: 2,000 films (from 6,047 available)
- **Patterns**: 20 distinct narrative patterns discovered
- **Effect Sizes**: 
  - Strongest: -0.899 to +0.631 (very strong)
  - Median: 0.40 (strong narrative signal)
- **Statistical Significance**: 100% of patterns significant
- **Finding**: Narrative structure significantly predicts movie success
  - Pattern 0 (high success): 0.617 vs 0.547 baseline (+12.8%)
  - Pattern 2 (low success): 0.469 vs 0.569 baseline (-17.6%)
- **Status**: ENTERTAINMENT ANALYSIS (not betting, cultural prediction)
- **Routes**: `/movies`, `/imdb`, `/movie-results`, `/imdb-results`
- **File**: `narrative_optimization/results/domains/movies/n2000_analysis.json`

---

## PENDING VALIDATION

### High Priority Sports

**Tennis** ⏳
- Previous: 93.1% R², 127% ROI
- Status: AWAITING revalidation
- Expected: High value betting domain

**Golf** ✅ VALIDATED
- Validated: Nov 17, 2025
- Result: 20 patterns, median effect 0.07 (7%)
- Status: LIVE ON WEBSITE
- Note: 97.7% R² from previous analysis not replicated - different methodology

**UFC** ⏳
- Previous: High π (0.72) but low Δ (0.025)
- Status: AWAITING revalidation
- Expected: Performance-dominated confirmation

**MLB** ⏳
- Previous: Not validated
- Status: AWAITING revalidation
- Expected: Betting potential testing

### Business Domains

**Startups** ⚠️ VALIDATED (MARGINAL)
- Validated: Nov 17, 2025
- Result: 4 patterns (expected 21), effect 0.13, n=258
- Status: LIVE ON WEBSITE WITH WARNINGS
- Note: Previous r=0.980 NOT replicated. Small sample (258) limits conclusions. Needs 1000+ samples.

**Crypto** ⏳
- Previous: AUC=0.925
- Status: AWAITING revalidation
- Expected: Speculation domain validation

### Entertainment

**Movies** ✅ VALIDATED
- Validated: Nov 17, 2025
- Result: 20 patterns, median effect 0.40
- Status: LIVE ON WEBSITE

**Oscars** ⏳
- Previous: AUC=1.00 (perfect)
- Status: AWAITING revalidation
- Expected: Overfit testing

**Music** ⏳
- Previous: Low R²
- Status: AWAITING revalidation

### Research/Theory

**Dinosaurs** ⏳
- Previous: 62.3% R² from names (156x > science)
- Status: AWAITING revalidation
- Expected: Educational transmission theory

**Mythology** ⏳
- Previous: π=0.85 (pure narrative)
- Status: AWAITING revalidation

**Poker** ⏳
- Previous: Highest π (0.83) but variance-dominated
- Status: AWAITING revalidation

**Boxing** ⏳
- Previous: Not validated
- Status: AWAITING revalidation

**Supreme Court** ✅ VALIDATED
- Validated: Nov 17, 2025
- Result: r=0.785, R²=61.6%, n=26 opinions
- Status: LIVE ON WEBSITE
- Note: Small sample but strong correlation between narrative quality and citations

---

## VALIDATION WORKFLOW

### For Each Domain:

1. **Run Universal Pipeline**:
   ```bash
   python3 narrative_optimization/universal_domain_processor.py \
     --domain [NAME] --sample_size [N] --use_transformers --save_results
   ```

2. **Extract Results**:
   - Check: `narrative_optimization/results/domains/[domain]_results.json`
   - Get: π, Δ, R², sample size, patterns

3. **Update Website**:
   - Uncomment blueprint import in `app.py`
   - Uncomment blueprint registration
   - Add/update results route
   - Verify template exists

4. **Document Here**:
   - Move from PENDING to VALIDATED
   - Record all metrics
   - Note key findings
   - Update status

5. **Test**:
   - Run `python3 app.py`
   - Visit route in browser
   - Verify data displays correctly

---

## VALIDATION CRITERIA

**Production Ready (Betting):**
- ✅ Δ/π > 0.5 (agency exceeds threshold)
- ✅ ROI > 20% on holdout data
- ✅ Win rate > 58%
- ✅ Statistical significance
- ✅ Temporal validation (recent season)

**Validated Research:**
- ✅ Pipeline completed successfully
- ✅ Sample size adequate (>200)
- ✅ Key metrics calculated (π, Δ, R²)
- ✅ Findings documented
- ✅ Theory tested/validated

**Marginal:**
- ⚠️ Δ/π between 0.3-0.5
- ⚠️ ROI 5-20%
- ⚠️ Win rate 53-58%
- ⚠️ Small sample size
- ⚠️ Needs more testing

---

## QUICK STATUS SUMMARY

**Live on Website:** 9 domains
- NHL (betting - 69.4% win, 32.5% ROI)
- NFL (betting - 66.7% win, 27.3% ROI)
- NBA (betting - 54.5% win, 7.6% ROI)
- Golf (individual sport - 7% median effect)
- Startups (business - MARGINAL, 4 patterns, 13% effect)
- Hurricanes (research - dual π, name effects)
- Movies (entertainment - 40% median effect)
- IMDB (entertainment - same as movies)
- Supreme Court (legal - r=0.785, R²=61.6%)

**Awaiting Revalidation:** 11+ domains
- Sports: Tennis, UFC, MLB, Boxing
- Business: Crypto
- Entertainment: Oscars, Music
- Research: Dinosaurs, Mythology, Poker, Mental Health, Housing

**Total Registered:** 20+ domains in `domain_registry.py`

---

## NEXT STEPS

1. **Validate High-Priority Sports** (Tennis, Golf first)
2. **Validate Business Domains** (Startups, Crypto)
3. **Validate Entertainment** (Movies, Oscars)
4. **Complete Research Domains** (Dinosaurs, etc.)

As each completes, update this file and enable on website.

---

**Last Updated**: November 17, 2025  
**Validated Domains**: 9/20 (8 unique domains + IMDB alias)
**Next Target**: Tennis (high ROI potential), UFC, MLB, Crypto

