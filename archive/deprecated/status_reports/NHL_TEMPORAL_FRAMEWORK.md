# NHL Temporal Framework - Complete Implementation Plan

**Created:** November 19, 2025  
**Purpose:** Perfect NHL as template for all sports domains with three-scale temporal modeling

---

## Current State

### NHL Production Model (Validated Nov 17, 2025)
- **Win Rate:** 69.4% (59-26 record)
- **ROI:** 32.5%
- **Features:** 900 total (79 structured + 821 narrative/transformer)
- **Temporal Features:** Only 6 (L10 wins, rest days, rest advantage)
- **Models:** Narrative Logistic, Gradient Boosting, Random Forest, Meta-Ensemble

### What's Missing
The current model lacks deep temporal context at three critical scales:

1. **MACRO-TEMPORAL** (Season-Long, 0-82 games)
   - Playoff push intensity
   - Underdog momentum arcs
   - Post-trade deadline effects
   - Coach change impacts
   - Season trajectory (improving/declining)

2. **MESO-TEMPORAL** (Recent Form, 5-20 games)
   - Multi-window streaks (L5, L10, L20)
   - Home/away splits (recent)
   - Divisional performance
   - Scoring trends (goals for/against)
   - Special teams momentum
   - Goalie rotation patterns

3. **MICRO-TEMPORAL** (In-Game, period-by-period)
   - Period momentum shifts
   - Comeback patterns
   - Lead protection rates
   - Empty net tendencies
   - Overtime/shootout history

---

## Implementation Created

### New Files
1. **`temporal_narrative_features.py`** - Three-scale feature extractor
   - `NHLTemporalExtractor` class
   - `extract_macro_temporal()` - 18 season-long features
   - `extract_meso_temporal()` - 22 recent form features
   - `extract_micro_temporal()` - 10 in-game features
   - **Total:** 50 new temporal features

2. **`integrate_temporal_features.py`** - Integration with existing pipeline
   - Loads historical season data
   - Enriches games with temporal context
   - Combines with existing 79-feature baseline
   - **Result:** 79 + 50 = 129 total features

---

## Integration Steps (To Complete)

### Step 1: Load Full Season Data
```python
# Need to load complete 2024-25 season data with:
# - All games to date
# - Team records at each point in time
# - Goalie assignments
# - Playoff standings
# - Trade deadline moves (if available)
```

**Data Source:** Either scrape from ESPN/NHL.com or use existing `nhl_games_with_odds.json` if it has season field

### Step 2: Enrich Training Dataset
```bash
cd narrative_optimization/domains/nhl
python3 enrich_training_with_temporal.py \
  --input nhl_narrative_betting_dataset.parquet \
  --output nhl_narrative_betting_temporal_dataset.parquet \
  --seasons 2023-24 2024-25
```

This will:
- Load 15,927 training games
- Add 50 temporal features to each
- Save as new parquet: 900 → 950 features

### Step 3: Retrain Models
```bash
python3 train_temporal_models.py \
  --data nhl_narrative_betting_temporal_dataset.parquet \
  --models logistic gradient forest meta \
  --output models/temporal/
```

Expected improvements:
- Win rate: 69.4% → 72-75%
- ROI: 32.5% → 38-42%
- Confidence calibration: Better edge detection on playoff push games

### Step 4: Update Daily Pipeline
Modify `scripts/data_generation/generate_all_predictions.py`:
```python
# Add temporal enrichment before feature extraction
from narrative_optimization.domains.nhl.temporal_narrative_features import NHLTemporalExtractor

def predict_nhl(date: str) -> List[Dict]:
    games = fetch_nhl_games_with_context(date)
    
    # NEW: Load season context
    season_data = load_current_season_data()
    
    # NEW: Enrich with temporal features
    extractor = NHLTemporalExtractor()
    for game in games:
        game['temporal_features'] = extractor.extract_all_temporal_features(game, season_data)
    
    # Build feature matrix (now includes temporal)
    feature_matrix = build_feature_matrix_with_temporal(games, feature_columns)
    
    # Rest of pipeline unchanged...
```

---

## Three-Scale Temporal Use Cases

### MACRO-TEMPORAL: Long-Term Betting Strategies

**Example 1: Playoff Push Underdog**
```
Game: Arizona Coyotes @ Colorado Avalanche (March 15)
Macro Features:
  - away_playoff_push: 0.85 (Coyotes fighting for wild card)
  - away_vs_expectation: +0.45 (15 wins above preseason projection)
  - away_desperation: 0.92 (bubble team, 10 games left)
  - home_playoff_push: 0.20 (Avalanche safely in)

Narrative: "Desperate underdog with season-long momentum vs complacent favorite"
Betting Angle: Take Coyotes +odds (market undervalues desperation)
```

**Example 2: Post-Trade Deadline Boost**
```
Game: Toronto Maple Leafs @ Boston Bruins (March 10)
Macro Features:
  - home_post_trade_deadline: 1.0 (Bruins acquired top defenseman 3 days ago)
  - days_since_trade_deadline: 0.10 (fresh acquisition)
  - home_trajectory: +0.30 (improving before trade)

Narrative: "Newly upgraded contender with integration momentum"
Betting Angle: Fade Bruins short-term (integration lag), back them after 10 games
```

### MESO-TEMPORAL: Streak Exploitation

**Example 3: Hot Streak Continuation**
```
Game: Vegas Golden Knights @ Seattle Kraken
Meso Features:
  - home_l5_wins: 5 (perfect 5-0)
  - home_l10_wins: 9 (9-1 run)
  - home_home_win_pct_l10: 1.00 (undefeated at home recently)
  - home_goals_per_game_l10: 4.8 (offensive explosion)

Narrative: "Peak-form home team with scoring surge"
Betting Angle: Back Seattle (market slow to adjust to hot streaks)
```

**Example 4: Goalie Rotation Edge**
```
Game: Tampa Bay Lightning @ Florida Panthers
Meso Features:
  - home_goalie_games_l5: 5 (starter overworked)
  - away_goalie_games_l5: 1 (backup, fresh)

Narrative: "Fatigued starter vs rested backup"
Betting Angle: Fade Panthers (goalie fatigue underpriced)
```

### MICRO-TEMPORAL: Live Betting Opportunities

**Example 5: Comeback Pattern**
```
Game: Edmonton Oilers @ Calgary Flames (LIVE - End of 2nd Period)
Micro Features (Checkpoint):
  - current_score_diff: -2 (Oilers trailing 1-3)
  - home_comeback_rate: 0.35 (Oilers strong 3rd period team)
  - home_recent_momentum: 1.0 (scored last goal)
  - period: 0.67 (2/3 complete)

Narrative: "Trailing team with 3rd period strength and late momentum"
Betting Angle: Live bet Oilers +odds (comeback narrative forming)
```

**Example 6: Lead Protection**
```
Game: Boston Bruins @ Montreal Canadiens (LIVE - End of 1st Period)
Micro Features:
  - current_score_diff: +2 (Bruins leading 3-1)
  - away_lead_protection_rate: 0.82 (Bruins elite at holding leads)
  - away_3rd_period_strength: 0.68 (prevent defense)

Narrative: "Elite team with early lead and strong close-out ability"
Betting Angle: Live bet Bruins to win (market overvalues Canadiens comeback chance)
```

---

## Why This Matters for Betting

### Current Model Limitations
The existing 69.4% win rate NHL model treats every game as isolated. It knows:
- Team records (snapshot)
- Last 10 games (crude momentum)
- Rest advantage (fatigue)

But it DOESN'T know:
- **Playoff desperation** (teams on bubble play harder)
- **Season arcs** (underdog momentum compounds)
- **Trade impacts** (roster changes take time to integrate)
- **Multi-window streaks** (5-game hot streak different from 20-game)
- **In-game momentum** (period-by-period dynamics)

### Expected Improvements with Temporal Framework

**Pre-Game Betting:**
- **Macro features** identify undervalued underdogs (playoff push, vs expectation)
- **Meso features** catch hot/cold streaks before market adjusts
- **Combined:** 69.4% → 72-75% win rate, 32.5% → 38-42% ROI

**Live Betting:**
- **Micro features** enable real-time edge detection
- Comeback patterns, lead protection, momentum shifts
- New market: Live betting recommendations (currently not offered)

---

## Next Steps to Complete NHL Template

### Immediate (This Session)
1. ✓ Create three-scale temporal extractor
2. ✓ Write integration framework
3. ⏳ Load full season data (need to scrape or find existing)
4. ⏳ Enrich training dataset with 50 temporal features
5. ⏳ Retrain models on 129-feature set
6. ⏳ Validate on holdout (expect 72%+ win rate)
7. ⏳ Update daily pipeline to use temporal models

### Short-Term (This Week)
8. Build live betting module using micro-temporal features
9. Create temporal feature importance analysis
10. Document which temporal scales matter most for NHL

### Template Replication (Next Week)
11. Apply same three-scale framework to NBA
12. Apply to NFL
13. Apply to MLB
14. Validate cross-sport temporal patterns

---

## Template for Other Sports

This NHL framework serves as the blueprint:

### NBA Temporal Scales
- **Macro:** Playoff seeding battles, trade deadline, tanking dynamics
- **Meso:** Back-to-back games, road trip fatigue, rest advantage
- **Micro:** Quarter-by-quarter momentum, comeback patterns (down 10+ at half)

### NFL Temporal Scales
- **Macro:** Playoff clinching scenarios, division races, bye week effects
- **Meso:** 3-game rolling performance, home/road splits, primetime records
- **Micro:** Half-by-half momentum, 4th quarter performance, overtime tendencies

### MLB Temporal Scales
- **Macro:** Pennant races, trade deadline acquisitions, September call-ups
- **Meso:** 10-game streaks, home stand vs road trip, bullpen usage
- **Micro:** Inning-by-inning momentum, late-inning comebacks, extra innings

---

## Key Insight: Temporal Narratives Are Underpriced

Markets are efficient at incorporating **current state** (records, recent form) but slow to price:
1. **Trajectory** (improving vs declining teams)
2. **Desperation** (playoff bubble urgency)
3. **Integration lag** (post-trade, post-injury, new coach)
4. **Multi-scale momentum** (5-game hot streak ≠ 20-game hot streak)
5. **In-game dynamics** (comeback patterns, lead protection)

The temporal framework captures these underpriced narratives and converts them to betting edge.

---

**Status:** Framework designed and coded. Awaiting full season data integration and model retraining.

