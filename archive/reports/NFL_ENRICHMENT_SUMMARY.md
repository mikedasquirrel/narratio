# NFL Data Enrichment Summary
## Rosters, QBs, Key Players & Matchup History

**Created**: November 16, 2025  
**Source**: [nflverse-data](https://github.com/nflverse/nflverse-data) rosters & stats  
**Base Data**: `nfl_complete_nflverse.json` (3,160 games)  
**Enriched Data**: `nfl_enriched_with_rosters.json` (3.6 MB)

---

## What Was Added

### 1. **Quarterback Information** (95%+ coverage)
For each game:
- **Home QB**: Name, player ID, jersey number, status
- **Away QB**: Name, player ID, jersey number, status

Example:
```json
"home_qb": {
  "qb_name": "Patrick Mahomes",
  "qb_id": "00-0033873",
  "qb_jersey": 15,
  "qb_status": "Active"
}
```

### 2. **Key Players** (Top 5 per team)
Priority positions for narrative analysis:
- **QB** - Quarterback
- **RB** - Running back
- **WR** - Wide receiver
- **TE** - Tight end
- **DE** - Defensive end
- **LB** - Linebacker
- **CB** - Cornerback

Example:
```json
"home_key_players": [
  {"name": "Patrick Mahomes", "position": "QB", "jersey": 15},
  {"name": "Travis Kelce", "position": "TE", "jersey": 87},
  {"name": "Chris Jones", "position": "DE", "jersey": 95}
]
```

### 3. **Team Records** (100% coverage)
Win-loss records **before** each game:
- Shows team performance entering the matchup
- Enables underdog/favorite narrative analysis
- Tracks season progression

Example:
```json
"home_record_before": "8-3",
"away_record_before": "5-6"
```

### 4. **Head-to-Head Matchup History**
Historical context between teams:
- Total previous matchups
- Win-loss breakdown
- Last meeting season
- Previous winner

Example:
```json
"matchup_history": {
  "total_games": 52,
  "home_wins": 28,
  "away_wins": 24,
  "last_meeting_season": 2023,
  "last_winner": "KC"
}
```

---

## Coverage Statistics

| Data Type | Games | Coverage |
|-----------|-------|----------|
| **Total Games** | 3,160 | 100% |
| **QB Data** | 3,010+ | 95%+ |
| **Key Players** | 3,010+ | 95%+ |
| **Team Records** | 3,160 | 100% |
| **Matchup History** | 2,800+ | 89% |

**Note**: Some early season 2014 games may have incomplete roster data due to data availability.

---

## Narrative Analysis Value

### QB Narratives
- **Star QB matchups**: Mahomes vs. Allen, Burrow vs. Jackson
- **Backup QB storylines**: Injury replacements, breakout performances
- **QB prestige**: Name recognition and career accomplishments
- **Nominative analysis**: QB name semantic fields

### Player Narratives
- **Star power**: Pro Bowl and All-Pro players
- **Position battles**: RB1 vs. RB2 dynamics
- **Defensive stars**: Pass rushers, shutdown corners
- **Chemistry**: QB-WR connections (Mahomes-Kelce, etc.)

### Team Context Narratives
- **Underdog stories**: Weak records upsetting strong teams
- **Momentum**: Win/loss streaks
- **Playoff implications**: Must-win scenarios
- **Season arcs**: Early season vs. late season performance

### Rivalry Narratives
- **Historic matchups**: Cowboys-Eagles, Steelers-Ravens
- **Division battles**: 2x per year intensity
- **Playoff rematches**: Previous postseason meetings
- **Coaching matchups**: Reid vs. Belichick history

---

## Use Cases

### 1. Enhanced Feature Extraction
```python
# Extract QB prestige features
qb_features = extract_qb_prestige(game['home_qb'], game['away_qb'])

# Calculate star player differential
star_diff = calculate_star_power(game['home_key_players'], game['away_key_players'])

# Rivalry intensity from matchup history
rivalry_score = calculate_rivalry(game['matchup_history'])
```

### 2. Nominative Analysis
```python
# QB name semantic analysis
home_qb_sem = analyze_name_semantics(game['home_qb']['qb_name'])
away_qb_sem = analyze_name_semantics(game['away_qb']['qb_name'])

# Name field fit analysis
qb_fit = calculate_nominative_fit(qb_name, position='QB')
```

### 3. Contextual Betting
```python
# Underdog with star QB
if game['home_record_before'] < game['away_record_before']:
    if is_elite_qb(game['home_qb']['qb_name']):
        # Potential upset scenario
        betting_edge += 0.15
```

### 4. Narrative Quality Scoring
```python
# Calculate story quality
story_score = (
    qb_star_power * 0.3 +
    record_differential * 0.2 +
    rivalry_history * 0.2 +
    playoff_implications * 0.3
)
```

---

## Data Structure

### Complete Game Object
```json
{
  "game_id": "2024_11_KC_BUF",
  "season": 2024,
  "week": 11,
  "home_team": "BUF",
  "away_team": "KC",
  
  "home_qb": {
    "qb_name": "Josh Allen",
    "qb_id": "00-0035228",
    "qb_jersey": 17,
    "qb_status": "Active"
  },
  "away_qb": {
    "qb_name": "Patrick Mahomes",
    "qb_id": "00-0033873",
    "qb_jersey": 15,
    "qb_status": "Active"
  },
  
  "home_key_players": [
    {"name": "Josh Allen", "position": "QB", "jersey": 17},
    {"name": "James Cook", "position": "RB", "jersey": 4},
    {"name": "Stefon Diggs", "position": "WR", "jersey": 14}
  ],
  
  "home_record_before": "8-2",
  "away_record_before": "9-1",
  
  "matchup_history": {
    "total_games": 6,
    "home_wins": 2,
    "away_wins": 4,
    "last_meeting_season": 2023,
    "last_winner": "KC"
  },
  
  "home_score": 27,
  "away_score": 30,
  "home_won": false,
  "spread_line": -2.5,
  "playoff": false
}
```

---

## Next Steps

### 1. Extract Narrative Features
Apply transformers to enriched data:
```bash
cd narrative_optimization/domains/nfl
python3 extract_enriched_features.py
```

### 2. QB Nominative Analysis
Analyze QB name semantics:
```bash
python3 analyze_qb_nominatives.py
```

### 3. Rivalry Detection
Identify key rivalries:
```bash
python3 detect_rivalries.py
```

### 4. Update Production Model
Retrain with enriched features:
```bash
python3 train_with_enriched_data.py
```

---

## Comparison

**Basic Data** (`nfl_complete_nflverse.json` - 2.3 MB):
- Game scores and outcomes
- Basic metadata (stadium, weather)
- Betting lines

**Enriched Data** (`nfl_enriched_with_rosters.json` - 3.6 MB):
- **+ QB names and IDs** → Nominative analysis
- **+ Key players** → Star power differential
- **+ Team records** → Underdog narratives
- **+ Matchup history** → Rivalry intensity
- **= Complete narrative context**

---

## Data Quality

### Strengths
✅ **95%+ QB coverage** - Nearly complete QB data  
✅ **100% team records** - Full season context  
✅ **Rich player rosters** - Key position players  
✅ **Historic matchups** - Head-to-head context  
✅ **Automated updates** - Script ready for new data

### Known Gaps
⚠️ **Early 2014 data** - Some roster gaps  
⚠️ **Practice squad changes** - Mid-week roster moves not captured  
⚠️ **Injury data** - Player status incomplete for older games

---

## File Locations

**Enriched Data**: `data/domains/nfl_enriched_with_rosters.json`  
**Enrichment Script**: `scripts/enrich_nfl_with_rosters_matchups.py`  
**Base Data**: `data/domains/nfl_complete_nflverse.json`

---

## Update Process

To refresh with latest rosters:
```bash
python3 scripts/update_nfl_data_from_nflverse.py      # Update base data
python3 scripts/enrich_nfl_with_rosters_matchups.py   # Add rosters
```

**Recommended**: Run after each week during season

---

**Created**: November 16, 2025  
**Status**: ✅ Complete  
**Coverage**: 3,160 games, 12 seasons (2014-2025)  
**Size**: 3.6 MB (enriched) vs 2.3 MB (base)

