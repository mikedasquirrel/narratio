# NFL Database Update Summary
## Data Source: nflverse-data

**Update Date**: November 16, 2025  
**Source**: [https://github.com/nflverse/nflverse-data](https://github.com/nflverse/nflverse-data)  
**Method**: Play-by-play aggregation to game-level data

---

## Dataset Overview

**Total Games**: 3,160  
**Seasons Covered**: 2014-2025 (12 seasons)  
**Data File**: `data/domains/nfl_complete_nflverse.json` (2.3 MB)

### Games by Season

| Season | Games | Type |
|--------|-------|------|
| 2014 | 267 | Complete |
| 2015 | 267 | Complete |
| 2016 | 267 | Complete |
| 2017 | 267 | Complete |
| 2018 | 267 | Complete |
| 2019 | 267 | Complete |
| 2020 | 269 | Complete (COVID-adjusted) |
| 2021 | 285 | Complete (17-game season) |
| 2022 | 284 | Complete |
| 2023 | 285 | Complete |
| 2024 | 285 | Complete |
| 2025 | 150 | In Progress |

---

## Data Fields

Each game includes:

### Game Identification
- `game_id` - Unique identifier
- `season` - Year
- `week` - Week number
- `game_type` - REG or POST
- `gameday` - Date
- `gametime` - Time (when available)

### Teams & Scores
- `home_team`, `away_team` - Team abbreviations
- `home_score`, `away_score` - Final scores
- `home_won` - Boolean outcome
- `winner`, `loser` - Team identifiers
- `result` - Point differential

### Betting Lines
- `spread_line` - Point spread (when available)
- `total_line` - Over/under total (when available)

### Stadium & Weather
- `stadium` - Stadium name
- `location` - City/location
- `roof` - Dome/retractable/open
- `surface` - Field type
- `temp` - Temperature
- `wind` - Wind speed

### Coaches
- `home_coach` - Home team coach
- `away_coach` - Away team coach

### Context
- `div_game` - Division rivalry
- `playoff` - Playoff game flag
- `overtime` - OT flag
- `total_plays` - Play count
- `season_progress` - % through season

---

## Special Game Types

**Playoff Games**: 131 (2014-2024)  
**Overtime Games**: 178  
**Division Games**: (tracked in metadata)

---

## Data Quality

### Strengths
✅ **Complete play-by-play aggregation** - High accuracy  
✅ **Official nflverse data** - Industry standard  
✅ **Rich contextual data** - Stadium, weather, coaches  
✅ **Comprehensive coverage** - 12 seasons, 3,160+ games  
✅ **Current data** - Includes 2025 season in progress

### Limitations
⚠️ **Betting odds incomplete** - Not all games have spread/total data  
⚠️ **2025 season ongoing** - Only 150 games (as of Nov 2025)

---

## Comparison to Previous Data

**Previous Dataset**: `nfl_games_with_odds.json`
- Games: 3,010
- Seasons: 2014-2024

**New Dataset**: `nfl_complete_nflverse.json`
- Games: 3,160 (+150 games)
- Seasons: 2014-2025 (+1 season in progress)
- **Improvement**: +5% more games, more complete field data

---

## Next Steps

### 1. Feature Extraction
Run narrative transformers on the new data:
```bash
cd narrative_optimization/domains/nfl
python3 extract_nfl_features.py
```

### 2. Update Analysis
Recalculate domain formula with expanded dataset:
```bash
python3 analyze_nfl_complete.py
```

### 3. Betting Integration
For games with spread data, integrate with betting validation:
```bash
python3 validate_betting_patterns.py
```

### 4. Production Deployment
Update production model with new data:
```bash
python3 train_production_model.py
```

---

## Data Source Details

**Repository**: [nflverse/nflverse-data](https://github.com/nflverse/nflverse-data)  
**License**: CC-BY-4.0  
**Documentation**: [nflreadr.nflverse.com](https://nflreadr.nflverse.com/)

**Data Format**: Aggregated from play-by-play CSVs  
**Update Frequency**: Automated daily during season  
**Reliability**: High - used by NFL analytics community

---

## Automation

To update the database with latest data:
```bash
python3 scripts/update_nfl_data_from_nflverse.py
```

This script:
1. Downloads latest play-by-play data from nflverse
2. Aggregates plays into game-level data
3. Enriches with contextual information
4. Saves to `data/domains/nfl_complete_nflverse.json`

**Recommended frequency**: Weekly during season, monthly off-season

---

**Updated**: November 16, 2025  
**Status**: ✅ Complete  
**Next Update**: After Week 12, 2025

