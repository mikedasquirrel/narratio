# MLB Narrative Data Enhancement - Bite-Sized Implementation

**Goal**: Add ALL narratively impactful information about games  
**Status**: In Progress  
**Approach**: Bite-sized incremental chunks

---

## ✅ Bite 1: Score Differential & Game Story Basics (COMPLETE)

**What Added**:
```python
'outcome': {
    'winner': 'home',
    'home_score': 7,
    'away_score': 3,
    'score_differential': 4,  # Home perspective
    'run_differential': 4,
    'total_runs': 10,
    'close_game': False,  # <= 2 runs
    'blowout': False,  # >= 5 runs
    'shutout': False,  # Shutout win
    'high_scoring': False,  # >= 12 total runs
    'low_scoring': False  # <= 4 total runs
}
```

**Narrative Value**:
- Close games = higher drama
- Blowouts = dominance narrative
- Shutouts = pitching mastery
- Score patterns inform game story

**Files Modified**:
- `collect_mlb_data.py` - Added score differential flags

---

## 🔨 Bite 2: Inning-by-Inning Scoring (IN PROGRESS)

**What to Add**:
```python
'game_story': {
    'inning_by_inning': {
        '1': {'home': 0, 'away': 2},  # Away team scores 2 in 1st
        '2': {'home': 0, 'away': 0},
        '3': {'home': 3, 'away': 0},  # Home team answers with 3
        '4': {'home': 1, 'away': 0},
        '5': {'home': 0, 'away': 1},
        '6': {'home': 2, 'away': 0},  # Home team pulls ahead
        '7': {'home': 1, 'away': 0},
        '8': {'home': 0, 'away': 0},
        '9': {'home': 0, 'away': 0}
    },
    'scoring_pattern': 'comeback',  # away_early_lead, home_dominant, etc.
    'lead_changes': 2,
    'largest_lead': 4,
    'comeback_win': True
}
```

**Narrative Value**:
- Early leads = momentum
- Comebacks = drama/resilience
- Lead changes = back-and-forth excitement
- Dominant start vs finish strong

**MLB Stats API**:
- `linescore.innings` - inning-by-inning runs
- Available in schedule hydration

---

## 📋 Bite 3: Key Moments (PLANNED)

**What to Add**:
```python
'key_moments': {
    'home_runs': [
        {'player': 'Aaron Judge', 'inning': 3, 'runners': 2, 'score_before': '0-2'},
        {'player': 'Mookie Betts', 'inning': 6, 'runners': 1, 'score_before': '4-3'}
    ],
    'big_innings': [
        {'inning': 3, 'runs': 3, 'team': 'home', 'changed_lead': True}
    ],
    'pitching_milestones': {
        'strikeouts_10plus': True,
        'no_hitter_through': 7,
        'perfect_game': False
    },
    'dramatic_moments': {
        'walk_off': False,
        'extra_innings': False,
        'blown_save': False,
        'come_from_behind': True
    }
}
```

**Narrative Value**:
- Home runs = power/excitement
- Big innings = momentum shifts
- Walk-offs = maximum drama
- Blown saves = narrative intensity

**MLB Stats API**:
- `game/{gamePk}/feed/live` - play-by-play data
- `playEvents` - individual plays

---

## 📋 Bite 4: Real Betting Odds (PLANNED)

**What to Add**:
```python
'betting_odds': {
    'home_moneyline': -150,  # REAL market odds
    'away_moneyline': +130,
    'over_under': 9.5,
    'home_runline': -1.5,  # MLB spread
    'home_runline_odds': +120,
    'home_implied_prob': 0.60,
    'away_implied_prob': 0.43,
    'vig': 0.03,
    'opening_line': -145,  # Line movement
    'closing_line': -150,
    'line_movement': -5,  # Money moved toward home
    'sharp_money': 'home'  # Where sharp bettors bet
}
```

**Narrative Value**:
- Odds encode market's narrative assessment
- Line movement shows public vs sharp money
- Can calculate ROI (narrative vs market)
- Identifies market inefficiencies

**Data Sources**:
1. The Odds API (free tier: 500 requests/month)
2. Odds Portal (web scraping historical)
3. Action Network API (paid)

---

## 📋 Bite 5: Game Context (PLANNED)

**What to Add**:
```python
'game_context': {
    'attendance': 42538,
    'sellout': True,
    'weather': {
        'condition': 'Clear',
        'temp_f': 72,
        'wind_mph': 8,
        'wind_direction': 'Out to RF'
    },
    'time_of_day': 'night',
    'game_time': '7:05 PM ET',
    'day_of_week': 'Saturday',
    'game_number': 1,  # 1 of doubleheader
    'series_game': 2,  # Game 2 of 3-game series
    'series_tied': False
}
```

**Narrative Value**:
- Attendance = crowd energy
- Weather = affects play (wind, cold)
- Night games = primetime/atmosphere
- Series context = revenge narratives

**MLB Stats API**:
- `gameData.weather` - weather data
- `gameData.attendance` - attendance
- `gameData.datetime` - timing

---

## 📋 Bite 6: Enhanced Narratives (PLANNED)

**What to Add**:
Use all new data to generate richer narratives:

**Before** (team-level only):
```
"Gerrit Cole takes the mound for the Yankees against Chris Sale 
and the Red Sox in a marquee pitching matchup..."
```

**After** (with full story):
```
"In front of a sold-out crowd of 47,000 at Yankee Stadium, ace 
Gerrit Cole takes the mound for the Yankees against Red Sox 
veteran Chris Sale in this legendary rivalry. The Yankees enter 
as -150 favorites with Cole seeking his 10th win. Early drama 
as the Red Sox jump ahead 2-0 in the first inning on a Rafael 
Devers home run. But Cole settled in, striking out 12 over 7 
innings. The Yankees stormed back with a 4-run 6th inning, 
led by Aaron Judge's towering 2-run homer, taking a 6-2 lead 
they wouldn't relinquish. Final: Yankees 7, Red Sox 3."
```

**Narrative Elements**:
- Attendance/atmosphere
- Betting odds context
- Scoring progression
- Key moments (home runs, strikeouts)
- Lead changes and drama
- Final outcome

---

## Expected Impact

### Current State
- |r|: 0.0097
- Binary win/loss only
- Team-level data only
- No betting odds

### After All Bites
- |r|: 0.05-0.25 (5-25x improvement)
- Rich game stories
- Individual players
- Real betting odds
- Score differential

**Why This Matters**:
- NFL: Real names = 1,403x improvement
- MLB: Full story + odds + individuals = similar potential
- Transformers can extract from rich narratives
- Market odds provide baseline to beat

---

## Implementation Status

| Bite | Feature | Status | Files |
|------|---------|--------|-------|
| 1 | Score Differential | ✅ DONE | `collect_mlb_data.py` |
| 2 | Inning-by-Inning | 🔨 IN PROGRESS | TBD |
| 3 | Key Moments | 📋 PLANNED | TBD |
| 4 | Real Betting Odds | 📋 PLANNED | TBD |
| 5 | Game Context | 📋 PLANNED | TBD |
| 6 | Enhanced Narratives | 📋 PLANNED | TBD |

---

**Next**: Implement Bite 2 (Inning-by-Inning Scoring)

