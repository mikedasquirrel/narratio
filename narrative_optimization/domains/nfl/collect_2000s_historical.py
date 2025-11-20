"""
NFL Historical Data Collection: 2000-2013
Expand database from 11 seasons to 25 seasons

APPROACH:
1. Check for existing CSV/JSON datasets (nflverse, Kaggle, etc.)
2. Process and standardize to our format
3. Generate nominative features (QB, Coach names)
4. Merge with existing 2014-2024 data

KNOWN SOURCES:
- nflverse R package (has 1999-present)
- nflfastR data (play-by-play historical)
- Pro Football Reference (via sportsipy or manual)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("NFL HISTORICAL DATA COLLECTION (2000-2013)")
print("="*80)

# ============================================================================
# CHECK FOR EXISTING DATASETS
# ============================================================================

print("\n[1/5] Checking for existing historical datasets...")

# Common locations to check
potential_sources = [
    Path(__file__).parent.parent.parent.parent / 'data' / 'nfl_historical',
    Path(__file__).parent / 'historical_data',
    Path.home() / 'Downloads' / 'nfl_data'
]

found_sources = []
for source in potential_sources:
    if source.exists():
        csv_files = list(source.glob('*.csv'))
        json_files = list(source.glob('*.json'))
        if csv_files or json_files:
            found_sources.append(source)
            print(f"  ✓ Found data in: {source}")
            print(f"    CSV: {len(csv_files)}, JSON: {len(json_files)}")

if not found_sources:
    print(f"  ⚠ No existing datasets found in common locations")
    print(f"\n  MANUAL COLLECTION NEEDED:")
    print(f"  1. Download nflverse data: https://github.com/nflverse/nflverse-data")
    print(f"  2. Or use nflfastR R package")
    print(f"  3. Or scrape Pro Football Reference")
    print(f"\n  Place CSV/JSON files in:")
    print(f"    {Path(__file__).parent / 'historical_data'}")
    print(f"\n  Required columns:")
    print(f"    - game_id, season, week, gameday")
    print(f"    - home_team, away_team")
    print(f"    - home_score, away_score")
    print(f"    - home_qb_name, away_qb_name (if available)")
    print(f"    - spread (if available)")

# ============================================================================
# GENERATE PLACEHOLDER STRUCTURE
# ============================================================================

print("\n[2/5] Generating placeholder structure for 2000-2013...")

# NFL teams and their historical head coaches (2000-2013)
historical_coaches = {
    # This would need to be filled with actual data
    # Showing structure
    2000: {
        'NE': 'Bill Belichick',
        'GB': 'Mike Sherman',
        'TB': 'Tony Dungy',
        'PIT': 'Bill Cowher',
        # ... would need all 32 teams
    },
    # ... would need all years 2000-2013
}

print(f"✓ Historical coach data structure defined")
print(f"  Note: Needs actual data filled in")

# ============================================================================
# PROCESSING PIPELINE
# ============================================================================

print("\n[3/5] Data processing pipeline...")

def process_historical_game(raw_data):
    """Convert raw historical data to our format"""
    
    # Extract basic info
    game = {
        'game_id': raw_data.get('game_id', f"{raw_data['season']}_{raw_data['week']}_{raw_data['away_team']}_{raw_data['home_team']}"),
        'season': int(raw_data['season']),
        'week': int(raw_data['week']),
        'gameday': raw_data.get('gameday', ''),
        'gametime': raw_data.get('gametime', '13:00'),
        'home_team': raw_data['home_team'],
        'away_team': raw_data['away_team'],
        'home_score': int(raw_data.get('home_score', 0)),
        'away_score': int(raw_data.get('away_score', 0)),
        'home_won': raw_data.get('home_score', 0) > raw_data.get('away_score', 0),
    }
    
    # Add rosters (if available, else placeholder)
    game['home_roster'] = {
        'starting_qb': {'name': raw_data.get('home_qb', f"{game['home_team']} QB"), 'position': 'QB'},
        'starting_rb': {'name': raw_data.get('home_rb', f"{game['home_team']} RB"), 'position': 'RB'},
        'starting_wr1': {'name': raw_data.get('home_wr1', f"{game['home_team']} WR1"), 'position': 'WR'},
        'starting_wr2': {'name': raw_data.get('home_wr2', f"{game['home_team']} WR2"), 'position': 'WR'},
        'starting_te': {'name': raw_data.get('home_te', f"{game['home_team']} TE"), 'position': 'TE'},
    }
    
    game['away_roster'] = {
        'starting_qb': {'name': raw_data.get('away_qb', f"{game['away_team']} QB"), 'position': 'QB'},
        'starting_rb': {'name': raw_data.get('away_rb', f"{game['away_team']} RB"), 'position': 'RB'},
        'starting_wr1': {'name': raw_data.get('away_wr1', f"{game['away_team']} WR1"), 'position': 'WR'},
        'starting_wr2': {'name': raw_data.get('away_wr2', f"{game['away_team']} WR2"), 'position': 'WR'},
        'starting_te': {'name': raw_data.get('away_te', f"{game['away_team']} TE"), 'position': 'TE'},
    }
    
    # Add coaches
    season = game['season']
    home_team = game['home_team']
    away_team = game['away_team']
    
    game['home_coaches'] = {
        'head_coach': historical_coaches.get(season, {}).get(home_team, f"{home_team} Head Coach {season}"),
        'offensive_coordinator': f"{home_team} OC {season}",
        'defensive_coordinator': f"{home_team} DC {season}"
    }
    
    game['away_coaches'] = {
        'head_coach': historical_coaches.get(season, {}).get(away_team, f"{away_team} Head Coach {season}"),
        'offensive_coordinator': f"{away_team} OC {season}",
        'defensive_coordinator': f"{away_team} DC {season}"
    }
    
    # Add betting odds (if available)
    game['betting_odds'] = {
        'spread': raw_data.get('spread', 0.0),
        'moneyline_home': raw_data.get('moneyline_home', None),
        'moneyline_away': raw_data.get('moneyline_away', None),
        'over_under': raw_data.get('over_under', 0.0),
    }
    
    # Add context
    game['context'] = {
        'is_division_game': raw_data.get('is_division', False),
        'is_playoff': raw_data.get('is_playoff', False),
    }
    
    # Add ensemble placeholder
    game['home_ensemble'] = {}
    game['away_ensemble'] = {}
    
    return game

print(f"✓ Processing pipeline defined")

# ============================================================================
# WHAT WE NEED FROM USER/EXTERNAL
# ============================================================================

print("\n[4/5] What we need to complete expansion...")

print(f"""
TO EXPAND DATABASE, NEED ONE OF:

A. EXISTING DATASET:
   Place CSV/JSON file in: {Path(__file__).parent / 'historical_data'}
   With columns: season, week, home_team, away_team, home_score, away_score
   Optional: home_qb, away_qb, spread, head_coach_home, head_coach_away
   
   Then run:
   ```python
   python3 process_historical_dataset.py --input historical_data/nfl_2000_2013.csv
   ```

B. API ACCESS:
   Install nflfastR (R) or sportsipy (Python):
   ```bash
   pip install sportsipy
   ```
   
   Then run collector with API

C. MANUAL ENTRY:
   Key games only (can focus on playoffs, important matchups)
   ~50-100 key games per season for pattern validation
   
   Total: ~700 games instead of ~3,700

MINIMAL VIABLE:
Just need QB names + Coach names + outcomes for 2000-2013.
Betting odds optional (can validate without them initially).
With ~700 key games, can test if patterns hold across full 2000s.
""")

# ============================================================================
# ESTIMATE EXPANSION VALUE
# ============================================================================

print("\n[5/5] Expansion value estimation...")

print(f"""
CURRENT:
• Training data: 2014-2019 (6 seasons, ~1,600 games)
• Test data: 2020-2024 (5 seasons, ~1,400 games)
• Model accuracy: 66.1%
• Profitable patterns: 56

WITH EXPANSION (2000-2024):
• Training data: 2000-2019 (20 seasons, ~5,300 games)
• Test data: 2020-2024 (5 seasons, ~1,400 games)
• Expected improvement: +3-5% accuracy (more training data)
• Pattern stability: Can test across 20+ years (not just 6)
• Historical validation: See if patterns held in 2000s

VALUE:
• More robust model (3x more training data)
• Better pattern validation (20-year consistency check)
• Historical QB/Coach analysis (Brady early career, Manning, etc.)
• Understand if patterns are time-invariant or era-specific

EFFORT:
• If dataset available: 2-3 hours to process
• If manual collection: Several days
• If API scraping: 1 day with rate limits

RECOMMENDATION:
Worth doing if dataset available.
Even partial collection (playoffs only) adds value.
""")

print(f"\n✓ Expansion plan complete")
print("="*80)

