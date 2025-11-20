"""
Expand NFL Database to All 2000s Seasons (2000-2024)

Currently: 2014-2024 (11 seasons, 3,010 games)
Target: 2000-2024 (25 seasons, ~7,000 games)

DATA SOURCES:
- Pro Football Reference (has historical data back to 1920s)
- ESPN API
- Sports Reference datasets
- Historical odds archives

WILL COLLECT:
- Game outcomes (scores, winners)
- Betting odds (spread, moneyline, O/U)
- QB names (starters for each game)
- Coach names (head coach, coordinators if available)
- Roster information (key players)
- Game context (week, playoff, division)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time

print("="*80)
print("EXPANDING NFL DATABASE TO 2000-2024")
print("="*80)

# Load existing data
existing_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_games_with_odds.json'
with open(existing_path) as f:
    existing_games = json.load(f)

print(f"\nCurrent database:")
print(f"  Games: {len(existing_games)}")
seasons_existing = sorted(set(g.get('season', 0) for g in existing_games))
print(f"  Seasons: {min(seasons_existing)}-{max(seasons_existing)} ({len(seasons_existing)} seasons)")

# ============================================================================
# PHASE 1: COLLECT 2000-2013 SEASONS
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: Collecting 2000-2013 Data")
print("="*80)

missing_seasons = list(range(2000, 2014))
print(f"\nMissing seasons: {missing_seasons}")
print(f"  Need: ~{len(missing_seasons) * 267} games (267 per season avg)")

print("""
DATA COLLECTION APPROACH:

Option 1: Pro Football Reference Scraping
  - Most complete historical data
  - Has scores, basic rosters
  - May have odds for recent years
  - Respectful scraping with delays

Option 2: Sports Reference Python API
  - Clean API access
  - Structured data
  - May require subscription

Option 3: Existing Datasets
  - Kaggle NFL datasets
  - Sports databases
  - Academic datasets

RECOMMENDED: Start with existing datasets (faster), supplement with scraping
""")

# ============================================================================
# DATA STRUCTURE TEMPLATE
# ============================================================================

print("\n[TEMPLATE] Required fields for historical games...")

template_game = {
    'game_id': '2000_01_TB_WAS',
    'season': 2000,
    'week': 1,
    'gameday': '2000-09-03',
    'gametime': '13:00',
    'home_team': 'WAS',
    'away_team': 'TB',
    'home_score': None,  # To collect
    'away_score': None,  # To collect
    'home_won': None,  # To calculate
    'home_roster': {
        'starting_qb': {'name': 'Unknown', 'position': 'QB'},  # To collect
        'starting_rb': {'name': 'Unknown', 'position': 'RB'},
        'starting_wr1': {'name': 'Unknown', 'position': 'WR'},
    },
    'away_roster': {
        'starting_qb': {'name': 'Unknown', 'position': 'QB'},
        'starting_rb': {'name': 'Unknown', 'position': 'RB'},
        'starting_wr1': {'name': 'Unknown', 'position': 'WR'},
    },
    'home_coaches': {
        'head_coach': 'Unknown',  # To collect
        'offensive_coordinator': 'Unknown',
        'defensive_coordinator': 'Unknown'
    },
    'away_coaches': {
        'head_coach': 'Unknown',
        'offensive_coordinator': 'Unknown',
        'defensive_coordinator': 'Unknown'
    },
    'betting_odds': {
        'spread': None,  # To collect if available
        'moneyline_home': None,
        'moneyline_away': None,
        'over_under': None,
    },
    'context': {
        'is_division_game': False,
        'is_playoff': False,
    }
}

print(f"✓ Template structure defined")

# ============================================================================
# COLLECTION PLAN
# ============================================================================

print("\n" + "="*80)
print("DATA COLLECTION PLAN")
print("="*80)

print("""
PRIORITY FIELDS (Must Have):
1. Game outcomes (scores, winner) - CRITICAL
2. QB names (starters) - CRITICAL for our model
3. Coach names (head coach minimum) - CRITICAL for our model
4. Week, date, teams - CRITICAL
5. Betting odds (spread at minimum) - For validation

NICE TO HAVE:
- Full rosters (RB, WR, etc.)
- Coordinators (OC, DC)
- Moneyline, Over/Under odds
- Division game flags
- Weather, venue

COLLECTION SOURCES:
1. Check for existing Kaggle/GitHub NFL datasets (2000-2013)
2. Pro Football Reference for scores/rosters
3. Historical odds archives (harder to find for 2000s)

TIMELINE:
- Existing datasets: 1-2 hours to find and process
- Manual collection: Several days
- With assistance: Can be parallelized

RECOMMENDATION:
Start with what's available in structured datasets.
Can add odds later - scores and QB/Coach names are priority.
""")

print("\n[NEXT STEPS]")
print("1. Search for 'nfl historical data 2000-2013' datasets")
print("2. Process and standardize to our format")
print("3. Merge with existing 2014-2024 data")
print("4. Regenerate nominative features for full dataset")
print("5. Retrain model on expanded data (2000-2019 train, 2020-2024 test)")

print("\n✓ Collection plan ready")
print("="*80)

# Save collection plan
plan = {
    'target_seasons': list(range(2000, 2014)),
    'games_needed': len(missing_seasons) * 267,
    'current_coverage': f"{min(seasons_existing)}-{max(seasons_existing)}",
    'expansion_coverage': "2000-2024",
    'priority_fields': ['scores', 'QB_names', 'coach_names', 'week', 'date'],
    'optional_fields': ['spread', 'rosters', 'coordinators', 'odds'],
    'collection_methods': [
        'Search existing datasets (Kaggle, GitHub)',
        'Pro Football Reference API/scraping',
        'Sports database subscriptions'
    ]
}

with open('nfl_expansion_plan.json', 'w') as f:
    json.dump(plan, f, indent=2)

print(f"✓ Expansion plan saved to: nfl_expansion_plan.json")

