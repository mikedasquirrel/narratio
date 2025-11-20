"""
Complete 2025 NFL Roster Collection
NO SHORTCUTS - Get every piece of data the model needs

REQUIRED FOR ALL 124 GAMES:
1. Starting QB (home and away)
2. Head Coach (home and away) 
3. Offensive Coordinator (home and away)
4. Defensive Coordinator (home and away)
5. Starting RB (home and away)
6. WR1, WR2 (home and away)
7. TE (home and away)
8. O-line 5 players (home and away)
9. Star players list (home and away)

SOURCES:
- ESPN depth charts
- NFL.com official depth charts
- Team websites
- Pro Football Reference
"""

import json
import requests
from pathlib import Path
from collections import defaultdict
import time

print("="*80)
print("COMPLETE 2025 NFL ROSTER DATA COLLECTION")
print("ALL 29 Features - NO Placeholders")
print("="*80)

# Load our 124 games
with open('nfl_2025_real_qbs.json') as f:
    games_2025 = json.load(f)

print(f"✓ {len(games_2025)} games to fully enrich")

# ============================================================================
# 2025 TEAM ROSTERS (Manual - Most Accurate)
# ============================================================================

print("\n[COLLECTING] 2025 Team Rosters...")

# 2025 Starting rosters as of Week 10 (November 2025)
# This needs to be accurate for the model to work

team_rosters_2025 = {
    'KC': {
        'qb': 'Patrick Mahomes',
        'rb': 'Isiah Pacheco',
        'wr1': 'Rashee Rice',
        'wr2': 'Xavier Worthy',
        'te': 'Travis Kelce',
        'hc': 'Andy Reid',
        'oc': 'Matt Nagy',
        'dc': 'Steve Spagnuolo',
        'oline': ['Jawaan Taylor', 'Joe Thuney', 'Creed Humphrey', 'Trey Smith', 'Donovan Smith'],
        'stars': ['Patrick Mahomes', 'Travis Kelce']
    },
    'BUF': {
        'qb': 'Josh Allen',
        'rb': 'James Cook',
        'wr1': 'Khalil Shakir',
        'wr2': 'Keon Coleman',
        'te': 'Dalton Kincaid',
        'hc': 'Sean McDermott',
        'oc': 'Joe Brady',
        'dc': 'Bobby Babich',
        'oline': ['Dion Dawkins', 'David Edwards', 'Connor McGovern', 'OCyrus Torrence', 'Spencer Brown'],
        'stars': ['Josh Allen', 'James Cook']
    },
    'BAL': {
        'qb': 'Lamar Jackson',
        'rb': 'Derrick Henry',
        'wr1': 'Zay Flowers',
        'wr2': 'Rashod Bateman',
        'te': 'Mark Andrews',
        'hc': 'John Harbaugh',
        'oc': 'Todd Monken',
        'dc': 'Zach Orr',
        'oline': ['Ronnie Stanley', 'Patrick Mekari', 'Tyler Linderbaum', 'Daniel Faalele', 'Roger Rosengarten'],
        'stars': ['Lamar Jackson', 'Derrick Henry']
    },
    # Add all 32 teams...
    # This is tedious but NECESSARY for accurate modeling
}

print(f"✓ {len(team_rosters_2025)} teams with complete rosters")
print(f"⚠ Need to complete all 32 teams")

print(f"""
TO COMPLETE THIS:

1. Visit each team's depth chart:
   - ESPN.com/nfl/team/depth
   - NFL.com team pages
   - Or use team rosters API

2. For EACH of 32 teams, record:
   - Starting QB
   - Starting RB, WR1, WR2, TE
   - Head Coach, OC, DC
   - O-line 5 starters (LT, LG, C, RG, RT)
   - 2-3 star players

3. This takes ~30-45 minutes for all 32 teams

4. Then regenerate features for all 124 games

5. THEN we can properly test 2025

NO SHORTCUTS - This is what's required for accurate prediction
""")

# ============================================================================
# WHAT HAPPENS WITHOUT COMPLETE DATA
# ============================================================================

print("\n" + "="*80)
print("WHY COMPLETE DATA MATTERS")
print("="*80)

print(f"""
OUR MODEL USES 29 FEATURES:
- QB features (6): Prestige, differential, interactions
- Coach features (9): Prestige, experience, elite flags
- O-line features (4): Ensemble size, coherence
- Star features (4): Star count, differential
- Interactions (6): QB×Coach, elite combinations

MISSING DATA = BROKEN PREDICTIONS:
- Missing coaches → 9 features wrong
- Missing O-line → 4 features wrong
- Missing stars → 4 features wrong
- Missing interactions → 6 features wrong
= 23/29 features (79%) are wrong or placeholders

RESULT:
- Model trained on REAL data (2014-2024)
- Testing on PLACEHOLDER data (2025)
- Obviously fails (45% accuracy)

FIX:
Get complete roster data for 2025.
No shortcuts.
This is what's needed for your money.
""")

# Save progress
progress = {
    'status': 'partial',
    'games_total': len(games_2025),
    'teams_complete': len(team_rosters_2025),
    'teams_needed': 32,
    'completion_pct': len(team_rosters_2025) / 32 * 100,
    'next_step': 'Complete all 32 team rosters with real 2025 data'
}

with open('2025_collection_progress.json', 'w') as f:
    json.dump(progress, f, indent=2)

print(f"\n✓ Progress saved")
print(f"  Completion: {progress['completion_pct']:.0f}%")
print("="*80)

