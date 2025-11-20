"""
Enrich 2025 Data with QB/Coach Names
Use 2024 rosters as baseline (most carry over season-to-season)
"""

import json
from pathlib import Path
from collections import defaultdict

print("="*80)
print("ENRICHING 2025 SEASON DATA")
print("="*80)

# Load 2025 games
with open('nfl_2025_current.json') as f:
    games_2025 = json.load(f)

print(f"✓ {len(games_2025)} games from 2025")

# Load 2024 games to get QB/Coach rosters
existing_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_games_with_odds.json'
with open(existing_path) as f:
    all_games = json.load(f)

games_2024 = [g for g in all_games if g.get('season') == 2024]

print(f"✓ {len(games_2024)} games from 2024 (for roster reference)")

# Build 2024 rosters by team (most recent)
team_rosters_2024 = {}
team_coaches_2024 = {}

for game in sorted(games_2024, key=lambda x: x.get('gameday', ''), reverse=True):
    home = game['home_team']
    away = game['away_team']
    
    if home not in team_rosters_2024:
        team_rosters_2024[home] = {
            'qb': game['home_roster']['starting_qb']['name'],
            'rb': game['home_roster'].get('starting_rb', {}).get('name', f"{home} RB"),
            'wr1': game['home_roster'].get('starting_wr1', {}).get('name', f"{home} WR1"),
        }
        team_coaches_2024[home] = game['home_coaches']['head_coach']
    
    if away not in team_rosters_2024:
        team_rosters_2024[away] = {
            'qb': game['away_roster']['starting_qb']['name'],
            'rb': game['away_roster'].get('starting_rb', {}).get('name', f"{away} RB"),
            'wr1': game['away_roster'].get('starting_wr1', {}).get('name', f"{away} WR1"),
        }
        team_coaches_2024[away] = game['away_coaches']['head_coach']

print(f"✓ Extracted rosters for {len(team_rosters_2024)} teams")

# Known 2025 changes (major QB/Coach changes)
changes_2025 = {
    'qb': {
        # Add known QB changes here
        # 'ATL': 'Kirk Cousins',  # Example
    },
    'coach': {
        # Add known coach changes here
        # 'DAL': 'New Coach Name',  # Example
    }
}

print(f"  Known 2025 changes: {len(changes_2025['qb'])} QBs, {len(changes_2025['coach'])} coaches")

# Enrich 2025 games
enriched_2025 = []

for game in games_2025:
    home = game['home_team']
    away = game['away_team']
    
    # Get QB (use 2024 as baseline, apply known changes)
    home_qb = changes_2025['qb'].get(home, team_rosters_2024.get(home, {}).get('qb', f"{home} QB 2025"))
    away_qb = changes_2025['qb'].get(away, team_rosters_2024.get(away, {}).get('qb', f"{away} QB 2025"))
    
    # Get Coach
    home_coach = changes_2025['coach'].get(home, team_coaches_2024.get(home, f"{home} Head Coach 2025"))
    away_coach = changes_2025['coach'].get(away, team_coaches_2024.get(away, f"{away} Head Coach 2025"))
    
    # Build enriched game
    enriched_game = {
        **game,
        'game_id': f"{game['season']}_{game['week']:02d}_{game['away_team']}_{game['home_team']}",
        'gametime': '13:00',
        'home_roster': {
            'starting_qb': {'name': home_qb, 'position': 'QB'},
            'starting_rb': {'name': team_rosters_2024.get(home, {}).get('rb', f"{home} RB"), 'position': 'RB'},
            'starting_wr1': {'name': team_rosters_2024.get(home, {}).get('wr1', f"{home} WR1"), 'position': 'WR'},
            'starting_wr2': {'name': f"{home} WR2", 'position': 'WR'},
            'starting_te': {'name': f"{home} TE", 'position': 'TE'},
        },
        'away_roster': {
            'starting_qb': {'name': away_qb, 'position': 'QB'},
            'starting_rb': {'name': team_rosters_2024.get(away, {}).get('rb', f"{away} RB"), 'position': 'RB'},
            'starting_wr1': {'name': team_rosters_2024.get(away, {}).get('wr1', f"{away} WR1"), 'position': 'WR'},
            'starting_wr2': {'name': f"{away} WR2", 'position': 'WR'},
            'starting_te': {'name': f"{away} TE", 'position': 'TE'},
        },
        'home_coaches': {
            'head_coach': home_coach,
            'offensive_coordinator': f"{home} OC 2025",
            'defensive_coordinator': f"{home} DC 2025"
        },
        'away_coaches': {
            'head_coach': away_coach,
            'offensive_coordinator': f"{away} OC 2025",
            'defensive_coordinator': f"{away} DC 2025"
        },
        'betting_odds': {
            'spread': 0.0,  # Would need from odds API
            'moneyline_home': None,
            'moneyline_away': None,
            'over_under': 0.0,
        },
        'home_ensemble': {},
        'away_ensemble': {},
        'context': {
            'is_division_game': False,
            'is_playoff': False,
        }
    }
    
    enriched_2025.append(enriched_game)

print(f"✓ Enriched {len(enriched_2025)} games")

# Save enriched
output_path = Path(__file__).parent / 'nfl_2025_enriched.json'
with open(output_path, 'w') as f:
    json.dump(enriched_2025, f, indent=2)

print(f"✓ Saved to: {output_path.name}")

# Show sample
print(f"\nSample enriched game:")
print(f"  {enriched_2025[0]['away_team']} @ {enriched_2025[0]['home_team']}")
print(f"  QB: {enriched_2025[0]['away_roster']['starting_qb']['name']} vs {enriched_2025[0]['home_roster']['starting_qb']['name']}")
print(f"  Coach: {enriched_2025[0]['away_coaches']['head_coach']} vs {enriched_2025[0]['home_coaches']['head_coach']}")
print(f"  Score: {enriched_2025[0]['away_score']}-{enriched_2025[0]['home_score']}")

print("\n✓ Ready to generate features and test model")
print("="*80)

