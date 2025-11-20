"""
NFL Rich Narrative Generator
Creates detailed narratives for each game using all available context

NARRATIVE ELEMENTS:
- Team identities (nominative)
- Season context (temporal)
- Matchup history (competitive)
- Stakes/implications (championship)
- Momentum patterns (temporal)
- Key players (character)
- Coaching narratives (reputation)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("NFL RICH NARRATIVE GENERATOR")
print("="*80)

# Load data
print("\n[1/3] Loading NFL data...")
data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_games_with_odds.json'

with open(data_path) as f:
    games = json.load(f)

print(f"✓ Loaded {len(games):,} NFL games")

# ============================================================================
# ANALYZE TEAM PATTERNS (for context)
# ============================================================================

print("\n[2/3] Analyzing team patterns...")

team_records = defaultdict(lambda: {'wins': 0, 'losses': 0, 'games': []})
team_streaks = defaultdict(lambda: {'current': 0, 'max': 0})

# Sort by date
games_sorted = sorted(games, key=lambda x: (x['season'], x['week']))

for game in games_sorted:
    home = game['home_team']
    away = game['away_team']
    
    # Track records
    team_records[home]['games'].append(game)
    team_records[away]['games'].append(game)
    
    if game['home_won']:
        team_records[home]['wins'] += 1
        team_records[away]['losses'] += 1
    else:
        team_records[away]['wins'] += 1
        team_records[home]['losses'] += 1

print(f"✓ Analyzed {len(team_records)} teams")

# ============================================================================
# GENERATE RICH NARRATIVES
# ============================================================================

print("\n[3/3] Generating rich narratives...")

def generate_rich_narrative(game, team_history):
    """Generate detailed narrative for a game"""
    
    home = game['home_team']
    away = game['away_team']
    season = game['season']
    week = game['week']
    
    home_rec = team_history[home]
    away_rec = team_history[away]
    
    # Calculate records up to this game
    home_wins = home_rec['wins']
    home_losses = home_rec['losses']
    away_wins = away_rec['wins']
    away_losses = away_rec['losses']
    
    # Narrative components
    narrative_parts = []
    
    # 1. MATCHUP INTRODUCTION (Nominative)
    narrative_parts.append(f"The {home} host the {away} in Week {week} of the {season} season.")
    
    # 2. TEAM IDENTITIES (Nominative + Reputation)
    home_identity = get_team_identity(home)
    away_identity = get_team_identity(away)
    
    if home_identity:
        narrative_parts.append(home_identity)
    if away_identity:
        narrative_parts.append(away_identity)
    
    # 3. RECORDS & STANDINGS (Competitive Context)
    if home_wins + home_losses > 0:
        home_record_str = f"The {home} enter with a {home_wins}-{home_losses} record"
        away_record_str = f"while the {away} are {away_wins}-{away_losses}"
        
        # Add playoff implications if late season
        if week >= 14:
            if home_wins > 8:
                home_record_str += ", battling for playoff position"
            if away_wins > 8:
                away_record_str += ", also in playoff contention"
        
        narrative_parts.append(f"{home_record_str}, {away_record_str}.")
    
    # 4. KEY PLAYERS (Character)
    home_roster = game.get('home_roster', {})
    away_roster = game.get('away_roster', {})
    
    if home_roster and isinstance(home_roster, dict) and home_roster.get('starting_qb'):
        qb_data = home_roster['starting_qb']
        if isinstance(qb_data, dict):
            qb_home = qb_data.get('name', '')
            if qb_home:
                narrative_parts.append(f"{qb_home} leads the {home} offense.")
        elif isinstance(qb_data, str):
            narrative_parts.append(f"{qb_data} leads the {home} offense.")
    
    if away_roster and isinstance(away_roster, dict) and away_roster.get('starting_qb'):
        qb_data = away_roster['starting_qb']
        if isinstance(qb_data, dict):
            qb_away = qb_data.get('name', '')
            if qb_away:
                narrative_parts.append(f"{qb_away} directs the {away} attack.")
        elif isinstance(qb_data, str):
            narrative_parts.append(f"{qb_data} directs the {away} attack.")
    
    # 5. COACHING (Reputation/Prestige)
    home_coaches = game.get('home_coaches', {})
    away_coaches = game.get('away_coaches', {})
    
    if home_coaches and isinstance(home_coaches, dict) and home_coaches.get('head_coach'):
        coach_data = home_coaches['head_coach']
        if isinstance(coach_data, dict):
            coach_home = coach_data.get('name', '')
            if coach_home:
                narrative_parts.append(f"Head coach {coach_home} guides the home squad.")
        elif isinstance(coach_data, str):
            narrative_parts.append(f"Head coach {coach_data} guides the home squad.")
    
    if away_coaches and isinstance(away_coaches, dict) and away_coaches.get('head_coach'):
        coach_data = away_coaches['head_coach']
        if isinstance(coach_data, dict):
            coach_away = coach_data.get('name', '')
            if coach_away:
                narrative_parts.append(f"{coach_away} leads the visiting {away}.")
        elif isinstance(coach_data, str):
            narrative_parts.append(f"{coach_data} leads the visiting {away}.")
    
    # 6. MATCHUP TYPE (Competitive Context + Stakes)
    context = game.get('context', {})
    if context.get('is_division_game'):
        narrative_parts.append("This critical division matchup carries extra weight.")
    if context.get('is_rivalry'):
        narrative_parts.append("The historic rivalry adds intensity to this clash.")
    
    # 7. SEASON CONTEXT (Temporal)
    if week <= 4:
        narrative_parts.append("Early in the season, teams establish their identity.")
    elif 5 <= week <= 13:
        narrative_parts.append("Mid-season momentum becomes crucial.")
    elif week >= 14:
        narrative_parts.append("Late-season intensity peaks as playoffs loom.")
    
    # 8. BETTING LINE (Competitive Context)
    spread = game.get('betting_odds', {}).get('spread', 0)
    if spread != 0:
        if spread < -7:
            narrative_parts.append(f"The {home} are heavy favorites with a {abs(spread):.1f}-point spread.")
        elif spread > 7:
            narrative_parts.append(f"The {away} face a daunting {spread:.1f}-point deficit as road underdogs.")
        elif spread < -3:
            narrative_parts.append(f"The {home} are favored by {abs(spread):.1f} points at home.")
        elif spread > 3:
            narrative_parts.append(f"The {away} are {spread:.1f}-point underdogs on the road.")
        else:
            narrative_parts.append("This is expected to be a closely contested battle.")
    
    # 9. STAKES (if available)
    over_under = game.get('betting_odds', {}).get('over_under', 0)
    if over_under > 0:
        if over_under > 50:
            narrative_parts.append(f"Expect an offensive showcase with a {over_under:.1f} point total.")
        elif over_under < 42:
            narrative_parts.append(f"A defensive struggle looms with the low {over_under:.1f} point total.")
    
    return " ".join(narrative_parts)

def get_team_identity(team_abbr):
    """Get narrative identity for team"""
    identities = {
        'NE': "The New England Patriots, a dynasty of excellence and championship pedigree",
        'GB': "The Green Bay Packers bring legendary tradition and cold-weather toughness",
        'DAL': "America's Team, the Dallas Cowboys carry immense prestige and expectation",
        'PIT': "The Pittsburgh Steelers embody blue-collar grit and defensive dominance",
        'SF': "The San Francisco 49ers represent West Coast innovation and historic success",
        'KC': "The Kansas City Chiefs combine explosive offense with championship ambition",
        'BUF': "The Buffalo Bills have transformed from underdog to contender",
        'TB': "The Tampa Bay Buccaneers seek to maintain their championship standard",
        'LAR': "The Los Angeles Rams bring Hollywood flash and aggressive gameplay",
        'BAL': "The Baltimore Ravens pride themselves on physical, dominant defense",
        'SEA': "The Seattle Seahawks create an intimidating fortress at home",
        'DEN': "The Denver Broncos leverage altitude and quarterback excellence",
        'CLE': "The Cleveland Browns battle to overcome decades of narrative weight",
        'DET': "The Detroit Lions fight to shed underdog expectations",
        'JAX': "The Jacksonville Jaguars seek to establish sustained relevance",
    }
    return identities.get(team_abbr, None)

# Generate for all
enriched_games = []

for i, game in enumerate(games):
    if i % 500 == 0:
        print(f"  Generated {i:,}/{len(games):,} narratives...")
    
    # Build team history up to this point
    team_history_snapshot = {}
    for team, data in team_records.items():
        # Count games before this one
        games_before = [g for g in data['games'] 
                       if (g['season'], g['week']) < (game['season'], game['week'])]
        
        wins_before = sum(1 for g in games_before 
                         if (g['home_team'] == team and g['home_won']) 
                         or (g['away_team'] == team and not g['home_won']))
        
        team_history_snapshot[team] = {
            'wins': wins_before,
            'losses': len(games_before) - wins_before,
            'games': games_before
        }
    
    narrative = generate_rich_narrative(game, team_history_snapshot)
    
    game['rich_narrative'] = narrative
    enriched_games.append(game)

print(f"✓ Generated narratives for all {len(games):,} games")

# ============================================================================
# SAVE ENRICHED DATA
# ============================================================================

output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_games_rich_narratives.json'

with open(output_path, 'w') as f:
    json.dump(enriched_games, f, indent=2)

print(f"\n✓ Saved to: {output_path.name}")
print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# Show samples
print(f"\n" + "="*80)
print("SAMPLE NARRATIVES")
print("="*80)

for i in range(min(3, len(enriched_games))):
    game = enriched_games[i]
    print(f"\nGame {i+1}: {game['home_team']} vs {game['away_team']}")
    print(f"Week {game['week']}, {game['season']}")
    print(f"Narrative: {game['rich_narrative'][:300]}...")

print("\n" + "="*80)
print("✓ COMPLETE - Ready for narrative transformer validation")
print("="*80)

