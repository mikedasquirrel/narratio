"""
NBA Rich Narrative Generator
Creates detailed narratives from team-level data into proper matchups

CONVERTS:
- Team-perspective records (LAL won, BOS lost)
INTO:
- Matchup-level narratives (LAL vs BOS game)

WITH RICH CONTEXT:
- Team identities, records, momentum, key players, stakes
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("NBA RICH NARRATIVE GENERATOR")
print("="*80)

# Load data
print("\n[1/4] Loading NBA data...")
data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_with_temporal_context.json'

with open(data_path) as f:
    team_games = json.load(f)

print(f"✓ Loaded {len(team_games):,} team-game records")

# ============================================================================
# BUILD MATCHUPS
# ============================================================================

print("\n[2/4] Building matchups from team records...")

matchups = defaultdict(lambda: {'home': None, 'away': None})

for game in team_games:
    game_id = game.get('game_id')
    if not game_id:
        continue
    
    is_home = game.get('home_game', False)
    
    if is_home:
        matchups[game_id]['home'] = game
    else:
        matchups[game_id]['away'] = game

# Filter complete
complete_matchups = [(gid, m) for gid, m in matchups.items() if m['home'] and m['away']]

print(f"✓ Built {len(complete_matchups):,} complete matchups")

# ============================================================================
# GENERATE RICH NARRATIVES
# ============================================================================

print("\n[3/4] Generating rich narratives...")

def get_team_identity(team_abbr):
    """NBA team identities"""
    identities = {
        'LAL': "The Los Angeles Lakers, legendary franchise with championship pedigree and Hollywood prestige",
        'BOS': "The Boston Celtics embody tradition, Celtic pride, and historic excellence",
        'GSW': "The Golden State Warriors revolutionized basketball with their championship dynasty",
        'MIA': "The Miami Heat bring South Beach swagger and big-game experience",
        'CHI': "The Chicago Bulls carry Michael Jordan's legacy and championship expectations",
        'SAS': "The San Antonio Spurs exemplify fundamental excellence and sustained success",
        'NYK': "The New York Knicks represent Madison Square Garden mystique and big-market pressure",
        'BRK': "The Brooklyn Nets seek to establish themselves in the nation's biggest market",
        'PHI': "The Philadelphia 76ers rebuild toward the Process and championship dreams",
        'MIL': "The Milwaukee Bucks transformed from overlooked to championship contenders",
        'TOR': "The Toronto Raptors represent basketball's global expansion and championship potential",
        'DEN': "The Denver Broncos leverage altitude advantage and emerging young talent",
        'PHX': "The Phoenix Suns combine fast-paced offense with championship aspirations",
        'DAL': "The Dallas Mavericks pair international flair with championship experience",
        'CLE': "The Cleveland Cavaliers chase redemption and LeBron's legacy",
    }
    return identities.get(team_abbr, f"The {team_abbr}")

def generate_matchup_narrative(matchup_data):
    """Generate rich narrative for matchup"""
    
    home_game = matchup_data['home']
    away_game = matchup_data['away']
    
    home_team = home_game['team_name']
    away_team = away_game['team_name']
    
    parts = []
    
    # 1. MATCHUP INTRODUCTION
    parts.append(f"{home_team} host {away_team}.")
    
    # 2. TEAM IDENTITIES
    home_identity = get_team_identity(home_game['team_abbreviation'])
    away_identity = get_team_identity(away_game['team_abbreviation'])
    
    if "legendary" in home_identity or "championship" in home_identity:
        parts.append(home_identity + ".")
    if "legendary" in away_identity or "championship" in away_identity:
        parts.append(away_identity + ".")
    
    # 3. RECORDS & MOMENTUM
    home_tc = home_game.get('temporal_context', {})
    away_tc = away_game.get('temporal_context', {})
    
    home_record = home_tc.get('season_record', 'unknown')
    away_record = away_tc.get('season_record', 'unknown')
    
    if home_record != 'unknown':
        parts.append(f"{home_team} enter {home_record}")
        
    if away_record != 'unknown':
        if home_record != 'unknown':
            parts[-1] = parts[-1] + f" while {away_team} are {away_record}."
        else:
            parts.append(f"{away_team} are {away_record}.")
    
    # 4. RECENT FORM
    home_l10 = home_tc.get('l10_wins', 0)
    away_l10 = away_tc.get('l10_wins', 0)
    
    if home_l10 >= 7:
        parts.append(f"{home_team} ride momentum with {home_l10} wins in their last 10 games.")
    elif home_l10 <= 3:
        parts.append(f"{home_team} struggle with only {home_l10} wins in their last 10 games.")
    
    if away_l10 >= 7:
        parts.append(f"{away_team} bring heat with {away_l10} wins in their last 10 games.")
    elif away_l10 <= 3:
        parts.append(f"{away_team} battle adversity with {away_l10} wins in their last 10.")
    
    # 5. SEASON CONTEXT
    games_played = home_tc.get('games_played', 0)
    
    if games_played <= 20:
        parts.append("Early season basketball establishes identity and rhythm.")
    elif games_played >= 60:
        parts.append("Late-season intensity builds as playoff positions crystallize.")
    else:
        parts.append("Mid-season battles shape playoff positioning.")
    
    # 6. STAKES
    home_win_pct = home_tc.get('season_win_pct', 0.5)
    away_win_pct = away_tc.get('season_win_pct', 0.5)
    
    if games_played >= 70:
        if home_win_pct > 0.6 and away_win_pct > 0.6:
            parts.append("Both teams fight for playoff seeding advantage.")
        elif home_win_pct < 0.4 and away_win_pct < 0.4:
            parts.append("Draft positioning implications loom for both squads.")
    
    # 7. HOME COURT
    parts.append(f"{home_team} look to leverage home court advantage.")
    
    # 8. EXPECTATION
    if abs(home_win_pct - away_win_pct) > 0.20:
        if home_win_pct > away_win_pct:
            parts.append(f"{home_team} are heavily favored given the record disparity.")
        else:
            parts.append(f"{away_team} bring superior record into hostile territory.")
    else:
        parts.append("This figures to be a closely contested matchup between comparable teams.")
    
    return " ".join(parts)

enriched_matchups = []

for i, (game_id, matchup) in enumerate(complete_matchups):
    if i % 1000 == 0:
        print(f"  Generated {i:,}/{len(complete_matchups):,} narratives...")
    
    narrative = generate_matchup_narrative(matchup)
    
    enriched_matchups.append({
        'game_id': game_id,
        'season': matchup['home']['season'],
        'date': matchup['home']['date'],
        'home_team': matchup['home']['team_name'],
        'away_team': matchup['away']['team_name'],
        'home_won': matchup['home']['won'],
        'home_score': matchup['home']['points'],
        'away_score': matchup['away']['points'],
        'rich_narrative': narrative,
        'home_temporal_context': matchup['home'].get('temporal_context', {}),
        'away_temporal_context': matchup['away'].get('temporal_context', {}),
    })

print(f"✓ Generated {len(enriched_matchups):,} rich matchup narratives")

# ============================================================================
# SAVE
# ============================================================================

print("\n[4/4] Saving enriched matchups...")

output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_matchups_rich_narratives.json'

with open(output_path, 'w') as f:
    json.dump(enriched_matchups, f, indent=2)

print(f"✓ Saved to: {output_path.name}")
print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Matchups: {len(enriched_matchups):,}")

# Show samples
print(f"\n" + "="*80)
print("SAMPLE NARRATIVES")
print("="*80)

for i in range(min(3, len(enriched_matchups))):
    game = enriched_matchups[i]
    print(f"\nGame {i+1}: {game['home_team']} vs {game['away_team']}")
    print(f"Date: {game['date']}")
    print(f"Narrative: {game['rich_narrative'][:250]}...")

print("\n" + "="*80)
print("✓ COMPLETE - Ready for narrative betting validation")
print("="*80)
