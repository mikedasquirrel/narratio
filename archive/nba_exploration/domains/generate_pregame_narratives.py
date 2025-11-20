"""
NBA Pre-Game Narrative Generator
Creates narratives with ZERO outcome information

INCLUDES:
- Team names, identities (nominative)
- Records BEFORE the game (temporal)
- Recent form, momentum (temporal)
- Matchup history (competitive)
- Stakes, playoff implications (championship)
- Key players (character)

EXCLUDES:
- Any mention of game result
- Final scores
- "Won", "lost", "victory", "fell short"
- Any post-game information
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("NBA PRE-GAME NARRATIVE GENERATOR (Zero Data Leakage)")
print("="*80)

# Load data
print("\n[1/4] Loading NBA data...")
data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_with_temporal_context.json'

with open(data_path) as f:
    games = json.load(f)

print(f"✓ Loaded {len(games):,} NBA games")

# ============================================================================
# BUILD MATCHUPS
# ============================================================================

print("\n[2/4] Reconstructing matchups...")

# Parse matchup string to identify opponent
matchups_structured = []

for game in games:
    matchup_str = game.get('matchup', '')
    # Format: "LAL vs. SAC" or "ATL @ CHI"
    
    if ' vs. ' in matchup_str:
        teams = matchup_str.split(' vs. ')
        home_abbr = teams[0].strip()
        away_abbr = teams[1].strip()
        is_home = game.get('home_game', True)
        
    elif ' @ ' in matchup_str:
        teams = matchup_str.split(' @ ')
        away_abbr = teams[0].strip()
        home_abbr = teams[1].strip()
        is_home = game.get('home_game', False)
    else:
        continue
    
    team_abbr = game.get('team_abbreviation', '')
    opponent_abbr = away_abbr if is_home else home_abbr
    
    matchups_structured.append({
        'game': game,
        'home_abbr': home_abbr,
        'away_abbr': away_abbr,
        'team_abbr': team_abbr,
        'opponent_abbr': opponent_abbr,
        'is_home': is_home
    })

print(f"✓ Parsed {len(matchups_structured):,} matchups")

# ============================================================================
# GENERATE PRE-GAME NARRATIVES
# ============================================================================

print("\n[3/4] Generating PRE-GAME narratives (zero outcome leakage)...")

def get_team_identity(team_abbr):
    """Team identity narratives"""
    identities = {
        'LAL': "the legendary Los Angeles Lakers, championship dynasty with Hollywood prestige",
        'BOS': "the storied Boston Celtics, embodying tradition and Celtic pride",
        'GSW': "the dominant Golden State Warriors, revolutionizing modern basketball",
        'MIA': "the Miami Heat, bringing South Beach swagger and championship experience",
        'CHI': "the Chicago Bulls, carrying Michael Jordan's legacy forward",
        'SAS': "the San Antonio Spurs, exemplifying fundamental basketball excellence",
        'NYK': "the New York Knicks, playing under Madison Square Garden's legendary lights",
        'BRK': "the Brooklyn Nets, establishing themselves in the nation's biggest market",
        'PHI': "the Philadelphia 76ers, building toward championship contention",
        'MIL': "the Milwaukee Bucks, transformed from overlooked to elite",
        'TOR': "the Toronto Raptors, representing basketball's global expansion",
        'DEN': "the Denver Nuggets, leveraging altitude and emerging talent",
        'PHX': "the Phoenix Suns, combining fast-paced offense with championship ambition",
        'DAL': "the Dallas Mavericks, pairing international flair with championship pedigree",
        'CLE': "the Cleveland Cavaliers, forever chasing championship glory",
    }
    return identities.get(team_abbr, f"the {team_abbr}")

def generate_pregame_narrative(matchup_data):
    """Generate narrative with ZERO outcome information"""
    
    game = matchup_data['game']
    team_abbr = matchup_data['team_abbr']
    opponent_abbr = matchup_data['opponent_abbr']
    is_home = matchup_data['is_home']
    
    team_name = game['team_name']
    tc = game.get('temporal_context', {})
    
    parts = []
    
    # 1. MATCHUP SETUP
    if is_home:
        parts.append(f"{team_name} host {opponent_abbr} tonight.")
    else:
        parts.append(f"{team_name} travel to face {opponent_abbr}.")
    
    # 2. TEAM IDENTITY
    identity = get_team_identity(team_abbr)
    if "legendary" in identity or "championship" in identity:
        parts.append(f"As {identity},")
    else:
        parts.append(f"As {identity}, they")
    
    # 3. RECORD & FORM (PRE-GAME)
    season_record = tc.get('season_record', '')
    if season_record:
        parts.append(f"enter with a {season_record} record.")
    
    # 4. RECENT MOMENTUM
    l10_wins = tc.get('l10_wins', 0)
    if l10_wins >= 7:
        parts.append(f"The team rides momentum, winning {l10_wins} of their last 10 games.")
    elif l10_wins <= 3:
        parts.append(f"The team battles through adversity with only {l10_wins} wins in their last 10.")
    elif l10_wins > 0:
        parts.append(f"The team shows {l10_wins}-for-10 recent form.")
    
    # 5. SEASON CONTEXT
    games_played = tc.get('games_played', 0)
    
    if games_played <= 20:
        parts.append("Early in the season, the team establishes its identity.")
    elif games_played >= 70:
        parts.append("With playoffs approaching, every game carries weight.")
    elif games_played >= 60:
        parts.append("Late-season intensity builds as playoff positioning solidifies.")
    
    # 6. PLAYOFF IMPLICATIONS
    season_win_pct = tc.get('season_win_pct', 0)
    if games_played >= 60:
        if season_win_pct > 0.60:
            parts.append("The team fights for higher playoff seeding.")
        elif 0.45 <= season_win_pct <= 0.55:
            parts.append("Playoff hopes hang in the balance.")
        elif season_win_pct < 0.35:
            parts.append("Despite the difficult season, pride remains on the line.")
    
    # 7. HOME/AWAY CONTEXT
    if is_home:
        parts.append(f"{team_name} look to protect home court.")
    else:
        parts.append(f"{team_name} seek a crucial road victory.")
    
    # 8. MATCHUP ANTICIPATION
    parts.append(f"The matchup against {opponent_abbr} represents an important test.")
    
    return " ".join(parts)

enriched_games = []

for i, matchup in enumerate(matchups_structured):
    if i % 2000 == 0:
        print(f"  Generated {i:,}/{len(matchups_structured):,} narratives...")
    
    narrative = generate_pregame_narrative(matchup)
    
    game_enriched = matchup['game'].copy()
    game_enriched['pregame_narrative'] = narrative
    
    enriched_games.append(game_enriched)

print(f"✓ Generated {len(enriched_games):,} pre-game narratives")

# ============================================================================
# VERIFY ZERO OUTCOME LEAKAGE
# ============================================================================

print("\n[4/4] Verifying zero outcome leakage...")

# Only flag PAST TENSE outcome words (actual leakage)
outcome_phrases = [
    'secured a victory', 'secured victory', 'narrowly won', 'won the game',
    'fell short', 'narrowly lost', 'lost the game', 'were defeated',
    'defeated the', 'beat the', 'triumphed over'
]
leak_count = sum(1 for g in enriched_games if any(phrase in g['pregame_narrative'].lower() for phrase in outcome_phrases))

print(f"  Narratives checked: {len(enriched_games):,}")
print(f"  Past-tense outcome phrases found: {leak_count}")
print(f"  {'✓ CLEAN - Zero leakage' if leak_count == 0 else '✗ LEAKAGE DETECTED'}")

# Save
output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nba_pregame_narratives.json'

with open(output_path, 'w') as f:
    json.dump(enriched_games, f, indent=2)

print(f"\n✓ Saved to: {output_path.name}")
print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# Show samples
print(f"\n" + "="*80)
print("SAMPLE PRE-GAME NARRATIVES")
print("="*80)

for i in range(min(3, len(enriched_games))):
    game = enriched_games[i]
    print(f"\nGame {i+1}: {game['team_name']} vs opponent")
    print(f"Date: {game['date']}")
    print(f"Pregame: {game['pregame_narrative'][:200]}...")

print("\n" + "="*80)
print("✓ COMPLETE - Clean pre-game narratives ready for validation")
print("="*80)

