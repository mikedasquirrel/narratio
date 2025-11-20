"""
NFL Situational Performance Narratives

Generate narratives that combine:
- Player names (Mahomes, Kelce)
- Situational terms (3rd and long, red zone)
- Performance stats (68% conversion, 12 TDs)

This is the REAL goldmine: "Mahomes excels in 3rd-and-long with 68% conversion"
Not just "Team had 15 3rd downs" (0% R²) or "Mahomes is QB" (14% R²)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("SITUATIONAL PERFORMANCE NARRATIVE GENERATION")
print("="*80)
print("\nCombining: Player names + Situations + Performance")
print("  'Mahomes excels in 3rd-and-long (68% conversion)'")
print("  'Kelce dominates red zone (12 TDs this season)'")

# Load plays
pbp_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_playbyplay_sample.json'
with open(pbp_path) as f:
    plays = json.load(f)

# Load games
games_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
with open(games_path) as f:
    games = json.load(f)

print(f"\n✓ Loaded {len(plays)} plays, {len(games)} games")

# Calculate player situational performance
print(f"\n[1/3] Calculating player situational performance...", end=" ", flush=True)

player_situation_stats = defaultdict(lambda: defaultdict(lambda: {'attempts': 0, 'successes': 0}))

for play in plays:
    passer = play.get('passer_player_name')
    receiver = play.get('receiver_player_name')
    down = play.get('down')
    ydstogo = play.get('ydstogo')
    yardline = play.get('yardline_100')
    complete = play.get('complete_pass')
    yards = play.get('yards_gained', 0)
    
    # Track QB in 3rd down
    if passer and down == 3:
        if ydstogo and ydstogo >= 7:
            situation = '3rd_and_long'
        elif ydstogo and ydstogo <= 3:
            situation = '3rd_and_short'
        else:
            situation = '3rd_and_medium'
        
        player_situation_stats[passer][situation]['attempts'] += 1
        if complete or (yards and yards > ydstogo):
            player_situation_stats[passer][situation]['successes'] += 1
    
    # Track receiver in red zone
    if receiver and yardline and yardline <= 20:
        player_situation_stats[receiver]['red_zone']['attempts'] += 1
        if play.get('touchdown'):
            player_situation_stats[receiver]['red_zone']['successes'] += 1

print(f"✓ {len(player_situation_stats)} players")

# Generate enhanced narratives
print(f"\n[2/3] Generating situational performance narratives...")

enhanced_games = []
narratives_generated = 0

for game in games:
    game_id = game.get('game_id')
    
    # Get QBs
    home_qb = game['home_roster']['starting_qb']['name']
    away_qb = game['away_roster']['starting_qb']['name']
    
    # Get top receivers
    home_wr1 = game['home_roster']['starting_wr1']['name']
    away_wr1 = game['away_roster']['starting_wr1']['name']
    
    # Build situational narrative
    narrative_parts = []
    
    # Base
    narrative_parts.append(game.get('narrative', ''))
    
    # Add QB situational performance
    if home_qb in player_situation_stats:
        qb_stats = player_situation_stats[home_qb]
        if '3rd_and_long' in qb_stats and qb_stats['3rd_and_long']['attempts'] >= 5:
            attempts = qb_stats['3rd_and_long']['attempts']
            successes = qb_stats['3rd_and_long']['successes']
            pct = (successes / attempts * 100) if attempts > 0 else 0
            narrative_parts.append(f"{home_qb} converts {pct:.0f}% of 3rd-and-long situations.")
    
    # Add WR red zone performance
    if home_wr1 in player_situation_stats:
        wr_stats = player_situation_stats[home_wr1]
        if 'red_zone' in wr_stats and wr_stats['red_zone']['attempts'] >= 3:
            targets = wr_stats['red_zone']['attempts']
            tds = wr_stats['red_zone']['successes']
            narrative_parts.append(f"{home_wr1} has {tds} red zone touchdowns on {targets} targets.")
    
    # Combine
    enhanced_narrative = " ".join(narrative_parts)
    game['enhanced_narrative'] = enhanced_narrative
    enhanced_games.append(game)
    
    if len(narrative_parts) > 1:  # Has situational additions
        narratives_generated += 1
    
    if (len(enhanced_games) % 500 == 0):
        print(f"  Generated {len(enhanced_games)}/3010...", end="\r", flush=True)

print(f"\n✓ Enhanced {narratives_generated} narratives with situational performance")

# Save
print(f"\n[3/3] Saving enhanced dataset...", end=" ", flush=True)

output_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_situational_narratives.json'
with open(output_path, 'w') as f:
    json.dump(enhanced_games, f, indent=2)

print(f"✓")

print(f"\n{'='*80}")
print("SITUATIONAL PERFORMANCE NARRATIVES COMPLETE")
print(f"{'='*80}")

print(f"\nEnhanced narratives: {narratives_generated}/{len(games)}")
print(f"Saved to: {output_path}")

print(f"\nSample enhanced narrative:")
sample = next((g for g in enhanced_games if 'Mahomes' in g.get('enhanced_narrative', '')), enhanced_games[0])
print(f"{sample['enhanced_narrative'][:500]}...")

print(f"\nNext: Apply transformers to enhanced narratives and test R² improvement")

