"""
Parse NFL Play-by-Play Descriptions

Extract nominative elements from 531K play descriptions:
- QB-to-WR relational phrases ("Mahomes to Kelce")
- Situational terms ("3rd and long", "red zone")
- Player action patterns
- Play type terminology

THIS is the real nominative goldmine.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

print("="*80)
print("PARSING 531K PLAY DESCRIPTIONS")
print("="*80)

# Load play-by-play data
pbp_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_playbyplay_sample.json'

print(f"\n[1/6] Loading play data...")
with open(pbp_path) as f:
    plays = json.load(f)

print(f"✓ Loaded {len(plays)} plays")

# Sample descriptions
print(f"\n[2/6] Sample play descriptions:")
for i in range(3):
    print(f"  {i+1}. {plays[i]['desc'][:100]}...")

# Parse relational phrases
print(f"\n[3/6] Extracting QB-to-WR relational phrases...", end=" ", flush=True)

qb_to_wr_connections = defaultdict(int)
qb_to_te_connections = defaultdict(int)

for play in plays:
    desc = play.get('desc', '')
    passer = play.get('passer_player_name')
    receiver = play.get('receiver_player_name')
    
    if passer and receiver and 'pass' in desc.lower():
        # Create relational phrase
        connection = f"{passer}-to-{receiver}"
        
        # Classify by position (crude heuristic)
        if 'TE' in desc or any(te in receiver for te in ['Kelce', 'Gronk', 'Kittle']):
            qb_to_te_connections[connection] += 1
        else:
            qb_to_wr_connections[connection] += 1

print(f"✓ {len(qb_to_wr_connections)} QB-WR connections, {len(qb_to_te_connections)} QB-TE")

# Top connections
top_wr = sorted(qb_to_wr_connections.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"  Top QB-WR: {top_wr[0][0]} ({top_wr[0][1]} times)")

# Extract situational terminology
print(f"\n[4/6] Extracting situational terms...", end=" ", flush=True)

situational_terms = {
    '3rd_and_long': 0,
    '3rd_and_short': 0,
    'red_zone': 0,
    '2_minute': 0,
    'goal_line': 0,
    'shotgun': 0,
    'no_huddle': 0,
    'play_action': 0
}

for play in plays:
    desc = play.get('desc', '').lower()
    down = play.get('down')
    ydstogo = play.get('ydstogo')
    yardline = play.get('yardline_100')
    time_remaining = play.get('game_seconds_remaining')
    
    # Identify situations
    if down == 3:
        if ydstogo and ydstogo >= 7:
            situational_terms['3rd_and_long'] += 1
        elif ydstogo and ydstogo <= 3:
            situational_terms['3rd_and_short'] += 1
    
    if yardline and yardline <= 20:
        situational_terms['red_zone'] += 1
    
    if yardline and yardline <= 5:
        situational_terms['goal_line'] += 1
    
    if time_remaining and time_remaining <= 120:
        situational_terms['2_minute'] += 1
    
    if 'shotgun' in desc:
        situational_terms['shotgun'] += 1
    
    if 'no huddle' in desc or 'no-huddle' in desc:
        situational_terms['no_huddle'] += 1
    
    if 'play action' in desc or 'play-action' in desc:
        situational_terms['play_action'] += 1

print(f"✓ Identified")
for term, count in situational_terms.items():
    if count > 0:
        print(f"    {term}: {count}")

# Aggregate to game level
print(f"\n[5/6] Aggregating plays to game level...", end=" ", flush=True)

game_aggregates = defaultdict(lambda: {
    'plays': [],
    'qb_wr_connections': defaultdict(int),
    'situations': defaultdict(int),
    'total_plays': 0
})

for play in plays:
    game_id = play.get('game_id')
    if game_id:
        game_aggregates[game_id]['plays'].append(play)
        game_aggregates[game_id]['total_plays'] += 1

print(f"✓ {len(game_aggregates)} games")

# Save parsed data
print(f"\n[6/6] Saving parsed data...", end=" ", flush=True)

output = {
    'qb_wr_connections': dict(qb_to_wr_connections),
    'qb_te_connections': dict(qb_to_te_connections),
    'situational_terms': situational_terms,
    'top_connections': [
        {'pair': conn, 'frequency': freq} 
        for conn, freq in sorted(qb_to_wr_connections.items(), key=lambda x: x[1], reverse=True)[:20]
    ],
    'games_with_plays': len(game_aggregates),
    'total_plays_parsed': len(plays)
}

output_path = Path(__file__).parent / 'play_descriptions_parsed.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Saved to {output_path}")

print(f"\n{'='*80}")
print("PLAY DESCRIPTION PARSING COMPLETE")
print(f"{'='*80}")

print(f"\nKey Findings:")
print(f"  • {len(qb_to_wr_connections)} unique QB-WR connections")
print(f"  • {situational_terms['3rd_and_long']} third-and-long situations")
print(f"  • {situational_terms['red_zone']} red zone plays")
print(f"  • {situational_terms['2_minute']} two-minute drill plays")
print(f"\nThese are the NOMINATIVE units tennis captured!")
print(f"Next: Build features from these relational and situational elements")

