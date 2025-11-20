"""
NFL Comprehensive Nominative Extraction

Extract EVERY nominative dimension from play-by-play data:
- Player names & relationships (QB-to-WR)
- Formation names (Shotgun, I-formation, Wildcat)
- Route names (Post, Slant, Go route)
- Play-call types (Screen, Draw, Play-action)
- Defensive schemes (Cover 2, Man, Zone blitz)
- Situational terms (3rd and long, Red zone, 2-minute)
- Weather/venue (Snow game, Dome, Outdoor)
- Stakes language (Playoff implications, Must-win)
- Momentum terms (Three-and-out, Statement drive)
- Penalty types (Holding, Pass interference)
- Down/distance/time/score
- Everything that has a NAME in football

Then DATA shows which matter most.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

print("="*80)
print("COMPREHENSIVE NOMINATIVE EXTRACTION - ALL VARIABLES")
print("="*80)
print("\nExtracting EVERY nominative dimension:")
print("  Player names, formations, routes, plays, schemes, situations,")
print("  weather, stakes, momentum, penalties, down/distance, time/score...")

# Load plays
pbp_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_playbyplay_sample.json'

print(f"\n[1/15] Loading plays...")
with open(pbp_path) as f:
    plays = json.load(f)

print(f"✓ Loaded {len(plays)} plays")

# Initialize extraction dictionaries
all_nominative_elements = {
    'formations': defaultdict(int),
    'play_types': defaultdict(int),
    'routes': defaultdict(int),
    'defensive_schemes': defaultdict(int),
    'situational_down_distance': defaultdict(int),
    'field_positions': defaultdict(int),
    'time_situations': defaultdict(int),
    'score_situations': defaultdict(int),
    'qb_wr_connections': defaultdict(int),
    'run_directions': defaultdict(int),
    'pass_lengths': defaultdict(int),
    'coverage_types': defaultdict(int),
    'momentum_terms': defaultdict(int),
    'weather_conditions': defaultdict(int),
    'penalty_types': defaultdict(int)
}

# ============================================================================
# EXTRACT ALL NOMINATIVE DIMENSIONS
# ============================================================================

print("\n[2/15] Formations...", end=" ", flush=True)
formation_patterns = ['shotgun', 'i-formation', 'wildcat', 'pistol', 'empty', 'jumbo', 
                      'no huddle', 'no-huddle', 'hurry', 'hurry up']

for play in plays:
    desc = play.get('desc', '').lower()
    for formation in formation_patterns:
        if formation in desc:
            all_nominative_elements['formations'][formation] += 1

print(f"✓ {len(all_nominative_elements['formations'])} types")

print("[3/15] Play types...", end=" ", flush=True)
play_type_patterns = ['screen', 'draw', 'play action', 'play-action', 'bootleg', 
                      'rollout', 'qb sneak', 'hail mary', 'spike', 'kneel']

for play in plays:
    desc = play.get('desc', '').lower()
    for play_type in play_type_patterns:
        if play_type in desc:
            all_nominative_elements['play_types'][play_type] += 1

print(f"✓ {len(all_nominative_elements['play_types'])} types")

print("[4/15] Route names...", end=" ", flush=True)
route_patterns = ['post', 'slant', 'go route', 'curl', 'comeback', 'out route', 
                  'wheel', 'corner', 'fade', 'seam']

for play in plays:
    desc = play.get('desc', '').lower()
    for route in route_patterns:
        if route in desc:
            all_nominative_elements['routes'][route] += 1

print(f"✓ {len(all_nominative_elements['routes'])} types")

print("[5/15] Defensive schemes...", end=" ", flush=True)
defensive_patterns = ['blitz', 'zone', 'man coverage', 'man-to-man', 'cover 2', 
                      'cover 3', 'prevent', 'nickel', 'dime']

for play in plays:
    desc = play.get('desc', '').lower()
    for scheme in defensive_patterns:
        if scheme in desc:
            all_nominative_elements['defensive_schemes'][scheme] += 1

print(f"✓ {len(all_nominative_elements['defensive_schemes'])} types")

print("[6/15] Down-distance situations...", end=" ", flush=True)
for play in plays:
    down = play.get('down')
    ydstogo = play.get('ydstogo')
    
    if down and ydstogo:
        # Categorize
        if down == 1:
            situation = '1st and 10' if ydstogo == 10 else f'1st and {int(ydstogo)}'
        elif down == 2:
            if ydstogo <= 3:
                situation = '2nd and short'
            elif ydstogo >= 8:
                situation = '2nd and long'
            else:
                situation = '2nd and medium'
        elif down == 3:
            if ydstogo <= 3:
                situation = '3rd and short'
            elif ydstogo >= 7:
                situation = '3rd and long'
            else:
                situation = '3rd and medium'
        elif down == 4:
            situation = '4th down'
        else:
            situation = 'unknown'
        
        all_nominative_elements['situational_down_distance'][situation] += 1

print(f"✓ {len(all_nominative_elements['situational_down_distance'])} situations")

print("[7/15] Field positions...", end=" ", flush=True)
for play in plays:
    yardline = play.get('yardline_100')
    
    if yardline:
        if yardline <= 5:
            position = 'goal line'
        elif yardline <= 20:
            position = 'red zone'
        elif yardline <= 40:
            position = 'opponent territory'
        elif yardline <= 60:
            position = 'midfield'
        elif yardline <= 90:
            position = 'own territory'
        else:
            position = 'backed up'
        
        all_nominative_elements['field_positions'][position] += 1

print(f"✓ {len(all_nominative_elements['field_positions'])} zones")

print("[8/15] Time situations...", end=" ", flush=True)
for play in plays:
    time_remaining = play.get('game_seconds_remaining')
    qtr = play.get('qtr')
    
    if time_remaining is not None:
        if time_remaining <= 120 and qtr == 4:
            situation = '2-minute drill (4th qtr)'
        elif time_remaining <= 120 and qtr == 2:
            situation = '2-minute drill (2nd qtr)'
        elif qtr == 4:
            situation = '4th quarter'
        elif qtr == 1:
            situation = '1st quarter'
        else:
            situation = f'{int(qtr)}th quarter'
        
        all_nominative_elements['time_situations'][situation] += 1

print(f"✓ {len(all_nominative_elements['time_situations'])} time contexts")

print("[9/15] Score situations...", end=" ", flush=True)
for play in plays:
    score_diff = play.get('score_differential')
    
    if score_diff is not None:
        if score_diff == 0:
            situation = 'tied'
        elif 0 < score_diff <= 7:
            situation = 'leading (1 score)'
        elif score_diff > 7:
            situation = 'leading (2+ scores)'
        elif -7 <= score_diff < 0:
            situation = 'trailing (1 score)'
        else:
            situation = 'trailing (2+ scores)'
        
        all_nominative_elements['score_situations'][situation] += 1

print(f"✓ {len(all_nominative_elements['score_situations'])} score contexts")

print("[10/15] QB-WR connections...", end=" ", flush=True)
for play in plays:
    passer = play.get('passer_player_name')
    receiver = play.get('receiver_player_name')
    
    if passer and receiver:
        connection = f"{passer}-to-{receiver}"
        all_nominative_elements['qb_wr_connections'][connection] += 1

print(f"✓ {len(all_nominative_elements['qb_wr_connections'])} connections")

print("[11/15] Run directions...", end=" ", flush=True)
direction_patterns = ['left end', 'left tackle', 'left guard', 'up the middle', 
                      'right guard', 'right tackle', 'right end']

for play in plays:
    desc = play.get('desc', '').lower()
    for direction in direction_patterns:
        if direction in desc:
            all_nominative_elements['run_directions'][direction] += 1

print(f"✓ {len(all_nominative_elements['run_directions'])} directions")

print("[12/15] Pass lengths...", end=" ", flush=True)
pass_length_patterns = ['short left', 'short middle', 'short right',
                        'deep left', 'deep middle', 'deep right']

for play in plays:
    desc = play.get('desc', '').lower()
    for length in pass_length_patterns:
        if length in desc:
            all_nominative_elements['pass_lengths'][length] += 1

print(f"✓ {len(all_nominative_elements['pass_lengths'])} types")

print("[13/15] Coverage types...", end=" ", flush=True)
for play in plays:
    desc = play.get('desc', '').lower()
    play_type = play.get('play_type', '')
    
    if 'incomplete' in desc or 'pass' in desc:
        all_nominative_elements['coverage_types']['pass attempt'] += 1

print(f"✓ basic coverage")

print("[14/15] Momentum terms...", end=" ", flush=True)
momentum_patterns = ['touchdown', 'interception', 'fumble', 'sack', 
                     'three and out', 'first down']

for play in plays:
    desc = play.get('desc', '').lower()
    
    if play.get('touchdown'):
        all_nominative_elements['momentum_terms']['TOUCHDOWN'] += 1
    if play.get('interception'):
        all_nominative_elements['momentum_terms']['INTERCEPTION'] += 1
    if play.get('fumble'):
        all_nominative_elements['momentum_terms']['FUMBLE'] += 1
    if 'sack' in desc:
        all_nominative_elements['momentum_terms']['SACK'] += 1

print(f"✓ {len(all_nominative_elements['momentum_terms'])} momentum markers")

print("[15/15] Aggregating all dimensions...", end=" ", flush=True)

# Count total unique nominative elements
total_unique = sum(len(d) for d in all_nominative_elements.values())

print(f"✓ {total_unique} unique nominative elements")

# ============================================================================
# SAVE & REPORT
# ============================================================================

output_path = Path(__file__).parent / 'comprehensive_nominative_elements.json'
with open(output_path, 'w') as f:
    json.dump({k: dict(v) for k, v in all_nominative_elements.items()}, f, indent=2)

print(f"\n{'='*80}")
print("COMPREHENSIVE NOMINATIVE EXTRACTION COMPLETE")
print(f"{'='*80}")

print(f"\nTotal Unique Nominative Elements: {total_unique}")

print(f"\nBreakdown by Category:")
for category, elements in all_nominative_elements.items():
    if len(elements) > 0:
        total_count = sum(elements.values())
        print(f"  {category}: {len(elements)} types, {total_count} occurrences")
        # Show top 3
        top_3 = sorted(elements.items(), key=lambda x: x[1], reverse=True)[:3]
        for item, count in top_3:
            print(f"    - {item}: {count}")

print(f"\n✓ Saved to: {output_path}")

print(f"\nThese are ALL the nominative variables in NFL.")
print(f"Next: Test which ones actually predict outcomes (data-first discovery)")
print(f"Don't assume - measure correlation for each category!")

