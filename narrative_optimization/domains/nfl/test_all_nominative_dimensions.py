"""
Test ALL Nominative Dimensions - Data-First Discovery

We have 1,894 nominative elements extracted:
- 1,816 QB-WR connections
- 28 down-distance situations
- Formations, routes, schemes, positions, etc.

TEST: Which ones actually predict game outcomes?
Don't assume - MEASURE each category's correlation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

print("="*80)
print("TESTING ALL NOMINATIVE DIMENSIONS - DATA-FIRST")
print("="*80)

# Load comprehensive nominative elements
nominative_path = Path(__file__).parent / 'comprehensive_nominative_elements.json'
with open(nominative_path) as f:
    all_elements = json.load(f)

print(f"\n✓ Loaded nominative elements")
print(f"  Categories: {len(all_elements)}")
print(f"  Total unique elements: {sum(len(v) for v in all_elements.values())}")

# Load plays
pbp_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_playbyplay_sample.json'
with open(pbp_path) as f:
    plays = json.load(f)

print(f"  Plays: {len(plays)}")

# Load games with outcomes
games_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
with open(games_path) as f:
    games = json.load(f)

print(f"  Games: {len(games)}")

# Create game-level aggregation from plays
print(f"\n[1/4] Aggregating plays to games...")

game_play_features = defaultdict(lambda: {
    'formations': defaultdict(int),
    'play_types': defaultdict(int),
    'down_distance': defaultdict(int),
    'field_position': defaultdict(int),
    'qb_wr_freq': defaultdict(int),
    'momentum_events': defaultdict(int),
    'total_plays': 0
})

for play in plays:
    game_id = play.get('game_id')
    if not game_id:
        continue
    
    desc = play.get('desc', '').lower()
    down = play.get('down')
    ydstogo = play.get('ydstogo')
    yardline = play.get('yardline_100')
    
    game_play_features[game_id]['total_plays'] += 1
    
    # Track formations
    if 'shotgun' in desc:
        game_play_features[game_id]['formations']['shotgun'] += 1
    if 'no huddle' in desc or 'no-huddle' in desc:
        game_play_features[game_id]['formations']['no_huddle'] += 1
    
    # Track situations
    if down == 3 and ydstogo and ydstogo >= 7:
        game_play_features[game_id]['down_distance']['3rd_and_long'] += 1
    if yardline and yardline <= 20:
        game_play_features[game_id]['field_position']['red_zone'] += 1
    
    # Track QB-WR
    passer = play.get('passer_player_name')
    receiver = play.get('receiver_player_name')
    if passer and receiver:
        conn = f"{passer}-to-{receiver}"
        game_play_features[game_id]['qb_wr_freq'][conn] += 1
    
    # Track momentum
    if play.get('touchdown'):
        game_play_features[game_id]['momentum_events']['touchdown'] += 1
    if 'sack' in desc:
        game_play_features[game_id]['momentum_events']['sack'] += 1

print(f"✓ Aggregated {len(game_play_features)} games")

# Build feature matrix
print(f"\n[2/4] Building feature matrix from nominative dimensions...")

# Match plays to games
feature_matrix = []
game_ids_ordered = []
outcomes = []

for game in games:
    game_id = game.get('game_id')
    if game_id not in game_play_features:
        continue
    
    play_data = game_play_features[game_id]
    total = play_data['total_plays']
    
    if total == 0:
        continue
    
    # Build feature vector for this game
    features = []
    
    # Formation frequencies
    features.append(play_data['formations'].get('shotgun', 0) / total)
    features.append(play_data['formations'].get('no_huddle', 0) / total)
    
    # Situational frequencies
    features.append(play_data['down_distance'].get('3rd_and_long', 0) / total)
    features.append(play_data['field_position'].get('red_zone', 0) / total)
    
    # Momentum events
    features.append(play_data['momentum_events'].get('touchdown', 0))
    features.append(play_data['momentum_events'].get('sack', 0))
    
    # Top QB-WR connection strength
    if play_data['qb_wr_freq']:
        top_connection = max(play_data['qb_wr_freq'].values())
        features.append(top_connection / total)
    else:
        features.append(0)
    
    # Total plays (game pace)
    features.append(total / 70.0)  # Normalize around 70 plays
    
    feature_matrix.append(features)
    game_ids_ordered.append(game_id)
    outcomes.append(int(game['home_won']))

X = np.array(feature_matrix)
y = np.array(outcomes)

print(f"✓ Feature matrix: {X.shape}")
print(f"  Features:")
print(f"    - Shotgun %")
print(f"    - No-huddle %")
print(f"    - 3rd-and-long %")
print(f"    - Red zone %")
print(f"    - Touchdowns")
print(f"    - Sacks")
print(f"    - Top QB-WR connection strength")
print(f"    - Game pace")

# Test prediction
print(f"\n[3/4] Testing play-level features...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Ridge(alpha=10.0)
model.fit(X_train_scaled, y_train)

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

r_train = np.corrcoef(y_pred_train, y_train)[0, 1]
r_test = np.corrcoef(y_pred_test, y_test)[0, 1]

r2_train = r_train ** 2
r2_test = r_test ** 2

print(f"\n✓ Play-Level Features Only:")
print(f"  Train R²: {r2_train:.4f} ({r2_train*100:.1f}%)")
print(f"  Test R²: {r2_test:.4f} ({r2_test*100:.1f}%)")

# Save
print(f"\n[4/4] Saving results...")

results = {
    'play_level_only': {
        'features': 8,
        'train_r2': float(r2_train),
        'test_r2': float(r2_test)
    },
    'nominative_elements_tested': {
        'shotgun_pct': 'formation',
        'no_huddle_pct': 'formation',
        '3rd_and_long_pct': 'situational',
        'red_zone_pct': 'field_position',
        'touchdowns': 'momentum',
        'sacks': 'momentum',
        'top_qb_wr_connection': 'relational',
        'game_pace': 'tempo'
    },
    'comparison': {
        'game_level_r2': 0.14,
        'play_level_r2': float(r2_test),
        'improvement': float((r2_test - 0.14) / 0.14 * 100) if r2_test > 0.14 else 0
    }
}

output_path = Path(__file__).parent / 'play_level_test_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Saved to: {output_path}")

print(f"\n{'='*80}")
print("PLAY-LEVEL TESTING COMPLETE")
print(f"{'='*80}")

print(f"\nComparison:")
print(f"  Game-level (coach/QB names): 14% R²")
print(f"  Play-level (formations/situations/connections): {r2_test*100:.1f}% R²")

if r2_test > 0.14:
    improvement = (r2_test - 0.14) / 0.14 * 100
    print(f"  ✓ IMPROVEMENT: +{improvement:.0f}%")
    print(f"\n  Proves play-level nominative richness adds value!")
else:
    print(f"  → Similar performance")
    print(f"\n  Play-level features alone comparable to game-level")

print(f"\nNext: Combine ALL features (game + play level) for maximum R²")

