"""
NFL-Specific Nominative Feature Engineering

Extract 100-150 features exploiting NFL's nominative goldmine:
- QB prestige and reputation (Brady=0.98, Mahomes=0.90)
- Coach prestige with REAL NAMES (Belichick=0.98, Reid=0.92)
- O-line ensemble cohesion (5 players)
- Position group patterns
- Star player effects

This is what tennis did (+46 features) applied to NFL's richer space (22 players + coach).
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# DYNAMIC prestige calculation - NO HARDCODING
# Calculate from actual data: win rate, tenure, consistency

print("="*80)
print("NFL-SPECIFIC NOMINATIVE FEATURE ENGINEERING")
print("="*80)

# Load data
print("\n[1/7] Loading data...")
dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
with open(dataset_path) as f:
    games = json.load(f)

print(f"✓ Loaded {len(games)} games with REAL COACH NAMES")

# Verify real names
sample_coaches = set()
for g in games[:100]:
    sample_coaches.add(g['home_coaches']['head_coach'])
    sample_coaches.add(g['away_coaches']['head_coach'])

print(f"  Sample coaches: {list(sample_coaches)[:5]}")

# Initialize feature matrix
all_features = []

print("\n[2/7] Calculating QB prestige DYNAMICALLY from data...", end=" ", flush=True)

# Calculate QB prestige from actual performance in dataset
qb_stats = defaultdict(lambda: {'games': 0, 'wins': 0})

for game in games:
    home_qb = game['home_roster']['starting_qb']['name']
    away_qb = game['away_roster']['starting_qb']['name']
    
    qb_stats[home_qb]['games'] += 1
    qb_stats[away_qb]['games'] += 1
    
    if game['home_won']:
        qb_stats[home_qb]['wins'] += 1
    else:
        qb_stats[away_qb]['wins'] += 1

# Calculate prestige: win rate × log(games) normalized
qb_prestige_dynamic = {}
for qb, stats in qb_stats.items():
    if stats['games'] >= 5:
        win_rate = stats['wins'] / stats['games']
        experience = np.log1p(stats['games']) / np.log1p(100)  # Normalize to 0-1
        prestige = (win_rate * 0.7 + experience * 0.3)  # Blend win rate and experience
        qb_prestige_dynamic[qb] = prestige

print(f"✓ {len(qb_prestige_dynamic)} QBs with prestige scores")

# Extract features for each game
qb_features_list = []
for game in games:
    home_qb = game['home_roster']['starting_qb']['name']
    away_qb = game['away_roster']['starting_qb']['name']
    
    home_qb_prestige = qb_prestige_dynamic.get(home_qb, 0.50)
    away_qb_prestige = qb_prestige_dynamic.get(away_qb, 0.50)
    
    qb_features = [
        home_qb_prestige,
        away_qb_prestige,
        home_qb_prestige - away_qb_prestige,  # Differential
        max(home_qb_prestige, away_qb_prestige),  # Best QB
        min(home_qb_prestige, away_qb_prestige),  # Worst QB
        home_qb_prestige * away_qb_prestige,  # Product (both elite?)
    ]
    qb_features_list.append(qb_features)

qb_features_array = np.array(qb_features_list)
print(f"✓ {qb_features_array.shape[1]} QB features")

print("[3/7] Calculating coach prestige DYNAMICALLY from data...", end=" ", flush=True)

# Calculate coach prestige from actual performance
coach_stats = defaultdict(lambda: {'games': 0, 'wins': 0})

for game in games:
    home_coach = game['home_coaches']['head_coach']
    away_coach = game['away_coaches']['head_coach']
    
    coach_stats[home_coach]['games'] += 1
    coach_stats[away_coach]['games'] += 1
    
    if game['home_won']:
        coach_stats[home_coach]['wins'] += 1
    else:
        coach_stats[away_coach]['wins'] += 1

# Calculate prestige: win rate × log(games) + consistency
coach_prestige_dynamic = {}
for coach, stats in coach_stats.items():
    if stats['games'] >= 5:
        win_rate = stats['wins'] / stats['games']
        experience = np.log1p(stats['games']) / np.log1p(200)  # More games than QBs
        prestige = (win_rate * 0.6 + experience * 0.4)  # Experience matters more for coaches
        coach_prestige_dynamic[coach] = prestige

print(f"✓ {len(coach_prestige_dynamic)} coaches with prestige scores")

# Show top coaches by calculated prestige
top_coaches = sorted(coach_prestige_dynamic.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"    Top 5: {[f'{c[0]}({c[1]:.2f})' for c in top_coaches]}")

coach_features_list = []
for game in games:
    home_coach = game['home_coaches']['head_coach']
    away_coach = game['away_coaches']['head_coach']
    
    home_coach_prestige = coach_prestige_dynamic.get(home_coach, 0.50)
    away_coach_prestige = coach_prestige_dynamic.get(away_coach, 0.50)
    
    # Get games coached (from stats)
    home_coach_games = coach_stats[home_coach]['games']
    away_coach_games = coach_stats[away_coach]['games']
    
    coach_features = [
        home_coach_prestige,
        away_coach_prestige,
        home_coach_prestige - away_coach_prestige,
        max(home_coach_prestige, away_coach_prestige),
        home_coach_prestige * away_coach_prestige,
        1.0 if home_coach_prestige > 0.70 else 0.0,  # High prestige coach (data-driven)
        1.0 if away_coach_prestige > 0.70 else 0.0,
        home_coach_games / 200.0,  # Experience (normalized)
        away_coach_games / 200.0,
    ]
    coach_features_list.append(coach_features)

coach_features_array = np.array(coach_features_list)
print(f"✓ {coach_features_array.shape[1]} coach features")

print("[4/7] Extracting O-line ensemble features...", end=" ", flush=True)
oline_features_list = []
for game in games:
    home_oline = game['home_ensemble'].get('offensive_unit', [])[:5]
    away_oline = game['away_ensemble'].get('offensive_unit', [])[:5]
    
    # O-line features
    home_oline_count = len(home_oline)
    away_oline_count = len(away_oline)
    
    # Name diversity (unique vs repeated)
    home_unique = len(set(home_oline))
    away_unique = len(set(away_oline))
    
    oline_features = [
        home_oline_count,
        away_oline_count,
        home_unique / home_oline_count if home_oline_count > 0 else 0,
        away_unique / away_oline_count if away_oline_count > 0 else 0,
        home_oline_count - away_oline_count,
    ]
    oline_features_list.append(oline_features)

oline_features_array = np.array(oline_features_list)
print(f"✓ {oline_features_array.shape[1]} O-line features")

print("[5/7] Extracting star player features...", end=" ", flush=True)
star_features_list = []
for game in games:
    home_stars = game['home_ensemble'].get('star_players', [])
    away_stars = game['away_ensemble'].get('star_players', [])
    
    star_features = [
        len(home_stars),
        len(away_stars),
        len(home_stars) - len(away_stars),
        max(len(home_stars), len(away_stars)),
    ]
    star_features_list.append(star_features)

star_features_array = np.array(star_features_list)
print(f"✓ {star_features_array.shape[1]} star features")

print("[6/7] Creating interaction terms...", end=" ", flush=True)
interaction_features_list = []
for i in range(len(games)):
    interactions = [
        qb_features_array[i, 0] * coach_features_array[i, 0],  # Home QB × Coach
        qb_features_array[i, 1] * coach_features_array[i, 1],  # Away QB × Coach
        qb_features_array[i, 2] * coach_features_array[i, 2],  # QB diff × Coach diff
        qb_features_array[i, 0] * star_features_array[i, 0],  # QB × star count
        coach_features_array[i, 5] * coach_features_array[i, 6],  # Both legendary?
    ]
    interaction_features_list.append(interactions)

interaction_features_array = np.array(interaction_features_list)
print(f"✓ {interaction_features_array.shape[1]} interactions")

print("[7/7] Combining all NFL-specific features...", end=" ", flush=True)
# Combine all
nfl_specific_features = np.hstack([
    qb_features_array,
    coach_features_array,
    oline_features_array,
    star_features_array,
    interaction_features_array
])
print(f"✓ Total {nfl_specific_features.shape[1]} features")

# Save
output_path = Path(__file__).parent / 'nfl_specific_features.npz'
np.savez_compressed(
    output_path,
    features=nfl_specific_features,
    game_ids=[g['game_id'] for g in games]
)

print(f"\n{'='*80}")
print(f"✓ NFL-SPECIFIC FEATURES ENGINEERED")
print(f"  Total features: {nfl_specific_features.shape[1]}")
print(f"  Games: {nfl_specific_features.shape[0]}")
print(f"  Breakdown:")
print(f"    QB features: {qb_features_array.shape[1]}")
print(f"    Coach features: {coach_features_array.shape[1]} (WITH REAL NAMES!)")
print(f"    O-line features: {oline_features_array.shape[1]}")
print(f"    Star features: {star_features_array.shape[1]}")
print(f"    Interactions: {interaction_features_array.shape[1]}")
print(f"{'='*80}")

print(f"\n✓ Saved to: {output_path}")
print(f"\nNext: Combine with transformer features (1,044) for total ~1,070 features")
print("Then: Build coach-specific models (Belichick, Reid, Carroll)")

