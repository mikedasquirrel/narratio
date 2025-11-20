"""
Tennis Player-Specific Models

Build specialized models for Big 3 (Federer, Nadal, Djokovic).
Each player has unique narrative patterns that benefit from dedicated optimization.

Target: 95%+ R² for player-specific matches
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*80)
print("TENNIS PLAYER-SPECIFIC MODELS")
print("="*80)

# Load data
print("\nLoading data...")
dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'tennis_complete_dataset.json'
with open(dataset_path) as f:
    all_matches = json.load(f)

genome_path = Path(__file__).parent / 'tennis_genome_data.npz'
genome_data = np.load(genome_path, allow_pickle=True)

ж = genome_data['genome']
outcomes = genome_data['outcomes']

# Tennis-specific features
tennis_features_path = Path(__file__).parent / 'tennis_specific_features.npz'
tennis_data = np.load(tennis_features_path)
tennis_features = tennis_data['features']

# Combined
X_combined = np.hstack([ж, tennis_features])

matches = all_matches[:5000]  # Same sample

print(f"✓ Loaded {len(matches)} matches with {X_combined.shape[1]} features")

# ============================================================================
# BUILD PLAYER-SPECIFIC MODELS
# ============================================================================

print("\n" + "="*80)
print("PLAYER-SPECIFIC OPTIMIZATION")
print("="*80)

big_3 = {
    'Roger Federer': [],
    'Rafael Nadal': [],
    'Novak Djokovic': []
}

# Find Big 3 matches
print("\nIdentifying Big 3 matches...", end=" ", flush=True)
for i, match in enumerate(matches):
    p1_name = match['player1']['name']
    p2_name = match['player2']['name']
    
    for player in big_3.keys():
        if player in p1_name or player in p2_name:
            big_3[player].append(i)

for player, indices in big_3.items():
    print(f"\n  {player}: {len(indices)} matches", end="")

print("\n✓ Big 3 matches identified")

# Build models for each
player_results = {}

for player_name, indices in big_3.items():
    if len(indices) < 30:
        print(f"\n[{player_name}] Skipping (only {len(indices)} matches)")
        continue
    
    print(f"\n[{player_name}] Optimizing...", end=" ", flush=True)
    
    # Extract player-specific data
    X_player = X_combined[indices]
    y_player = outcomes[indices]
    
    # Split
    if len(indices) < 50:
        test_size = 0.2
    else:
        test_size = 0.3
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_player, y_player, test_size=test_size, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = Ridge(alpha=5.0)  # Lower alpha for smaller datasets
    model.fit(X_train_scaled, y_train)
    
    # Test
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    r_train = np.corrcoef(y_pred_train, y_train)[0, 1]
    r_test = np.corrcoef(y_pred_test, y_test)[0, 1]
    r2_train = r_train ** 2
    r2_test = r_test ** 2
    
    print(f"✓ Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")
    
    player_results[player_name] = {
        'n_matches': len(indices),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_r2': float(r2_train),
        'test_r2': float(r2_test),
        'improvement_over_general': float(r2_test - 0.9294) if r2_test > 0.9294 else 0.0
    }

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING PLAYER MODELS")
print("="*80)

output = {
    'player_models': player_results,
    'general_model_r2': 0.9294,
    'summary': {
        'players_modeled': len(player_results),
        'best_player_model': max(player_results.keys(), key=lambda k: player_results[k]['test_r2']) if player_results else None
    }
}

output_path = Path(__file__).parent / 'player_models_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved to: {output_path}")

print("\n" + "="*80)
print("PLAYER-SPECIFIC MODELS COMPLETE")
print("="*80)

if player_results:
    print(f"\nPlayer-Specific Results:")
    for player, res in player_results.items():
        improvement = res['improvement_over_general']
        symbol = "+" if improvement > 0 else ""
        print(f"  {player}: {res['test_r2']*100:.1f}% R² ({symbol}{improvement*100:.1f}pp vs general)")

