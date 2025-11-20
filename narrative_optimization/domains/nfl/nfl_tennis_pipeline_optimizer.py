"""
NFL Tennis-Style Optimization Pipeline

Apply EXACT tennis methodology to NFL:
1. Combine transformer (1,044) + NFL-specific (29) features
2. Feature selection (top 400 like tennis 300)
3. StandardScaler
4. Ridge(α=10)
5. Test on coach-specific contexts

Target: 40-60% R² overall, 70-85% for top coaches
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from collections import defaultdict

print("="*80)
print("NFL OPTIMIZED - TENNIS PIPELINE APPLICATION")
print("="*80)

# Load genome + NFL features
print("\n[1/5] Loading features...")
genome_path = Path(__file__).parent / 'nfl_genome_data.npz'
genome_data = np.load(genome_path, allow_pickle=True)
ж = genome_data['genome']
outcomes = genome_data['outcomes']

nfl_features_path = Path(__file__).parent / 'nfl_specific_features.npz'
nfl_data = np.load(nfl_features_path)
nfl_features = nfl_data['features']

print(f"✓ Transformer features: {ж.shape}")
print(f"✓ NFL-specific features: {nfl_features.shape}")

# Combine
X_combined = np.hstack([ж, nfl_features])
y = outcomes

print(f"✓ Combined: {X_combined.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.3, random_state=42
)

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# Load games for context analysis
print("\n[2/5] Loading game metadata...")
dataset_path = Path(__file__).parent.parent.parent.parent / 'data' / 'domains' / 'nfl_complete_dataset.json'
with open(dataset_path) as f:
    games = json.load(f)

print(f"✓ Loaded {len(games)} games with REAL coach names")

# ============================================================================
# OVERALL OPTIMIZATION (Tennis Style)
# ============================================================================

print("\n[3/5] Applying tennis optimization pipeline...")
print("  Scaling...", end=" ", flush=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓")

print("  Selecting features (top 400)...", end=" ", flush=True)
selector = SelectKBest(mutual_info_regression, k=400)
selector.fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)
print("✓")

print("  Training Ridge(α=10)...", end=" ", flush=True)
model = Ridge(alpha=10.0)
model.fit(X_train_selected, y_train)
print("✓")

print("  Testing...", end=" ", flush=True)
y_pred_train = model.predict(X_train_selected)
y_pred_test = model.predict(X_test_selected)

r_train = np.corrcoef(y_pred_train, y_train)[0, 1]
r_test = np.corrcoef(y_pred_test, y_test)[0, 1]
r2_train = r_train ** 2
r2_test = r_test ** 2
print("✓")

print(f"\n{'='*80}")
print("NFL OPTIMIZED RESULTS")
print(f"{'='*80}")
print(f"\nOverall Performance:")
print(f"  Train R²: {r2_train:.4f} ({r2_train*100:.1f}%)")
print(f"  Test R²: {r2_test:.4f} ({r2_test*100:.1f}%)")
print(f"  Train |r|: {abs(r_train):.4f}")
print(f"  Test |r|: {abs(r_test):.4f}")

# ============================================================================
# COACH-SPECIFIC OPTIMIZATION (Like Tennis Surfaces)
# ============================================================================

print(f"\n[4/5] Coach-specific models (top coaches)...")

# Find coaches with most games
coach_game_counts = defaultdict(list)
for idx, game in enumerate(games):
    coach = game['home_coaches']['head_coach']
    coach_game_counts[coach].append(idx)

# Top coaches by game count
top_coaches = sorted(coach_game_counts.items(), key=lambda x: len(x[1]), reverse=True)[:5]

coach_results = {}

for coach, indices in top_coaches:
    if len(indices) < 50:
        continue
    
    print(f"\n  {coach} ({len(indices)} games)...", end=" ", flush=True)
    
    # Extract coach-specific data
    X_coach = X_combined[indices]
    y_coach = outcomes[indices]
    
    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_coach, y_coach, test_size=0.3, random_state=42
    )
    
    # Scale
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    
    # Select features
    sel = SelectKBest(mutual_info_regression, k=min(200, X_tr_sc.shape[1]))
    sel.fit(X_tr_sc, y_tr)
    X_tr_sel = sel.transform(X_tr_sc)
    X_te_sel = sel.transform(X_te_sc)
    
    # Train
    mod = Ridge(alpha=10.0)
    mod.fit(X_tr_sel, y_tr)
    
    # Test
    y_pred_tr = mod.predict(X_tr_sel)
    y_pred_te = mod.predict(X_te_sel)
    
    r_tr = np.corrcoef(y_pred_tr, y_tr)[0, 1]
    r_te = np.corrcoef(y_pred_te, y_te)[0, 1]
    
    r2_tr = r_tr ** 2
    r2_te = r_te ** 2
    
    print(f"Train R²: {r2_tr:.3f}, Test R²: {r2_te:.3f}")
    
    coach_results[coach] = {
        'n_games': len(indices),
        'train_r2': float(r2_tr),
        'test_r2': float(r2_te)
    }

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n[5/5] Saving results...")

output = {
    'overall': {
        'train_r2': float(r2_train),
        'test_r2': float(r2_test),
        'train_r': float(r_train),
        'test_r': float(r_test),
        'features_total': int(X_combined.shape[1]),
        'features_selected': 400
    },
    'coach_specific': coach_results,
    'improvement_over_original': {
        'original_test_r2': 0.0001,  # 0.97% |r| = 0.0001 R²
        'optimized_test_r2': float(r2_test),
        'improvement_factor': float(r2_test / 0.0001) if r2_test > 0 else 0
    }
}

output_path = Path(__file__).parent / 'nfl_optimized_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Saved to: {output_path}")

print(f"\n{'='*80}")
print("NFL OPTIMIZED COMPLETE")
print(f"{'='*80}")

print(f"\nComparison:")
print(f"  Original NFL: 0.01% R²")
print(f"  Optimized NFL: {r2_test*100:.1f}% R²")
print(f"  Improvement: {r2_test/0.0001:.0f}x" if r2_test > 0 else "  N/A")

print(f"\nCoach-Specific Results:")
for coach, res in coach_results.items():
    print(f"  {coach}: {res['test_r2']*100:.1f}% R² (n={res['n_games']})")

print(f"\nVs Tennis:")
print(f"  Tennis: 93.1% R²")
print(f"  NFL Optimized: {r2_test*100:.1f}% R²")
print(f"  Gap: {(93.1 - r2_test*100):.1f}pp (expected due to team vs individual)")

