"""
NBA V6 - FIXED (No Data Leakage)

Removes:
- Outcome-revealing narrative text ("fell short", "narrowly lost")
- Points and plus_minus fields

Uses ONLY:
- Historical form (win%, L10)
- Player names (nominative features)
- Context (home/away, season position)

Author: Narrative Optimization Framework  
Date: November 13, 2025
"""

import json
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'narrative_optimization'))

print("="*80)
print("NBA V6 - FIXED (No Outcome Leakage)")
print("="*80)

from src.transformers import (
    NominativeAnalysisTransformer, PhoneticTransformer,
    UniversalNominativeTransformer
)

dataset_path = project_root / 'data' / 'domains' / 'nba_with_temporal_context.json'
with open(dataset_path) as f:
    all_games = json.load(f)

print(f"✓ Loaded {len(all_games):,} NBA games\n")

train_games = [g for g in all_games if g.get('season', '2020-21') <= '2020-21']
val_games = [g for g in all_games if '2021-22' <= g.get('season', '2020-21') <= '2022-23']
test_games = [g for g in all_games if g.get('season', '2020-21') >= '2023-24']

y_train = np.array([float(g.get('won', False)) for g in train_games])
y_val = np.array([float(g.get('won', False)) for g in val_games])
y_test = np.array([float(g.get('won', False)) for g in test_games])

print(f"Train: {len(train_games):,}, Val: {len(val_games):,}, Test: {len(test_games):,}\n")

# STRUCTURAL FEATURES (no outcomes!)
def extract_structural_features(game: dict) -> dict:
    features = {}
    temporal = game.get('temporal_context', {})
    
    features['season'] = float(game.get('season', '2020-21').split('-')[0]) / 2024.0
    features['win_pct'] = temporal.get('season_win_pct', 0.5)
    features['l10_win_pct'] = temporal.get('l10_win_pct', 0.5)
    
    form = temporal.get('form', 'neutral')
    features['form_hot'] = float(form == 'hot')
    features['form_cold'] = float(form == 'cold')
    
    games_played = temporal.get('games_played', 41)
    features['games_played'] = games_played / 82.0
    features['late_season'] = float(games_played >= 60)
    features['home_game'] = float(game.get('home_game', False))
    
    nom = game.get('nominative_coverage', {})
    features['nom_total'] = nom.get('total', 10) / 30.0
    features['nom_players'] = nom.get('players', 5) / 15.0
    
    features['momentum'] = features['l10_win_pct'] - features['win_pct']
    
    return features

print("Extracting structural features...")
train_structural = [extract_structural_features(g) for g in train_games]
val_structural = [extract_structural_features(g) for g in val_games]
test_structural = [extract_structural_features(g) for g in test_games]

structural_features = list(train_structural[0].keys())
X_train_struct = np.array([[s[k] for k in structural_features] for s in train_structural])
X_val_struct = np.array([[s[k] for k in structural_features] for s in val_structural])
X_test_struct = np.array([[s[k] for k in structural_features] for s in test_structural])

print(f"✓ Extracted {len(structural_features)} structural features\n")

# TEXT: ONLY player names (NO outcome-revealing narratives!)
def build_clean_narrative(game: dict) -> str:
    matchup = game.get('matchup', 'Unknown')
    team = game.get('team_name', 'Unknown')
    # Just team names and matchup - NO "fell short" or outcome text
    return f"{team} {matchup}"

train_narratives = [build_clean_narrative(g) for g in train_games]
val_narratives = [build_clean_narrative(g) for g in val_games]
test_narratives = [build_clean_narrative(g) for g in test_games]

print("Extracting nominative features (names only)...")
transformers = [
    ('nominative', NominativeAnalysisTransformer()),
    ('phonetic', PhoneticTransformer())
    # Skip UniversalNominativeTransformer - too slow on 8K+ samples
]

features_list = []
for name, transformer in transformers:
    print(f"[{name}]...", end=' ', flush=True)
    try:
        transformer.fit(train_narratives)
        train_feats = transformer.transform(train_narratives)
        val_feats = transformer.transform(val_narratives)
        test_feats = transformer.transform(test_narratives)
        
        if not isinstance(train_feats, np.ndarray):
            train_feats = np.array(train_feats)
        if train_feats.ndim == 1:
            train_feats = train_feats.reshape(-1, 1)
        if not isinstance(val_feats, np.ndarray):
            val_feats = np.array(val_feats)
        if val_feats.ndim == 1:
            val_feats = val_feats.reshape(-1, 1)
        if not isinstance(test_feats, np.ndarray):
            test_feats = np.array(test_feats)
        if test_feats.ndim == 1:
            test_feats = test_feats.reshape(-1, 1)
        
        features_list.append((train_feats, val_feats, test_feats))
        print(f"✓ ({train_feats.shape[1]})")
    except Exception as e:
        print(f"✗")

X_train_text = np.hstack([f[0] for f in features_list if f[0].ndim == 2])
X_val_text = np.hstack([f[1] for f in features_list if f[1].ndim == 2])
X_test_text = np.hstack([f[2] for f in features_list if f[2].ndim == 2])

print(f"\n✓ TEXT: {X_train_text.shape[1]} features\n")

# COMBINE
X_train = np.hstack([X_train_struct, X_train_text])
X_val = np.hstack([X_val_struct, X_val_text])
X_test = np.hstack([X_test_struct, X_test_text])

print(f"✓ TOTAL: {X_train.shape[1]} features\n")

# TRAIN
X_train = np.nan_to_num(X_train, nan=0.0)
X_val = np.nan_to_num(X_val, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Training...")
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ Best: R²={grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test_scaled)
y_pred_binary = (y_pred_test > 0.5).astype(int)

r2_test = r2_score(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_binary)

print(f"\n📊 TEST: R²={r2_test:.4f}, Acc={accuracy_test:.4f}")

# Feature importance
importance = best_model.feature_importances_
all_features = structural_features + [f"text_{i}" for i in range(X_train_text.shape[1])]
top_idx = np.argsort(importance)[-10:][::-1]

print("\n🔝 TOP 10 FEATURES:")
for idx in top_idx:
    if idx < len(all_features):
        print(f"  {all_features[idx]}: {importance[idx]:.4f}")

# SAVE
results_dir = Path(__file__).parent / 'results'
results_dir.mkdir(exist_ok=True)

import pickle
with open(results_dir / 'nba_v6_fixed.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'scaler': scaler}, f)

with open(results_dir / 'nba_v6_fixed_results.json', 'w') as f:
    json.dump({
        'r2_train': float(grid_search.best_score_),
        'r2_test': float(r2_test),
        'accuracy': float(accuracy_test),
        'n_features': int(X_train.shape[1])
    }, f, indent=2)

if r2_test > 0.2:
    print("\n✅ NBA V6 SUCCESS (no leakage)!")
else:
    print(f"\n⚠️  NBA V6 R²: {r2_test:.4f}")

print("="*80)

