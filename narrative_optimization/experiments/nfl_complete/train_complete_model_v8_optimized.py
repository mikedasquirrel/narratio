"""
NFL V8 - OPTIMIZED (Feature Selection + Hyperparameter Tuning)

**INSIGHT FROM V7**: Adding features hurt performance (R²: 0.88 → 0.86)
**WHY**: Betting odds already encode ALL public narrativity (EMH)
**STRATEGY**: Use ONLY high-signal features + aggressive hyperparameter search

Features to KEEP:
- Betting odds (97% importance)
- Context (primetime, division, playoffs)
- Record differentials (win%, streaks)

Features to DROP:
- Text features (1% importance = noise)
- Roster details (already in spread)
- Coaching (already in spread)

Target: R² = 0.89-0.91 via optimization, not feature engineering

Author: Narrative Optimization Framework  
Date: November 13, 2025
"""

import json
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, accuracy_score
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("NFL V8 - OPTIMIZED (Feature Selection + Tuning)")
print("="*80)
print("\n🏈 Strategy: LESS features + BETTER tuning")
print("📊 Target: R² = 0.89-0.91\n")

dataset_path = project_root / 'data' / 'domains' / 'nfl_with_temporal_context.json'
with open(dataset_path) as f:
    all_games = json.load(f)

print(f"✓ Loaded {len(all_games):,} NFL games\n")

train_games = [g for g in all_games if g.get('season', 2020) <= 2020]
val_games = [g for g in all_games if 2021 <= g.get('season', 2020) <= 2022]
test_games = [g for g in all_games if g.get('season', 2020) >= 2023]

y_train = np.array([float(g.get('home_won', False)) for g in train_games])
y_val = np.array([float(g.get('home_won', False)) for g in val_games])
y_test = np.array([float(g.get('home_won', False)) for g in test_games])

print(f"Train: {len(train_games):,}, Val: {len(val_games):,}, Test: {len(test_games):,}\n")

# ==============================================================================
# HIGH-SIGNAL FEATURES ONLY
# ==============================================================================

print("="*80)
print("EXTRACTING HIGH-SIGNAL FEATURES ONLY")
print("="*80)

def extract_optimized_features(game: dict) -> dict:
    """Only features with proven signal"""
    features = {}
    
    # ===== BETTING ODDS (97% importance) =====
    odds = game.get('betting_odds', {})
    
    spread = odds.get('spread', 0)
    features['spread'] = spread / 20.0
    
    # Implied probabilities (TOP 2 FEATURES!)
    ml_home = odds.get('moneyline_home', -110)
    ml_away = odds.get('moneyline_away', -110)
    
    if ml_home < 0:
        prob_home = abs(ml_home) / (abs(ml_home) + 100)
    else:
        prob_home = 100 / (ml_home + 100)
    
    if ml_away < 0:
        prob_away = abs(ml_away) / (abs(ml_away) + 100)
    else:
        prob_away = 100 / (ml_away + 100)
    
    features['implied_prob_home'] = prob_home  # TOP FEATURE!
    features['implied_prob_away'] = prob_away
    features['confidence_gap'] = abs(prob_home - prob_away)
    
    # Over/under
    ou = odds.get('over_under', 45.0)
    features['over_under'] = ou / 60.0
    
    # ===== CONTEXT (small but clean signal) =====
    context = game.get('context', {})
    temporal = game.get('temporal_context', {})
    
    week = context.get('week', 1)
    features['week'] = week / 18.0
    features['late_season'] = float(week >= 14)
    features['primetime'] = float(context.get('primetime', False))
    features['division_game'] = float(context.get('division_game', False))
    features['playoff_implications'] = float(temporal.get('playoff_implications', False))
    
    # ===== RECORD DIFFERENTIALS (learnable signal) =====
    home_context = temporal.get('home_context', {})
    away_context = temporal.get('away_context', {})
    
    home_record = home_context.get('season_record', '0-0')
    away_record = away_context.get('season_record', '0-0')
    
    try:
        h_w, h_l = map(int, home_record.split('-'))
        home_win_pct = h_w / max(h_w + h_l, 1)
    except:
        home_win_pct = 0.5
    
    try:
        a_w, a_l = map(int, away_record.split('-'))
        away_win_pct = a_w / max(a_w + a_l, 1)
    except:
        away_win_pct = 0.5
    
    features['win_pct_diff'] = home_win_pct - away_win_pct
    
    # Streaks
    home_streak = home_context.get('win_streak', 0)
    away_streak = away_context.get('win_streak', 0)
    features['streak_diff'] = (home_streak - away_streak) / 10.0
    
    # ===== INTERACTIONS (conditional effects) =====
    # These capture non-linear relationships
    features['spread_x_late'] = features['spread'] * features['late_season']
    features['prob_x_playoff'] = features['implied_prob_home'] * features['playoff_implications']
    features['spread_x_division'] = features['spread'] * features['division_game']
    
    return features

print("Extracting features...")
train_feats = [extract_optimized_features(g) for g in train_games]
val_feats = [extract_optimized_features(g) for g in val_games]
test_feats = [extract_optimized_features(g) for g in test_games]

feature_names = list(train_feats[0].keys())
X_train = np.array([[f[k] for k in feature_names] for f in train_feats])
X_val = np.array([[f[k] for k in feature_names] for f in val_feats])
X_test = np.array([[f[k] for k in feature_names] for f in test_feats])

print(f"✓ Extracted {len(feature_names)} optimized features")
print(f"  Features: {', '.join(feature_names)}\n")

# ==============================================================================
# SCALE
# ==============================================================================

X_train = np.nan_to_num(X_train, nan=0.0)
X_val = np.nan_to_num(X_val, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# HYPERPARAMETER OPTIMIZATION (AGGRESSIVE)
# ==============================================================================

print("="*80)
print("AGGRESSIVE HYPERPARAMETER SEARCH")
print("="*80)

print("\nPhase 1: Broad random search (100 candidates)...")

param_distributions = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.005, 0.145),  # 0.005 to 0.15
    'max_depth': randint(2, 10),
    'min_samples_split': randint(5, 30),
    'min_samples_leaf': randint(1, 10),
    'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
    'max_features': uniform(0.5, 0.5)  # 0.5 to 1.0
}

model = GradientBoostingRegressor(random_state=42)
random_search = RandomizedSearchCV(
    model, param_distributions, n_iter=100, cv=3, 
    scoring='r2', n_jobs=-1, random_state=42, verbose=1
)
random_search.fit(X_train_scaled, y_train)

print(f"\n✓ Best from random search: R²={random_search.best_score_:.4f}")
print(f"  Params: {random_search.best_params_}")

# Phase 2: Fine-tune around best from phase 1
best_params = random_search.best_params_

print("\nPhase 2: Grid search refinement around best params...")

param_grid = {
    'n_estimators': [
        max(100, best_params['n_estimators'] - 100),
        best_params['n_estimators'],
        best_params['n_estimators'] + 100
    ],
    'learning_rate': [
        max(0.005, best_params['learning_rate'] - 0.01),
        best_params['learning_rate'],
        min(0.15, best_params['learning_rate'] + 0.01)
    ],
    'max_depth': [
        max(2, best_params['max_depth'] - 1),
        best_params['max_depth'],
        min(10, best_params['max_depth'] + 1)
    ],
    'subsample': [best_params['subsample']],
    'max_features': [best_params['max_features']],
    'min_samples_split': [best_params['min_samples_split']],
    'min_samples_leaf': [best_params['min_samples_leaf']]
}

model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
)
grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ Best after refinement: R²={grid_search.best_score_:.4f}")
print(f"  Final params: {grid_search.best_params_}")

# ==============================================================================
# TEST EVALUATION
# ==============================================================================

print("\n" + "="*80)
print("NFL V8 TEST RESULTS")
print("="*80)

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test_scaled)
y_pred_binary = (y_pred_test > 0.5).astype(int)

r2_test = r2_score(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_binary)

print(f"\n📊 TEST PERFORMANCE:")
print(f"   R²: {r2_test:.4f}")
print(f"   Accuracy: {accuracy_test:.4f}")

print(f"\n📈 PROGRESSION:")
print(f"   V6: R²=0.8768, Acc=96.32%")
print(f"   V7: R²=0.8632, Acc=95.26% (WORSE - too many features)")
print(f"   V8: R²={r2_test:.4f}, Acc={accuracy_test:.4f}")

improvement = r2_test - 0.8768
print(f"\n   V8 vs V6: {improvement:+.4f} R²")

# Feature importance
importance = best_model.feature_importances_
print("\n🔝 FEATURE IMPORTANCE:")
for i, (name, imp) in enumerate(zip(feature_names, importance)):
    print(f"  {i+1:2d}. {name:30s}: {imp:.4f}")

# ==============================================================================
# SAVE
# ==============================================================================

results_dir = Path(__file__).parent / 'results'
results_dir.mkdir(exist_ok=True)

import pickle
with open(results_dir / 'nfl_v8_optimized.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': {
            'r2_train': grid_search.best_score_,
            'r2_test': r2_test,
            'accuracy': accuracy_test
        }
    }, f)

with open(results_dir / 'nfl_v8_results.json', 'w') as f:
    json.dump({
        'version': 'V8_optimized',
        'r2_train': float(grid_search.best_score_),
        'r2_test': float(r2_test),
        'accuracy': float(accuracy_test),
        'n_features': len(feature_names),
        'improvement_over_v6': float(improvement),
        'best_params': grid_search.best_params_
    }, f, indent=2)

print(f"\n✓ Saved to: {results_dir}/nfl_v8_optimized.pkl")

if r2_test >= 0.89:
    print("\n✅ NFL V8 SUCCESS! R² >= 0.89")
    print("   Feature selection + tuning worked!")
elif r2_test > 0.8768:
    print("\n✅ NFL V8 IMPROVEMENT!")
    print(f"   Better than V6 by {improvement:.4f}")
else:
    print("\n📊 NFL V8: Peak performance likely at V6")
    print("   Betting odds capture maximum extractable signal")

print("="*80)

