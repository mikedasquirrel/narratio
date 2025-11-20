"""
Tennis V7 - MAXIMUM PERFORMANCE

Tennis V6 achieved R²=1.0, but let's validate with MORE features:
1. BETTING ODDS: player1/2 odds, implied probabilities (like NFL!)
2. MATCH STATS: Aces, double faults differentials
3. PHYSICAL: Height, handedness matchup advantages
4. OFFICIALS: Chair umpire narrativity
5. SURFACE SPECIALIST: Domain-specific advantages

If V6's R²=1.0 holds with 300+ features → true perfect prediction
If it drops → V6 was overfitting on limited features

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
print("TENNIS V7 - MAXIMUM NARRATIVITY EXTRACTION")
print("="*80)
print("\n🎾 Adding betting odds + match stats + player attributes")
print("📊 Validating V6's R²=1.0 with 300+ features\n")

from src.transformers import NominativeAnalysisTransformer, PhoneticTransformer

dataset_path = project_root / 'data' / 'domains' / 'tennis_with_temporal_context.json'

print("Loading tennis data...")
with open(dataset_path) as f:
    all_matches = json.load(f)

print(f"✓ Loaded {len(all_matches):,} matches\n")

train_matches = [m for m in all_matches if m.get('year', 2020) <= 2020]
val_matches = [m for m in all_matches if 2021 <= m.get('year', 2020) <= 2022]
test_matches = [m for m in all_matches if m.get('year', 2020) >= 2023]

y_train = np.array([float(m.get('player1_won', False)) for m in train_matches])
y_val = np.array([float(m.get('player1_won', False)) for m in val_matches])
y_test = np.array([float(m.get('player1_won', False)) for m in test_matches])

print(f"Train: {len(train_matches):,}, Val: {len(val_matches):,}, Test: {len(test_matches):,}\n")

# ==============================================================================
# FEATURE 1: BETTING ODDS (CRITICAL - LIKE NFL!)
# ==============================================================================

print("="*80)
print("FEATURE 1: BETTING ODDS (Quantified Narrative)")
print("="*80)

def extract_betting_features(match: dict) -> dict:
    """Extract betting odds - market's narrative quantification"""
    features = {}
    
    odds = match.get('betting_odds', {})
    
    # Direct odds
    p1_odds = odds.get('player1_odds', 2.0)
    p2_odds = odds.get('player2_odds', 2.0)
    
    features['p1_odds'] = p1_odds / 10.0  # Normalize
    features['p2_odds'] = p2_odds / 10.0
    features['odds_diff'] = (p2_odds - p1_odds) / 10.0  # Positive = p1 favored
    
    # Implied probabilities (LIKE NFL!)
    impl_p1 = odds.get('implied_prob_p1', 0.5)
    impl_p2 = odds.get('implied_prob_p2', 0.5)
    
    features['implied_prob_p1'] = impl_p1
    features['implied_prob_p2'] = impl_p2
    features['confidence_gap'] = abs(impl_p1 - impl_p2)
    
    # Favorite/underdog flags
    features['p1_favorite'] = float(odds.get('favorite', '') == 'player1')
    features['p2_favorite'] = float(odds.get('favorite', '') == 'player2')
    features['upset_potential'] = float(odds.get('upset', False))
    
    # Heavy favorite (>80% implied prob)
    features['heavy_favorite'] = float(max(impl_p1, impl_p2) > 0.8)
    
    return features

print("Extracting betting odds...")
train_betting = [extract_betting_features(m) for m in train_matches]
val_betting = [extract_betting_features(m) for m in val_matches]
test_betting = [extract_betting_features(m) for m in test_matches]

betting_features = list(train_betting[0].keys())
X_train_betting = np.array([[b[k] for k in betting_features] for b in train_betting])
X_val_betting = np.array([[b[k] for k in betting_features] for b in val_betting])
X_test_betting = np.array([[b[k] for k in betting_features] for b in test_betting])

print(f"✓ Extracted {len(betting_features)} betting features\n")

# ==============================================================================
# FEATURE 2: RANKINGS & H2H (V6 BASELINE)
# ==============================================================================

print("="*80)
print("FEATURE 2: RANKINGS & H2H (V6 Baseline)")
print("="*80)

def extract_ranking_features(match: dict) -> dict:
    """Rankings and head-to-head (V6 baseline)"""
    features = {}
    
    p1 = match.get('player1', {})
    p2 = match.get('player2', {})
    h2h = match.get('head_to_head', {})
    
    p1_rank = p1.get('ranking') if p1.get('ranking') is not None else 100
    p2_rank = p2.get('ranking') if p2.get('ranking') is not None else 100
    
    features['p1_ranking'] = p1_rank / 200.0
    features['p2_ranking'] = p2_rank / 200.0
    features['ranking_diff'] = (p2_rank - p1_rank) / 200.0  # KEY FEATURE
    
    p1_points = p1.get('rank_points') if p1.get('rank_points') is not None else 1000
    p2_points = p2.get('rank_points') if p2.get('rank_points') is not None else 1000
    
    features['p1_points'] = p1_points / 10000.0
    features['p2_points'] = p2_points / 10000.0
    features['points_diff'] = (p1_points - p2_points) / 10000.0
    
    # H2H
    total_h2h = h2h.get('total_matches', 0)
    features['h2h_exists'] = float(total_h2h > 0)
    features['h2h_p1_win_pct'] = (h2h.get('player1_wins', 0) / total_h2h) if total_h2h > 0 else 0.5
    
    # Surface H2H
    surface_record = h2h.get('surface_record', {})
    surface_total = surface_record.get('p1', 0) + surface_record.get('p2', 0)
    features['h2h_surface_p1_win_pct'] = (surface_record.get('p1', 0) / surface_total) if surface_total > 0 else 0.5
    
    return features

print("Extracting ranking features...")
train_ranking = [extract_ranking_features(m) for m in train_matches]
val_ranking = [extract_ranking_features(m) for m in val_matches]
test_ranking = [extract_ranking_features(m) for m in test_matches]

ranking_features = list(train_ranking[0].keys())
X_train_ranking = np.array([[r[k] for k in ranking_features] for r in train_ranking])
X_val_ranking = np.array([[r[k] for k in ranking_features] for r in val_ranking])
X_test_ranking = np.array([[r[k] for k in ranking_features] for r in test_ranking])

print(f"✓ Extracted {len(ranking_features)} ranking features\n")

# ==============================================================================
# FEATURE 3: PHYSICAL ATTRIBUTES & MATCH STATS
# ==============================================================================

print("="*80)
print("FEATURE 3: PHYSICAL ATTRIBUTES & MATCH STATS")
print("="*80)

def extract_physical_and_stats(match: dict) -> dict:
    """Height, hand, age, match stats"""
    features = {}
    
    p1 = match.get('player1', {})
    p2 = match.get('player2', {})
    stats = match.get('match_stats', {})
    
    # Age (experience)
    p1_age = p1.get('age') if p1.get('age') is not None else 25
    p2_age = p2.get('age') if p2.get('age') is not None else 25
    features['p1_age'] = p1_age / 40.0
    features['p2_age'] = p2_age / 40.0
    features['age_diff'] = (p1_age - p2_age) / 20.0
    
    # Height (advantage in serve)
    p1_height = p1.get('height_cm') if p1.get('height_cm') is not None else 180
    p2_height = p2.get('height_cm') if p2.get('height_cm') is not None else 180
    features['p1_height'] = p1_height / 210.0
    features['p2_height'] = p2_height / 210.0
    features['height_diff'] = (p1_height - p2_height) / 30.0
    
    # Handedness (lefty vs righty narrativity)
    p1_hand = p1.get('hand', 'R')
    p2_hand = p2.get('hand', 'R')
    features['p1_lefty'] = float(p1_hand == 'L')
    features['p2_lefty'] = float(p2_hand == 'L')
    features['both_lefty'] = features['p1_lefty'] * features['p2_lefty']
    features['hand_mismatch'] = float(p1_hand != p2_hand)
    
    # Match stats - DON'T USE (these are outcomes, not pre-match data!)
    # Aces/DFs only known AFTER the match
    # Skip to avoid data leakage
    
    return features

print("Extracting physical & stats features...")
train_physical = [extract_physical_and_stats(m) for m in train_matches]
val_physical = [extract_physical_and_stats(m) for m in val_matches]
test_physical = [extract_physical_and_stats(m) for m in test_matches]

physical_features = list(train_physical[0].keys())
X_train_physical = np.array([[p[k] for k in physical_features] for p in train_physical])
X_val_physical = np.array([[p[k] for k in physical_features] for p in val_physical])
X_test_physical = np.array([[p[k] for k in physical_features] for p in test_physical])

print(f"✓ Extracted {len(physical_features)} physical/stats features\n")

# ==============================================================================
# FEATURE 4: TOURNAMENT & SURFACE CONTEXT
# ==============================================================================

print("="*80)
print("FEATURE 4: TOURNAMENT & SURFACE CONTEXT")
print("="*80)

def extract_context_features(match: dict) -> dict:
    """Tournament tier, surface, round importance"""
    features = {}
    
    context = match.get('context', {})
    
    # Surface
    surface = match.get('surface', 'hard')
    features['surface_hard'] = float(surface == 'hard')
    features['surface_clay'] = float(surface == 'clay')
    features['surface_grass'] = float(surface == 'grass')
    
    # Tournament level
    level = match.get('level', 'atp_250')
    features['grand_slam'] = float(level == 'grand_slam')
    features['masters'] = float(level == 'masters_1000')
    features['atp_500'] = float(level == 'atp_500')
    
    # Round
    round_name = match.get('round', 'R32')
    features['final'] = float(round_name == 'F')
    features['semifinal'] = float(round_name == 'SF')
    features['quarterfinal'] = float(round_name == 'QF')
    
    # Context flags
    features['top_10_match'] = float(context.get('top_10_match', False))
    features['rivalry'] = float(context.get('rivalry', False))
    features['ranking_upset_potential'] = float(context.get('ranking_upset', False))
    
    # Surface specialist advantage
    specialist = context.get('surface_specialist_advantage')
    features['p1_surface_specialist'] = float(specialist == 'player1')
    features['p2_surface_specialist'] = float(specialist == 'player2')
    
    # Seeding
    p1 = match.get('player1', {})
    p2 = match.get('player2', {})
    features['p1_seeded'] = float(p1.get('seed') is not None)
    features['p2_seeded'] = float(p2.get('seed') is not None)
    
    return features

print("Extracting context features...")
train_context = [extract_context_features(m) for m in train_matches]
val_context = [extract_context_features(m) for m in val_matches]
test_context = [extract_context_features(m) for m in test_matches]

context_features = list(train_context[0].keys())
X_train_context = np.array([[c[k] for k in context_features] for c in train_context])
X_val_context = np.array([[c[k] for k in context_features] for c in val_context])
X_test_context = np.array([[c[k] for k in context_features] for c in test_context])

print(f"✓ Extracted {len(context_features)} context features\n")

# ==============================================================================
# FEATURE 5: OFFICIALS NARRATIVITY
# ==============================================================================

print("="*80)
print("FEATURE 5: OFFICIALS NARRATIVITY")
print("="*80)

def extract_officials_features(match: dict) -> dict:
    """Chair umpire and line judges"""
    features = {}
    
    officials = match.get('officials', {})
    chair = officials.get('chair_umpire', {})
    
    # Chair umpire attributes
    umpire_style = chair.get('style', 'neutral')
    features['umpire_strict'] = float(umpire_style == 'strict')
    features['umpire_lenient'] = float(umpire_style == 'lenient')
    
    # Line judges count (more officials = higher profile match?)
    line_judges = officials.get('line_judges', [])
    features['line_judge_count'] = len(line_judges) / 10.0
    
    # Umpire country (home advantage?)
    umpire_country = chair.get('country', 'UNK')
    p1_country = match.get('player1', {}).get('country', 'UNK')
    p2_country = match.get('player2', {}).get('country', 'UNK')
    
    features['umpire_p1_country'] = float(umpire_country == p1_country)
    features['umpire_p2_country'] = float(umpire_country == p2_country)
    
    return features

print("Extracting officials features...")
train_officials = [extract_officials_features(m) for m in train_matches]
val_officials = [extract_officials_features(m) for m in val_matches]
test_officials = [extract_officials_features(m) for m in test_matches]

officials_features = list(train_officials[0].keys())
X_train_officials = np.array([[o[k] for k in officials_features] for o in train_officials])
X_val_officials = np.array([[o[k] for k in officials_features] for o in val_officials])
X_test_officials = np.array([[o[k] for k in officials_features] for o in test_officials])

print(f"✓ Extracted {len(officials_features)} officials features\n")

# ==============================================================================
# FEATURE 6: NOMINATIVE (Player Names)
# ==============================================================================

print("="*80)
print("FEATURE 6: NOMINATIVE (Player Names)")
print("="*80)

def build_narrative(match: dict) -> str:
    p1 = match.get('player1', {}).get('name', 'Unknown')
    p2 = match.get('player2', {}).get('name', 'Unknown')
    tournament = match.get('tournament', 'Unknown')
    return f"{p1} versus {p2} at {tournament}"

print("Building narratives...")
train_narratives = [build_narrative(m) for m in train_matches]
val_narratives = [build_narrative(m) for m in val_matches]
test_narratives = [build_narrative(m) for m in test_matches]

print("Extracting nominative features...")
transformers = [
    ('nominative', NominativeAnalysisTransformer()),
    ('phonetic', PhoneticTransformer())
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

# ==============================================================================
# COMBINE ALL
# ==============================================================================

print("="*80)
print("COMBINING ALL FEATURES")
print("="*80)

X_train = np.hstack([X_train_betting, X_train_ranking, X_train_physical,
                     X_train_context, X_train_officials, X_train_text])
X_val = np.hstack([X_val_betting, X_val_ranking, X_val_physical,
                   X_val_context, X_val_officials, X_val_text])
X_test = np.hstack([X_test_betting, X_test_ranking, X_test_physical,
                    X_test_context, X_test_officials, X_test_text])

print(f"✓ TOTAL FEATURES: {X_train.shape[1]}")
print(f"  Betting: {X_train_betting.shape[1]}")
print(f"  Rankings: {X_train_ranking.shape[1]}")
print(f"  Physical/Stats: {X_train_physical.shape[1]}")
print(f"  Context: {X_train_context.shape[1]}")
print(f"  Officials: {X_train_officials.shape[1]}")
print(f"  Text: {X_train_text.shape[1]}\n")

# ==============================================================================
# TRAIN
# ==============================================================================

print("="*80)
print("TRAINING TENNIS V7")
print("="*80)

X_train = np.nan_to_num(X_train, nan=0.0)
X_val = np.nan_to_num(X_val, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Running grid search...")
param_grid = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ Best Train: R²={grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test_scaled)
y_pred_binary = (y_pred_test > 0.5).astype(int)

r2_test = r2_score(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_binary)

print(f"\n📊 TEST: R²={r2_test:.4f}, Acc={accuracy_test:.4f}")

# Feature importance
all_feature_names = (betting_features + ranking_features + physical_features + 
                     context_features + officials_features + 
                     [f"text_{i}" for i in range(X_train_text.shape[1])])

importance = best_model.feature_importances_
top_idx = np.argsort(importance)[-15:][::-1]

print("\n🔝 TOP 15 FEATURES:")
for idx in top_idx:
    if idx < len(all_feature_names):
        print(f"  {all_feature_names[idx]}: {importance[idx]:.4f}")

# SAVE
results_dir = Path(__file__).parent / 'results'
results_dir.mkdir(exist_ok=True)

import pickle
with open(results_dir / 'tennis_v7_maximum.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_names': all_feature_names,
        'metrics': {'r2_train': grid_search.best_score_, 'r2_test': r2_test, 'accuracy': accuracy_test}
    }, f)

with open(results_dir / 'tennis_v7_results.json', 'w') as f:
    json.dump({
        'version': 'V7',
        'r2_train': float(grid_search.best_score_),
        'r2_test': float(r2_test),
        'accuracy': float(accuracy_test),
        'n_features': int(X_train.shape[1])
    }, f, indent=2)

print(f"\n✓ Saved to: {results_dir}/tennis_v7_maximum.pkl")

if r2_test >= 1.0:
    print("\n✅ TENNIS V7: PERFECT PREDICTION VALIDATED!")
    print("   With 300+ features, still R²=1.0 → Rankings truly deterministic")
elif r2_test >= 0.95:
    print("\n✅ TENNIS V7: EXCELLENT!")
else:
    print(f"\n⚠️  Tennis V7: R²={r2_test:.4f}")

print("="*80)

