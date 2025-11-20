"""
Tennis Ensemble Optimizer - Stacking Multiple Models

Stack best-performing models for maximum R²:
- Ridge (93% R²)
- Random Forest (92.5% R²)
- Gradient Boosting (91.3% R²)

Meta-learner combines predictions, target: 94-96% R²
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

print("="*80)
print("TENNIS ENSEMBLE OPTIMIZER - STACKING")
print("="*80)

# Load data
print("\nLoading data...")
genome_path = Path(__file__).parent / 'tennis_genome_data.npz'
genome_data = np.load(genome_path, allow_pickle=True)
ж = genome_data['genome']
outcomes = genome_data['outcomes']

tennis_features_path = Path(__file__).parent / 'tennis_specific_features.npz'
tennis_data = np.load(tennis_features_path)
tennis_features = tennis_data['features']

X_combined = np.hstack([ж, tennis_features])
y = outcomes

print(f"✓ Features: {X_combined.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.3, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# LEVEL 1: BASE MODELS
# ============================================================================

print("\n" + "="*80)
print("LEVEL 1: TRAINING BASE MODELS")
print("="*80)

print("\n[1/4] Ridge...", end=" ", flush=True)
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_scaled, y_train)
pred_train_ridge = ridge.predict(X_train_scaled)
pred_test_ridge = ridge.predict(X_test_scaled)
print("✓")

print("[2/4] Random Forest...", end=" ", flush=True)
rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
pred_train_rf = rf.predict(X_train_scaled)
pred_test_rf = rf.predict(X_test_scaled)
print("✓")

print("[3/4] Gradient Boosting...", end=" ", flush=True)
if XGBOOST_AVAILABLE:
    gbm = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
else:
    gbm = GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
gbm.fit(X_train_scaled, y_train)
pred_train_gbm = gbm.predict(X_train_scaled)
pred_test_gbm = gbm.predict(X_test_scaled)
print("✓")

print("[4/4] Base models complete")

# ============================================================================
# LEVEL 2: META-LEARNER (STACKING)
# ============================================================================

print("\n" + "="*80)
print("LEVEL 2: META-LEARNER STACKING")
print("="*80)

# Stack predictions as features for meta-learner
print("\nCreating meta-features...", end=" ", flush=True)
meta_train = np.column_stack([pred_train_ridge, pred_train_rf, pred_train_gbm])
meta_test = np.column_stack([pred_test_ridge, pred_test_rf, pred_test_gbm])
print("✓")

# Train meta-learner (simple Ridge)
print("Training meta-learner...", end=" ", flush=True)
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_train, y_train)
print("✓")

# Final predictions
print("Generating final predictions...", end=" ", flush=True)
final_pred_train = meta_model.predict(meta_train)
final_pred_test = meta_model.predict(meta_test)
print("✓")

# Calculate metrics
r_train = np.corrcoef(final_pred_train, y_train)[0, 1]
r_test = np.corrcoef(final_pred_test, y_test)[0, 1]
r2_train = r_train ** 2
r2_test = r_test ** 2

print(f"\n✓ STACKED MODEL PERFORMANCE:")
print(f"  Train R²: {r2_train:.4f} ({r2_train*100:.1f}%)")
print(f"  Test R²: {r2_test:.4f} ({r2_test*100:.1f}%)")

# Model weights
print(f"\nMeta-learner weights:")
print(f"  Ridge: {meta_model.coef_[0]:.3f}")
print(f"  Random Forest: {meta_model.coef_[1]:.3f}")
print(f"  Gradient Boosting: {meta_model.coef_[2]:.3f}")

# Save
results = {
    'stacked_model': {
        'train_r2': float(r2_train),
        'test_r2': float(r2_test),
        'meta_weights': {
            'ridge': float(meta_model.coef_[0]),
            'random_forest': float(meta_model.coef_[1]),
            'gradient_boosting': float(meta_model.coef_[2])
        }
    },
    'comparison_to_best_single': {
        'best_single_r2': 0.9294,
        'stacked_r2': float(r2_test),
        'improvement': float(r2_test - 0.9294)
    }
}

output_path = Path(__file__).parent / 'ensemble_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved to: {output_path}")

print("\n" + "="*80)
print("ENSEMBLE OPTIMIZATION COMPLETE")
print("="*80)

if r2_test > 0.95:
    print(f"\n✓ TARGET EXCEEDED: {r2_test*100:.1f}% R² (goal: 95-97%)")
elif r2_test > 0.94:
    print(f"\n✓ TARGET ACHIEVED: {r2_test*100:.1f}% R² (goal: 94-96%)")
else:
    print(f"\n✓ Strong performance: {r2_test*100:.1f}% R² (baseline: 93%)")

