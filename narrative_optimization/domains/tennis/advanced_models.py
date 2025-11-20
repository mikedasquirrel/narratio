"""
Tennis Advanced Models - Beyond Ridge Regression

Test multiple model architectures to push beyond 93% R²:
- XGBoost (gradient boosting)
- Random Forest (ensemble trees)  
- Neural Network (deep learning)
- Compare all to Ridge baseline

Target: 94-96% R² with best architecture
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Try XGBoost if available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available, using GradientBoosting instead")

print("="*80)
print("TENNIS ADVANCED MODELS - ARCHITECTURE COMPARISON")
print("="*80)

# Load data
print("\n[$(date +%H:%M:%S)] Loading data...")
genome_path = Path(__file__).parent / 'tennis_genome_data.npz'
genome_data = np.load(genome_path, allow_pickle=True)

ж = genome_data['genome']
outcomes = genome_data['outcomes']

# Load tennis-specific features
tennis_features_path = Path(__file__).parent / 'tennis_specific_features.npz'
tennis_data = np.load(tennis_features_path)
tennis_features = tennis_data['features']

# Combine: Transformer features + Tennis-specific
print(f"✓ Transformer features: {ж.shape}")
print(f"✓ Tennis-specific features: {tennis_features.shape}")

X_combined = np.hstack([ж, tennis_features])
y = outcomes

print(f"✓ Combined features: {X_combined.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.3, random_state=42
)

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled")

# ============================================================================
# TEST ALL MODELS
# ============================================================================

print("\n" + "="*80)
print("TESTING MODEL ARCHITECTURES")
print("="*80)

results = {}

# MODEL 1: Ridge Regression (Baseline)
print("\n[1/5] Ridge Regression (baseline)...", end=" ", flush=True)
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_scaled, y_train)

y_pred_train = ridge.predict(X_train_scaled)
y_pred_test = ridge.predict(X_test_scaled)

r_train = np.corrcoef(y_pred_train, y_train)[0, 1]
r_test = np.corrcoef(y_pred_test, y_test)[0, 1]
r2_train = r_train ** 2
r2_test = r_test ** 2

print(f"✓ Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")

results['ridge'] = {
    'train_r2': float(r2_train),
    'test_r2': float(r2_test),
    'train_r': float(r_train),
    'test_r': float(r_test)
}

# MODEL 2: Random Forest
print("[2/5] Random Forest...", end=" ", flush=True)
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

y_pred_train_rf = rf.predict(X_train_scaled)
y_pred_test_rf = rf.predict(X_test_scaled)

r_train_rf = np.corrcoef(y_pred_train_rf, y_train)[0, 1]
r_test_rf = np.corrcoef(y_pred_test_rf, y_test)[0, 1]
r2_train_rf = r_train_rf ** 2
r2_test_rf = r_test_rf ** 2

print(f"✓ Train R²: {r2_train_rf:.4f}, Test R²: {r2_test_rf:.4f}")

results['random_forest'] = {
    'train_r2': float(r2_train_rf),
    'test_r2': float(r2_test_rf),
    'train_r': float(r_train_rf),
    'test_r': float(r_test_rf)
}

# MODEL 3: Gradient Boosting / XGBoost
print("[3/5] Gradient Boosting...", end=" ", flush=True)

if XGBOOST_AVAILABLE:
    gbm = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
else:
    gbm = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

gbm.fit(X_train_scaled, y_train)

y_pred_train_gbm = gbm.predict(X_train_scaled)
y_pred_test_gbm = gbm.predict(X_test_scaled)

r_train_gbm = np.corrcoef(y_pred_train_gbm, y_train)[0, 1]
r_test_gbm = np.corrcoef(y_pred_test_gbm, y_test)[0, 1]
r2_train_gbm = r_train_gbm ** 2
r2_test_gbm = r_test_gbm ** 2

print(f"✓ Train R²: {r2_train_gbm:.4f}, Test R²: {r2_test_gbm:.4f}")

results['gradient_boosting'] = {
    'train_r2': float(r2_train_gbm),
    'test_r2': float(r2_test_gbm),
    'train_r': float(r_train_gbm),
    'test_r': float(r_test_gbm),
    'model_type': 'XGBoost' if XGBOOST_AVAILABLE else 'GradientBoosting'
}

# MODEL 4: Neural Network
print("[4/5] Neural Network...", end=" ", flush=True)
nn = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42,
    early_stopping=True
)
nn.fit(X_train_scaled, y_train)

y_pred_train_nn = nn.predict(X_train_scaled)
y_pred_test_nn = nn.predict(X_test_scaled)

r_train_nn = np.corrcoef(y_pred_train_nn, y_train)[0, 1]
r_test_nn = np.corrcoef(y_pred_test_nn, y_test)[0, 1]
r2_train_nn = r_train_nn ** 2
r2_test_nn = r_test_nn ** 2

print(f"✓ Train R²: {r2_train_nn:.4f}, Test R²: {r2_test_nn:.4f}")

results['neural_network'] = {
    'train_r2': float(r2_train_nn),
    'test_r2': float(r2_test_nn),
    'train_r': float(r_train_nn),
    'test_r': float(r_test_nn)
}

# MODEL 5: Simple Ensemble (Average)
print("[5/5] Simple Ensemble (average)...", end=" ", flush=True)

y_pred_train_ens = (y_pred_train + y_pred_train_rf + y_pred_train_gbm + y_pred_train_nn) / 4
y_pred_test_ens = (y_pred_test + y_pred_test_rf + y_pred_test_gbm + y_pred_test_nn) / 4

r_train_ens = np.corrcoef(y_pred_train_ens, y_train)[0, 1]
r_test_ens = np.corrcoef(y_pred_test_ens, y_test)[0, 1]
r2_train_ens = r_train_ens ** 2
r2_test_ens = r_test_ens ** 2

print(f"✓ Train R²: {r2_train_ens:.4f}, Test R²: {r2_test_ens:.4f}")

results['simple_ensemble'] = {
    'train_r2': float(r2_train_ens),
    'test_r2': float(r2_test_ens),
    'train_r': float(r_train_ens),
    'test_r': float(r_test_ens)
}

# ============================================================================
# COMPARISON & SELECTION
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

print(f"\n{'Model':<20} {'Train R²':<12} {'Test R²':<12} {'Generalization':<15}")
print("-" * 60)

for name, res in results.items():
    gen_gap = res['train_r2'] - res['test_r2']
    gen_quality = "Excellent" if gen_gap < 0.02 else "Good" if gen_gap < 0.05 else "Fair"
    print(f"{name:<20} {res['train_r2']:>10.4f}  {res['test_r2']:>10.4f}  {gen_quality:<15}")

# Find best model
best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
best_test_r2 = results[best_model_name]['test_r2']

print(f"\n✓ BEST MODEL: {best_model_name}")
print(f"  Test R²: {best_test_r2:.4f} ({best_test_r2*100:.1f}%)")

# Save results
output = {
    'models_tested': results,
    'best_model': best_model_name,
    'best_test_r2': float(best_test_r2),
    'improvement_over_ridge': float(best_test_r2 - results['ridge']['test_r2']),
    'feature_count': int(X_combined.shape[1])
}

output_path = Path(__file__).parent / 'advanced_models_comparison.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved to: {output_path}")

print("\n" + "="*80)
print("ADVANCED MODELS COMPLETE")
print("="*80)

if best_test_r2 > 0.94:
    print(f"\n✓ TARGET ACHIEVED: {best_test_r2*100:.1f}% R² (goal: 94-96%)")
else:
    print(f"\n✓ Strong performance: {best_test_r2*100:.1f}% R²")

