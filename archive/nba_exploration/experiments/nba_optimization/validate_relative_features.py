"""
Validate Relative Features

Tests if RELATIVE/COMPARATIVE features predict better than absolute.

Theory: "Narratives are relative to the scene"
- Not "Team has 17 championships"
- But "Team has 17 vs league average 2.7" = +14.3 advantage

David vs Goliath = SIZE DIFFERENTIAL, not David's size alone.
"""

import sys
from pathlib import Path
import numpy as np
import json
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy import stats as scipy_stats

print("\n" + "="*70)
print("VALIDATING RELATIVE/COMPARATIVE FEATURES")
print("Testing if relativity improves prediction")
print("="*70)

# Load relative features
print("\nLoading relative feature dataset...")
with open('data/domains/nba_relative_features.json', 'r') as f:
    all_data = json.load(f)

print(f"✅ Loaded {len(all_data)} games with relative features")

# Temporal split
train_data = [g for g in all_data if g['season'] < '2022']
test_data = [g for g in all_data if g['season'] >= '2022']

print(f"\nTemporal split:")
print(f"  Train: {len(train_data)} games (2014-2021)")
print(f"  Test: {len(test_data)} games (2022-2024)")

# Extract relative features
def extract_relative_features(games):
    features = []
    for g in games:
        rf = g['relative_features']
        f = [
            rf['championships_relative'],
            rf['momentum_relative'],
            rf['win_pct_relative'],
            rf['syllables_relative'],
            rf['harshness_relative'],
            rf['memorability_relative'],
            rf['name_power_relative'],
            rf['harshness_moderated_relative'],
            rf['syllables_moderated_relative'],
            rf['memorability_moderated_relative'],
            rf['stakes_weight'],
            rf['rivalry_score'],
            rf['streak_length'] / 10.0,
            rf['is_powerhouse'],
            rf['has_aggressive_colors']
        ]
        features.append(f)
    return np.array(features)

X_train = extract_relative_features(train_data)
X_test = extract_relative_features(test_data)
y_train = np.array([g['won'] for g in train_data])
y_test = np.array([g['won'] for g in test_data])

print(f"\n✅ Relative features: {X_train.shape[1]} dimensions")

baseline = max(y_test.mean(), 1 - y_test.mean())
print(f"\nBaseline: {baseline:.1%}")

# Test relative model
print("\n" + "="*70)
print("TESTING RELATIVE FEATURES")
print("="*70)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

p_val = scipy_stats.binom_test(
    (pred == y_test).sum(),
    len(y_test),
    baseline,
    alternative='greater'
)

print(f"\nRELATIVE FEATURES MODEL:")
print(f"   Accuracy: {acc:.1%}")
print(f"   AUC: {auc:.3f}")
print(f"   p-value: {p_val:.6f} {'✅ SIGNIFICANT' if p_val < 0.05 else '❌ NOT SIGNIFICANT'}")

# Feature importance
print(f"\n📊 TOP RELATIVE FEATURE COEFFICIENTS:")
feature_names = [
    'championships_relative',
    'momentum_relative',
    'win_pct_relative',
    'syllables_relative',
    'harshness_relative',
    'memorability_relative',
    'name_power_relative',
    'harshness_moderated_rel',
    'syllables_moderated_rel',
    'memorability_moderated_rel',
    'stakes_weight',
    'rivalry_score',
    'streak_length',
    'is_powerhouse',
    'aggressive_colors'
]

coeffs = model.coef_[0]
sorted_indices = np.argsort(np.abs(coeffs))[::-1]

print(f"   {'Feature':<35} {'Coefficient':<12} {'Effect'}")
print(f"   {'-'*65}")
for idx in sorted_indices[:15]:
    name = feature_names[idx]
    coef = coeffs[idx]
    effect = "+" if coef > 0 else "-"
    print(f"   {name:<35} {coef:>10.3f}   {effect}")

# Compare to baselines
print("\n" + "="*70)
print("COMPARISON TO PRIOR RESULTS")
print("="*70)

print(f"\n{'Model':<40} {'Accuracy':<10} {'p-value':<12} {'Status'}")
print(f"{'-'*70}")
print(f"{'Baseline (random)':<40} {baseline:.1%}        —            —")
print(f"{'Generic narrative':<40} {'50.9%':<10} {'0.40':<12} ❌")
print(f"{'Rich nominative (absolute)':<40} {'54.4%':<10} {'0.0001':<12} ✅")
print(f"{'Maximum nominative (absolute)':<40} {'54.1%':<10} {'0.0002':<12} ✅")
print(f"{'RELATIVE FEATURES':<40} {f'{acc:.1%}':<10} {f'{p_val:.6f}':<12} {'✅' if p_val < 0.05 else '❌'}")

if acc > 54.4 and p_val < 0.05:
    improvement = (acc - 0.544) * 100
    print(f"\n🎉 RELATIVITY IMPROVES PREDICTION!")
    print(f"   Improvement: +{improvement:.1f} percentage points")
    print(f"   Your theory validated: Narratives are relative to scene!")
elif acc > 54.1:
    print(f"\n✅ Relative features work comparably to absolute")
    print(f"   No significant difference")
else:
    print(f"\n   Relative formulation doesn't improve prediction")
    print(f"   May need opponent-specific comparisons (not league average)")

# Save
with open('narrative_optimization/results/nba_relative_validation.json', 'w') as f:
    json.dump({
        'relative_accuracy': float(acc),
        'relative_p_value': float(p_val),
        'baseline': float(baseline),
        'comparison_to_absolute': {
            'absolute_rich': 0.544,
            'relative': float(acc),
            'difference': float(acc - 0.544)
        }
    }, f, indent=2)

print(f"\n✅ Results saved")

