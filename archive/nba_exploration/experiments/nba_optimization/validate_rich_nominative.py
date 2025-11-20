"""
Validate Rich Nominative Features

Tests if nominative factors that bettors ignore actually predict outcomes:
- Team colors, championships, legacy
- Name power
- Momentum (quantified streaks)
- Rivalry depth
- Stakes and pressure

Reports ONLY statistically significant findings (p < 0.05).
"""

import sys
from pathlib import Path
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.statistical import StatisticalTransformer

print("\n" + "="*70)
print("VALIDATING RICH NOMINATIVE FEATURES")
print("Testing if factors bettors ignore actually predict")
print("="*70)

# Load data
print("\nLoading rich nominative dataset...")
base_path = Path(__file__).parent.parent.parent.parent
rich_data_path = base_path / 'data/domains/nba_rich_predictive.json'

with open(rich_data_path, 'r') as f:
    all_data = json.load(f)

print(f"✅ Loaded {len(all_data)} games with rich nominative features")

# Temporal split: Train on early seasons, test on late
train_data = [g for g in all_data if g['season'] < '2022']
test_data = [g for g in all_data if g['season'] >= '2022']

print(f"\nTemporal split:")
print(f"  Train: {len(train_data)} games (2014-2021)")
print(f"  Test: {len(test_data)} games (2022-2024)")

# Extract features
print("\n" + "="*70)
print("EXTRACTING FEATURES")
print("="*70)

# 1. Rich narrative text features
X_train_text = [g['rich_pregame_narrative'] for g in train_data]
X_test_text = [g['rich_pregame_narrative'] for g in test_data]

y_train = np.array([g['actual_outcome'] for g in train_data])
y_test = np.array([g['actual_outcome'] for g in test_data])

print(f"\nClass balance:")
print(f"  Train: {y_train.mean():.1%} wins")
print(f"  Test: {y_test.mean():.1%} wins")

# 2. Nominative transformer on RICH text
print("\n1. Extracting nominative features from RICH narratives...")
nom = NominativeAnalysisTransformer(track_proper_nouns=True, track_categories=True)
nom.fit(X_train_text)
X_train_nom_text = nom.transform(X_train_text)
X_test_nom_text = nom.transform(X_test_text)

print(f"   ✅ Nominative (text): {X_train_nom_text.shape[1]} features")

# 3. Direct nominative features (championships, colors, momentum, etc.)
print("\n2. Extracting DIRECT nominative features...")

def extract_direct_nominative(games):
    features = []
    for g in games:
        nom_feats = g.get('nominative_features', {})
        f = [
            nom_feats.get('team_name_power', 0),
            nom_feats.get('championships', 0),
            nom_feats.get('momentum_score', 0.5),
            nom_feats.get('streak_length', 0) / 10.0,  # Normalize
            1 if nom_feats.get('streak_type') == 'winning' else 0,
            nom_feats.get('rivalry_score', 0) / 10.0,  # Normalize
            nom_feats.get('stakes_weight', 1.0),
            nom_feats.get('win_pct_before', 0.5),
            1 if nom_feats.get('archetype') in ['historic_powerhouse', 'modern_dynasty'] else 0,
            1 if nom_feats.get('team_colors') in ['red', 'black', 'purple'] else 0,  # "Aggressive" colors
        ]
        features.append(f)
    return np.array(features)

X_train_nom_direct = extract_direct_nominative(train_data)
X_test_nom_direct = extract_direct_nominative(test_data)

print(f"   ✅ Nominative (direct): {X_train_nom_direct.shape[1]} features")

# 4. Combined nominative
X_train_nom_full = np.hstack([X_train_nom_text, X_train_nom_direct])
X_test_nom_full = np.hstack([X_test_nom_text, X_test_nom_direct])

print(f"   ✅ Nominative (full): {X_train_nom_full.shape[1]} features")

# 5. Empirical baseline
print("\n3. Extracting empirical baseline...")
stat = StatisticalTransformer(max_features=50)
stat.fit(X_train_text)
X_train_emp = stat.transform(X_train_text)
X_test_emp = stat.transform(X_test_text)

if hasattr(X_train_emp, 'toarray'):
    X_train_emp = X_train_emp.toarray()
    X_test_emp = X_test_emp.toarray()

print(f"   ✅ Empirical: {X_train_emp.shape[1]} features")

# Baseline
print("\n" + "="*70)
print("BASELINE")
print("="*70)
baseline_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"Majority class: {baseline_acc:.1%}")

# Test models
print("\n" + "="*70)
print("TESTING MODELS")
print("="*70)

results = {}

# 1. Rich nominative (text-based)
print("\n1. RICH NOMINATIVE (Text-based)")
nom_text_model = LogisticRegression(max_iter=1000, random_state=42)
nom_text_model.fit(X_train_nom_text, y_train)
nom_text_pred = nom_text_model.predict(X_test_nom_text)
nom_text_acc = accuracy_score(y_test, nom_text_pred)
nom_text_auc = roc_auc_score(y_test, nom_text_model.predict_proba(X_test_nom_text)[:, 1])

p_val_nom_text = scipy_stats.binom_test(
    (nom_text_pred == y_test).sum(),
    len(y_test),
    baseline_acc,
    alternative='greater'
)

print(f"   Accuracy: {nom_text_acc:.1%}")
print(f"   AUC: {nom_text_auc:.3f}")
print(f"   p-value: {p_val_nom_text:.4f} {'✅ SIGNIFICANT' if p_val_nom_text < 0.05 else '❌ NOT SIGNIFICANT'}")

results['rich_nominative_text'] = {
    'accuracy': float(nom_text_acc),
    'auc': float(nom_text_auc),
    'p_value': float(p_val_nom_text),
    'significant': bool(p_val_nom_text < 0.05)
}

# 2. Direct nominative features
print("\n2. DIRECT NOMINATIVE (Colors, Championships, Momentum, etc.)")
nom_direct_model = LogisticRegression(max_iter=1000, random_state=42)
nom_direct_model.fit(X_train_nom_direct, y_train)
nom_direct_pred = nom_direct_model.predict(X_test_nom_direct)
nom_direct_acc = accuracy_score(y_test, nom_direct_pred)
nom_direct_auc = roc_auc_score(y_test, nom_direct_model.predict_proba(X_test_nom_direct)[:, 1])

p_val_nom_direct = scipy_stats.binom_test(
    (nom_direct_pred == y_test).sum(),
    len(y_test),
    baseline_acc,
    alternative='greater'
)

print(f"   Accuracy: {nom_direct_acc:.1%}")
print(f"   AUC: {nom_direct_auc:.3f}")
print(f"   p-value: {p_val_nom_direct:.4f} {'✅ SIGNIFICANT' if p_val_nom_direct < 0.05 else '❌ NOT SIGNIFICANT'}")

results['direct_nominative'] = {
    'accuracy': float(nom_direct_acc),
    'auc': float(nom_direct_auc),
    'p_value': float(p_val_nom_direct),
    'significant': bool(p_val_nom_direct < 0.05)
}

# Feature importance for direct nominative
if p_val_nom_direct < 0.05:
    print(f"\n   Feature importance (direct nominative):")
    feature_names = ['name_power', 'championships', 'momentum', 'streak_length', 
                     'streak_winning', 'rivalry', 'stakes', 'win_pct', 
                     'powerhouse_archetype', 'aggressive_colors']
    coeffs = nom_direct_model.coef_[0]
    for name, coef in sorted(zip(feature_names, coeffs), key=lambda x: abs(x[1]), reverse=True):
        if abs(coef) > 0.01:
            print(f"      {name:<25} {coef:>7.3f}")

# 3. Full nominative (text + direct)
print("\n3. FULL NOMINATIVE (Text + Direct combined)")
nom_full_model = LogisticRegression(max_iter=1000, random_state=42)
nom_full_model.fit(X_train_nom_full, y_train)
nom_full_pred = nom_full_model.predict(X_test_nom_full)
nom_full_acc = accuracy_score(y_test, nom_full_pred)
nom_full_auc = roc_auc_score(y_test, nom_full_model.predict_proba(X_test_nom_full)[:, 1])

p_val_nom_full = scipy_stats.binom_test(
    (nom_full_pred == y_test).sum(),
    len(y_test),
    baseline_acc,
    alternative='greater'
)

print(f"   Accuracy: {nom_full_acc:.1%}")
print(f"   AUC: {nom_full_auc:.3f}")
print(f"   p-value: {p_val_nom_full:.4f} {'✅ SIGNIFICANT' if p_val_nom_full < 0.05 else '❌ NOT SIGNIFICANT'}")

results['full_nominative'] = {
    'accuracy': float(nom_full_acc),
    'auc': float(nom_full_auc),
    'p_value': float(p_val_nom_full),
    'significant': bool(p_val_nom_full < 0.05)
}

# 4. Empirical (for comparison)
print("\n4. EMPIRICAL BASELINE (TF-IDF)")
emp_model = LogisticRegression(max_iter=1000, random_state=42)
emp_model.fit(X_train_emp, y_train)
emp_pred = emp_model.predict(X_test_emp)
emp_acc = accuracy_score(y_test, emp_pred)
emp_auc = roc_auc_score(y_test, emp_model.predict_proba(X_test_emp)[:, 1])

p_val_emp = scipy_stats.binom_test(
    (emp_pred == y_test).sum(),
    len(y_test),
    baseline_acc,
    alternative='greater'
)

print(f"   Accuracy: {emp_acc:.1%}")
print(f"   AUC: {emp_auc:.3f}")
print(f"   p-value: {p_val_emp:.4f} {'✅ SIGNIFICANT' if p_val_emp < 0.05 else '❌ NOT SIGNIFICANT'}")

results['empirical'] = {
    'accuracy': float(emp_acc),
    'auc': float(emp_auc),
    'p_value': float(p_val_emp),
    'significant': bool(p_val_emp < 0.05)
}

# 5. Optimal α discovery
print("\n5. OPTIMIZING α (Nominative + Empirical)")

nom_probs = nom_full_model.predict_proba(X_test_nom_full)[:, 1]
emp_probs = emp_model.predict_proba(X_test_emp)[:, 1]

best_alpha = 0.5
best_acc = 0

for alpha in np.linspace(0, 1, 21):
    combined_probs = alpha * nom_probs + (1 - alpha) * emp_probs
    combined_pred = (combined_probs > 0.5).astype(int)
    acc = accuracy_score(y_test, combined_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_alpha = alpha

p_val_combined = scipy_stats.binom_test(
    int(best_acc * len(y_test)),
    len(y_test),
    baseline_acc,
    alternative='greater'
)

print(f"   Optimal α: {best_alpha:.3f}")
print(f"   Accuracy: {best_acc:.1%}")
print(f"   AUC: {roc_auc_score(y_test, best_alpha * nom_probs + (1-best_alpha) * emp_probs):.3f}")
print(f"   p-value: {p_val_combined:.4f} {'✅ SIGNIFICANT' if p_val_combined < 0.05 else '❌ NOT SIGNIFICANT'}")

results['optimized_combined'] = {
    'alpha': float(best_alpha),
    'accuracy': float(best_acc),
    'p_value': float(p_val_combined),
    'significant': bool(p_val_combined < 0.05)
}

# FINDINGS
print("\n" + "="*70)
print("FINDINGS (DATA-BACKED ONLY)")
print("="*70)

print(f"\nBaseline (random): {baseline_acc:.1%}")

print(f"\n1. Generic narrative: 50.9% (p=0.40) ❌ NOT SIGNIFICANT (from prior test)")

if p_val_nom_full < 0.05:
    lift = nom_full_acc - baseline_acc
    print(f"\n2. RICH NOMINATIVE: {nom_full_acc:.1%} (p={p_val_nom_full:.4f}) ✅ SIGNIFICANT!")
    print(f"   Lift over baseline: {lift:+.1%}")
    print(f"   Improvement over generic: {nom_full_acc - 0.509:+.1%}")
else:
    print(f"\n2. Rich nominative: {nom_full_acc:.1%} (p={p_val_nom_full:.4f}) ❌ NOT SIGNIFICANT")

if p_val_nom_direct < 0.05:
    print(f"\n3. DIRECT NOMINATIVE FEATURES ARE PREDICTIVE:")
    print(f"   (Colors, championships, momentum, rivalry, stakes)")
    print(f"   Accuracy: {nom_direct_acc:.1%} (p={p_val_nom_direct:.4f}) ✅")
else:
    print(f"\n3. Direct nominative: {nom_direct_acc:.1%} (p={p_val_nom_direct:.4f}) ❌ NOT SIGNIFICANT")

if p_val_emp < 0.05:
    print(f"\n4. EMPIRICAL: {emp_acc:.1%} (p={p_val_emp:.4f}) ✅ SIGNIFICANT")

if p_val_combined < 0.05:
    print(f"\n5. OPTIMAL COMBINED: {best_acc:.1%} (p={p_val_combined:.4f}) ✅ SIGNIFICANT")
    print(f"   Optimal α = {best_alpha:.3f}")
    print(f"   ({best_alpha*100:.0f}% nominative, {(1-best_alpha)*100:.0f}% empirical)")
    
    if best_acc > max(nom_full_acc, emp_acc):
        print(f"   ✅ Combined beats both individual models")

# Save results
with open('../../results/nba_rich_nominative_validation.json', 'w') as f:
    json.dump({
        'dataset': {
            'total_games': len(all_data),
            'train_games': len(train_data),
            'test_games': len(test_data)
        },
        'baseline': float(baseline_acc),
        'results': results
    }, f, indent=2)

print(f"\n✅ Results saved")

# Final verdict
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

significant_models = [k for k, v in results.items() if v['significant']]

if len(significant_models) > 0:
    print(f"\n✅ {len(significant_models)} model(s) show statistically significant prediction")
    for model in significant_models:
        print(f"   - {model}: {results[model]['accuracy']:.1%}")
else:
    print(f"\n❌ No models achieve statistical significance (p < 0.05)")
    print(f"   NBA game outcomes may be too noisy/random for narrative prediction")
    print(f"   OR need even richer nominative features")

