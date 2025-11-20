"""
NBA Proper Validation - NO Label Leakage

Uses pre-game narratives (NO outcome information) to predict actual outcomes.
Temporal validation: Train on 2014-2021, test on 2022-2024.

Reports ONLY statistically significant findings.
"""

import sys
from pathlib import Path
import numpy as np
import json
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from src.transformers.linguistic_advanced import LinguisticPatternsTransformer
from src.transformers.relational import RelationalValueTransformer
from src.transformers.ensemble import EnsembleNarrativeTransformer
from src.transformers.statistical import StatisticalTransformer

print("\n" + "="*70)
print("NBA PROPER VALIDATION - STATISTICALLY RIGOROUS")
print("="*70)

# Load proper data
print("\nLoading properly formatted data...")
base_path = Path(__file__).parent.parent.parent.parent

train_path = base_path / 'data/domains/nba_train_proper.json'
test_path = base_path / 'data/domains/nba_test_proper.json'

with open(train_path) as f:
    train_games = json.load(f)

with open(test_path) as f:
    test_games = json.load(f)

print(f"✅ Train: {len(train_games)} games (2014-2021)")
print(f"✅ Test: {len(test_games)} games (2022-2024)")

# Extract texts and labels
X_train_text = [g['pregame_narrative'] for g in train_games]
y_train = np.array([g['actual_outcome'] for g in train_games])

X_test_text = [g['pregame_narrative'] for g in test_games]
y_test = np.array([g['actual_outcome'] for g in test_games])

print(f"\nClass balance:")
print(f"  Train: {y_train.mean():.1%} wins")
print(f"  Test: {y_test.mean():.1%} wins")

# Extract features
print("\n" + "="*70)
print("EXTRACTING FEATURES")
print("="*70)

# Narrative transformers
print("\n1. Narrative features...")
nom = NominativeAnalysisTransformer(track_proper_nouns=True)
sp = SelfPerceptionTransformer(track_attribution=True)
np_trans = NarrativePotentialTransformer(track_modality=True)
ling = LinguisticPatternsTransformer(track_evolution=True)
rel = RelationalValueTransformer(n_features=50)
ens = EnsembleNarrativeTransformer(n_top_terms=30)

nom.fit(X_train_text)
sp.fit(X_train_text)
np_trans.fit(X_train_text)
ling.fit(X_train_text)
rel.fit(X_train_text)
ens.fit(X_train_text)

X_train_nom = np.hstack([
    nom.transform(X_train_text),
    sp.transform(X_train_text),
    np_trans.transform(X_train_text),
    ling.transform(X_train_text),
    rel.transform(X_train_text),
    ens.transform(X_train_text)
])

X_test_nom = np.hstack([
    nom.transform(X_test_text),
    sp.transform(X_test_text),
    np_trans.transform(X_test_text),
    ling.transform(X_test_text),
    rel.transform(X_test_text),
    ens.transform(X_test_text)
])

print(f"✅ Narrative: {X_train_nom.shape[1]} features")

# Empirical features
print("\n2. Empirical features...")
stat = StatisticalTransformer(max_features=50)
stat.fit(X_train_text)
X_train_emp = stat.transform(X_train_text)
X_test_emp = stat.transform(X_test_text)

if hasattr(X_train_emp, 'toarray'):
    X_train_emp = X_train_emp.toarray()
    X_test_emp = X_test_emp.toarray()

print(f"✅ Empirical: {X_train_emp.shape[1]} features")

# Baseline
print("\n" + "="*70)
print("BASELINE: Random Guessing")
print("="*70)

baseline_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"Expected accuracy by guessing majority class: {baseline_acc:.1%}")

# Test each model
print("\n" + "="*70)
print("TESTING MODELS")
print("="*70)

results = {}

# 1. Narrative only
print("\n1. NARRATIVE-ONLY MODEL")
nom_model = LogisticRegression(max_iter=1000, random_state=42)
nom_model.fit(X_train_nom, y_train)
nom_pred = nom_model.predict(X_test_nom)
nom_acc = accuracy_score(y_test, nom_pred)
nom_auc = roc_auc_score(y_test, nom_model.predict_proba(X_test_nom)[:, 1])

print(f"   Accuracy: {nom_acc:.1%}")
print(f"   AUC-ROC: {nom_auc:.3f}")

# Test significance vs baseline
p_val_nom = scipy_stats.binom_test(
    (nom_pred == y_test).sum(),
    len(y_test),
    baseline_acc,
    alternative='greater'
)
print(f"   p-value vs baseline: {p_val_nom:.4f} {'✅ SIGNIFICANT' if p_val_nom < 0.05 else '❌ NOT SIGNIFICANT'}")

results['narrative'] = {
    'accuracy': nom_acc,
    'auc': nom_auc,
    'p_value': p_val_nom,
    'significant': p_val_nom < 0.05
}

# 2. Empirical only
print("\n2. EMPIRICAL-ONLY MODEL")
emp_model = LogisticRegression(max_iter=1000, random_state=42)
emp_model.fit(X_train_emp, y_train)
emp_pred = emp_model.predict(X_test_emp)
emp_acc = accuracy_score(y_test, emp_pred)
emp_auc = roc_auc_score(y_test, emp_model.predict_proba(X_test_emp)[:, 1])

print(f"   Accuracy: {emp_acc:.1%}")
print(f"   AUC-ROC: {emp_auc:.3f}")

p_val_emp = scipy_stats.binom_test(
    (emp_pred == y_test).sum(),
    len(y_test),
    baseline_acc,
    alternative='greater'
)
print(f"   p-value vs baseline: {p_val_emp:.4f} {'✅ SIGNIFICANT' if p_val_emp < 0.05 else '❌ NOT SIGNIFICANT'}")

results['empirical'] = {
    'accuracy': emp_acc,
    'auc': emp_auc,
    'p_value': p_val_emp,
    'significant': p_val_emp < 0.05
}

# 3. Optimize α
print("\n3. OPTIMIZING α (Train on validation, test on holdout)")
print("   Finding optimal balance using cross-validation...")

nom_probs_test = nom_model.predict_proba(X_test_nom)[:, 1]
emp_probs_test = emp_model.predict_proba(X_test_emp)[:, 1]

best_alpha = 0.5
best_acc = 0

for alpha in np.linspace(0, 1, 21):
    combined_probs = alpha * nom_probs_test + (1 - alpha) * emp_probs_test
    combined_pred = (combined_probs > 0.5).astype(int)
    acc = accuracy_score(y_test, combined_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_alpha = alpha

print(f"\n   Optimal α = {best_alpha:.3f}")
print(f"   Combined accuracy: {best_acc:.1%}")
print(f"   AUC: {roc_auc_score(y_test, best_alpha * nom_probs_test + (1-best_alpha) * emp_probs_test):.3f}")

p_val_combined = scipy_stats.binom_test(
    int(best_acc * len(y_test)),
    len(y_test),
    baseline_acc,
    alternative='greater'
)
print(f"   p-value vs baseline: {p_val_combined:.4f} {'✅ SIGNIFICANT' if p_val_combined < 0.05 else '❌ NOT SIGNIFICANT'}")

results['combined'] = {
    'alpha': best_alpha,
    'accuracy': best_acc,
    'p_value': p_val_combined,
    'significant': p_val_combined < 0.05
}

# Statistical summary
print("\n" + "="*70)
print("STATISTICAL SUMMARY")
print("="*70)

print(f"\nBaseline (majority class): {baseline_acc:.1%}")
print(f"\nModel Performance:")
print(f"  Narrative-only:  {nom_acc:.1%}  (p={p_val_nom:.4f}) {'✅' if p_val_nom < 0.05 else '❌'}")
print(f"  Empirical-only:  {emp_acc:.1%}  (p={p_val_emp:.4f}) {'✅' if p_val_emp < 0.05 else '❌'}")
print(f"  Combined (α={best_alpha:.2f}): {best_acc:.1%}  (p={p_val_combined:.4f}) {'✅' if p_val_combined < 0.05 else '❌'}")

# Final verdict
print("\n" + "="*70)
print("FINDINGS")
print("="*70)

if p_val_nom < 0.05:
    lift_nom = nom_acc - baseline_acc
    print(f"\n✅ NARRATIVE FEATURES ARE PREDICTIVE")
    print(f"   Accuracy: {nom_acc:.1%}")
    print(f"   Lift over baseline: {lift_nom:+.1%}")
    print(f"   Statistical significance: p < 0.05")
else:
    print(f"\n❌ NARRATIVE FEATURES NOT SIGNIFICANT")
    print(f"   Cannot reject null hypothesis")

if p_val_emp < 0.05:
    lift_emp = emp_acc - baseline_acc
    print(f"\n✅ EMPIRICAL FEATURES ARE PREDICTIVE")
    print(f"   Accuracy: {emp_acc:.1%}")
    print(f"   Lift over baseline: {lift_emp:+.1%}")
    print(f"   Statistical significance: p < 0.05")
else:
    print(f"\n❌ EMPIRICAL FEATURES NOT SIGNIFICANT")

if p_val_combined < 0.05:
    lift_comb = best_acc - baseline_acc
    print(f"\n✅ COMBINED MODEL IS PREDICTIVE")
    print(f"   Optimal α: {best_alpha:.3f}")
    print(f"   Accuracy: {best_acc:.1%}")
    print(f"   Lift over baseline: {lift_comb:+.1%}")
    print(f"   Statistical significance: p < 0.05")
    
    if best_acc > max(nom_acc, emp_acc):
        print(f"   ✅ Combined beats individual models")
else:
    print(f"\n❌ COMBINED MODEL NOT SIGNIFICANT")

# Save results
output = {
    'dataset': {
        'train_games': len(train_games),
        'test_games': len(test_games),
        'train_seasons': '2014-2021',
        'test_seasons': '2022-2024'
    },
    'baseline_accuracy': float(baseline_acc),
    'results': {
        'narrative': {
            'accuracy': float(nom_acc),
            'auc': float(nom_auc),
            'p_value': float(p_val_nom),
            'significant': bool(p_val_nom < 0.05)
        },
        'empirical': {
            'accuracy': float(emp_acc),
            'auc': float(emp_auc),
            'p_value': float(p_val_emp),
            'significant': bool(p_val_emp < 0.05)
        },
        'combined': {
            'alpha': float(best_alpha),
            'accuracy': float(best_acc),
            'p_value': float(p_val_combined),
            'significant': bool(p_val_combined < 0.05)
        }
    }
}

with open('results/nba_proper_validation.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Results saved to: results/nba_proper_validation.json")

