"""
UFC Narrative Delta Optimization

Maximize the incremental contribution of narrative features beyond physical.
Uses feature selection, ensemble methods, and hyperparameter tuning.

Current baseline: Δ = +0.0247
Target: Δ = 0.03-0.04 or higher
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import json
import warnings
warnings.filterwarnings('ignore')


def main():
    """Optimize narrative delta"""
    
    print("="*80)
    print("UFC NARRATIVE DELTA OPTIMIZATION")
    print("="*80)
    
    # Load features
    print("\n[1/6] Loading features...")
    
    data_dir = Path('narrative_optimization/domains/ufc')
    X_df = pd.read_csv(data_dir / 'ufc_comprehensive_features.csv')
    y = np.load(data_dir / 'ufc_comprehensive_outcomes.npy')
    
    with open(data_dir / 'ufc_feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    
    print(f"✓ {X_df.shape[1]} features, {X_df.shape[0]} fights")
    
    # Identify feature types
    physical_cols = [i for i, f in enumerate(feature_names) if any(x in f for x in ['strike', 'td', 'ctrl', 'kd', 'sub', 'rev'])]
    narrative_cols = [i for i, f in enumerate(feature_names) if any(x in f for x in ['name', 'nick', 'vowel', 'syllable', 'memorability', 'title', 'bonus'])]
    interaction_cols = [i for i, f in enumerate(feature_names) if '_x_' in f]
    
    print(f"  Physical: {len(physical_cols)}")
    print(f"  Narrative: {len(narrative_cols)}")
    print(f"  Interactions: {len(interaction_cols)}")
    
    X = X_df.values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_phys = X_scaled[:, physical_cols]
    X_narr = X_scaled[:, narrative_cols]
    X_inter = X_scaled[:, interaction_cols]
    
    # === BASELINE ===
    print("\n[2/6] Establishing baseline...")
    
    model_phys = LogisticRegression(random_state=42, max_iter=1000)
    cv_phys = cross_val_score(model_phys, X_phys, y, cv=5, scoring='roc_auc')
    baseline_phys = cv_phys.mean()
    
    model_all = LogisticRegression(random_state=42, max_iter=1000)
    cv_all = cross_val_score(model_all, X_scaled, y, cv=5, scoring='roc_auc')
    baseline_all = cv_all.mean()
    
    baseline_delta = baseline_all - baseline_phys
    
    print(f"  Baseline Physical AUC: {baseline_phys:.4f}")
    print(f"  Baseline Combined AUC: {baseline_all:.4f}")
    print(f"  Baseline Δ: {baseline_delta:+.4f}")
    
    results = {'baseline': {'phys': float(baseline_phys), 'all': float(baseline_all), 'delta': float(baseline_delta)}}
    
    # === OPTIMIZATION 1: Feature Selection ===
    print("\n[3/6] Optimizing with feature selection...")
    
    # Select best narrative features
    selector = SelectKBest(f_classif, k=min(20, len(narrative_cols)))
    selector.fit(X_narr, y)
    
    best_narr_idx = selector.get_support(indices=True)
    best_narr_features = [feature_names[narrative_cols[i]] for i in best_narr_idx]
    
    print(f"  Selected top {len(best_narr_idx)} narrative features:")
    for feat in best_narr_features[:5]:
        print(f"    - {feat}")
    
    # Test with selected features
    X_selected = np.hstack([X_phys, X_narr[:, best_narr_idx]])
    cv_selected = cross_val_score(model_all, X_selected, y, cv=5, scoring='roc_auc')
    delta_selected = cv_selected.mean() - baseline_phys
    
    print(f"  Selected Features AUC: {cv_selected.mean():.4f}")
    print(f"  Delta: {delta_selected:+.4f} (improvement: {delta_selected - baseline_delta:+.4f})")
    
    results['feature_selection'] = {'auc': float(cv_selected.mean()), 'delta': float(delta_selected), 'features': best_narr_features}
    
    # === OPTIMIZATION 2: Add Interactions ===
    print("\n[4/6] Optimizing with interaction features...")
    
    X_with_inter = np.hstack([X_phys, X_narr[:, best_narr_idx], X_inter])
    cv_inter = cross_val_score(model_all, X_with_inter, y, cv=5, scoring='roc_auc')
    delta_inter = cv_inter.mean() - baseline_phys
    
    print(f"  With Interactions AUC: {cv_inter.mean():.4f}")
    print(f"  Delta: {delta_inter:+.4f} (improvement: {delta_inter - baseline_delta:+.4f})")
    
    results['with_interactions'] = {'auc': float(cv_inter.mean()), 'delta': float(delta_inter)}
    
    # === OPTIMIZATION 3: Ensemble Methods ===
    print("\n[5/6] Testing ensemble methods...")
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
    cv_rf_phys = cross_val_score(rf, X_phys, y, cv=5, scoring='roc_auc')
    cv_rf_all = cross_val_score(rf, X_with_inter, y, cv=5, scoring='roc_auc')
    delta_rf = cv_rf_all.mean() - cv_rf_phys.mean()
    
    print(f"  Random Forest:")
    print(f"    Physical: {cv_rf_phys.mean():.4f}")
    print(f"    Combined: {cv_rf_all.mean():.4f}")
    print(f"    Δ: {delta_rf:+.4f}")
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    cv_gb_phys = cross_val_score(gb, X_phys, y, cv=5, scoring='roc_auc')
    cv_gb_all = cross_val_score(gb, X_with_inter, y, cv=5, scoring='roc_auc')
    delta_gb = cv_gb_all.mean() - cv_gb_phys.mean()
    
    print(f"  Gradient Boosting:")
    print(f"    Physical: {cv_gb_phys.mean():.4f}")
    print(f"    Combined: {cv_gb_all.mean():.4f}")
    print(f"    Δ: {delta_gb:+.4f}")
    
    # Voting Ensemble
    voting = VotingClassifier([
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=5))
    ], voting='soft')
    
    cv_voting_all = cross_val_score(voting, X_with_inter, y, cv=5, scoring='roc_auc')
    delta_voting = cv_voting_all.mean() - baseline_phys
    
    print(f"  Voting Ensemble:")
    print(f"    Combined: {cv_voting_all.mean():.4f}")
    print(f"    Δ: {delta_voting:+.4f}")
    
    results['ensemble'] = {
        'random_forest': {'auc': float(cv_rf_all.mean()), 'delta': float(delta_rf)},
        'gradient_boosting': {'auc': float(cv_gb_all.mean()), 'delta': float(delta_gb)},
        'voting': {'auc': float(cv_voting_all.mean()), 'delta': float(delta_voting)}
    }
    
    # === SUMMARY ===
    print("\n[6/6] OPTIMIZATION SUMMARY")
    print("="*80)
    
    all_deltas = {
        'Baseline': baseline_delta,
        'Feature Selection': delta_selected,
        'With Interactions': delta_inter,
        'Random Forest': delta_rf,
        'Gradient Boosting': delta_gb,
        'Voting Ensemble': delta_voting
    }
    
    best_method = max(all_deltas.items(), key=lambda x: x[1])
    
    print(f"\nNarrative Δ by Method:")
    for method, delta in sorted(all_deltas.items(), key=lambda x: x[1], reverse=True):
        marker = "★" if method == best_method[0] else " "
        print(f"  {marker} {method:20s}: Δ = {delta:+.4f}")
    
    print(f"\n✓ BEST METHOD: {best_method[0]}")
    print(f"  Δ = {best_method[1]:+.4f}")
    print(f"  Improvement over baseline: {best_method[1] - baseline_delta:+.4f}")
    
    if best_method[1] > 0.03:
        print(f"\n✓ TARGET ACHIEVED! Δ > 0.03")
    else:
        print(f"\n✗ Below target (Δ < 0.03)")
    
    results['best_method'] = best_method[0]
    results['best_delta'] = float(best_method[1])
    results['improvement'] = float(best_method[1] - baseline_delta)
    
    # Save
    output_path = data_dir / 'ufc_optimization_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")


if __name__ == "__main__":
    main()

