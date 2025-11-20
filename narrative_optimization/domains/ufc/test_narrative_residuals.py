"""
UFC: Testing if Narrative Predicts RESIDUALS

Hypothesis: Narrative doesn't predict outcomes directly because physical skill dominates.
BUT: Among evenly-matched fighters (after controlling for physical attributes),
     does narrative explain the RESIDUAL variance?

This tests: Physical skill → outcome (strong)
           + Narrative → outcome residual (weak but real?)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Import the rigorous feature extraction
import sys
sys.path.insert(0, 'narrative_optimization/domains/ufc')

def main():
    """Test if narrative predicts residuals after controlling for physical"""
    
    print("="*80)
    print("UFC: NARRATIVE RESIDUAL ANALYSIS")
    print("Testing if narrative matters AFTER controlling for physical attributes")
    print("="*80)
    
    # Load data
    print("\n[1/6] Loading data...")
    with open('data/domains/ufc_with_narratives.json') as f:
        fights = json.load(f)
    
    sample_size = 2000
    np.random.seed(42)
    sample_indices = np.random.choice(len(fights), size=sample_size, replace=False)
    fights_sample = [fights[i] for i in sample_indices]
    print(f"✓ {len(fights_sample)} fights")
    
    # Extract outcomes
    outcomes = np.array([1 if f['result']['winner'] == 'fighter_a' else 0 for f in fights_sample])
    
    # Extract ALL features
    print("\n[2/6] Extracting features...")
    
    features_list = []
    for fight in fights_sample:
        fa = fight['fighter_a']
        fb = fight['fighter_b']
        
        # PHYSICAL features (should dominate)
        phys = {
            'strike_pct_a': fa['sig_str_pct'] / 100,
            'strike_pct_b': fb['sig_str_pct'] / 100,
            'sub_threat_a': fa['sub_att'],
            'sub_threat_b': fb['sub_att'],
            'td_pct_a': fa['td_pct'] / 100,
            'td_pct_b': fb['td_pct'] / 100,
            'reach_a': fa['reach'],
            'reach_b': fb['reach'],
            'age_a': fa['age'],
            'age_b': fb['age'],
        }
        
        # NARRATIVE features
        narr = {
            'name_len_a': len(fa['name']),
            'name_len_b': len(fb['name']),
            'has_nick_a': 1 if fa['nickname'] else 0,
            'has_nick_b': 1 if fb['nickname'] else 0,
            'streak_a': fa['win_streak'],
            'streak_b': fb['win_streak'],
            'is_title': 1 if fight['title_fight'] else 0,
        }
        
        features_list.append({**phys, **narr})
    
    X_df = pd.DataFrame(features_list)
    X = X_df.values
    
    print(f"✓ {X.shape[1]} features extracted")
    print(f"  - Physical: 10 features")
    print(f"  - Narrative: 7 features")
    
    # Split feature sets
    phys_cols = ['strike_pct_a', 'strike_pct_b', 'sub_threat_a', 'sub_threat_b', 
                 'td_pct_a', 'td_pct_b', 'reach_a', 'reach_b', 'age_a', 'age_b']
    narr_cols = ['name_len_a', 'name_len_b', 'has_nick_a', 'has_nick_b', 
                 'streak_a', 'streak_b', 'is_title']
    
    phys_idx = [X_df.columns.get_loc(c) for c in phys_cols]
    narr_idx = [X_df.columns.get_loc(c) for c in narr_cols]
    
    X_phys = X[:, phys_idx]
    X_narr = X[:, narr_idx]
    
    # Standardize
    scaler_phys = StandardScaler()
    scaler_narr = StandardScaler()
    X_phys_scaled = scaler_phys.fit_transform(X_phys)
    X_narr_scaled = scaler_narr.fit_transform(X_narr)
    
    # TEST 1: Physical-only model
    print("\n[3/6] Building physical-only model...")
    
    model_phys = LogisticRegression(random_state=42, max_iter=1000)
    cv_phys = cross_val_score(model_phys, X_phys_scaled, outcomes, cv=5, scoring='roc_auc')
    
    model_phys.fit(X_phys_scaled, outcomes)
    pred_phys = model_phys.predict_proba(X_phys_scaled)[:, 1]
    
    print(f"  Physical-only AUC: {cv_phys.mean():.4f} ± {cv_phys.std():.4f}")
    
    # TEST 2: Narrative-only model
    print("\n[4/6] Building narrative-only model...")
    
    model_narr = LogisticRegression(random_state=42, max_iter=1000)
    cv_narr = cross_val_score(model_narr, X_narr_scaled, outcomes, cv=5, scoring='roc_auc')
    
    print(f"  Narrative-only AUC: {cv_narr.mean():.4f} ± {cv_narr.std():.4f}")
    
    # TEST 3: Combined model
    print("\n[5/6] Building combined model...")
    
    X_combined = np.hstack([X_phys_scaled, X_narr_scaled])
    model_combined = LogisticRegression(random_state=42, max_iter=1000)
    cv_combined = cross_val_score(model_combined, X_combined, outcomes, cv=5, scoring='roc_auc')
    
    print(f"  Combined AUC: {cv_combined.mean():.4f} ± {cv_combined.std():.4f}")
    
    # TEST 4: Does narrative add value?
    print("\n[6/6] Testing narrative added value...")
    
    delta_auc = cv_combined.mean() - cv_phys.mean()
    
    print(f"\n  Physical alone: {cv_phys.mean():.4f}")
    print(f"  Combined:       {cv_combined.mean():.4f}")
    print(f"  Δ AUC:          {delta_auc:+.4f}")
    
    # Test statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(cv_combined, cv_phys)
    
    print(f"  p-value:        {p_value:.4f}")
    print(f"  Significant:    {'YES' if p_value < 0.05 else 'NO'}")
    
    # TEST 5: Narrative on even matchups
    print("\n" + "="*80)
    print("TESTING: Narrative effect on EVEN MATCHUPS")
    print("="*80)
    
    # Find even matchups (predicted probability close to 0.5)
    even_mask = np.abs(pred_phys - 0.5) < 0.1  # Within 10% of 50/50
    
    print(f"\nEven matchups: {even_mask.sum()} fights (predicted 40-60%)")
    
    if even_mask.sum() > 50:
        outcomes_even = outcomes[even_mask]
        X_narr_even = X_narr_scaled[even_mask]
        
        # Narrative effect on even matchups
        narr_score_even = X_narr_even.mean(axis=1)
        r_even = np.corrcoef(narr_score_even, outcomes_even)[0, 1]
        
        print(f"  Narrative correlation on even fights: r = {r_even:+.4f}, |r| = {abs(r_even):.4f}")
        
        # Compare to overall
        narr_score_all = X_narr_scaled.mean(axis=1)
        r_all = np.corrcoef(narr_score_all, outcomes)[0, 1]
        
        print(f"  Narrative correlation overall:        r = {r_all:+.4f}, |r| = {abs(r_all):.4f}")
        print(f"  Improvement on even matchups:         {abs(r_even)/abs(r_all):.2f}x")
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    print(f"\n1. Physical attributes predict outcomes: AUC = {cv_phys.mean():.3f}")
    print(f"2. Narrative alone barely predicts:      AUC = {cv_narr.mean():.3f}")
    print(f"3. Narrative adds to physical model:     Δ = {delta_auc:+.4f} (p={p_value:.3f})")
    
    if delta_auc > 0.01 and p_value < 0.05:
        print(f"\n✓ NARRATIVE HAS MEASURABLE EFFECT!")
        print(f"  - Small but statistically significant")
        print(f"  - Adds {delta_auc*100:.2f}% to prediction accuracy")
        print(f"  - Matters AFTER controlling for physical skill")
    else:
        print(f"\n✗ NARRATIVE EFFECT IS NEGLIGIBLE")
        print(f"  - Not statistically significant")
        print(f"  - Physical performance overwhelms narrative")
        print(f"  - UFC is pure performance domain")
    
    # Save results
    results = {
        'physical_only_auc': float(cv_phys.mean()),
        'narrative_only_auc': float(cv_narr.mean()),
        'combined_auc': float(cv_combined.mean()),
        'delta_auc': float(delta_auc),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
        'narrative_adds_value': bool(delta_auc > 0.01 and p_value < 0.05)
    }
    
    with open('narrative_optimization/domains/ufc/ufc_residual_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: ufc_residual_analysis.json")

if __name__ == "__main__":
    main()

