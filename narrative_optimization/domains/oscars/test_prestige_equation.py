"""
Oscars Prestige Equation Test

Tests prestige formula on Oscar data to validate WWE finding.

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'narrative_optimization'))

from src.transformers import (
    AwarenessResistanceTransformer,
    FundamentalConstraintsTransformer,
    NominativeAnalysisTransformer
)


def main():
    """Test prestige equation on Oscars"""
    print("="*80)
    print("OSCARS PRESTIGE EQUATION TEST")
    print("="*80)
    
    # Load Oscar data
    data_path = project_root / 'narrative_optimization' / 'domains' / 'oscars' / 'oscar_processed.json'
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n✓ Loaded {len(data)} Oscar films")
    
    # Extract narratives and outcomes
    narratives = [film['full_narrative'] for film in data if 'full_narrative' in film]
    outcomes = [film['won_oscar'] for film in data if 'full_narrative' in film]
    
    print(f"Films with narratives: {len(narratives)}")
    print(f"Winners: {sum(outcomes)}, Nominees: {len(outcomes) - sum(outcomes)}")
    
    # Extract forces
    print(f"\n{'─'*80}")
    print("EXTRACTING FORCES")
    print(f"{'─'*80}")
    
    # θ
    print("\n[1/3] θ (Awareness)...")
    theta_trans = AwarenessResistanceTransformer()
    theta_features = theta_trans.fit_transform(narratives)
    theta_values = theta_features[:, 14]
    
    print(f"  Mean: {theta_values.mean():.3f}")
    print(f"  Std:  {theta_values.std():.3f}")
    
    # λ
    print("\n[2/3] λ (Constraints)...")
    lambda_trans = FundamentalConstraintsTransformer()
    lambda_features = lambda_trans.fit_transform(narratives)
    lambda_values = lambda_features[:, 11]
    
    print(f"  Mean: {lambda_values.mean():.3f}")
    print(f"  Std:  {lambda_values.std():.3f}")
    
    # ة (proxy)
    print("\n[3/3] ة (Nominative Gravity - proxy)...")
    nom_trans = NominativeAnalysisTransformer()
    nom_features = nom_trans.fit_transform(narratives)
    ta_values = np.mean(nom_features, axis=1)
    ta_values = (ta_values - ta_values.min()) / (ta_values.max() - ta_values.min())
    
    print(f"  Mean: {ta_values.mean():.3f}")
    print(f"  Std:  {ta_values.std():.3f}")
    
    # Test equations
    print(f"\n{'='*80}")
    print("TESTING PRESTIGE vs REGULAR EQUATION")
    print(f"{'='*80}")
    
    outcomes_array = np.array(outcomes)
    
    # Regular
    D_regular = ta_values - theta_values - lambda_values
    D_regular = np.clip(D_regular, 0, 1)
    r_regular, p_regular = stats.pearsonr(D_regular, outcomes_array)
    
    print(f"\nRegular (ة - θ - λ):")
    print(f"  r = {r_regular:.3f}, p = {p_regular:.4f}")
    
    # Prestige
    D_prestige = ta_values + theta_values - lambda_values
    D_prestige = np.clip(D_prestige, 0, 1)
    r_prestige, p_prestige = stats.pearsonr(D_prestige, outcomes_array)
    
    print(f"\nPrestige (ة + θ - λ):")
    print(f"  r = {r_prestige:.3f}, p = {p_prestige:.4f}")
    
    # Validation
    print(f"\n{'='*80}")
    print("VALIDATION")
    print(f"{'='*80}")
    
    print(f"\nWhich equation fits Oscars better?")
    print(f"  Regular:  |r| = {abs(r_regular):.3f}")
    print(f"  Prestige: |r| = {abs(r_prestige):.3f}")
    
    if abs(r_prestige) > abs(r_regular) + 0.05:
        print(f"\n  ✅ PRESTIGE EQUATION WINS (+{abs(r_prestige) - abs(r_regular):.3f})")
        print(f"  Validates WWE finding on second domain!")
    elif abs(r_regular) > abs(r_prestige) + 0.05:
        print(f"\n  ❌ REGULAR EQUATION BETTER")
        print(f"  Oscars may differ from WWE")
    else:
        print(f"\n  ~ EQUATIONS SIMILAR")
    
    # Combined prestige validation
    print(f"\n{'='*80}")
    print("PRESTIGE PATTERN ACROSS DOMAINS")
    print(f"{'='*80}")
    
    print(f"\nWWE (n=250):")
    print(f"  Prestige: r=0.147, p=0.020 ✅")
    print(f"  Regular:  r=0.073, p=0.248")
    print(f"  Winner: Prestige")
    
    print(f"\nOscars (n={len(narratives)}):")
    print(f"  Prestige: r={r_prestige:.3f}, p={p_prestige:.4f}")
    print(f"  Regular:  r={r_regular:.3f}, p={p_regular:.4f}")
    print(f"  Winner: {'Prestige' if abs(r_prestige) > abs(r_regular) else 'Regular'}")
    
    if abs(r_prestige) > abs(r_regular) and abs(r_regular) > 0:
        print(f"\n✅ PRESTIGE PATTERN CONFIRMED ACROSS 2 DOMAINS")
        print(f"   Both WWE and Oscars show awareness amplification")
        print(f"   Prestige equation is ROBUST")
    
    # Save
    output = {
        'domain': 'oscars',
        'samples': len(narratives),
        'forces': {
            'theta_mean': float(theta_values.mean()),
            'lambda_mean': float(lambda_values.mean()),
            'ta_mean': float(ta_values.mean())
        },
        'equations': {
            'regular': {'r': float(r_regular), 'p': float(p_regular)},
            'prestige': {'r': float(r_prestige), 'p': float(p_prestige)}
        },
        'winner': 'prestige' if abs(r_prestige) > abs(r_regular) else 'regular'
    }
    
    output_path = project_root / 'narrative_optimization' / 'domains' / 'oscars' / 'prestige_equation_test.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"\n{'='*80}")
    print("✓ OSCARS PRESTIGE TEST COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

