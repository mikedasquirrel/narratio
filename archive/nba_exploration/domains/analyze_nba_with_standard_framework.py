"""
NBA Analysis with Standard Framework

Applies complete framework to 11,979 real games:
- Extract ж using standard transformers
- Compute ю
- Measure r (correlation)
- Calculate Д = п × r × κ
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
from scipy import stats


def analyze_nba():
    """Complete NBA analysis with Д calculation."""
    print("=" * 80)
    print("NBA ANALYSIS - Complete Framework")
    print("=" * 80)
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'data/domains/nba_all_seasons_real.json'
    
    with open(data_path, 'r') as f:
        games = json.load(f)
    
    print(f"\n✓ Loaded {len(games)} games")
    
    # Extract narratives and outcomes
    narratives = [g.get('narrative', '') for g in games if g.get('narrative')]
    outcomes = [int(g.get('won', 0)) for g in games if g.get('narrative')]
    
    print(f"✓ {len(narratives)} games with narratives")
    
    # Sample for speed (use all 11979 in production)
    sample_size = min(2000, len(narratives))
    narratives_sample = narratives[:sample_size]
    outcomes_sample = outcomes[:sample_size]
    
    X = np.array(narratives_sample)
    y = np.array(outcomes_sample)
    
    print(f"✓ Analyzing {len(X)} games")
    
    # Apply transformers
    print("\nApplying standard transformers...")
    
    transformers = {
        'nominative': NominativeAnalysisTransformer(),
        'self_perception': SelfPerceptionTransformer(),
        'narrative_potential': NarrativePotentialTransformer()
    }
    
    all_features = []
    
    for name, transformer in transformers.items():
        try:
            transformer.fit(X)
            features = transformer.transform(X)
            all_features.append(features)
            print(f"  ✓ {name}: {features.shape[1]} features")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # Combine
    ж = np.hstack(all_features)
    ю = np.mean(ж, axis=1)
    
    # Measure r
    r, p = stats.pearsonr(ю, y)
    
    print(f"\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"r (predictive correlation): {r:.4f}, p={p:.4f}")
    print(f"R²: {r**2:.4f}")
    
    # Calculate Д
    п = 0.48  # NBA narrativity (from structural analysis)
    κ = 0.7   # Player performs (participant), stats judge (somewhat coupled)
    Д = п * r * κ
    
    print(f"\nNarrative Agency Calculation:")
    print(f"  п (narrativity): {п}")
    print(f"  r (correlation): {r:.4f}")
    print(f"  κ (coupling): {κ}")
    print(f"  Д = п × r × κ = {Д:.4f}")
    print(f"\n  Efficiency: Д/п = {Д/п:.4f}")
    print(f"  Threshold: {'✓ PASS' if Д/п > 0.5 else '✗ FAIL'} (need > 0.5)")
    
    results = {
        'domain': 'nba',
        'n_games': len(X),
        'narrativity': п,
        'coupling': κ,
        'r_measured': float(r),
        'p_value': float(p),
        'D_agency': float(Д),
        'efficiency': float(Д/п),
        'passes_threshold': bool(Д/п > 0.5)
    }
    
    # Save
    output_path = Path(__file__).parent / 'nba_D_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved: {output_path}")
    
    return results


if __name__ == "__main__":
    analyze_nba()

