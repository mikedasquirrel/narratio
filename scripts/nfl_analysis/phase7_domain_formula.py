#!/usr/bin/env python3
"""
Phase 7: Domain Formula Calculation
Calculates NFL domain formula: п, Д, r, κ
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def calculate_narrativity():
    """Calculate narrativity (п) for NFL"""
    print("  📐 Calculating narrativity (п)...")
    
    # From config: NFL has structural constraints
    components = {
        'subjectivity': 0.45,      # Some interpretation but performance matters
        'agency': 0.60,            # Coaches/players have some control
        'observability': 0.90,     # Highly visible outcomes
        'generativity': 0.50,      # Moderate narrative generation
        'constraint': 0.45,        # Constrained by performance/physics
    }
    
    pi = sum(components.values()) / len(components)
    
    print(f"    Components:")
    for name, value in components.items():
        print(f"      {name}: {value:.2f}")
    print(f"    п = {pi:.3f}")
    
    return pi, components

def calculate_correlation(story_quality, outcomes):
    """Calculate correlation (r) between story quality and outcomes"""
    print("\n  📊 Calculating correlation (r)...")
    
    # Pearson correlation
    r, p_value = stats.pearsonr(story_quality, outcomes)
    
    print(f"    r = {r:.4f}")
    print(f"    p-value = {p_value:.4f}")
    print(f"    Interpretation: {interpretation_r(r)}")
    
    return r, p_value

def interpretation_r(r):
    """Interpret correlation strength"""
    if abs(r) < 0.1:
        return "Very weak/no correlation"
    elif abs(r) < 0.3:
        return "Weak correlation"
    elif abs(r) < 0.5:
        return "Moderate correlation"
    elif abs(r) < 0.7:
        return "Strong correlation"
    else:
        return "Very strong correlation"

def calculate_coupling():
    """Calculate coupling (κ) for NFL"""
    print("\n  🔗 Calculating coupling (κ)...")
    
    # NFL: Partial coupling
    # - Narrator = Media, fans, teams creating narrative
    # - Narrated = Players, games, outcomes
    # - Coupling ~0.6 (narrative influences but doesn't control)
    
    kappa = 0.60
    
    print(f"    κ = {kappa:.2f}")
    print(f"    Rationale: Narrative influences but performance dominates")
    
    return kappa

def calculate_narrative_agency(pi, r, kappa):
    """Calculate narrative agency (Д)"""
    print("\n  ⚡ Calculating narrative agency (Д)...")
    
    delta = pi * abs(r) * kappa
    
    print(f"    Д = п × |r| × κ")
    print(f"    Д = {pi:.3f} × {abs(r):.4f} × {kappa:.2f}")
    print(f"    Д = {delta:.4f}")
    
    return delta

def test_threshold(delta, pi):
    """Test if narrative matters (Д/п > 0.5)"""
    print("\n  🎯 Testing threshold...")
    
    ratio = delta / pi
    threshold = 0.5
    passes = ratio > threshold
    
    print(f"    Д/п = {delta:.4f} / {pi:.3f} = {ratio:.4f}")
    print(f"    Threshold: {threshold}")
    print(f"    Result: {'✓ PASSES' if passes else '✗ FAILS'}")
    
    if passes:
        print(f"    Interpretation: Narrative DOES control outcomes in NFL")
    else:
        print(f"    Interpretation: Narrative does NOT control outcomes in NFL")
        print(f"    Note: Performance/skill dominates, but narrative may create inefficiencies")
    
    return passes, ratio

def main():
    print("="*60)
    print(f"PHASE 7: DOMAIN FORMULA - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "domains"
    
    # Load story scores
    print("\n📂 Loading story scores...")
    with open(data_dir / "nfl_story_scores.json", 'r') as f:
        data = json.load(f)
    
    games = data['games']
    print(f"  ✓ {len(games):,} games with story quality scores")
    
    # Extract arrays
    story_quality = np.array([g['story_quality'] for g in games])
    outcomes = np.array([1 if g['home_won'] else 0 for g in games])
    
    print(f"  Mean story quality: {story_quality.mean():.3f}")
    print(f"  Home win rate: {outcomes.mean():.3f}")
    
    # Calculate formula components
    print("\n🔄 Calculating domain formula...")
    
    pi, pi_components = calculate_narrativity()
    r, p_value = calculate_correlation(story_quality, outcomes)
    kappa = calculate_coupling()
    delta = calculate_narrative_agency(pi, r, kappa)
    passes_threshold, efficiency = test_threshold(delta, pi)
    
    # Compile results
    formula_results = {
        'timestamp': datetime.now().isoformat(),
        'domain': 'NFL',
        'total_games': len(games),
        'narrativity': {
            'pi': float(pi),
            'components': pi_components,
        },
        'correlation': {
            'r': float(r),
            'p_value': float(p_value),
            'interpretation': interpretation_r(r),
        },
        'coupling': {
            'kappa': float(kappa),
            'rationale': 'Narrative influences but performance dominates',
        },
        'narrative_agency': {
            'delta': float(delta),
            'formula': 'Д = п × |r| × κ',
            'calculation': f'{pi:.3f} × {abs(r):.4f} × {kappa:.2f} = {delta:.4f}',
        },
        'threshold_test': {
            'ratio': float(efficiency),
            'threshold': 0.5,
            'passes': bool(passes_threshold),
            'interpretation': 'Narrative controls outcomes' if passes_threshold else 'Performance dominates, narrative creates market inefficiencies',
        },
        'verdict': {
            'narrative_matters': bool(passes_threshold),
            'efficiency': float(efficiency),
            'conclusion': 'Better stories win' if passes_threshold else 'Better performance wins, but narrative creates exploitable patterns',
        }
    }
    
    # Save formula results
    output_path = data_dir / "nfl_domain_formula.json"
    with open(output_path, 'w') as f:
        json.dump(formula_results, f, indent=2)
    
    print(f"\n✓ Domain formula saved: {output_path.name}")
    
    print(f"\n{'='*60}")
    print("FORMULA SUMMARY")
    print(f"{'='*60}")
    print(f"  п (narrativity):     {pi:.3f}")
    print(f"  r (correlation):     {r:.4f}")
    print(f"  κ (coupling):        {kappa:.2f}")
    print(f"  Д (agency):          {delta:.4f}")
    print(f"  Д/п (efficiency):    {efficiency:.4f}")
    print(f"  Threshold test:      {'PASS ✓' if passes_threshold else 'FAIL ✗'}")
    print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print("PHASE 7 COMPLETE ✓")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

