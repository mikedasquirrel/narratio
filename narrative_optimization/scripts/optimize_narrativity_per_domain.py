"""
Optimize Narrativity (π) Per Domain

Uses empirical data to optimize π calculation for each domain:
1. Current π values (from website/theory)
2. Observed performance (R², Д)
3. Force measurements (θ, λ from Phase 7)
4. Back-calculate optimal π components

Optimizes the 5-component formula per domain type.

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Current domain data from website
DOMAIN_DATA = {
    'lottery': {'pi': 0.04, 'arch': 0.000, 'r': 0.000, 'type': 'Pure Randomness'},
    'aviation': {'pi': 0.12, 'arch': 0.000, 'r': 0.000, 'type': 'Engineering'},
    'hurricanes': {'pi': 0.30, 'arch': 0.036, 'r': 0.19, 'type': 'Natural'},
    'nba': {'pi': 0.49, 'arch': 0.018, 'r': 0.20, 'type': 'Physical Skill'},
    'nfl': {'pi': 0.57, 'arch': 0.014, 'r': 0.25, 'type': 'Team Sport'},
    'mental_health': {'pi': 0.55, 'arch': 0.066, 'r': 0.27, 'type': 'Medical'},
    'movies': {'pi': 0.65, 'arch': 0.026, 'r': 0.29, 'type': 'Entertainment'},
    'oscars': {'pi': 0.75, 'arch': 1.000, 'r': 0.95, 'type': 'Prestige'},
    'crypto': {'pi': 0.76, 'arch': 0.423, 'r': 0.65, 'type': 'Speculation'},
    'startups': {'pi': 0.76, 'arch': 0.223, 'r': 0.980, 'type': 'Business'},
    'character': {'pi': 0.85, 'arch': 0.617, 'r': 0.725, 'type': 'Character'},
    'tennis': {'pi': 0.75, 'arch': 0.865, 'r': 0.93, 'type': 'Individual Sport'},
    'ufc': {'pi': 0.722, 'arch': 0.025, 'r': 0.17, 'type': 'Combat Sport'},
    'golf': {'pi': 0.70, 'arch': 0.953, 'r': 0.988, 'type': 'Individual Sport'},
    'music': {'pi': 0.702, 'arch': 0.031, 'r': 0.18, 'type': 'Entertainment'},
    'housing': {'pi': 0.92, 'arch': 0.156, 'r': 0.40, 'type': 'Pure Nominative'},
    'self_rated': {'pi': 0.95, 'arch': 0.564, 'r': 0.594, 'type': 'Identity'},
    'wwe': {'pi': 0.974, 'arch': 1.800, 'r': 0.40, 'type': 'Prestige'}
}


def load_phase7_data():
    """Load Phase 7 force measurements"""
    summary_path = project_root / 'narrative_optimization' / 'data' / 'phase7_extraction_summary.json'
    
    if not summary_path.exists():
        return {}
    
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    # Map to domain names
    mapping = {
        'tennisset': 'tennis',
        'golf_enhanced_narratives': 'golf',
        'ufc_with_narratives': 'ufc',
        'nba_all_seasons_real': 'nba',
        'nflset': 'nfl',
        'mental_health': 'mental_health',
        'crypto_with_competitive_context': 'crypto',
        'startups_real': 'startups',
        'mlbset': 'mlb'
    }
    
    phase7 = {}
    for result in data['results']:
        domain = result['domain']
        if domain in mapping:
            framework_name = mapping[domain]
            if framework_name not in phase7:  # Take first
                phase7[framework_name] = {
                    'theta': result['theta_mean'],
                    'lambda': result['lambda_mean'],
                    'samples': result['samples']
                }
    
    return phase7


def calculate_pi_from_components(structural, temporal, agency, interpretive, format_flex):
    """Standard π calculation"""
    return (0.30 * structural + 
            0.20 * temporal + 
            0.25 * agency + 
            0.15 * interpretive + 
            0.10 * format_flex)


def estimate_pi_components_from_data(domain_name, domain_data, phase7_data):
    """
    Estimate optimal π components from empirical data.
    
    Uses:
    - Observed R²/correlation
    - θ and λ measurements
    - Domain type
    """
    pi_current = domain_data['pi']
    r_observed = domain_data['r']
    domain_type = domain_data['type']
    
    # Get Phase 7 data if available
    theta = phase7_data.get(domain_name, {}).get('theta', 0.5)
    lambda_val = phase7_data.get(domain_name, {}).get('lambda', 0.5)
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZING: {domain_name.upper()}")
    print(f"{'='*80}")
    print(f"Current π: {pi_current:.3f}")
    print(f"Observed r: {r_observed:.3f}")
    print(f"Type: {domain_type}")
    if domain_name in phase7_data:
        print(f"θ: {theta:.3f}, λ: {lambda_val:.3f}")
    
    # Estimate components based on domain characteristics
    components = {}
    
    # Structural: Inverse of λ (high λ → low structural freedom)
    components['structural'] = max(0.0, min(1.0, 1.0 - lambda_val))
    
    # Temporal: Based on domain type
    if 'Sport' in domain_type:
        components['temporal'] = 0.6  # Games unfold over time
    elif domain_type in ['Prestige', 'Character', 'Identity']:
        components['temporal'] = 0.9  # Long-term reputation
    elif domain_type in ['Pure Randomness', 'Engineering']:
        components['temporal'] = 0.1  # Instant/determined
    else:
        components['temporal'] = 0.5  # Default
    
    # Agency: Based on domain type  
    if 'Individual Sport' in domain_type:
        components['agency'] = 1.0  # Complete individual control
    elif 'Team Sport' in domain_type:
        components['agency'] = 0.7  # Shared control
    elif domain_type in ['Pure Randomness', 'Natural']:
        components['agency'] = 0.0  # No control
    elif domain_type in ['Character', 'Identity', 'Self-Rated']:
        components['agency'] = 0.95  # Near-total control
    else:
        components['agency'] = 0.6  # Moderate
    
    # Interpretive: Correlation with r_observed (high r → more interpretive)
    components['interpretive'] = min(0.95, r_observed)
    
    # Format: Based on domain type
    if domain_type == 'Prestige':
        components['format'] = 0.95  # Very flexible
    elif 'Entertainment' in domain_type:
        components['format'] = 0.7  # Flexible
    elif domain_type in ['Engineering', 'Pure Randomness']:
        components['format'] = 0.1  # Rigid
    else:
        components['format'] = 0.5  # Moderate
    
    # Calculate optimized π
    pi_optimized = calculate_pi_from_components(
        components['structural'],
        components['temporal'],
        components['agency'],
        components['interpretive'],
        components['format']
    )
    
    # Constraint: Don't deviate too much from observed performance
    # Use θ and λ to inform π
    # Theory: π should correlate inversely with λ
    pi_from_lambda = 1.0 - lambda_val  # Simple inverse
    
    # Blend current, calculated, and lambda-based
    pi_final = 0.5 * pi_current + 0.3 * pi_optimized + 0.2 * pi_from_lambda
    
    print(f"\nComponent Estimates:")
    print(f"  Structural: {components['structural']:.2f} (from λ)")
    print(f"  Temporal: {components['temporal']:.2f} (from type)")
    print(f"  Agency: {components['agency']:.2f} (from type)")
    print(f"  Interpretive: {components['interpretive']:.2f} (from r)")
    print(f"  Format: {components['format']:.2f} (from type)")
    
    print(f"\nCalculations:")
    print(f"  π_calculated: {pi_optimized:.3f}")
    print(f"  π_from_λ: {pi_from_lambda:.3f}")
    print(f"  π_blended: {pi_final:.3f}")
    print(f"  Δπ: {pi_final - pi_current:+.3f}")
    
    if abs(pi_final - pi_current) > 0.15:
        print(f"  ⚠️  Large deviation - recommend review")
    elif abs(pi_final - pi_current) > 0.05:
        print(f"  ~ Moderate adjustment suggested")
    else:
        print(f"  ✓ Current π well-calibrated")
    
    return {
        'domain': domain_name,
        'pi_current': pi_current,
        'pi_optimized': pi_final,
        'delta_pi': pi_final - pi_current,
        'components': components,
        'theta': theta,
        'lambda': lambda_val,
        'r_observed': r_observed
    }


def main():
    """Optimize π for all domains"""
    print("="*80)
    print("NARRATIVITY OPTIMIZATION - PER-DOMAIN DIMENSIONAL TUNING")
    print("="*80)
    print("\nUsing empirical data to optimize π components:")
    print("  • Phase 7 force measurements (θ, λ)")
    print("  • Observed performance (R², correlations)")
    print("  • Domain type characteristics")
    print("  • Back-calculated from outcomes")
    
    # Load Phase 7 data
    phase7_data = load_phase7_data()
    print(f"\n✓ Loaded Phase 7 data for {len(phase7_data)} domains")
    
    # Optimize each domain
    optimizations = []
    
    for domain_name, domain_data in DOMAIN_DATA.items():
        result = estimate_pi_components_from_data(domain_name, domain_data, phase7_data)
        optimizations.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Domain':<20} {'π_curr':<8} {'π_opt':<8} {'Δπ':<8} {'θ':<8} {'λ':<8} {'r':<8}")
    print("-" * 80)
    
    for opt in sorted(optimizations, key=lambda x: abs(x['delta_pi']), reverse=True):
        print(f"{opt['domain']:<20} {opt['pi_current']:<8.3f} {opt['pi_optimized']:<8.3f} {opt['delta_pi']:<+8.3f} {opt['theta']:<8.3f} {opt['lambda']:<8.3f} {opt['r_observed']:<8.3f}")
    
    # Identify domains needing adjustment
    large_adjustments = [o for o in optimizations if abs(o['delta_pi']) > 0.10]
    moderate_adjustments = [o for o in optimizations if 0.05 < abs(o['delta_pi']) <= 0.10]
    
    print(f"\n{'='*80}")
    print("ADJUSTMENT RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if large_adjustments:
        print(f"\n⚠️  LARGE ADJUSTMENTS (Δπ > 0.10): {len(large_adjustments)} domains")
        for opt in large_adjustments:
            print(f"  • {opt['domain']}: {opt['pi_current']:.3f} → {opt['pi_optimized']:.3f} ({opt['delta_pi']:+.3f})")
            print(f"    Reason: θ={opt['theta']:.3f}, λ={opt['lambda']:.3f}, r={opt['r_observed']:.3f}")
    
    if moderate_adjustments:
        print(f"\n~ MODERATE ADJUSTMENTS (0.05 < Δπ ≤ 0.10): {len(moderate_adjustments)} domains")
        for opt in moderate_adjustments:
            print(f"  • {opt['domain']}: {opt['pi_current']:.3f} → {opt['pi_optimized']:.3f} ({opt['delta_pi']:+.3f})")
    
    well_calibrated = [o for o in optimizations if abs(o['delta_pi']) <= 0.05]
    print(f"\n✓ WELL CALIBRATED (Δπ ≤ 0.05): {len(well_calibrated)} domains")
    for opt in well_calibrated[:5]:
        print(f"  • {opt['domain']}: π={opt['pi_current']:.3f} (Δ={opt['delta_pi']:+.3f})")
    
    # Theoretical insights
    print(f"\n{'='*80}")
    print("THEORETICAL INSIGHTS")
    print(f"{'='*80}")
    
    # Test π vs observed performance
    pi_values = [o['pi_current'] for o in optimizations]
    r_values = [o['r_observed'] for o in optimizations]
    theta_values = [o['theta'] for o in optimizations]
    lambda_values = [o['lambda'] for o in optimizations]
    
    r_pi_r, p = stats.pearsonr(pi_values, r_values)
    r_pi_theta, _ = stats.pearsonr(pi_values, theta_values)
    r_pi_lambda, _ = stats.pearsonr(pi_values, lambda_values)
    
    print(f"\nCurrent π Relationships:")
    print(f"  π ↔ r (performance): {r_pi_r:.3f}")
    print(f"  π ↔ θ (awareness): {r_pi_theta:.3f}")
    print(f"  π ↔ λ (constraints): {r_pi_lambda:.3f}")
    
    print(f"\nExpected:")
    print(f"  π ↔ r: Positive (higher π → higher potential r)")
    print(f"  π ↔ θ: Weak (domain-specific)")
    print(f"  π ↔ λ: Negative (higher π → lower λ)")
    
    if r_pi_r > 0.5:
        print(f"  ✓ π ↔ r validates (positive correlation)")
    if r_pi_lambda < 0:
        print(f"  ✓ π ↔ λ validates (negative correlation)")
    
    # Save results
    output = {
        'timestamp': '2025-11-12',
        'method': 'empirical_optimization',
        'domains_optimized': len(optimizations),
        'optimizations': optimizations,
        'summary': {
            'large_adjustments': len(large_adjustments),
            'moderate_adjustments': len(moderate_adjustments),
            'well_calibrated': len(well_calibrated)
        },
        'correlations': {
            'pi_r': float(r_pi_r),
            'pi_theta': float(r_pi_theta),
            'pi_lambda': float(r_pi_lambda)
        }
    }
    
    output_path = project_root / 'narrative_optimization' / 'data' / 'narrativity_optimization_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved optimization results to: {output_path}")
    
    print(f"\n{'='*80}")
    print("✓ NARRATIVITY OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"  1. Review large adjustments")
    print(f"  2. Validate component estimates per domain")
    print(f"  3. Update domain configs with optimized π")
    print(f"  4. Re-run analyses with new π values")


if __name__ == '__main__':
    main()

