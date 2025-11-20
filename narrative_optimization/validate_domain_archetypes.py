"""
Domain Archetype Validation Script

Validates domain-specific Ξ architecture on priority domains:
1. Golf (should maintain 97.7% R² with POSITIVE Д)
2. Tennis (should maintain 93% R²)
3. Boxing (test if proper Ξ measurement improves from 0.4%)
4. NBA (test if proper Ξ measurement improves from 15%)
5. WWE (should maintain 74.3% R² with prestige equation)

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer


def load_domain_data(domain_name: str) -> dict:
    """
    Load domain data for validation.
    
    Parameters
    ----------
    domain_name : str
        Domain name
    
    Returns
    -------
    dict
        Data with texts, outcomes, and optional fields
    """
    print(f"\nLoading data for {domain_name}...")
    
    # Domain data paths
    domain_paths = {
        'golf': 'narrative_optimization/domains/golf/data',
        'tennis': 'narrative_optimization/domains/tennis/data',
        'boxing': 'narrative_optimization/domains/boxing/data',
        'nba': 'narrative_optimization/domains/nba/data',
        'wwe': 'narrative_optimization/domains/wwe/data'
    }
    
    data_dir = Path(__file__).parent / domain_paths.get(domain_name, f'data/domains/{domain_name}')
    
    # Try to load data (implementation depends on data format)
    # This is a template - adjust to your actual data structure
    try:
        # Example: load from JSON
        data_file = data_dir / f'{domain_name}_data.json'
        if data_file.exists():
            with open(data_file) as f:
                data = json.load(f)
            
            texts = data.get('texts', data.get('narratives', []))
            outcomes = np.array(data.get('outcomes', data.get('results', [])))
            
            print(f"  ✓ Loaded {len(texts)} samples")
            return {
                'texts': texts,
                'outcomes': outcomes,
                'names': data.get('names', None),
                'timestamps': data.get('timestamps', None)
            }
    except Exception as e:
        print(f"  ✗ Could not load data: {e}")
        print(f"  Note: Implement load_domain_data() for your data structure")
        return None


def validate_domain(domain_name: str, expected_r2: float, description: str):
    """
    Validate a single domain.
    
    Parameters
    ----------
    domain_name : str
        Domain name
    expected_r2 : float
        Expected R² to validate against
    description : str
        Description of what we're testing
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING: {domain_name.upper()}")
    print(f"{'='*80}")
    print(f"Test: {description}")
    print(f"Expected R²: {expected_r2:.1%}")
    
    # Load data
    data = load_domain_data(domain_name)
    if data is None:
        print(f"\n✗ SKIPPED: No data available")
        return False
    
    # Run analysis with domain-specific Ξ
    analyzer = DomainSpecificAnalyzer(domain_name)
    
    try:
        results = analyzer.analyze_complete(
            texts=data['texts'],
            outcomes=data['outcomes'],
            names=data.get('names'),
            timestamps=data.get('timestamps')
        )
        
        # Validate results
        r2_achieved = results['r_squared']
        delta = results['delta']
        efficiency = results['efficiency']
        
        print(f"\n{'='*80}")
        print("VALIDATION RESULTS")
        print(f"{'='*80}")
        
        print(f"\nPerformance:")
        print(f"  Expected R²: {expected_r2:.1%}")
        print(f"  Achieved R²: {r2_achieved:.1%}")
        print(f"  Difference: {(r2_achieved - expected_r2):.1%}")
        
        print(f"\nΔ (The Bridge):")
        print(f"  Д: {delta:.4f}")
        print(f"  Sign: {'POSITIVE ✓' if delta > 0 else 'NEGATIVE ✗'}")
        print(f"  Efficiency (Д/π): {efficiency:.4f}")
        print(f"  Passes threshold (>0.5): {'YES ✓' if efficiency > 0.5 else 'NO ✗'}")
        
        # Validation criteria
        meets_r2 = r2_achieved >= expected_r2 * 0.90  # Within 10% of expected
        has_positive_delta = delta > 0 if domain_name in ['golf', 'tennis'] else True
        
        if meets_r2 and has_positive_delta:
            print(f"\n{'='*80}")
            print("✓ VALIDATION PASSED")
            print(f"{'='*80}")
            return True
        else:
            print(f"\n{'='*80}")
            print("✗ VALIDATION FAILED")
            print(f"{'='*80}")
            if not meets_r2:
                print(f"  - R² below expected ({r2_achieved:.1%} < {expected_r2 * 0.90:.1%})")
            if not has_positive_delta and domain_name in ['golf', 'tennis']:
                print(f"  - Д is negative (should be positive for high performers)")
            return False
            
    except Exception as e:
        print(f"\n✗ VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run validation on all priority domains."""
    print("="*80)
    print("DOMAIN ARCHETYPE VALIDATION")
    print("="*80)
    print("\nValidating domain-specific Ξ architecture on 5 priority domains")
    
    # Priority domains with expected performance
    domains_to_validate = [
        ('golf', 0.977, "Should maintain 97.7% R² with POSITIVE Д"),
        ('tennis', 0.931, "Should maintain 93.1% R² similar to Golf"),
        ('boxing', 0.004, "Tests if proper Ξ measurement improves from 0.4%"),
        ('nba', 0.15, "Tests if proper Ξ measurement improves from 15%"),
        ('wwe', 0.743, "Should maintain 74.3% R² with prestige equation")
    ]
    
    results = []
    
    for domain, expected_r2, description in domains_to_validate:
        passed = validate_domain(domain, expected_r2, description)
        results.append((domain, passed))
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}\n")
    
    for domain, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {domain.upper():15s}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} passed")
    
    if total_passed == len(results):
        print("\n✓ ALL VALIDATIONS PASSED - Architecture is working!")
    else:
        print(f"\n⚠ {len(results) - total_passed} validation(s) failed - Review needed")


if __name__ == '__main__':
    main()

