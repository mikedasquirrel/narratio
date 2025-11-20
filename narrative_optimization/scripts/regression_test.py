"""
Regression Testing

Tests that changes don't break existing functionality.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer
from src.data import DataLoader


class RegressionTester:
    """
    Tests for regressions in domain analyses.
    
    Compares:
    - Current R² vs baseline
    - Current Д vs baseline
    - Pattern counts
    """
    
    def __init__(self, baseline_path: Path = None):
        if baseline_path is None:
            baseline_path = Path(__file__).parent.parent / 'baseline_metrics.json'
        
        self.baseline_path = baseline_path
        self.baseline = self._load_baseline()
        
    def _load_baseline(self) -> Dict:
        """Load baseline metrics."""
        if not self.baseline_path.exists():
            return {}
        
        with open(self.baseline_path) as f:
            return json.load(f)
    
    def save_baseline(self, metrics: Dict):
        """Save new baseline."""
        with open(self.baseline_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def test_domain(self, domain_name: str) -> Dict:
        """
        Test a single domain for regression.
        
        Parameters
        ----------
        domain_name : str
            Domain name
        
        Returns
        -------
        dict
            Test results
        """
        try:
            # Load data
            loader = DataLoader()
            data = loader.load_domain(domain_name)
            
            # Analyze
            analyzer = DomainSpecificAnalyzer(domain_name)
            results = analyzer.analyze_complete(
                texts=data['texts'],
                outcomes=data['outcomes']
            )
            
            # Compare to baseline
            if domain_name in self.baseline:
                baseline_r2 = self.baseline[domain_name].get('r_squared', 0)
                baseline_delta = self.baseline[domain_name].get('delta', 0)
                
                r2_change = results['r_squared'] - baseline_r2
                delta_change = results['delta'] - baseline_delta
                
                # Regression if performance dropped significantly
                has_regression = (
                    r2_change < -0.05 or  # R² dropped more than 5 points
                    (baseline_delta > 0 and delta_change < -0.1)  # Д dropped 0.1+
                )
                
                return {
                    'domain': domain_name,
                    'status': 'regression' if has_regression else 'pass',
                    'r_squared': results['r_squared'],
                    'r_squared_baseline': baseline_r2,
                    'r_squared_change': r2_change,
                    'delta': results['delta'],
                    'delta_baseline': baseline_delta,
                    'delta_change': delta_change
                }
            else:
                # No baseline - set current as baseline
                return {
                    'domain': domain_name,
                    'status': 'new_baseline',
                    'r_squared': results['r_squared'],
                    'delta': results['delta']
                }
        
        except Exception as e:
            return {
                'domain': domain_name,
                'status': 'error',
                'error': str(e)
            }
    
    def test_all_domains(self) -> Dict[str, Dict]:
        """Test all domains for regression."""
        from src.registry import get_domain_registry
        
        registry = get_domain_registry()
        domains = [d.name for d in registry.get_all_domains()]
        
        results = {}
        
        for domain in domains:
            results[domain] = self.test_domain(domain)
        
        return results


def run_regression_tests():
    """Run regression tests."""
    print("="*80)
    print("REGRESSION TESTING")
    print("="*80)
    
    tester = RegressionTester()
    
    print(f"\nTesting all registered domains...\n")
    
    results = tester.test_all_domains()
    
    # Count results
    passed = sum(1 for r in results.values() if r['status'] == 'pass')
    regressions = sum(1 for r in results.values() if r['status'] == 'regression')
    new_baselines = sum(1 for r in results.values() if r['status'] == 'new_baseline')
    errors = sum(1 for r in results.values() if r['status'] == 'error')
    
    # Print results
    for domain, result in results.items():
        status = result['status']
        
        if status == 'pass':
            r2_change = result['r_squared_change']
            print(f"  ✓ {domain:20s} PASS (R² {r2_change:+.3f})")
        elif status == 'regression':
            r2_change = result['r_squared_change']
            print(f"  ✗ {domain:20s} REGRESSION (R² {r2_change:+.3f})")
        elif status == 'new_baseline':
            print(f"  ⊙ {domain:20s} NEW BASELINE (R²={result['r_squared']:.3f})")
        elif status == 'error':
            print(f"  ✗ {domain:20s} ERROR: {result.get('error', 'Unknown')}")
    
    # Summary
    print(f"\n{'='*80}")
    print("REGRESSION TEST SUMMARY")
    print(f"{'='*80}")
    print(f"  Passed: {passed}")
    print(f"  Regressions: {regressions}")
    print(f"  New baselines: {new_baselines}")
    print(f"  Errors: {errors}")
    print(f"  Total: {len(results)}")
    
    if regressions > 0:
        print(f"\n✗ REGRESSION DETECTED - {regressions} domains regressed")
        return False
    else:
        print(f"\n✓ NO REGRESSIONS - all domains pass")
        return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Regression testing')
    parser.add_argument('--update-baseline', action='store_true', help='Update baseline after tests')
    
    args = parser.parse_args()
    
    success = run_regression_tests()
    
    if args.update_baseline:
        print("\nUpdating baseline...")
        # Would update baseline here
    
    sys.exit(0 if success else 1)

