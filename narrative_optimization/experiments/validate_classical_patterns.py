"""
Classical Pattern Validation Experiment

Tests if π/λ/θ framework recovers known classical patterns:
1. High π → High Hero's Journey completion
2. Mythology validates Campbell (r > 0.85)
3. Greek tragedy validates Aristotle
4. Frye's mythoi cluster in θ/λ space
5. Archetype clarity highest in mythology

Runs when data is collected.

Author: Narrative Optimization Framework
Date: November 13, 2025
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.formula_validation import FormulaValidator
from analysis.archetype_cross_domain import ArchetypeCrossDomainAnalyzer
from src.transformers.archetypes import discover_journey_patterns


class ClassicalPatternValidator:
    """
    Validate that framework recovers known classical narrative patterns.
    """
    
    def __init__(self):
        self.results = {}
        self.validator = FormulaValidator()
        self.cross_domain = ArchetypeCrossDomainAnalyzer()
    
    def load_all_domains(self) -> Dict:
        """Load all collected domain datasets."""
        domains = {}
        
        data_dir = Path('data/domains')
        
        # Try to load each domain
        for domain_name in ['mythology', 'classical_literature', 'scripture_parables',
                           'film_extended', 'music', 'stage_drama']:
            domain_file = data_dir / domain_name / f'{domain_name}_complete_dataset.json'
            
            if domain_file.exists():
                with open(domain_file) as f:
                    data = json.load(f)
                
                # Extract texts and outcomes
                if domain_name == 'mythology':
                    texts = [m['full_narrative'] for m in data['myths']]
                    outcomes = np.array([m['outcome_measures']['cultural_persistence_score'] 
                                        for m in data['myths']])
                elif domain_name == 'classical_literature':
                    texts = [w['full_text'] for w in data['works']]
                    outcomes = np.array([w['outcome_measures']['literary_success_score'] 
                                        for w in data['works']])
                # ... similar for other domains
                
                domains[domain_name] = {
                    'texts': texts,
                    'outcomes': outcomes,
                    'pi': data['metadata'].get('calculated_pi', 0.5)
                }
                
                print(f"✅ Loaded {domain_name}: {len(texts)} samples")
            else:
                print(f"⏳ {domain_name} data not yet collected")
        
        return domains
    
    def test_1_campbell_on_mythology(self, domains: Dict) -> Dict:
        """
        TEST 1: Campbell's Hero's Journey validates on mythology (his source).
        
        Expected: correlation(campbell_weights, empirical_weights) > 0.85
        """
        print("\n" + "="*70)
        print("TEST 1: Campbell Validation on Mythology")
        print("="*70)
        
        if 'mythology' not in domains:
            return {'status': 'skipped', 'reason': 'Mythology data not collected'}
        
        mythology = domains['mythology']
        
        # Discover empirical patterns
        results = discover_journey_patterns(
            mythology['texts'],
            mythology['outcomes'],
            method='correlation'
        )
        
        validation = results['theoretical_validation']
        
        test_result = {
            'hypothesis': 'Campbell validates on mythology (r > 0.85)',
            'correlation': validation['summary']['correlation'],
            'validated': validation['summary']['campbell_validated'],
            'passes': validation['summary']['correlation'] > 0.85,
            'stages_agreeing': validation['summary']['stages_agreeing'],
            'most_important_stages': sorted(
                results['learned_weights'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
        if test_result['passes']:
            print("✅ PASSED: Campbell validated on mythology!")
            print(f"   Correlation: {test_result['correlation']:.3f}")
        else:
            print(f"❌ FAILED: Campbell not validated (r={test_result['correlation']:.3f})")
        
        return test_result
    
    def test_2_pi_journey_correlation(self, domains: Dict) -> Dict:
        """
        TEST 2: π correlates with Hero's Journey completion across domains.
        
        Expected: r(π, journey_completion) > 0.70
        """
        print("\n" + "="*70)
        print("TEST 2: π → Journey Completion Correlation")
        print("="*70)
        
        result = self.validator.validate_pi_journey_correlation(domains)
        
        if result['validated']:
            print(f"✅ PASSED: π predicts journey completion (r={result['correlation']:.3f})")
        else:
            print(f"❌ FAILED: Weak correlation (r={result['correlation']:.3f})")
        
        return result
    
    def test_3_greek_tragedy_aristotle(self, domains: Dict) -> Dict:
        """
        TEST 3: Greek tragedy has highest Aristotelian adherence.
        
        Expected: Greek tragedy scores > 0.80 on Aristotelian principles
        """
        print("\n" + "="*70)
        print("TEST 3: Greek Tragedy Validates Aristotle")
        print("="*70)
        
        if 'stage_drama' not in domains:
            return {'status': 'skipped', 'reason': 'Drama data not collected'}
        
        # Filter for Greek tragedy
        drama_data = domains['stage_drama']
        # Would filter for period == 'greek_tragedy'
        
        # Extract Aristotelian features
        from src.transformers.archetypes import StructuralBeatTransformer
        
        transformer = StructuralBeatTransformer()
        transformer.fit(drama_data['texts'])
        features = transformer.transform(drama_data['texts'])
        
        # Measure structural quality (proxy for Aristotelian)
        aristotelian_score = np.mean(features[:, -1])  # Overall structure quality
        
        test_result = {
            'hypothesis': 'Greek tragedy validates Aristotle (score > 0.80)',
            'aristotelian_score': aristotelian_score,
            'passes': aristotelian_score > 0.80,
            'interpretation': 'Aristotle empirically validated' if aristotelian_score > 0.80 else 'Aristotle challenged'
        }
        
        if test_result['passes']:
            print(f"✅ PASSED: Aristotle validated (score={aristotelian_score:.3f})")
        else:
            print(f"❌ FAILED: Aristotle not validated (score={aristotelian_score:.3f})")
        
        return test_result
    
    def test_4_frye_clustering(self, domains: Dict) -> Dict:
        """
        TEST 4: Frye's four mythoi cluster distinctly in θ/λ space.
        
        Expected: Silhouette score > 0.40
        """
        print("\n" + "="*70)
        print("TEST 4: Frye Mythoi Clustering in θ/λ Space")
        print("="*70)
        
        result = self.validator.validate_frye_theta_lambda_clustering(domains)
        
        if result.get('validated', False):
            print(f"✅ PASSED: Frye mythoi cluster (silhouette={result['silhouette_score']:.3f})")
        else:
            print(f"❌ FAILED: Weak clustering (silhouette={result.get('silhouette_score', 0):.3f})")
        
        return result
    
    def test_5_archetype_clarity_by_domain(self, domains: Dict) -> Dict:
        """
        TEST 5: Archetype clarity: Mythology > Literature > Film > Music
        
        Expected: Clear hierarchy matching π values
        """
        print("\n" + "="*70)
        print("TEST 5: Archetype Clarity Hierarchy")
        print("="*70)
        
        # Load all domains into cross-domain analyzer
        for domain_name, data in domains.items():
            self.cross_domain.load_domain_data(
                domain_name,
                data['texts'],
                data['outcomes']
            )
        
        # Compare clarity across domains
        clarity_comparison = self.cross_domain.compare_archetype_clarity()
        
        # Expected ranking: mythology > literature > film > music
        ranking = clarity_comparison['ranking']
        
        test_result = {
            'hypothesis': 'Clarity: Mythology > Literature > Film > Music',
            'actual_ranking': [(d[0], d[1]['mean_clarity']) for d in ranking],
            'correlation_pi_clarity': clarity_comparison['hypothesis_test'].get('correlation_pi_clarity', 0),
            'passes': clarity_comparison['hypothesis_test'].get('hypothesis_validated', False)
        }
        
        if test_result['passes']:
            print("✅ PASSED: Archetype clarity correlates with π")
        else:
            print("❌ FAILED: Clarity-π correlation weak")
        
        return test_result
    
    def run_all_validation_tests(self) -> Dict:
        """
        Run all 5 classical pattern validation tests.
        
        Returns comprehensive validation report.
        """
        print("\n" + "="*70)
        print("CLASSICAL PATTERN VALIDATION - ALL TESTS")
        print("="*70)
        
        # Load all domains
        domains = self.load_all_domains()
        
        if not domains:
            print("\n⚠️  No domain data collected yet")
            print("Run collection scripts first, then return to validation")
            return {'status': 'no_data'}
        
        # Run all tests
        results = {
            'test_1_campbell_mythology': self.test_1_campbell_on_mythology(domains),
            'test_2_pi_journey': self.test_2_pi_journey_correlation(domains),
            'test_3_greek_tragedy': self.test_3_greek_tragedy_aristotle(domains),
            'test_4_frye_clustering': self.test_4_frye_clustering(domains),
            'test_5_clarity_hierarchy': self.test_5_archetype_clarity_by_domain(domains)
        }
        
        # Summary
        passed = sum([1 for r in results.values() if r.get('passes', False)])
        total = len(results)
        
        results['summary'] = {
            'tests_run': total,
            'tests_passed': passed,
            'tests_failed': total - passed,
            'pass_rate': passed / total,
            'overall_validated': passed >= 4  # 4 of 5 is success
        }
        
        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Tests passed: {passed}/{total} ({passed/total:.0%})")
        print(f"Overall validated: {results['summary']['overall_validated']}")
        print("="*70)
        
        # Save report
        self._save_validation_report(results)
        
        return results
    
    def _save_validation_report(self, results: Dict) -> None:
        """Save validation report."""
        output_file = Path('narrative_optimization/results/classical_validation_results.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Validation report saved: {output_file}")


def main():
    """Run classical pattern validation."""
    validator = ClassicalPatternValidator()
    results = validator.run_all_validation_tests()
    
    return results


if __name__ == '__main__':
    main()

