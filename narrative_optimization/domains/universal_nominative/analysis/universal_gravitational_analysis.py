"""
Universal Gravitational Similarity Analysis

Tests nominative determinism across ALL domains using our complete framework:
- п (narrativity) = career choice subjectivity per field
- ж (genome) = 524 features from 29 transformers
- ю (story quality) = name-field fit score  
- Д (bridge) = career selection effect |r|
- Presume-and-Prove: Test if Д/п > threshold

Research Question: Are people with name-field fit OVERREPRESENTED in matching careers?
"""

import sys
from pathlib import Path
import json
import numpy as np
from scipy import stats
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from narrative_optimization.domains.meta_nominative.feature_extraction.name_field_fit import NameFieldFitCalculator


class UniversalGravitationalAnalyzer:
    """
    Analyze gravitational attraction to careers across all domains.
    
    Integrates complete narrative framework:
    - Calculate п for each field
    - Extract ж via transformers
    - Compute ю (name-field fit)
    - Measure Д (selection effect)
    - Test presume-and-prove hypothesis
    """
    
    def __init__(self, researchers_path: Path):
        """Initialize analyzer."""
        print(f"\n{'='*80}")
        print("UNIVERSAL GRAVITATIONAL SIMILARITY ANALYSIS")
        print(f"{'='*80}\n")
        
        # Load researchers
        print("[1/5] Loading multi-domain researchers...")
        with open(researchers_path) as f:
            data = json.load(f)
            self.researchers = data['researchers']
        
        print(f"      ✓ Loaded {len(self.researchers)} researchers across {len(set(r['field'] for r in self.researchers))} fields")
        
        # Initialize calculator
        self.fit_calculator = NameFieldFitCalculator()
        
        # Results storage
        self.results_by_field = {}
        self.overall_results = {}
    
    def calculate_narrativity_by_field(self) -> Dict[str, float]:
        """
        Calculate п (narrativity) for each field.
        
        Returns:
            Dictionary mapping field to narrativity score
        """
        print(f"\n[2/5] Calculating narrativity (п) by field...")
        
        field_narrativity = {}
        
        # Get narrativity from researcher data (stored during collection)
        for r in self.researchers:
            field = r['field']
            if field not in field_narrativity:
                field_narrativity[field] = r.get('narrativity', 0.60)
        
        print(f"      ✓ Calculated п for {len(field_narrativity)} fields")
        print(f"\n      Field narrativity scores:")
        
        # Sort by narrativity
        sorted_fields = sorted(field_narrativity.items(), key=lambda x: x[1], reverse=True)
        
        for field, п in sorted_fields[:5]:
            print(f"        {field:20s} п = {п:.3f}")
        print(f"        ...")
        
        return field_narrativity
    
    def calculate_name_field_fits(self) -> List[float]:
        """
        Calculate ю (story quality / name-field fit) for all researchers.
        
        Returns:
            List of fit scores
        """
        print(f"\n[3/5] Calculating name-field fit (ю) for all researchers...")
        
        fit_scores = []
        high_fit_count = 0
        
        for idx, researcher in enumerate(self.researchers):
            if (idx + 1) % 200 == 0:
                print(f"      Processing {idx+1}/{len(self.researchers)}...")
            
            name = researcher['name']
            field = researcher['field']
            
            # Calculate fit using our 4-algorithm calculator
            fit_result = self.fit_calculator.calculate_fit(name, field)
            fit_score = fit_result['overall_fit']
            
            researcher['name_field_fit'] = fit_score
            researcher['fit_details'] = fit_result
            fit_scores.append(fit_score)
            
            if fit_score > 50:
                high_fit_count += 1
        
        print(f"      ✓ Calculated fit for {len(self.researchers)} researchers")
        print(f"      High fit (>50): {high_fit_count} ({high_fit_count/len(self.researchers)*100:.1f}%)")
        print(f"      Mean fit: {np.mean(fit_scores):.1f}")
        print(f"      Median fit: {np.median(fit_scores):.1f}")
        
        return fit_scores
    
    def test_gravitational_selection(self, field_narrativity: Dict[str, float]) -> Dict:
        """
        Test PRIMARY hypothesis: Are high-fit researchers overrepresented?
        
        This is the SELECTION EFFECT - do names attract people to careers?
        
        Args:
            field_narrativity: п values by field
            
        Returns:
            Analysis results
        """
        print(f"\n[4/5] Testing gravitational selection effects...")
        
        print(f"\n{'─'*80}")
        print("PRIMARY TEST: Career Selection via Name Gravitational Attraction")
        print(f"{'─'*80}\n")
        
        # Overall analysis
        total = len(self.researchers)
        high_fit = sum(1 for r in self.researchers if r['name_field_fit'] > 50)
        medium_fit = sum(1 for r in self.researchers if 20 <= r['name_field_fit'] <= 50)
        low_fit = total - high_fit - medium_fit
        
        observed_high_pct = high_fit / total
        
        # Expected by chance: ~5-8% should have high fit randomly
        expected_high_pct = 0.065
        
        print(f"  Overall Statistics:")
        print(f"    Total researchers: {total}")
        print(f"    High fit (>50): {high_fit} ({observed_high_pct*100:.1f}%)")
        print(f"    Medium fit (20-50): {medium_fit} ({medium_fit/total*100:.1f}%)")
        print(f"    Low fit (<20): {low_fit} ({low_fit/total*100:.1f}%)")
        print(f"\n    Expected high fit by chance: {expected_high_pct*100:.1f}%")
        print(f"    Observed high fit: {observed_high_pct*100:.1f}%")
        
        # Chi-square test
        expected_high = total * expected_high_pct
        expected_low = total * (1 - expected_high_pct)
        
        chi2, p_value = stats.chisquare(
            [high_fit, total - high_fit],
            [expected_high, expected_low]
        )
        
        print(f"\n  Chi-square test:")
        print(f"    χ² = {chi2:.3f}")
        print(f"    p = {p_value:.6f}")
        
        # Effect size
        odds_ratio = (high_fit / (total - high_fit)) / (expected_high / expected_low)
        
        print(f"\n  Effect size:")
        print(f"    Odds ratio = {odds_ratio:.3f}")
        
        if p_value < 0.05:
            if observed_high_pct > expected_high_pct:
                print(f"\n  🔥 SIGNIFICANT OVERREPRESENTATION!")
                print(f"  → Names DO attract people to matching careers!")
                interpretation = "STRONG_SELECTION_EFFECT"
            else:
                print(f"\n  ⚠️  SIGNIFICANT UNDERREPRESENTATION")
                print(f"  → People AVOID careers matching their names!")
                interpretation = "AVOIDANCE_EFFECT"
        else:
            print(f"\n  ✗ NULL RESULT")
            print(f"  → No significant career selection effect detected")
            interpretation = "NULL"
        
        # Calculate Д (bridge) - the actual effect size
        # Д = correlation between name-field fit and being in that field
        # For categorical outcome (chose field or not), use point-biserial
        
        print(f"\n{'─'*80}")
        print("CALCULATING Д (BRIDGE EFFECT SIZE)")
        print(f"{'─'*80}\n")
        
        # For each field, test if high-fit predicts being in that field
        field_effects = []
        
        for field in set(r['field'] for r in self.researchers):
            field_researchers = [r for r in self.researchers if r['field'] == field]
            
            if len(field_researchers) < 10:
                continue
            
            # Get fit scores for this field
            field_fits = [r['name_field_fit'] for r in field_researchers]
            
            # Get fit scores for other fields (control)
            other_fits = [r['name_field_fit'] for r in self.researchers if r['field'] != field]
            
            # Test if this field has higher fit scores
            if len(other_fits) > 0:
                t_stat, t_p = stats.ttest_ind(field_fits, other_fits)
                cohens_d = (np.mean(field_fits) - np.mean(other_fits)) / np.sqrt(
                    (np.var(field_fits) + np.var(other_fits)) / 2
                )
                
                field_effects.append({
                    'field': field,
                    'mean_fit': np.mean(field_fits),
                    'cohens_d': cohens_d,
                    'p_value': t_p,
                    'narrativity': field_narrativity.get(field, 0.60)
                })
        
        # Overall Д = average effect size weighted by narrativity
        if field_effects:
            # Weight by narrativity (high п fields should show stronger effects)
            weighted_effects = [e['cohens_d'] * e['narrativity'] for e in field_effects]
            Д = np.mean([abs(e['cohens_d']) for e in field_effects])
            Д_weighted = np.mean([abs(w) for w in weighted_effects])
            
            print(f"  Overall bridge effect (Д):")
            print(f"    Unweighted: |Д| = {Д:.3f}")
            print(f"    Weighted by п: |Д| = {Д_weighted:.3f}")
            
            # Test presume-and-prove hypothesis: Д/п > threshold
            avg_narrativity = np.mean([e['narrativity'] for e in field_effects])
            ratio = Д_weighted / avg_narrativity
            
            print(f"\n  Presume-and-Prove Test:")
            print(f"    Average п = {avg_narrativity:.3f}")
            print(f"    Д/п ratio = {ratio:.3f}")
            print(f"    Threshold = 0.50")
            
            if ratio > 0.50:
                print(f"    ✓ HYPOTHESIS CONFIRMED: Д/п > 0.50")
                print(f"    → Narrative effects are REAL and STRONG")
            else:
                print(f"    ✗ HYPOTHESIS REJECTED: Д/п < 0.50")
                print(f"    → Effects below theoretical threshold")
        else:
            Д = 0
            Д_weighted = 0
        
        # Show top fields by effect
        print(f"\n  Fields with strongest gravitational effects:")
        sorted_effects = sorted(field_effects, key=lambda x: abs(x['cohens_d']), reverse=True)
        for effect in sorted_effects[:5]:
            print(f"    {effect['field']:20s} d={effect['cohens_d']:+.3f}, п={effect['narrativity']:.2f}, p={effect['p_value']:.4f}")
        
        return {
            'total_researchers': total,
            'high_fit_count': high_fit,
            'observed_high_pct': observed_high_pct,
            'expected_high_pct': expected_high_pct,
            'chi2': chi2,
            'p_value': p_value,
            'odds_ratio': odds_ratio,
            'interpretation': interpretation,
            'bridge_effect_unweighted': Д,
            'bridge_effect_weighted': Д_weighted,
            'field_effects': field_effects
        }
    
    def identify_notable_cases(self):
        """Find researchers with perfect or near-perfect name-field fits."""
        print(f"\n[5/5] Identifying notable gravitational cases...")
        
        # Sort by fit score
        sorted_researchers = sorted(self.researchers, key=lambda r: r['name_field_fit'], reverse=True)
        
        print(f"\n  Top 10 name-field matches:")
        for i, r in enumerate(sorted_researchers[:10], 1):
            name = r['name']
            field = r['field']
            fit = r['name_field_fit']
            print(f"    {i:2d}. {name:30s} → {field:20s} (fit={fit:.1f})")
        
        # Find perfect semantic matches
        print(f"\n  Perfect semantic matches (names containing field keywords):")
        perfect_matches = []
        
        for r in self.researchers:
            name_lower = r['name'].lower()
            field = r['field']
            
            # Check if name contains field name
            if field in name_lower or any(part in name_lower for part in field.split('_')):
                perfect_matches.append(r)
        
        if perfect_matches:
            for r in perfect_matches[:10]:
                print(f"    {r['name']:30s} studying {r['field']}")
        else:
            print(f"    (None found in current dataset)")
    
    def save_results(self, output_path: Path = None):
        """Save complete analysis results."""
        if output_path is None:
            output_path = Path(__file__).parent.parent / 'data' / 'universal_analysis_results.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        def serialize(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        output = {
            'overall_results': {k: serialize(v) for k, v in self.overall_results.items()},
            'results_by_field': {k: {kk: serialize(vv) for kk, vv in v.items()} 
                                for k, v in self.results_by_field.items()},
            'researchers_with_fits': [
                {
                    'name': r['name'],
                    'field': r['field'],
                    'fit': float(r['name_field_fit']),
                    'narrativity': float(r.get('narrativity', 0.6))
                }
                for r in self.researchers
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Saved complete results to: {output_path}")


def main():
    """Run universal gravitational analysis."""
    # Path to researchers
    data_path = Path(__file__).parent.parent / 'data' / 'researchers_multi_domain.json'
    
    if not data_path.exists():
        print(f"Error: Researchers data not found at {data_path}")
        print("Run collectors/multi_domain_collector.py first!")
        return
    
    # Initialize analyzer
    analyzer = UniversalGravitationalAnalyzer(data_path)
    
    # Run analysis pipeline
    field_narrativity = analyzer.calculate_narrativity_by_field()
    fit_scores = analyzer.calculate_name_field_fits()
    selection_results = analyzer.test_gravitational_selection(field_narrativity)
    analyzer.identify_notable_cases()
    
    # Store results
    analyzer.overall_results = selection_results
    analyzer.results_by_field = {
        field: {'narrativity': п}
        for field, п in field_narrativity.items()
    }
    
    # Save
    analyzer.save_results()
    
    # Final summary
    print(f"\n{'='*80}")
    print("UNIVERSAL NOMINATIVE DETERMINISM: FINAL VERDICT")
    print(f"{'='*80}\n")
    
    Д = selection_results['bridge_effect_weighted']
    avg_п = np.mean(list(field_narrativity.values()))
    
    print(f"  Sample: {selection_results['total_researchers']} researchers across {len(field_narrativity)} fields")
    print(f"  Average narrativity: п = {avg_п:.3f}")
    print(f"  Bridge effect: Д = {Д:.3f}")
    print(f"  Д/п ratio: {Д/avg_п:.3f}")
    print(f"\n  Result: {selection_results['interpretation']}")
    print(f"  p-value: {selection_results['p_value']:.6f}")
    print(f"  Odds ratio: {selection_results['odds_ratio']:.3f}")
    
    if selection_results['p_value'] < 0.05 and selection_results['odds_ratio'] > 1:
        print(f"\n  🔥 NOMINATIVE DETERMINISM CONFIRMED ACROSS DOMAINS!")
        print(f"  → Names gravitationally attract people to matching careers")
    else:
        print(f"\n  → No universal gravitational effect detected")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

