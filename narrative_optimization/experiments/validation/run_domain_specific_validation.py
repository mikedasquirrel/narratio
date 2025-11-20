"""
Domain-Specific Validation (CORRECTED APPROACH)

Tests the CORRECT hypothesis:
- Each domain discovers its OWN best story archetype
- Formula works WITHIN that domain
- Then we do META-ANALYSIS across domain-specific formulas

This is NOT about cross-domain transfer.
This IS about discovering domain-specific patterns, then finding taxonomical relationships.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer
# Using available transformers
from sklearn.feature_extraction.text import TfidfVectorizer


class DomainSpecificValidator:
    """
    Validates that each domain can discover its own best story archetype.
    
    The CORRECT approach:
    1. For each domain, find which transformer(s) work best
    2. Discover domain-specific α parameter
    3. Validate WITHIN that domain
    4. Then compare formulas across domains (meta-analysis)
    """
    
    def __init__(self):
        self.domain_formulas = {}
        self.transformers = {
            'nominative': NominativeAnalysisTransformer(),
            'self_perception': SelfPerceptionTransformer(),
            'narrative_potential': NarrativePotentialTransformer()
        }
        # Add statistical baseline separately
        self.statistical_vectorizer = TfidfVectorizer(max_features=50)
    
    def discover_domain_formula(
        self,
        domain_name: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Discover the best story archetype for THIS SPECIFIC DOMAIN.
        
        Tests all transformers, finds which works best, calculates α parameter.
        
        Returns domain-specific formula that should work WITHIN this domain.
        """
        print(f"\n{'='*80}")
        print(f"DISCOVERING FORMULA FOR: {domain_name.upper()}")
        print(f"{'='*80}\n")
        
        results = {}
        
        # Test statistical baseline first
        try:
            print("Testing statistical (TF-IDF baseline)...")
            X_tfidf = self.statistical_vectorizer.fit_transform(X).toarray()
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            scores = cross_val_score(model, X_tfidf, y, cv=5)
            mean_score = np.mean(scores)
            results['statistical'] = {
                'accuracy': mean_score,
                'n_features': X_tfidf.shape[1]
            }
            print(f"  statistical: {mean_score:.3f} ({X_tfidf.shape[1]} features)")
        except Exception as e:
            print(f"  statistical: ERROR - {e}")
            results['statistical'] = {'accuracy': 0.0, 'error': str(e)}
        
        # Test each narrative transformer on THIS domain
        for trans_name, transformer in self.transformers.items():
            try:
                print(f"Testing {trans_name}...")
                
                # Fit and transform
                transformer.fit(X, y)
                X_trans = transformer.transform(X)
                
                # Cross-validate WITHIN domain
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                scores = cross_val_score(model, X_trans, y, cv=5)
                mean_score = np.mean(scores)
                
                results[trans_name] = {
                    'accuracy': mean_score,
                    'n_features': X_trans.shape[1]
                }
                
                print(f"  {trans_name}: {mean_score:.3f} ({X_trans.shape[1]} features)")
                
            except Exception as e:
                print(f"  {trans_name}: ERROR - {e}")
                results[trans_name] = {'accuracy': 0.0, 'error': str(e)}
        
        # Find best transformer for THIS domain
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {
                'domain': domain_name,
                'best_transformer': None,
                'alpha': None,
                'formula': 'No valid transformers',
                'error': 'All transformers failed'
            }
        
        best_name = max(valid_results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_score = valid_results[best_name]['accuracy']
        statistical_score = valid_results.get('statistical', {}).get('accuracy', 0)
        
        # Calculate α (balance between content and narrative)
        # α = 1.0 means pure statistical (plot-driven)
        # α = 0.0 means pure narrative (character-driven)
        if statistical_score > 0:
            alpha = statistical_score / (statistical_score + best_score) if best_name != 'statistical' else 1.0
        else:
            alpha = 0.0
        
        # Determine domain archetype
        if alpha > 0.7:
            archetype = "Plot-Driven (content matters most)"
        elif alpha > 0.5:
            archetype = "Hybrid (content + narrative)"
        elif alpha > 0.3:
            archetype = "Ensemble-Driven (relationships matter)"
        else:
            archetype = "Character-Driven (identity matters most)"
        
        formula = {
            'domain': domain_name,
            'best_transformer': best_name,
            'best_accuracy': best_score,
            'alpha': alpha,
            'archetype': archetype,
            'all_results': results,
            'recommendation': f"Use {best_name} transformer with α={alpha:.3f}"
        }
        
        self.domain_formulas[domain_name] = formula
        
        print(f"\n{'='*80}")
        print(f"FORMULA DISCOVERED FOR {domain_name.upper()}")
        print(f"{'='*80}")
        print(f"Best Transformer: {best_name}")
        print(f"Accuracy: {best_score:.3f}")
        print(f"α Parameter: {alpha:.3f}")
        print(f"Archetype: {archetype}")
        print(f"{'='*80}\n")
        
        return formula
    
    def validate_within_domain(
        self,
        domain_name: str,
        X: np.ndarray,
        y: np.ndarray,
        formula: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that the discovered formula works WITHIN this domain.
        
        This is the correct test: does the domain-specific formula predict
        well within its own domain?
        """
        print(f"\nValidating formula for {domain_name}...")
        
        best_transformer_name = formula['best_transformer']
        
        # Handle statistical separately
        if best_transformer_name == 'statistical':
            X_trans = self.statistical_vectorizer.transform(X).toarray()
        else:
            transformer = self.transformers[best_transformer_name]
            # Fit and transform
            transformer.fit(X, y)
            X_trans = transformer.transform(X)
        
        # Cross-validate
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X_trans, y, cv=5)
        
        validation = {
            'domain': domain_name,
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'min_accuracy': np.min(scores),
            'max_accuracy': np.max(scores),
            'validates': np.mean(scores) > 0.6,  # Threshold for validation
            'interpretation': self._interpret_validation(np.mean(scores))
        }
        
        print(f"  Mean Accuracy: {validation['mean_accuracy']:.3f} ± {validation['std_accuracy']:.3f}")
        print(f"  Range: [{validation['min_accuracy']:.3f}, {validation['max_accuracy']:.3f}]")
        print(f"  Validates: {'YES' if validation['validates'] else 'NO'}")
        
        return validation
    
    def meta_analysis_across_domains(self) -> Dict[str, Any]:
        """
        META-ANALYSIS: Compare domain-specific formulas taxonomically.
        
        This is where we find patterns:
        - Which domains cluster together?
        - Which α values group by domain type?
        - What are the taxonomical relationships?
        """
        print(f"\n{'='*80}")
        print("META-ANALYSIS: TAXONOMICAL PATTERNS ACROSS DOMAINS")
        print(f"{'='*80}\n")
        
        if not self.domain_formulas:
            return {'error': 'No domain formulas to analyze'}
        
        # Group by archetype
        archetypes = {}
        for domain, formula in self.domain_formulas.items():
            archetype = formula['archetype']
            if archetype not in archetypes:
                archetypes[archetype] = []
            archetypes[archetype].append({
                'domain': domain,
                'alpha': formula['alpha'],
                'transformer': formula['best_transformer'],
                'accuracy': formula['best_accuracy']
            })
        
        # Group by α ranges
        alpha_groups = {
            'plot_driven': [],  # α > 0.7
            'hybrid': [],  # 0.3 < α < 0.7
            'character_driven': []  # α < 0.3
        }
        
        for domain, formula in self.domain_formulas.items():
            alpha = formula['alpha']
            if alpha > 0.7:
                alpha_groups['plot_driven'].append(domain)
            elif alpha > 0.3:
                alpha_groups['hybrid'].append(domain)
            else:
                alpha_groups['character_driven'].append(domain)
        
        # Group by best transformer
        transformer_groups = {}
        for domain, formula in self.domain_formulas.items():
            trans = formula['best_transformer']
            if trans not in transformer_groups:
                transformer_groups[trans] = []
            transformer_groups[trans].append(domain)
        
        meta_analysis = {
            'n_domains_analyzed': len(self.domain_formulas),
            'archetypes': archetypes,
            'alpha_groups': alpha_groups,
            'transformer_groups': transformer_groups,
            'mean_alpha': np.mean([f['alpha'] for f in self.domain_formulas.values()]),
            'std_alpha': np.std([f['alpha'] for f in self.domain_formulas.values()]),
            'interpretation': self._interpret_meta_analysis(archetypes, alpha_groups)
        }
        
        # Print results
        print("ARCHETYPE DISTRIBUTION:")
        for archetype, domains in archetypes.items():
            print(f"  {archetype}: {len(domains)} domains")
            for d in domains:
                print(f"    - {d['domain']}: α={d['alpha']:.3f}, {d['transformer']}")
        
        print(f"\nα PARAMETER DISTRIBUTION:")
        print(f"  Mean α: {meta_analysis['mean_alpha']:.3f} ± {meta_analysis['std_alpha']:.3f}")
        print(f"  Plot-driven (α>0.7): {len(alpha_groups['plot_driven'])} domains")
        print(f"  Hybrid (0.3<α<0.7): {len(alpha_groups['hybrid'])} domains")
        print(f"  Character-driven (α<0.3): {len(alpha_groups['character_driven'])} domains")
        
        print(f"\nTRANSFORMER PREFERENCES:")
        for trans, domains in transformer_groups.items():
            print(f"  {trans}: {len(domains)} domains")
        
        print(f"\n{'='*80}")
        print("META-PATTERN:", meta_analysis['interpretation'])
        print(f"{'='*80}\n")
        
        return meta_analysis
    
    def _interpret_validation(self, accuracy: float) -> str:
        if accuracy > 0.7:
            return "STRONG validation - formula works well in this domain"
        elif accuracy > 0.6:
            return "MODERATE validation - formula shows promise"
        else:
            return "WEAK validation - formula may not capture this domain well"
    
    def _interpret_meta_analysis(
        self,
        archetypes: Dict,
        alpha_groups: Dict
    ) -> str:
        """Interpret the meta-patterns across domains."""
        
        # Determine dominant pattern
        n_plot = len(alpha_groups['plot_driven'])
        n_hybrid = len(alpha_groups['hybrid'])
        n_character = len(alpha_groups['character_driven'])
        
        total = n_plot + n_hybrid + n_character
        
        if n_character > n_plot and n_character > n_hybrid:
            return "Domains are predominantly CHARACTER-DRIVEN - narrative features dominate"
        elif n_plot > n_character and n_plot > n_hybrid:
            return "Domains are predominantly PLOT-DRIVEN - statistical content dominates"
        else:
            return "Domains show DIVERSE archetypes - different narrative strategies for different contexts"
    
    def generate_report(self) -> str:
        """Generate comprehensive domain-specific validation report."""
        report = []
        report.append("=" * 80)
        report.append("DOMAIN-SPECIFIC VALIDATION REPORT")
        report.append("Testing: Each domain discovers its OWN best story formula")
        report.append("=" * 80)
        report.append("")
        
        if not self.domain_formulas:
            report.append("No domains analyzed yet.")
            return "\n".join(report)
        
        report.append("DOMAIN-SPECIFIC FORMULAS:")
        report.append("-" * 80)
        
        for domain, formula in self.domain_formulas.items():
            report.append(f"\n{domain.upper()}:")
            report.append(f"  Archetype: {formula['archetype']}")
            report.append(f"  Best Transformer: {formula['best_transformer']}")
            report.append(f"  Accuracy: {formula['best_accuracy']:.3f}")
            report.append(f"  α Parameter: {formula['alpha']:.3f}")
            report.append(f"  Recommendation: {formula['recommendation']}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def demo_correct_approach():
    """Demonstrate the CORRECT validation approach."""
    print("\n")
    print("=" * 80)
    print("DOMAIN-SPECIFIC VALIDATION (CORRECTED APPROACH)")
    print("=" * 80)
    print("\nTesting the CORRECT hypothesis:")
    print("- Each domain discovers its OWN formula")
    print("- Validates WITHIN that domain")
    print("- Then meta-analysis finds taxonomical patterns")
    print("")
    
    validator = DomainSpecificValidator()
    
    # Create demonstration datasets for different domain types
    np.random.seed(42)
    
    # Domain 1: Character-driven (like mental health, profiles)
    print("\nCreating CHARACTER-DRIVEN domain...")
    X_character = np.array([
        f"Person {i} has {'strong' if i % 2 == 0 else 'weak'} identity with "
        f"{'high' if i % 3 == 0 else 'low'} narrative potential"
        for i in range(100)
    ])
    y_character = np.array([1 if i % 2 == 0 else 0 for i in range(100)])
    
    # Domain 2: Plot-driven (like news, technical content)
    print("Creating PLOT-DRIVEN domain...")
    X_plot = np.array([
        f"Event {i} occurred at location {i % 10} with magnitude {i % 5} "
        f"affecting {i % 3} regions"
        for i in range(100)
    ])
    y_plot = np.array([1 if i % 5 == 0 else 0 for i in range(100)])
    
    # Discover formulas
    formula_character = validator.discover_domain_formula('character_domain', X_character, y_character)
    formula_plot = validator.discover_domain_formula('plot_domain', X_plot, y_plot)
    
    # Validate within domains
    validator.validate_within_domain('character_domain', X_character, y_character, formula_character)
    validator.validate_within_domain('plot_domain', X_plot, y_plot, formula_plot)
    
    # Meta-analysis
    meta = validator.meta_analysis_across_domains()
    
    # Generate report
    print("\n")
    print(validator.generate_report())
    
    # Save results
    output_path = Path(__file__).parent / 'domain_specific_validation_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'domain_formulas': validator.domain_formulas,
            'meta_analysis': meta
        }, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("Each domain has its OWN best story formula.")
    print("The framework doesn't TRANSFER across domains.")
    print("It DISCOVERS domain-specific patterns, then finds META-PATTERNS.")
    print("This is the CORRECT way to validate the theory.")
    print("=" * 80)


if __name__ == "__main__":
    demo_correct_approach()

