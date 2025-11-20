"""
Domain Name Tethering Experiment

Tests the hypothesis: How you NAME a domain affects which methods work best.

Theory predicts that calling the same dataset "character study" vs "profile analysis"
vs "identity patterns" will cause different transformers to perform better.

This is a strong claim that requires rigorous testing.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transformers.nominative import NominativeAnalysisTransformer
from src.transformers.self_perception import SelfPerceptionTransformer
from src.transformers.narrative_potential import NarrativePotentialTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class DomainNameTetheringExperiment:
    """
    Tests if domain naming affects which methods work.
    
    Experimental Design:
    1. Take same dataset
    2. Rename domain in different ways
    3. Test which transformers perform best under each name
    4. Check if performance patterns change with naming
    
    If tethering is real: naming should affect results
    If tethering is false: naming should not matter (good null hypothesis)
    """
    
    def __init__(self):
        self.results = {}
    
    def run_experiment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_domain_name: str,
        alternative_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Run tethering experiment with multiple domain names.
        
        Parameters
        ----------
        X : array-like
            Text data
        y : array-like
            Labels
        base_domain_name : str
            Original domain name
        alternative_names : List[str]
            Alternative names for same domain
            
        Returns
        -------
        Dict mapping domain_name to performance_scores
        """
        all_names = [base_domain_name] + alternative_names
        
        transformers = {
            'nominative': NominativeAnalysisTransformer(),
            'self_perception': SelfPerceptionTransformer(),
            'narrative_potential': NarrativePotentialTransformer()
        }
        
        results = {}
        
        for domain_name in all_names:
            print(f"\nTesting with domain name: '{domain_name}'")
            
            # Create modified texts that mention domain name
            # (Simulates how naming might affect analysis)
            X_modified = [
                f"{domain_name} context: {text}"
                for text in X
            ]
            
            scores = {}
            
            for trans_name, transformer in transformers.items():
                # Fit and transform
                transformer.fit(X_modified, y)
                X_trans = transformer.transform(X_modified)
                
                # Test performance
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                cv_scores = cross_val_score(model, X_trans, y, cv=3)
                mean_score = np.mean(cv_scores)
                
                scores[trans_name] = mean_score
                
                print(f"  {trans_name}: {mean_score:.3f}")
            
            results[domain_name] = scores
        
        # Analyze results
        self.results = results
        return results
    
    def test_tethering_effect(self) -> Dict[str, any]:
        """
        Test if naming caused meaningful performance differences.
        
        Returns
        -------
        Dict with tethering analysis
        """
        if not self.results:
            return {'tethering_detected': False, 'reason': 'No results yet'}
        
        # For each transformer, check variance across domain names
        transformer_names = list(next(iter(self.results.values())).keys())
        
        variances = {}
        for trans_name in transformer_names:
            scores = [
                results[trans_name]
                for results in self.results.values()
            ]
            variances[trans_name] = np.var(scores)
        
        # Total variance
        total_variance = sum(variances.values())
        mean_variance = np.mean(list(variances.values()))
        
        # If variance > 0.01, naming has meaningful effect
        tethering_detected = mean_variance > 0.01
        
        # Find which transformer was most affected
        most_affected = max(variances.items(), key=lambda x: x[1])
        
        return {
            'tethering_detected': tethering_detected,
            'mean_variance': mean_variance,
            'total_variance': total_variance,
            'variances_by_transformer': variances,
            'most_affected_transformer': most_affected[0],
            'max_variance': most_affected[1],
            'interpretation': self._interpret_tethering(
                tethering_detected, mean_variance, most_affected
            )
        }
    
    def _interpret_tethering(
        self,
        detected: bool,
        variance: float,
        most_affected: Tuple[str, float]
    ) -> str:
        if detected:
            return (
                f"TETHERING DETECTED: Domain naming affects results. "
                f"Mean variance across names: {variance:.4f}. "
                f"Most affected: {most_affected[0]} (var={most_affected[1]:.4f}). "
                f"This supports the theory that how you NAME research affects what you find."
            )
        else:
            return (
                f"NO TETHERING: Domain naming does not significantly affect results. "
                f"Mean variance: {variance:.4f} (below 0.01 threshold). "
                f"Methods perform consistently regardless of naming. "
                f"Theory is NOT supported in this test."
            )
    
    def generate_report(self) -> str:
        """Generate experiment report."""
        report = []
        report.append("=" * 80)
        report.append("DOMAIN NAME TETHERING EXPERIMENT")
        report.append("=" * 80)
        report.append("")
        
        if not self.results:
            report.append("No results yet.")
            return "\n".join(report)
        
        report.append("PERFORMANCE BY DOMAIN NAME:")
        report.append("-" * 80)
        
        for domain_name, scores in self.results.items():
            report.append(f"\n{domain_name}:")
            for trans_name, score in scores.items():
                report.append(f"  {trans_name}: {score:.3f}")
        
        report.append("")
        report.append("TETHERING ANALYSIS:")
        report.append("-" * 80)
        
        analysis = self.test_tethering_effect()
        report.append(analysis['interpretation'])
        report.append(f"\nVariance by transformer:")
        for trans, var in analysis['variances_by_transformer'].items():
            report.append(f"  {trans}: {var:.4f}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def demo_tethering_experiment():
    """Demo experiment with synthetic data."""
    print("Domain Name Tethering Experiment - Demo")
    print("=" * 80)
    
    # Create synthetic data
    np.random.seed(42)
    
    texts = [
        f"Sample text {i} with various content about patterns and characteristics"
        for i in range(100)
    ]
    
    labels = np.random.randint(0, 2, size=100)
    
    # Run experiment
    experiment = DomainNameTetheringExperiment()
    
    results = experiment.run_experiment(
        X=texts,
        y=labels,
        base_domain_name="identity_patterns",
        alternative_names=[
            "character_study",
            "profile_analysis",
            "behavioral_signatures",
            "personality_dimensions"
        ]
    )
    
    # Generate report
    print("\n" + experiment.generate_report())


if __name__ == "__main__":
    demo_tethering_experiment()

