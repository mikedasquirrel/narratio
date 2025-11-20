"""
Narrative Advantage Calculator

Computes Д = r_narrative - r_baseline

Measures how much narrative features improve prediction beyond
what objective/baseline features alone explain.

This preserves free will (not deterministic) while showing
narrative provides significant edge.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


class NarrativeAdvantageCalculator:
    """
    Calculates Д (narrative advantage over baseline).
    
    Tests: Stories tend to win MORE than baseline predicts.
    Not determinism, but meaningful edge.
    """
    
    def __init__(self):
        self.results = {}
    
    def compute_advantage(
        self,
        domain_name: str,
        X_narrative_features: np.ndarray,  # ж extracted by transformers
        X_baseline_features: np.ndarray,   # Objective features only
        y: np.ndarray,                     # Outcomes (❊)
        narrative_quality: np.ndarray = None  # Pre-computed ю if available
    ) -> Dict:
        """
        Compute Д for domain.
        
        Parameters
        ----------
        domain_name : str
        X_narrative_features : array
            Full narrative features (ж)
        X_baseline_features : array
            Objective features only (no narrative)
        y : array
            Outcomes (❊)
        narrative_quality : array, optional
            Pre-computed ю scores
            
        Returns
        -------
        Dict with Д calculation and interpretation
        """
        # Compute ю if not provided
        if narrative_quality is None:
            narrative_quality = np.mean(X_narrative_features, axis=1)
        
        # Test baseline
        r_baseline, p_baseline = stats.pearsonr(
            np.mean(X_baseline_features, axis=1),
            y
        )
        
        # Test narrative
        r_narrative, p_narrative = stats.pearsonr(
            narrative_quality,
            y
        )
        
        # THE ADVANTAGE
        D_advantage = r_narrative - r_baseline
        
        # Interpretation
        if D_advantage > 0.30:
            strength = "LARGE"
            interpretation = f"Narrative provides huge edge (+{D_advantage:.2f}) over baseline"
        elif D_advantage > 0.15:
            strength = "MODERATE"
            interpretation = f"Narrative provides meaningful edge (+{D_advantage:.2f})"
        elif D_advantage > 0.05:
            strength = "SMALL"
            interpretation = f"Narrative provides modest edge (+{D_advantage:.2f})"
        else:
            strength = "MINIMAL"
            interpretation = f"Narrative adds little (+{D_advantage:.2f}) beyond baseline"
        
        result = {
            'domain': domain_name,
            'r_baseline': float(r_baseline),
            'r_narrative': float(r_narrative),
            'D_advantage': float(D_advantage),
            'advantage_strength': strength,
            'interpretation': interpretation,
            'p_baseline': float(p_baseline),
            'p_narrative': float(p_narrative)
        }
        
        self.results[domain_name] = result
        
        return result
    
    def test_universal_hypothesis(self) -> Dict:
        """
        Test if Д > threshold across domains.
        
        Hypothesis: Narrative provides edge (>0.10) in most contexts,
        even if not deterministic.
        """
        if not self.results:
            return {}
        
        passing = [d for d in self.results.values() if d['D_advantage'] > 0.10]
        passing_rate = len(passing) / len(self.results)
        
        mean_advantage = np.mean([d['D_advantage'] for d in self.results.values()])
        
        return {
            'n_domains': len(self.results),
            'passing_threshold': len(passing),
            'passing_rate': passing_rate,
            'mean_advantage': mean_advantage,
            'interpretation': self._interpret_universal(passing_rate, mean_advantage)
        }
    
    def _interpret_universal(self, rate: float, mean: float) -> str:
        if rate > 0.7 and mean > 0.20:
            return "Narrative provides substantial advantage across most domains. Universal tendency validated."
        elif rate > 0.5 and mean > 0.10:
            return "Narrative provides edge in majority of domains. Bounded universality."
        elif mean > 0.05:
            return "Narrative adds modest value on average. Domain-specific effects."
        else:
            return "Narrative advantage minimal. Reality dominates most contexts."


def demo_narrative_advantage():
    """Demonstrate narrative advantage calculation."""
    
    # Simulated example
    np.random.seed(42)
    
    # Baseline features (objective)
    X_baseline = np.random.randn(100, 5)  # Just 5 objective features
    
    # Narrative features (ж)
    X_narrative = np.random.randn(100, 50)  # Rich narrative features
    
    # Outcomes where narrative helps
    y = (
        0.3 * np.mean(X_baseline, axis=1) +  # 30% from baseline
        0.7 * np.mean(X_narrative, axis=1) +  # 70% from narrative
        np.random.randn(100) * 0.1  # Noise (free will)
    )
    y = (y > np.median(y)).astype(int)
    
    # Calculate advantage
    calc = NarrativeAdvantageCalculator()
    result = calc.compute_advantage(
        domain_name='demo',
        X_narrative_features=X_narrative,
        X_baseline_features=X_baseline,
        y=y
    )
    
    print("Narrative Advantage Demo:")
    print(f"  Baseline r: {result['r_baseline']:.3f}")
    print(f"  Narrative r: {result['r_narrative']:.3f}")
    print(f"  Д (advantage): {result['D_advantage']:.3f}")
    print(f"  {result['interpretation']}")


if __name__ == "__main__":
    demo_narrative_advantage()

