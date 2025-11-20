"""
Adaptation Analyzer for Immigration Identity

Analyzes name adaptation patterns and their relationship to integration:
- Degree of adaptation (original → anglicized)
- Temporal trends (generational changes)
- Pattern types (full, partial, minimal, hybrid)

Author: Narrative Optimization Research
Date: November 2025
"""

from typing import Dict, List
import numpy as np
from scipy import stats


class AdaptationAnalyzer:
    """Analyze immigration name adaptation patterns."""
    
    def __init__(self):
        """Initialize adaptation analyzer."""
        pass
    
    def analyze_adaptation_integration_correlation(self, studies: List[Dict]) -> Dict:
        """
        Test: Adaptation degree → Integration success
        
        Parameters
        ----------
        studies : list of dict
            Immigration study records
        
        Returns
        -------
        dict
            Correlation analysis results
        """
        adaptations = []
        integrations = []
        
        for study in studies:
            adapt = study.get('adaptation_degree')
            integ = study.get('integration_score')
            
            if adapt is not None and integ is not None:
                adaptations.append(adapt)
                integrations.append(integ)
        
        if len(adaptations) < 3:
            return {'error': 'Insufficient data'}
        
        r, p = stats.pearsonr(adaptations, integrations)
        
        return {
            'n': len(adaptations),
            'correlation': r,
            'p_value': p,
            'significant': p < 0.05,
            'interpretation': f"Adaptation {'positively' if r > 0 else 'negatively'} correlates with integration (r={r:.3f})"
        }
    
    def analyze_generational_trends(self, studies: List[Dict]) -> Dict:
        """
        Analyze how adaptation changes across generations.
        
        Parameters
        ----------
        studies : list of dict
            Immigration studies with generation info
        
        Returns
        -------
        dict
            Generational analysis
        """
        by_generation = {}
        
        for study in studies:
            gen = study.get('generation', 1)
            adapt = study.get('adaptation_degree')
            
            if adapt is not None:
                if gen not in by_generation:
                    by_generation[gen] = []
                by_generation[gen].append(adapt)
        
        results = {}
        for gen, adaptations in sorted(by_generation.items()):
            results[f'generation_{gen}'] = {
                'mean_adaptation': np.mean(adaptations),
                'std': np.std(adaptations),
                'n': len(adaptations)
            }
        
        return {
            'generations': results,
            'trend': 'Adaptation increases with generations' if len(results) > 1 else 'Single generation'
        }


if __name__ == '__main__':
    # Demo
    analyzer = AdaptationAnalyzer()
    
    studies = [
        {'adaptation_degree': 0.3, 'integration_score': 0.5, 'generation': 1},
        {'adaptation_degree': 0.6, 'integration_score': 0.7, 'generation': 2},
        {'adaptation_degree': 0.9, 'integration_score': 0.8, 'generation': 3}
    ]
    
    result = analyzer.analyze_adaptation_integration_correlation(studies)
    print("Correlation:", result)
    
    trends = analyzer.analyze_generational_trends(studies)
    print("Trends:", trends)

