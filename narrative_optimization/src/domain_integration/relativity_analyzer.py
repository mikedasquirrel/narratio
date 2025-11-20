"""
Cross-Domain Relativity Analyzer

Analyzes how findings in one domain inform understanding of others.
Implements bidirectional relativity: discoveries update the entire framework.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict


class CrossDomainRelativityAnalyzer:
    """
    Analyze relativistic effects between domains.
    
    Core concept: Findings in Domain A inform interpretation of Domain B,
    and vice versa. The framework understanding evolves with each domain tested.
    """
    
    def __init__(self):
        self.domain_results = {}
        self.relativity_network = defaultdict(dict)
        self.theoretical_updates = []
    
    def register_domain(self, domain_name: str, results: Dict[str, Any]):
        """Register results from a domain."""
        self.domain_results[domain_name] = results
    
    def analyze_bidirectional_effects(
        self,
        domain_a: str,
        domain_b: str
    ) -> Dict[str, Any]:
        """
        Analyze how domains inform each other.
        
        Parameters
        ----------
        domain_a : str
            First domain name
        domain_b : str
            Second domain name
        
        Returns
        -------
        bidirectional_analysis : dict
            How they influence each other
        """
        if domain_a not in self.domain_results or domain_b not in self.domain_results:
            return {'error': 'One or both domains not registered'}
        
        results_a = self.domain_results[domain_a]
        results_b = self.domain_results[domain_b]
        
        analysis = {
            'a_to_b': self._analyze_influence(results_a, results_b, domain_a, domain_b),
            'b_to_a': self._analyze_influence(results_b, results_a, domain_b, domain_a),
            'mutual_insights': []
        }
        
        # Identify mutual insights
        best_a = self._get_best_transformer(results_a)
        best_b = self._get_best_transformer(results_b)
        
        if best_a == best_b:
            analysis['mutual_insights'].append(
                f"Both domains benefit from {best_a} - suggests universal pattern"
            )
        else:
            analysis['mutual_insights'].append(
                f"{domain_a} benefits from {best_a}, {domain_b} from {best_b} - confirms domain specificity"
            )
        
        return analysis
    
    def _analyze_influence(
        self,
        source_results: Dict[str, Any],
        target_results: Dict[str, Any],
        source_name: str,
        target_name: str
    ) -> Dict[str, Any]:
        """Analyze how source domain informs target domain."""
        influence = {
            'predictions': [],
            'confirmations': [],
            'contradictions': []
        }
        
        # What source predicts about target
        source_best = self._get_best_transformer(source_results)
        
        if source_best in target_results.get('transformer_scores', {}):
            target_score = target_results['transformer_scores'][source_best]
            
            if target_score > 0.6:
                influence['confirmations'].append(
                    f"{source_best} works in {source_name} AND {target_name} - transferable pattern"
                )
            else:
                influence['contradictions'].append(
                    f"{source_best} works in {source_name} but not {target_name} - domain-specific"
                )
        
        return influence
    
    def _get_best_transformer(self, results: Dict[str, Any]) -> str:
        """Get best performing transformer from results."""
        if 'transformer_scores' in results:
            return max(results['transformer_scores'].items(), key=lambda x: x[1])[0]
        elif 'results' in results:
            # Parse from nested results structure
            best = ('statistical', 0.0)
            for name, score in results['results'].items():
                if score > best[1]:
                    best = (name, score)
            return best[0]
        return 'unknown'
    
    def build_relativity_graph(self, all_domain_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Build network graph of domain relationships.
        
        Parameters
        ----------
        all_domain_results : dict
            Results from all tested domains
        
        Returns
        -------
        relativity_graph : dict
            Network structure showing domain relationships
        """
        graph = {
            'nodes': [],
            'edges': [],
            'clusters': []
        }
        
        # Nodes: Each domain
        for domain_name, results in all_domain_results.items():
            graph['nodes'].append({
                'id': domain_name,
                'type': results.get('type', 'unknown'),
                'best_transformer': self._get_best_transformer(results),
                'performance': results.get('best_score', 0.0)
            })
        
        # Edges: Relativity between domains
        domain_names = list(all_domain_results.keys())
        
        for i, domain_a in enumerate(domain_names):
            for domain_b in domain_names[i+1:]:
                # Calculate relativity strength
                similarity = self._calculate_domain_similarity(
                    all_domain_results[domain_a],
                    all_domain_results[domain_b]
                )
                
                if similarity > 0.3:  # Threshold for meaningful relativity
                    graph['edges'].append({
                        'source': domain_a,
                        'target': domain_b,
                        'weight': similarity,
                        'type': self._classify_relativity_type(
                            all_domain_results[domain_a],
                            all_domain_results[domain_b]
                        )
                    })
        
        return graph
    
    def _calculate_domain_similarity(self, results_a: Dict, results_b: Dict) -> float:
        """Calculate similarity between two domains based on results."""
        # Type similarity
        type_sim = 1.0 if results_a.get('type') == results_b.get('type') else 0.5
        
        # Best transformer similarity
        best_a = self._get_best_transformer(results_a)
        best_b = self._get_best_transformer(results_b)
        transformer_sim = 1.0 if best_a == best_b else 0.3
        
        # Weighted combination
        similarity = 0.5 * type_sim + 0.5 * transformer_sim
        
        return similarity
    
    def _classify_relativity_type(self, results_a: Dict, results_b: Dict) -> str:
        """Classify the type of relativity between domains."""
        best_a = self._get_best_transformer(results_a)
        best_b = self._get_best_transformer(results_b)
        
        if best_a == best_b:
            return 'transfer'  # Same approach works
        elif best_a == 'statistical' and best_b != 'statistical':
            return 'inverse'  # Opposite patterns
        else:
            return 'moderating'  # Context-dependent
    
    def predict_performance(
        self,
        new_domain_spec: Dict[str, Any],
        based_on_existing: str
    ) -> Dict[str, Any]:
        """
        Predict performance for new domain based on existing domain.
        
        Uses relativity to make informed predictions before testing.
        """
        if based_on_existing not in self.domain_results:
            return {'error': f'Domain {based_on_existing} not found'}
        
        existing_results = self.domain_results[based_on_existing]
        
        predictions = {
            'expected_best': [],
            'expected_worst': [],
            'confidence': '',
            'reasoning': []
        }
        
        # Calculate similarity
        new_dims = set(new_domain_spec.get('dimensions', []))
        existing_dims = set(self.existing_domains[based_on_existing].get('dimensions_matter', []))
        
        overlap = new_dims & existing_dims
        
        if len(overlap) / len(new_dims) > 0.7:
            predictions['confidence'] = 'high'
            predictions['reasoning'].append(
                f"High overlap with {based_on_existing} ({len(overlap)}/{len(new_dims)} dimensions shared)"
            )
            
            # Predict same transformers will work
            if 'results' in existing_results:
                sorted_transformers = sorted(
                    existing_results['results'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                predictions['expected_best'] = [t[0] for t in sorted_transformers[:2]]
                predictions['expected_worst'] = [t[0] for t in sorted_transformers[-2:]]
        else:
            predictions['confidence'] = 'low'
            predictions['reasoning'].append(
                f"Low overlap with {based_on_existing} - domain may behave differently"
            )
        
        return predictions
    
    def update_theoretical_framework(
        self,
        new_domain_name: str,
        new_results: Dict[str, Any]
    ):
        """
        Update framework understanding based on new domain results.
        
        Generates insights about when/why dimensions matter.
        """
        update = {
            'domain': new_domain_name,
            'timestamp': str(np.datetime64('now')),
            'insights': [],
            'theory_updates': []
        }
        
        # Analyze what we learned
        best_transformer = self._get_best_transformer(new_results)
        domain_type = new_results.get('type', 'unknown')
        
        # Generate insights
        update['insights'].append(
            f"In {new_domain_name} ({domain_type}), {best_transformer} performed best"
        )
        
        # Check if this confirms or contradicts expectations
        similar_domains = [
            name for name, info in self.existing_domains.items()
            if info['type'] == domain_type
        ]
        
        if similar_domains:
            # Compare to similar domains
            for similar in similar_domains:
                similar_best = self.existing_domains[similar].get('dimensions_matter', [''])[0]
                if similar_best == best_transformer:
                    update['theory_updates'].append(
                        f"Confirms: {best_transformer} matters for {domain_type} domains"
                    )
                else:
                    update['theory_updates'].append(
                        f"Nuance: {domain_type} domains vary - {best_transformer} here vs {similar_best} in {similar}"
                    )
        
        self.theoretical_updates.append(update)
        return update


if __name__ == '__main__':
    print("Cross-Domain Relativity Analyzer Ready")
    print("Analyzes how findings in one domain inform others")

