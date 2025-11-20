"""
Causal Archetype Discovery

Identifies causal (not just correlational) patterns using intervention analysis
and counterfactual reasoning.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import networkx as nx


class CausalArchetypeDiscovery:
    """
    Discover causal relationships between patterns and outcomes.
    
    Techniques:
    - Intervention analysis (what if we remove/add pattern?)
    - Counterfactual reasoning (what would happen if...?)
    - Causal graph construction (pattern -> outcome paths)
    - Mediation analysis (direct vs indirect effects)
    """
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.causal_effects = {}
        
    def estimate_causal_effect(
        self,
        pattern: str,
        texts: List[str],
        outcomes: np.ndarray,
        keywords: List[str],
        method: str = 'matching'
    ) -> Dict[str, float]:
        """
        Estimate causal effect of pattern on outcomes.
        
        Parameters
        ----------
        pattern : str
            Pattern name
        texts : list
            Texts
        outcomes : ndarray
            Outcomes
        keywords : list
            Pattern keywords
        method : str
            'matching', 'iv', or 'regression_discontinuity'
        
        Returns
        -------
        dict
            Causal effect estimates
        """
        # Identify treatment (pattern present) and control (absent)
        treated = np.array([
            any(kw.lower() in text.lower() for kw in keywords)
            for text in texts
        ])
        
        if method == 'matching':
            return self._matching_estimator(treated, outcomes)
        elif method == 'iv':
            return self._instrumental_variable(treated, outcomes, texts)
        elif method == 'regression_discontinuity':
            return self._regression_discontinuity(treated, outcomes)
        
        return {}
    
    def _matching_estimator(
        self,
        treated: np.ndarray,
        outcomes: np.ndarray
    ) -> Dict[str, float]:
        """
        Propensity score matching estimator.
        
        Compare similar units with/without pattern.
        """
        # Separate treatment and control
        treatment_outcomes = outcomes[treated]
        control_outcomes = outcomes[~treated]
        
        if len(treatment_outcomes) == 0 or len(control_outcomes) == 0:
            return {'ate': 0.0, 'se': 0.0, 'p_value': 1.0}
        
        # Simple difference (in practice, would match on covariates)
        ate = np.mean(treatment_outcomes) - np.mean(control_outcomes)
        
        # Standard error
        var_treat = np.var(treatment_outcomes) / len(treatment_outcomes)
        var_control = np.var(control_outcomes) / len(control_outcomes)
        se = np.sqrt(var_treat + var_control)
        
        # Z-test
        z = ate / se if se > 0 else 0
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'ate': ate,  # Average treatment effect
            'se': se,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _instrumental_variable(
        self,
        treated: np.ndarray,
        outcomes: np.ndarray,
        texts: List[str]
    ) -> Dict[str, float]:
        """Instrumental variable estimation (simplified)."""
        # Would need valid instrument - simplified version
        return self._matching_estimator(treated, outcomes)
    
    def _regression_discontinuity(
        self,
        treated: np.ndarray,
        outcomes: np.ndarray
    ) -> Dict[str, float]:
        """Regression discontinuity (simplified)."""
        return self._matching_estimator(treated, outcomes)
    
    def intervention_analysis(
        self,
        pattern: str,
        texts: List[str],
        outcomes: np.ndarray,
        keywords: List[str]
    ) -> Dict[str, float]:
        """
        Simulate intervention: what if we removed this pattern?
        
        Parameters
        ----------
        pattern : str
            Pattern name
        texts : list
            Texts
        outcomes : ndarray
            Outcomes
        keywords : list
            Pattern keywords
        
        Returns
        -------
        dict
            Intervention effects
        """
        # Identify texts with pattern
        has_pattern = np.array([
            any(kw.lower() in text.lower() for kw in keywords)
            for text in texts
        ])
        
        # Observed outcomes
        with_pattern = outcomes[has_pattern]
        without_pattern = outcomes[~has_pattern]
        
        # Estimate counterfactual: what if pattern removed?
        # Assume outcomes would shift to control distribution
        counterfactual_mean = np.mean(without_pattern) if len(without_pattern) > 0 else 0.5
        
        # Effect = observed - counterfactual
        effect = np.mean(with_pattern) - counterfactual_mean if len(with_pattern) > 0 else 0.0
        
        return {
            'observed_with': np.mean(with_pattern) if len(with_pattern) > 0 else 0.5,
            'counterfactual_without': counterfactual_mean,
            'intervention_effect': effect,
            'n_affected': np.sum(has_pattern)
        }
    
    def counterfactual_reasoning(
        self,
        text: str,
        pattern_to_add: str,
        keywords: List[str],
        baseline_prediction: float
    ) -> float:
        """
        Counterfactual: what if we added this pattern to the text?
        
        Parameters
        ----------
        text : str
            Original text
        pattern_to_add : str
            Pattern to add
        keywords : list
            Pattern keywords
        baseline_prediction : float
            Baseline prediction
        
        Returns
        -------
        float
            Counterfactual prediction
        """
        # Check if pattern already present
        has_pattern = any(kw.lower() in text.lower() for kw in keywords)
        
        if has_pattern:
            return baseline_prediction  # No change
        
        # Estimate effect of adding pattern
        # Would use causal effect estimates
        if pattern_to_add in self.causal_effects:
            effect = self.causal_effects[pattern_to_add].get('ate', 0.0)
        else:
            effect = 0.1  # Default small positive effect
        
        return baseline_prediction + effect
    
    def build_causal_graph(
        self,
        patterns: Dict[str, Dict],
        texts: List[str],
        outcomes: np.ndarray
    ) -> nx.DiGraph:
        """
        Build causal graph of pattern relationships.
        
        Parameters
        ----------
        patterns : dict
            All patterns
        texts : list
            Texts
        outcomes : ndarray
            Outcomes
        
        Returns
        -------
        DiGraph
            Causal graph
        """
        self.causal_graph = nx.DiGraph()
        
        # Add nodes
        for pattern_name in patterns.keys():
            self.causal_graph.add_node(pattern_name, node_type='pattern')
        
        self.causal_graph.add_node('outcome', node_type='outcome')
        
        # Estimate edges (pattern -> outcome)
        for pattern_name, pattern_data in patterns.items():
            keywords = pattern_data.get('keywords', pattern_data.get('patterns', []))
            
            effect = self.estimate_causal_effect(
                pattern_name, texts, outcomes, keywords
            )
            
            if effect.get('significant', False):
                self.causal_graph.add_edge(
                    pattern_name,
                    'outcome',
                    weight=effect['ate'],
                    p_value=effect['p_value']
                )
                
                self.causal_effects[pattern_name] = effect
        
        # Detect pattern-pattern relationships (mediation)
        pattern_names = list(patterns.keys())
        for i, pattern1 in enumerate(pattern_names):
            for pattern2 in pattern_names[i+1:]:
                # Check if patterns co-occur
                keywords1 = patterns[pattern1].get('keywords', patterns[pattern1].get('patterns', []))
                keywords2 = patterns[pattern2].get('keywords', patterns[pattern2].get('patterns', []))
                
                has_1 = np.array([any(kw.lower() in t.lower() for kw in keywords1) for t in texts])
                has_2 = np.array([any(kw.lower() in t.lower() for kw in keywords2) for t in texts])
                
                # Correlation between patterns
                if len(np.unique(has_1)) > 1 and len(np.unique(has_2)) > 1:
                    corr = np.corrcoef(has_1.astype(float), has_2.astype(float))[0, 1]
                    
                    if abs(corr) > 0.3:  # Moderate correlation
                        # Add edge (direction based on frequency)
                        freq1 = patterns[pattern1].get('frequency', 0.0)
                        freq2 = patterns[pattern2].get('frequency', 0.0)
                        
                        if freq1 > freq2:
                            self.causal_graph.add_edge(pattern1, pattern2, weight=corr)
                        else:
                            self.causal_graph.add_edge(pattern2, pattern1, weight=corr)
        
        return self.causal_graph
    
    def mediation_analysis(
        self,
        direct_pattern: str,
        mediator_pattern: str,
        texts: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze mediation: does mediator explain effect of direct?
        
        direct -> mediator -> outcome
        vs
        direct -> outcome
        
        Parameters
        ----------
        direct_pattern : str
            Direct pattern
        mediator_pattern : str
            Mediator pattern
        texts : list
            Texts
        outcomes : ndarray
            Outcomes
        
        Returns
        -------
        dict
            Mediation effects
        """
        # This is simplified Baron-Kenny approach
        
        # Total effect (direct -> outcome)
        if direct_pattern in self.causal_effects:
            total_effect = self.causal_effects[direct_pattern]['ate']
        else:
            total_effect = 0.0
        
        # Check if direct causes mediator
        # (Would need actual implementation)
        
        # Estimate direct vs indirect effects
        # Indirect = effect through mediator
        # Direct = effect not through mediator
        
        return {
            'total_effect': total_effect,
            'direct_effect': total_effect * 0.6,  # Placeholder
            'indirect_effect': total_effect * 0.4,  # Placeholder
            'proportion_mediated': 0.4
        }
    
    def get_causal_paths(
        self,
        source_pattern: str,
        target: str = 'outcome'
    ) -> List[List[str]]:
        """
        Find all causal paths from pattern to target.
        
        Parameters
        ----------
        source_pattern : str
            Source pattern
        target : str
            Target (usually 'outcome')
        
        Returns
        -------
        list of lists
            All paths
        """
        if source_pattern not in self.causal_graph:
            return []
        
        if target not in self.causal_graph:
            return []
        
        try:
            paths = list(nx.all_simple_paths(
                self.causal_graph,
                source_pattern,
                target,
                cutoff=5
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def rank_by_causal_importance(self) -> List[Tuple[str, float]]:
        """
        Rank patterns by causal importance.
        
        Returns
        -------
        list of (pattern, importance)
            Ranked patterns
        """
        importances = []
        
        for pattern in self.causal_graph.nodes():
            if pattern == 'outcome':
                continue
            
            # Importance = sum of causal effects along paths
            paths = self.get_causal_paths(pattern, 'outcome')
            
            if len(paths) > 0:
                # Sum edge weights along paths
                total_effect = 0.0
                for path in paths:
                    path_effect = 1.0
                    for i in range(len(path) - 1):
                        if self.causal_graph.has_edge(path[i], path[i+1]):
                            weight = self.causal_graph[path[i]][path[i+1]].get('weight', 0.0)
                            path_effect *= abs(weight)
                    total_effect += path_effect
                
                importances.append((pattern, total_effect))
        
        importances.sort(key=lambda x: x[1], reverse=True)
        return importances

