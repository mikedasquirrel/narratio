"""
Active Learning System

Identifies uncertain patterns and focuses learning on high-value areas.
Maximizes information gain per sample.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from scipy.stats import entropy


class ActiveLearner:
    """
    Active learning for efficient pattern discovery.
    
    Strategies:
    - Uncertainty sampling: Focus on ambiguous patterns
    - Query by committee: Disagreement between models
    - Expected information gain: Maximum learning value
    - Exploitation vs exploration: Balance refinement vs discovery
    """
    
    def __init__(self, exploration_rate: float = 0.2):
        self.exploration_rate = exploration_rate
        self.query_history = []
        
    def uncertainty_sampling(
        self,
        patterns: Dict[str, Dict],
        predictions: np.ndarray,
        n_queries: int = 5
    ) -> List[str]:
        """
        Select patterns with highest prediction uncertainty.
        
        Parameters
        ----------
        patterns : dict
            Available patterns
        predictions : ndarray
            Model predictions for each pattern
        n_queries : int
            Number of patterns to query
        
        Returns
        -------
        list
            Pattern names to focus on
        """
        uncertainties = {}
        
        for pattern_name in patterns.keys():
            # Uncertainty = entropy of prediction distribution
            # For binary: uncertainty highest near 0.5
            pred_entropy = self._calculate_uncertainty(predictions)
            uncertainties[pattern_name] = pred_entropy
        
        # Sort by uncertainty (highest first)
        sorted_patterns = sorted(
            uncertainties.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [name for name, _ in sorted_patterns[:n_queries]]
    
    def _calculate_uncertainty(self, predictions: np.ndarray) -> float:
        """Calculate prediction uncertainty."""
        if len(predictions) == 0:
            return 0.0
        
        # For continuous predictions: use variance
        if len(np.unique(predictions)) > 2:
            return np.var(predictions)
        
        # For binary: use entropy
        pos_rate = np.mean(predictions)
        if pos_rate == 0 or pos_rate == 1:
            return 0.0
        
        return -pos_rate * np.log2(pos_rate) - (1 - pos_rate) * np.log2(1 - pos_rate)
    
    def query_by_committee(
        self,
        patterns: Dict[str, Dict],
        committee_predictions: List[np.ndarray],
        n_queries: int = 5
    ) -> List[str]:
        """
        Select patterns where committee disagrees most.
        
        Parameters
        ----------
        patterns : dict
            Available patterns
        committee_predictions : list of ndarray
            Predictions from multiple models
        n_queries : int
            Number to query
        
        Returns
        -------
        list
            Pattern names with highest disagreement
        """
        disagreements = {}
        
        for pattern_name in patterns.keys():
            # Disagreement = variance across committee
            committee_preds = np.array([pred for pred in committee_predictions])
            disagreement = np.var(committee_preds, axis=0).mean()
            disagreements[pattern_name] = disagreement
        
        sorted_patterns = sorted(
            disagreements.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [name for name, _ in sorted_patterns[:n_queries]]
    
    def expected_information_gain(
        self,
        patterns: Dict[str, Dict],
        current_model_uncertainty: float,
        n_queries: int = 5
    ) -> List[str]:
        """
        Select patterns that maximize expected information gain.
        
        Parameters
        ----------
        patterns : dict
            Available patterns
        current_model_uncertainty : float
            Current model's uncertainty
        n_queries : int
            Number to query
        
        Returns
        -------
        list
            High-value patterns
        """
        gains = {}
        
        for pattern_name, pattern_data in patterns.items():
            # Information gain estimate
            frequency = pattern_data.get('frequency', 0.0)
            coherence = pattern_data.get('coherence', 0.0)
            
            # Gain = frequency * (1 - coherence) * current_uncertainty
            # High frequency, low coherence, high uncertainty = high gain
            gain = frequency * (1.0 - coherence) * current_model_uncertainty
            gains[pattern_name] = gain
        
        sorted_patterns = sorted(
            gains.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [name for name, _ in sorted_patterns[:n_queries]]
    
    def epsilon_greedy_selection(
        self,
        patterns: Dict[str, Dict],
        quality_scores: Dict[str, float],
        n_queries: int = 5
    ) -> List[str]:
        """
        Epsilon-greedy: exploit best patterns, explore new ones.
        
        Parameters
        ----------
        patterns : dict
            Available patterns
        quality_scores : dict
            Known quality of each pattern
        n_queries : int
            Number to select
        
        Returns
        -------
        list
            Mixed exploitation/exploration
        """
        n_exploit = int(n_queries * (1 - self.exploration_rate))
        n_explore = n_queries - n_exploit
        
        selected = []
        
        # Exploitation: best known patterns
        sorted_by_quality = sorted(
            quality_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        selected.extend([name for name, _ in sorted_by_quality[:n_exploit]])
        
        # Exploration: random from remaining
        remaining = [name for name in patterns.keys() if name not in selected]
        if len(remaining) > 0:
            explored = np.random.choice(
                remaining,
                size=min(n_explore, len(remaining)),
                replace=False
            )
            selected.extend(explored.tolist())
        
        return selected[:n_queries]
    
    def thompson_sampling(
        self,
        patterns: Dict[str, Dict],
        quality_estimates: Dict[str, Tuple[float, float]],
        n_queries: int = 5
    ) -> List[str]:
        """
        Thompson sampling: sample from quality distribution.
        
        Parameters
        ----------
        patterns : dict
            Available patterns
        quality_estimates : dict
            Pattern name -> (mean, std) of quality
        n_queries : int
            Number to select
        
        Returns
        -------
        list
            Sampled patterns
        """
        samples = {}
        
        for pattern_name in patterns.keys():
            if pattern_name in quality_estimates:
                mean, std = quality_estimates[pattern_name]
            else:
                mean, std = 0.5, 0.5  # Uninformative prior
            
            # Sample from distribution
            sample = np.random.normal(mean, std)
            samples[pattern_name] = sample
        
        # Select top sampled
        sorted_patterns = sorted(
            samples.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [name for name, _ in sorted_patterns[:n_queries]]
    
    def record_query(
        self,
        pattern_name: str,
        query_type: str,
        result: Dict
    ):
        """Record query for learning history."""
        self.query_history.append({
            'pattern': pattern_name,
            'type': query_type,
            'result': result
        })
    
    def get_query_statistics(self) -> Dict:
        """Get statistics on query efficiency."""
        if len(self.query_history) == 0:
            return {}
        
        by_type = {}
        for query in self.query_history:
            qtype = query['type']
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(query['result'].get('improvement', 0.0))
        
        stats = {}
        for qtype, improvements in by_type.items():
            stats[qtype] = {
                'count': len(improvements),
                'mean_improvement': np.mean(improvements),
                'total_improvement': np.sum(improvements)
            }
        
        return stats

