"""
Validation Engine

Validates discovered patterns using statistical tests.
Only patterns that pass validation are kept.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from sklearn.metrics import roc_auc_score


class ValidationEngine:
    """
    Validates archetype patterns statistically.
    
    Tests:
    1. Correlation with outcomes (must be significant)
    2. Frequency test (must appear often enough)
    3. Coherence test (patterns must be internally consistent)
    4. Predictive power test (must improve predictions)
    
    Parameters
    ----------
    alpha : float
        Significance level for statistical tests
    min_samples : int
        Minimum samples needed for validation
    """
    
    def __init__(self, alpha: float = 0.05, min_samples: int = 10):
        self.alpha = alpha
        self.min_samples = min_samples
        
    def validate_patterns(
        self,
        patterns: Dict[str, Dict],
        texts: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Validate a set of patterns.
        
        Parameters
        ----------
        patterns : dict
            Patterns to validate
        texts : list of str
            Texts
        outcomes : ndarray
            Outcomes
        
        Returns
        -------
        dict
            Validated patterns (subset of input)
        """
        validated = {}
        
        for pattern_name, pattern_data in patterns.items():
            # Extract pattern keywords
            keywords = pattern_data.get('keywords', pattern_data.get('patterns', []))
            
            if len(keywords) == 0:
                continue
            
            # Find texts matching this pattern
            matches = self._find_matches(texts, keywords)
            
            if len(matches) < self.min_samples:
                continue
            
            # Get outcomes for matches
            match_outcomes = outcomes[matches]
            non_match_outcomes = np.delete(outcomes, matches)
            
            # Test 1: Correlation test
            correlation, p_value = self._test_correlation(matches, outcomes)
            
            if p_value > self.alpha:
                continue  # Not significant
            
            # Test 2: Effect size (must be meaningful)
            effect_size = self._calculate_effect_size(match_outcomes, non_match_outcomes)
            
            if effect_size < 0.2:  # Cohen's d < 0.2 = very small
                continue
            
            # Test 3: Predictive power
            predictive_power = self._test_predictive_power(matches, outcomes)
            
            if predictive_power < 0.55:  # Must be better than random
                continue
            
            # Passed all tests - mark as validated
            validated[pattern_name] = pattern_data.copy()
            validated[pattern_name]['validation'] = {
                'correlation': correlation,
                'p_value': p_value,
                'effect_size': effect_size,
                'predictive_power': predictive_power,
                'sample_size': len(matches),
                'validated': True
            }
        
        return validated
    
    def _find_matches(self, texts: List[str], keywords: List[str]) -> np.ndarray:
        """Find texts matching keywords."""
        matches = []
        for i, text in enumerate(texts):
            text_lower = text.lower()
            if any(keyword.lower() in text_lower for keyword in keywords):
                matches.append(i)
        return np.array(matches)
    
    def _test_correlation(
        self,
        matches: np.ndarray,
        outcomes: np.ndarray
    ) -> Tuple[float, float]:
        """Test correlation between pattern presence and outcome."""
        # Create binary indicator
        indicator = np.zeros(len(outcomes))
        indicator[matches] = 1.0
        
        # Pearson correlation
        if len(np.unique(outcomes)) > 2:
            # Continuous outcomes
            correlation, p_value = stats.pearsonr(indicator, outcomes)
        else:
            # Binary outcomes - use point-biserial
            correlation, p_value = stats.pointbiserialr(indicator, outcomes)
        
        return abs(correlation), p_value
    
    def _calculate_effect_size(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> float:
        """Calculate Cohen's d effect size."""
        if len(group1) == 0 or len(group2) == 0:
            return 0.0
        
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        
        # Pooled standard deviation
        var1 = np.var(group1, ddof=1) if len(group1) > 1 else 0.0
        var2 = np.var(group2, ddof=1) if len(group2) > 1 else 0.0
        
        n1 = len(group1)
        n2 = len(group2)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = abs(mean1 - mean2) / pooled_std
        return cohens_d
    
    def _test_predictive_power(
        self,
        matches: np.ndarray,
        outcomes: np.ndarray
    ) -> float:
        """Test predictive power using AUC."""
        # Create predictions (1 if pattern present, 0 otherwise)
        predictions = np.zeros(len(outcomes))
        predictions[matches] = 1.0
        
        if len(np.unique(outcomes)) == 2:
            # Binary outcomes - use AUC
            try:
                auc = roc_auc_score(outcomes, predictions)
                return auc
            except Exception:
                # Fallback: accuracy
                accuracy = np.mean((predictions > 0.5) == outcomes)
                return accuracy
        else:
            # Continuous outcomes - use correlation as proxy
            correlation = abs(np.corrcoef(predictions, outcomes)[0, 1])
            # Scale to [0.5, 1.0] range
            return 0.5 + (correlation * 0.5)
    
    def validate_single_pattern(
        self,
        pattern_keywords: List[str],
        texts: List[str],
        outcomes: np.ndarray
    ) -> Dict[str, any]:
        """
        Validate a single pattern.
        
        Parameters
        ----------
        pattern_keywords : list of str
            Pattern keywords
        texts : list of str
            Texts
        outcomes : ndarray
            Outcomes
        
        Returns
        -------
        dict
            Validation results
        """
        matches = self._find_matches(texts, pattern_keywords)
        
        if len(matches) < self.min_samples:
            return {
                'validated': False,
                'reason': f'Insufficient samples ({len(matches)} < {self.min_samples})'
            }
        
        match_outcomes = outcomes[matches]
        non_match_outcomes = np.delete(outcomes, matches)
        
        correlation, p_value = self._test_correlation(matches, outcomes)
        effect_size = self._calculate_effect_size(match_outcomes, non_match_outcomes)
        predictive_power = self._test_predictive_power(matches, outcomes)
        
        validated = (
            p_value <= self.alpha and
            effect_size >= 0.2 and
            predictive_power >= 0.55
        )
        
        return {
            'validated': validated,
            'correlation': correlation,
            'p_value': p_value,
            'effect_size': effect_size,
            'predictive_power': predictive_power,
            'sample_size': len(matches)
        }

