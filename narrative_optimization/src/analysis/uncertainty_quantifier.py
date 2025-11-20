"""
Uncertainty Quantification

Quantifies uncertainty in predictions and pattern discovery.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats


class UncertaintyQuantifier:
    """
    Quantify uncertainty in predictions and patterns.
    
    Types of uncertainty:
    - Epistemic: Model uncertainty (reducible with more data)
    - Aleatoric: Data uncertainty (irreducible noise)
    - Prediction intervals
    - Confidence intervals
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def prediction_interval(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals.
        
        Parameters
        ----------
        predictions : ndarray
            Point predictions
        residuals : ndarray
            Residuals from training
        
        Returns
        -------
        tuple
            (lower_bounds, upper_bounds)
        """
        # Estimate standard deviation of residuals
        sigma = np.std(residuals)
        
        # Z-score for confidence level
        z = stats.norm.ppf(1 - self.alpha / 2)
        
        # Intervals
        margin = z * sigma
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper
    
    def confidence_interval_mean(
        self,
        values: np.ndarray
    ) -> Tuple[float, float]:
        """
        Confidence interval for mean.
        
        Parameters
        ----------
        values : ndarray
            Values
        
        Returns
        -------
        tuple
            (lower, upper)
        """
        mean = np.mean(values)
        sem = stats.sem(values)
        
        margin = sem * stats.t.ppf(1 - self.alpha / 2, len(values) - 1)
        
        return mean - margin, mean + margin
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: callable,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for any statistic.
        
        Parameters
        ----------
        data : ndarray
            Data
        statistic_func : callable
            Function to compute statistic
        n_bootstrap : int
            Number of bootstrap samples
        
        Returns
        -------
        tuple
            (lower, upper)
        """
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(data, size=len(data), replace=True)
            stat = statistic_func(sample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Percentile method
        lower = np.percentile(bootstrap_stats, self.alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - self.alpha / 2) * 100)
        
        return lower, upper
    
    def epistemic_uncertainty(
        self,
        ensemble_predictions: List[np.ndarray]
    ) -> np.ndarray:
        """
        Estimate epistemic uncertainty from ensemble.
        
        Parameters
        ----------
        ensemble_predictions : list of ndarray
            Predictions from ensemble members
        
        Returns
        -------
        ndarray
            Epistemic uncertainty (variance across models)
        """
        ensemble_array = np.array(ensemble_predictions)
        return np.var(ensemble_array, axis=0)
    
    def aleatoric_uncertainty(
        self,
        residuals: np.ndarray
    ) -> float:
        """
        Estimate aleatoric uncertainty from residuals.
        
        Parameters
        ----------
        residuals : ndarray
            Prediction residuals
        
        Returns
        -------
        float
            Aleatoric uncertainty (irreducible noise)
        """
        return np.var(residuals)
    
    def total_uncertainty(
        self,
        epistemic: np.ndarray,
        aleatoric: float
    ) -> np.ndarray:
        """
        Total uncertainty = epistemic + aleatoric.
        
        Parameters
        ----------
        epistemic : ndarray
            Epistemic uncertainty
        aleatoric : float
            Aleatoric uncertainty
        
        Returns
        -------
        ndarray
            Total uncertainty
        """
        return epistemic + aleatoric
    
    def uncertainty_decomposition(
        self,
        ensemble_predictions: List[np.ndarray],
        true_values: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Decompose total uncertainty into components.
        
        Parameters
        ----------
        ensemble_predictions : list of ndarray
            Ensemble predictions
        true_values : ndarray
            True values
        
        Returns
        -------
        dict
            Uncertainty components
        """
        ensemble_array = np.array(ensemble_predictions)
        
        # Mean prediction
        mean_pred = np.mean(ensemble_array, axis=0)
        
        # Epistemic: variance across models
        epistemic = np.var(ensemble_array, axis=0)
        
        # Aleatoric: mean squared error
        mse = np.mean((mean_pred - true_values) ** 2)
        aleatoric = mse - np.mean(epistemic)
        
        # Total
        total = epistemic + aleatoric
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total,
            'epistemic_fraction': np.mean(epistemic / (total + 1e-8))
        }
    
    def pattern_uncertainty(
        self,
        pattern_data: Dict,
        n_samples: int
    ) -> Dict[str, float]:
        """
        Quantify uncertainty in a discovered pattern.
        
        Parameters
        ----------
        pattern_data : dict
            Pattern data
        n_samples : int
            Number of samples pattern was discovered from
        
        Returns
        -------
        dict
            Uncertainty metrics
        """
        # Correlation uncertainty
        correlation = pattern_data.get('correlation', 0.0)
        
        # Fisher Z-transformation for correlation CI
        if abs(correlation) < 0.999:
            z = 0.5 * np.log((1 + correlation) / (1 - correlation))
            se_z = 1.0 / np.sqrt(n_samples - 3)
            
            z_margin = stats.norm.ppf(1 - self.alpha / 2) * se_z
            
            z_lower = z - z_margin
            z_upper = z + z_margin
            
            # Transform back
            corr_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            corr_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        else:
            corr_lower = correlation
            corr_upper = correlation
        
        # Frequency uncertainty (binomial proportion)
        frequency = pattern_data.get('frequency', 0.0)
        
        if 0 < frequency < 1:
            # Wilson score interval
            z = stats.norm.ppf(1 - self.alpha / 2)
            
            denominator = 1 + z**2 / n_samples
            centre_adjusted = frequency + z**2 / (2 * n_samples)
            adjusted_std = np.sqrt((frequency * (1 - frequency) + z**2 / (4 * n_samples)) / n_samples)
            
            freq_lower = (centre_adjusted - z * adjusted_std) / denominator
            freq_upper = (centre_adjusted + z * adjusted_std) / denominator
        else:
            freq_lower = frequency
            freq_upper = frequency
        
        return {
            'correlation_ci': (corr_lower, corr_upper),
            'correlation_uncertainty': (corr_upper - corr_lower) / 2,
            'frequency_ci': (freq_lower, freq_upper),
            'frequency_uncertainty': (freq_upper - freq_lower) / 2
        }
    
    def calibration_error(
        self,
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Parameters
        ----------
        predicted_probs : ndarray
            Predicted probabilities
        true_labels : ndarray
            True binary labels
        n_bins : int
            Number of bins
        
        Returns
        -------
        float
            ECE score
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (predicted_probs >= bin_lower) & (predicted_probs < bin_upper)
            
            if np.sum(in_bin) > 0:
                # Average confidence in bin
                avg_confidence = np.mean(predicted_probs[in_bin])
                
                # Accuracy in bin
                accuracy = np.mean(true_labels[in_bin])
                
                # Weighted absolute difference
                ece += np.abs(avg_confidence - accuracy) * np.sum(in_bin) / len(predicted_probs)
        
        return ece

