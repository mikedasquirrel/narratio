"""
Meta-Regression Tools for Cross-Domain Analysis

Implements meta-analytic models across all domains:
- Visibility moderation regression
- Heterogeneity analysis (I², τ²)
- Publication bias checks
- Subgroup meta-analyses

Author: Narrative Optimization Research
Date: November 2025
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MetaAnalysisResult:
    """Results from meta-analysis."""
    overall_effect: float
    ci_lower: float
    ci_upper: float
    heterogeneity_i2: float
    heterogeneity_tau2: float
    p_value: float
    n_studies: int


class MetaRegression:
    """
    Meta-regression for cross-domain narrative effects.
    
    Implements visibility moderation model and heterogeneity analysis.
    """
    
    def __init__(self):
        """Initialize meta-regression."""
        pass
    
    def fit_visibility_model(self, visibilities: List[float],
                            effects: List[float],
                            genre_congruences: Optional[List[float]] = None) -> Dict:
        """
        Fit visibility moderation model.
        
        Model: Effect = α - β₁(Visibility/100) + β₂(GenreCongruence)
        
        Parameters
        ----------
        visibilities : list of float
            Visibility scores for each domain
        effects : list of float
            Observed effect sizes (correlations)
        genre_congruences : list of float, optional
            Genre congruence scores
        
        Returns
        -------
        dict
            Regression results
        """
        visibilities = np.array(visibilities)
        effects = np.array(effects)
        
        # Normalize visibility to 0-1
        vis_norm = visibilities / 100
        
        if genre_congruences is not None:
            # Full model with genre congruence
            X = np.column_stack([np.ones(len(vis_norm)), vis_norm, np.array(genre_congruences)])
            
            # OLS regression
            beta = np.linalg.lstsq(X, effects, rcond=None)[0]
            
            # Predictions
            pred = X @ beta
            
            # R²
            ss_res = np.sum((effects - pred) ** 2)
            ss_tot = np.sum((effects - np.mean(effects)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'intercept': beta[0],
                'visibility_coef': beta[1],
                'genre_coef': beta[2],
                'r_squared': r_squared,
                'n_domains': len(effects),
                'formula': f"Effect = {beta[0]:.3f} {beta[1]:+.3f}(Vis/100) {beta[2]:+.3f}(Genre)"
            }
        else:
            # Simple model (visibility only)
            X = np.column_stack([np.ones(len(vis_norm)), vis_norm])
            
            beta = np.linalg.lstsq(X, effects, rcond=None)[0]
            pred = X @ beta
            
            ss_res = np.sum((effects - pred) ** 2)
            ss_tot = np.sum((effects - np.mean(effects)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'intercept': beta[0],
                'visibility_coef': beta[1],
                'r_squared': r_squared,
                'n_domains': len(effects),
                'formula': f"Effect = {beta[0]:.3f} {beta[1]:+.3f}(Vis/100)"
            }
    
    def calculate_heterogeneity(self, effects: List[float],
                               variances: Optional[List[float]] = None) -> Dict:
        """
        Calculate heterogeneity statistics (I², τ²).
        
        Parameters
        ----------
        effects : list of float
            Effect sizes from each study
        variances : list of float, optional
            Within-study variances
        
        Returns
        -------
        dict
            Heterogeneity statistics
        """
        effects = np.array(effects)
        n = len(effects)
        
        if variances is None:
            # Assume equal variances
            variances = np.ones(n) * 0.01
        else:
            variances = np.array(variances)
        
        # Calculate Q statistic
        weights = 1 / variances
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        Q = np.sum(weights * (effects - weighted_mean) ** 2)
        
        # Degrees of freedom
        df = n - 1
        
        # I² (percentage of variance due to heterogeneity)
        I2 = max(0, (Q - df) / Q) * 100
        
        # τ² (between-study variance)
        C = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
        tau2 = max(0, (Q - df) / C)
        
        # P-value for Q statistic
        p_value = 1 - stats.chi2.cdf(Q, df)
        
        return {
            'Q': Q,
            'df': df,
            'p_value': p_value,
            'I2': I2,
            'tau2': tau2,
            'interpretation': self._interpret_heterogeneity(I2)
        }
    
    def _interpret_heterogeneity(self, I2: float) -> str:
        """Interpret I² statistic."""
        if I2 < 25:
            return 'Low heterogeneity'
        elif I2 < 50:
            return 'Moderate heterogeneity'
        elif I2 < 75:
            return 'Substantial heterogeneity'
        else:
            return 'Considerable heterogeneity'
    
    def subgroup_analysis(self, effects: List[float],
                         subgroups: List[str]) -> Dict:
        """
        Perform subgroup meta-analysis.
        
        Parameters
        ----------
        effects : list of float
            Effect sizes
        subgroups : list of str
            Subgroup labels for each effect
        
        Returns
        -------
        dict
            Subgroup analysis results
        """
        effects = np.array(effects)
        unique_subgroups = list(set(subgroups))
        
        results = {}
        for subgroup in unique_subgroups:
            mask = np.array([s == subgroup for s in subgroups])
            sub_effects = effects[mask]
            
            if len(sub_effects) > 0:
                results[subgroup] = {
                    'n': len(sub_effects),
                    'mean': np.mean(sub_effects),
                    'std': np.std(sub_effects),
                    'min': np.min(sub_effects),
                    'max': np.max(sub_effects)
                }
        
        # Test differences between subgroups
        if len(unique_subgroups) > 1:
            groups = [effects[np.array([s == sg for s in subgroups])] 
                     for sg in unique_subgroups]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                results['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
        
        return results


if __name__ == '__main__':
    # Demo
    meta = MetaRegression()
    
    # Example data
    visibilities = [25, 30, 50, 75, 95]
    effects = [0.32, 0.28, 0.18, 0.24, 0.00]
    
    model = meta.fit_visibility_model(visibilities, effects)
    print("Model:", model['formula'])
    print(f"R² = {model['r_squared']:.3f}")
    
    het = meta.calculate_heterogeneity(effects)
    print(f"\nHeterogeneity: I² = {het['I2']:.1f}%")
    print(f"Interpretation: {het['interpretation']}")

