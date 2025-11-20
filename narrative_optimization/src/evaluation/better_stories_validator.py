"""
Better Stories Win Validator

Systematic validation of the central thesis across domains.

Tests whether "better stories" (higher narrative quality scores) actually
predict better outcomes, or whether this is wishful thinking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings


@dataclass
class BetterStoriesResult:
    """Result from testing 'better stories win' in one domain."""
    domain: str
    narrative_quality_metric: str
    correlation_with_outcome: float
    p_value: float
    effect_size_r2: float
    validates_thesis: bool
    evidence: Dict[str, Any]
    interpretation: str
    
    def __repr__(self):
        status = "✓ VALIDATED" if self.validates_thesis else "✗ REFUTED"
        return (
            f"{status} | {self.domain}: r={self.correlation_with_outcome:.3f}, "
            f"R²={self.effect_size_r2:.3f}, p={self.p_value:.4f}"
        )


class BetterStoriesValidator:
    """
    Tests the core optimistic thesis: "Better stories win over time."
    
    Operationalizes "better story" as:
    1. Higher narrative quality scores (from transformers)
    2. Higher coherence across narrative dimensions
    3. Better "fit" with domain genre expectations
    
    Tests if these predict:
    1. Better outcomes (wins, success, adoption, etc.)
    2. Over time (not just immediately)
    3. Across domains (not just one cherry-picked case)
    
    Honest assessment: Reports null findings, domain-specific failures,
    and alternative explanations (reverse causation, confounds, etc.)
    """
    
    def __init__(self, threshold_r: float = 0.20):
        """
        Parameters
        ----------
        threshold_r : float
            Minimum correlation to consider thesis validated
            (r=0.20 is "small" effect by Cohen's standards)
        """
        self.threshold_r = threshold_r
        self.results: List[BetterStoriesResult] = []
    
    def validate_domain(
        self,
        domain_name: str,
        narrative_quality: np.ndarray,
        outcomes: np.ndarray,
        quality_metric_name: str = "composite_narrative_score",
        control_variables: Optional[np.ndarray] = None
    ) -> BetterStoriesResult:
        """
        Test if better stories predict better outcomes in one domain.
        
        Parameters
        ----------
        domain_name : str
            Name of domain
        narrative_quality : array-like
            Narrative quality scores (higher = better story)
        outcomes : array-like
            Outcome measures (higher = better outcome)
        quality_metric_name : str
            Name of narrative quality metric used
        control_variables : array-like, optional
            Control variables to partial out confounds
            
        Returns
        -------
        BetterStoriesResult
        """
        # Basic correlation
        correlation, p_value = stats.pearsonr(narrative_quality, outcomes)
        
        # Effect size (R²)
        r2 = correlation ** 2
        
        # Partial correlation if controls provided
        if control_variables is not None:
            partial_corr, partial_p = self._partial_correlation(
                narrative_quality, outcomes, control_variables
            )
            evidence_partial = {
                'partial_correlation': partial_corr,
                'partial_p_value': partial_p,
                'partial_r2': partial_corr ** 2
            }
        else:
            partial_corr = correlation
            partial_p = p_value
            evidence_partial = {}
        
        # Test for non-linearity (maybe better stories win only at extremes?)
        quartiles = pd.qcut(narrative_quality, q=4, labels=False, duplicates='drop')
        
        if len(np.unique(quartiles)) >= 3:
            quartile_means = [
                np.mean(outcomes[quartiles == q])
                for q in range(len(np.unique(quartiles)))
            ]
            
            # Test for monotonic increase
            monotonic = all(quartile_means[i] <= quartile_means[i+1] 
                          for i in range(len(quartile_means)-1))
        else:
            quartile_means = []
            monotonic = None
        
        # Validation criteria:
        # 1. Significant positive correlation (p < 0.05)
        # 2. Effect size above threshold (r > 0.20)
        # 3. Survives controls if provided
        validates = (
            p_value < 0.05 and
            correlation > self.threshold_r and
            (control_variables is None or partial_corr > self.threshold_r * 0.7)
        )
        
        interpretation = self._interpret_result(
            correlation, p_value, r2, validates, quartile_means, monotonic
        )
        
        result = BetterStoriesResult(
            domain=domain_name,
            narrative_quality_metric=quality_metric_name,
            correlation_with_outcome=correlation,
            p_value=p_value,
            effect_size_r2=r2,
            validates_thesis=validates,
            evidence={
                'n_observations': len(narrative_quality),
                'narrative_mean': np.mean(narrative_quality),
                'narrative_std': np.std(narrative_quality),
                'outcome_mean': np.mean(outcomes),
                'outcome_std': np.std(outcomes),
                'quartile_means': quartile_means,
                'monotonic_increase': monotonic,
                **evidence_partial
            },
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def test_temporal_strengthening(
        self,
        domain_name: str,
        narrative_quality: np.ndarray,
        outcomes_immediate: np.ndarray,
        outcomes_delayed: np.ndarray,
        delay_description: str = "long-term"
    ) -> Dict[str, Any]:
        """
        Test if "better stories win over time" means the effect STRENGTHENS with time.
        
        Theory predicts: correlation(narrative, outcome) should increase from
        immediate to delayed measurements.
        
        Parameters
        ----------
        domain_name : str
        narrative_quality : array-like
        outcomes_immediate : array-like
            Outcomes measured immediately
        outcomes_delayed : array-like
            Outcomes measured after time has passed
        delay_description : str
            Description of time delay
            
        Returns
        -------
        Dict with temporal strengthening analysis
        """
        # Correlations at each time point
        r_immediate, p_immediate = stats.pearsonr(narrative_quality, outcomes_immediate)
        r_delayed, p_delayed = stats.pearsonr(narrative_quality, outcomes_delayed)
        
        # Test if correlation increased
        strengthening = r_delayed > r_immediate
        strengthening_amount = r_delayed - r_immediate
        
        # Statistical test: Do correlations differ significantly?
        # Using Fisher's z-transformation
        z1 = np.arctanh(r_immediate)
        z2 = np.arctanh(r_delayed)
        n = len(narrative_quality)
        
        se = np.sqrt(2 / (n - 3))
        z_diff = (z2 - z1) / se
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
        
        validates = strengthening and p_diff < 0.10
        
        return {
            'domain': domain_name,
            'r_immediate': r_immediate,
            'r_delayed': r_delayed,
            'strengthening': strengthening,
            'strengthening_amount': strengthening_amount,
            'p_value_difference': p_diff,
            'validates_temporal_thesis': validates,
            'interpretation': self._interpret_temporal_strengthening(
                r_immediate, r_delayed, strengthening_amount, p_diff, validates
            )
        }
    
    def cross_domain_synthesis(self) -> Dict[str, Any]:
        """
        Synthesize results across all tested domains.
        
        Returns
        -------
        Dict with cross-domain assessment
        """
        if not self.results:
            return {
                'n_domains': 0,
                'verdict': 'UNTESTED',
                'mean_correlation': 0.0,
                'validation_rate': 0.0
            }
        
        n_domains = len(self.results)
        n_validated = sum(1 for r in self.results if r.validates_thesis)
        validation_rate = n_validated / n_domains
        
        correlations = [r.correlation_with_outcome for r in self.results]
        mean_correlation = np.mean(correlations)
        std_correlation = np.std(correlations)
        
        # Meta-analysis: Average effect size weighted by sample size
        weights = [r.evidence['n_observations'] for r in self.results]
        weighted_avg_r = np.average(correlations, weights=weights)
        
        # Test for heterogeneity (do effects vary across domains?)
        if len(correlations) >= 3:
            # Q statistic for heterogeneity
            weighted_mean_r = weighted_avg_r
            q_stat = sum(
                w * (r - weighted_mean_r) ** 2
                for w, r in zip(weights, correlations)
            )
            
            # I² statistic (% of variation due to heterogeneity vs sampling error)
            df = len(correlations) - 1
            i_squared = max(0, (q_stat - df) / q_stat) if q_stat > 0 else 0
        else:
            q_stat = 0
            i_squared = 0
        
        # Overall verdict
        if validation_rate >= 0.75 and mean_correlation > self.threshold_r:
            verdict = "VALIDATED - Better stories win across domains"
        elif validation_rate >= 0.50:
            verdict = "PARTIAL - Better stories win in some domains"
        elif mean_correlation > 0:
            verdict = "WEAK EVIDENCE - Effect present but small/inconsistent"
        else:
            verdict = "REFUTED - Better stories do not predict better outcomes"
        
        # Identify strongest and weakest domains
        sorted_results = sorted(self.results, key=lambda r: r.correlation_with_outcome, reverse=True)
        
        return {
            'verdict': verdict,
            'n_domains': n_domains,
            'n_validated': n_validated,
            'validation_rate': validation_rate,
            'mean_correlation': mean_correlation,
            'std_correlation': std_correlation,
            'weighted_avg_correlation': weighted_avg_r,
            'heterogeneity_i_squared': i_squared,
            'strongest_domains': [r.domain for r in sorted_results[:3]],
            'weakest_domains': [r.domain for r in sorted_results[-3:]],
            'all_results': self.results
        }
    
    def test_reverse_causation(
        self,
        narrative_quality_t1: np.ndarray,
        narrative_quality_t2: np.ndarray,
        outcomes_t1: np.ndarray,
        outcomes_t2: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test for reverse causation: Do outcomes predict future narratives?
        
        If yes, suggests we construct good stories AFTER success, not before.
        
        Parameters
        ----------
        narrative_quality_t1, narrative_quality_t2 : array-like
            Narrative quality at time 1 and time 2
        outcomes_t1, outcomes_t2 : array-like
            Outcomes at time 1 and time 2
            
        Returns
        -------
        Dict with reverse causation analysis
        """
        # Forward: Does narrative(t1) predict outcome(t2)?
        r_forward, p_forward = stats.pearsonr(narrative_quality_t1, outcomes_t2)
        
        # Reverse: Does outcome(t1) predict narrative(t2)?
        r_reverse, p_reverse = stats.pearsonr(outcomes_t1, narrative_quality_t2)
        
        # Cross-lagged panel: Which direction is stronger?
        reverse_causation_detected = r_reverse > r_forward
        
        # Strength of evidence
        if reverse_causation_detected and p_reverse < 0.05:
            if r_reverse > 1.5 * r_forward:
                severity = 'severe'
            else:
                severity = 'moderate'
        else:
            severity = 'none'
        
        return {
            'r_forward': r_forward,
            'p_forward': p_forward,
            'r_reverse': r_reverse,
            'p_reverse': p_reverse,
            'reverse_causation_detected': reverse_causation_detected,
            'severity': severity,
            'interpretation': self._interpret_reverse_causation(
                r_forward, r_reverse, reverse_causation_detected, severity
            )
        }
    
    def _partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        controls: np.ndarray
    ) -> Tuple[float, float]:
        """Compute partial correlation controlling for confounds."""
        # Residualize both x and y with respect to controls
        from sklearn.linear_model import LinearRegression
        
        # Ensure 2D
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)
        
        # Residuals for x
        reg_x = LinearRegression().fit(controls, x)
        residuals_x = x - reg_x.predict(controls)
        
        # Residuals for y
        reg_y = LinearRegression().fit(controls, y)
        residuals_y = y - reg_y.predict(controls)
        
        # Correlation of residuals
        return stats.pearsonr(residuals_x, residuals_y)
    
    def _interpret_result(
        self,
        correlation: float,
        p_value: float,
        r2: float,
        validates: bool,
        quartile_means: List[float],
        monotonic: Optional[bool]
    ) -> str:
        if validates:
            practical_significance = "substantial" if r2 > 0.10 else "modest"
            monotonic_note = ""
            if monotonic is not None:
                monotonic_note = " Effect is monotonic across quality quartiles." if monotonic else ""
            
            return (
                f"Better stories DO predict better outcomes (r={correlation:.3f}, p={p_value:.4f}, R²={r2:.3f}). "
                f"Effect size is {practical_significance}.{monotonic_note} "
                f"This validates the core thesis in this domain."
            )
        elif correlation > 0 and p_value < 0.10:
            return (
                f"Weak evidence for better stories winning (r={correlation:.3f}, p={p_value:.4f}). "
                f"Effect is in predicted direction but small/marginally significant. "
                f"Theory receives weak support."
            )
        elif correlation > 0:
            return (
                f"No significant relationship (r={correlation:.3f}, p={p_value:.4f}). "
                f"Better stories do NOT predict better outcomes in this domain. "
                f"Core thesis is NOT supported."
            )
        else:
            return (
                f"NEGATIVE relationship (r={correlation:.3f}, p={p_value:.4f}). "
                f"Better stories predict WORSE outcomes. "
                f"This CONTRADICTS the core thesis."
            )
    
    def _interpret_temporal_strengthening(
        self,
        r_immediate: float,
        r_delayed: float,
        strengthening: float,
        p_diff: float,
        validates: bool
    ) -> str:
        if validates:
            return (
                f"Effect strengthens over time: r={r_immediate:.3f}→{r_delayed:.3f} "
                f"(Δ={strengthening:+.3f}, p={p_diff:.4f}). "
                f"This supports 'better stories win OVER TIME'."
            )
        elif strengthening > 0:
            return (
                f"Small increase over time: r={r_immediate:.3f}→{r_delayed:.3f} "
                f"(Δ={strengthening:+.3f}, p={p_diff:.4f}). "
                f"Trend is in predicted direction but not significant."
            )
        else:
            return (
                f"Effect WEAKENS over time: r={r_immediate:.3f}→{r_delayed:.3f} "
                f"(Δ={strengthening:+.3f}). "
                f"This REFUTES 'better stories win over time' thesis."
            )
    
    def _interpret_reverse_causation(
        self,
        r_forward: float,
        r_reverse: float,
        detected: bool,
        severity: str
    ) -> str:
        if severity == 'severe':
            return (
                f"STRONG reverse causation detected. "
                f"Outcomes predict future narratives (r={r_reverse:.3f}) "
                f"MORE than narratives predict outcomes (r={r_forward:.3f}). "
                f"This suggests we construct good stories AFTER success, not before."
            )
        elif detected:
            return (
                f"Moderate reverse causation. Both directions present but "
                f"outcome→narrative (r={r_reverse:.3f}) somewhat stronger than "
                f"narrative→outcome (r={r_forward:.3f}). "
                f"Bidirectional relationship - narratives both cause AND result from success."
            )
        else:
            return (
                f"Forward causation stronger. "
                f"Narrative→outcome (r={r_forward:.3f}) exceeds "
                f"outcome→narrative (r={r_reverse:.3f}). "
                f"This supports narratives predicting outcomes, not just explaining them."
            )
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("=" * 80)
        report.append("'BETTER STORIES WIN' VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        if not self.results:
            report.append("No domains tested yet.")
            return "\n".join(report)
        
        synthesis = self.cross_domain_synthesis()
        
        report.append("CROSS-DOMAIN SYNTHESIS:")
        report.append("-" * 80)
        report.append(f"Verdict: {synthesis['verdict']}")
        report.append(f"Domains Tested: {synthesis['n_domains']}")
        report.append(f"Domains Validated: {synthesis['n_validated']} ({synthesis['validation_rate']:.1%})")
        report.append(f"Mean Correlation: r={synthesis['mean_correlation']:.3f} ± {synthesis['std_correlation']:.3f}")
        report.append(f"Weighted Average: r={synthesis['weighted_avg_correlation']:.3f}")
        report.append(f"Heterogeneity: I²={synthesis['heterogeneity_i_squared']:.1%}")
        report.append("")
        
        report.append("DOMAIN-SPECIFIC RESULTS:")
        report.append("-" * 80)
        for result in sorted(self.results, key=lambda r: r.correlation_with_outcome, reverse=True):
            report.append(str(result))
            report.append(f"  {result.interpretation}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

