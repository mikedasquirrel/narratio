"""
Confirmation Bias Detector

Tests whether observed patterns are genuine or artifacts of confirmation bias,
p-hacking, or researcher degrees of freedom.

This module implements rigorous statistical tests to detect:
1. Patterns that disappear when data is randomized
2. Effect sizes that cluster suspiciously around expectations
3. Temporal precedence violations (outcome before narrative)
4. Publication bias (missing null findings)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.special import comb
import warnings


@dataclass
class BiasDetectionResult:
    """Results from a bias detection test."""
    test_name: str
    bias_detected: bool
    severity: str  # 'none', 'mild', 'moderate', 'severe'
    p_value: float
    effect_size: float
    evidence: Dict[str, Any]
    interpretation: str
    
    def __repr__(self):
        flag = "⚠️ BIAS" if self.bias_detected else "✓ CLEAN"
        return f"{flag} | {self.test_name} (severity: {self.severity}, p={self.p_value:.4f})"


class ConfirmationBiasDetector:
    """
    Detects confirmation bias and questionable research practices.
    
    Key principle: If patterns are genuine, they should:
    1. Survive randomization tests
    2. Show natural effect size distributions
    3. Maintain temporal precedence
    4. Include null findings at expected rates
    """
    
    def __init__(self, alpha: float = 0.05, n_permutations: int = 1000):
        """
        Parameters
        ----------
        alpha : float
            Significance threshold for bias detection
        n_permutations : int
            Number of randomization iterations
        """
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.results: List[BiasDetectionResult] = []
    
    def test_randomization_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        observed_score: float,
        metric: str = 'accuracy'
    ) -> BiasDetectionResult:
        """
        Test 1: Randomization Robustness
        
        Do patterns disappear when labels are randomized?
        
        If the model still performs well on randomized data, it suggests
        the "patterns" are just noise or data leakage, not genuine signal.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            True labels
        model : sklearn-compatible model
            The model to test
        observed_score : float
            Score on real data
        metric : str
            Evaluation metric
            
        Returns
        -------
        BiasDetectionResult
        """
        # Fit model on real data
        model.fit(X, y)
        
        # Test on permuted data
        permuted_scores = []
        
        for i in range(self.n_permutations):
            # Randomly shuffle labels
            y_permuted = np.random.permutation(y)
            
            # Fit and score
            model_permuted = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            model_permuted.fit(X, y_permuted)
            
            if metric == 'accuracy':
                score = model_permuted.score(X, y_permuted)
            else:
                predictions = model_permuted.predict(X)
                score = self._compute_metric(y_permuted, predictions, metric)
            
            permuted_scores.append(score)
        
        permuted_scores = np.array(permuted_scores)
        
        # How often do random labels match or beat observed performance?
        p_value = np.mean(permuted_scores >= observed_score)
        
        # Effect size: How much better than random?
        mean_random = np.mean(permuted_scores)
        std_random = np.std(permuted_scores)
        cohens_d = (observed_score - mean_random) / (std_random + 1e-10)
        
        # Detect bias
        # If p > 0.1, the pattern is not significantly better than random
        bias_detected = p_value > 0.1 or cohens_d < 0.5
        
        if bias_detected:
            if p_value > 0.5:
                severity = 'severe'
            elif p_value > 0.3:
                severity = 'moderate'
            else:
                severity = 'mild'
        else:
            severity = 'none'
        
        interpretation = self._interpret_randomization(
            observed_score, mean_random, p_value, cohens_d, bias_detected
        )
        
        result = BiasDetectionResult(
            test_name="Randomization Robustness Test",
            bias_detected=bias_detected,
            severity=severity,
            p_value=p_value,
            effect_size=cohens_d,
            evidence={
                'observed_score': observed_score,
                'mean_random_score': mean_random,
                'std_random_score': std_random,
                'cohens_d': cohens_d,
                'n_permutations': self.n_permutations
            },
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def test_effect_size_distribution(
        self,
        observed_effects: List[float],
        expected_mean: Optional[float] = None,
        expected_std: Optional[float] = None
    ) -> BiasDetectionResult:
        """
        Test 2: Effect Size Distribution
        
        Do effect sizes cluster suspiciously around expectations?
        
        In genuine research, effect sizes vary naturally. If they cluster
        tightly around theoretical predictions or desired thresholds (e.g., 0.05),
        it suggests cherry-picking or p-hacking.
        
        Parameters
        ----------
        observed_effects : List[float]
            Effect sizes from multiple analyses
        expected_mean : float, optional
            Theoretically predicted mean effect size
        expected_std : float, optional
            Expected natural variation
            
        Returns
        -------
        BiasDetectionResult
        """
        observed_effects = np.array(observed_effects)
        
        if len(observed_effects) < 3:
            return BiasDetectionResult(
                test_name="Effect Size Distribution Test",
                bias_detected=False,
                severity='none',
                p_value=1.0,
                effect_size=0.0,
                evidence={'error': 'Insufficient data (need >= 3 effects)'},
                interpretation="Cannot assess - need more effect sizes"
            )
        
        mean_effect = np.mean(observed_effects)
        std_effect = np.std(observed_effects)
        cv = std_effect / (abs(mean_effect) + 1e-10)  # Coefficient of variation
        
        # Test 1: Is variation suspiciously low? (suggests cherry-picking)
        # Natural effect sizes typically have CV > 0.3
        too_uniform = cv < 0.15
        
        # Test 2: Do effects cluster near significance threshold?
        # Check if many effects are just above p=0.05 threshold
        near_threshold = np.sum((observed_effects > 0.03) & (observed_effects < 0.08))
        threshold_clustering = near_threshold / len(observed_effects) > 0.5
        
        # Test 3: If expected values provided, test for suspicious alignment
        suspicious_alignment = False
        alignment_stat = 0.0
        
        if expected_mean is not None:
            # Are effects suspiciously close to expectation?
            deviation_from_expected = np.abs(mean_effect - expected_mean)
            expected_std_actual = expected_std if expected_std else 0.1
            
            alignment_stat = deviation_from_expected / expected_std_actual
            suspicious_alignment = alignment_stat < 0.5  # Too close to be true
        
        # Overall bias detection
        bias_detected = too_uniform or threshold_clustering or suspicious_alignment
        
        if bias_detected:
            if sum([too_uniform, threshold_clustering, suspicious_alignment]) >= 2:
                severity = 'severe'
            else:
                severity = 'moderate'
        else:
            severity = 'none'
        
        # P-value: Test if effects are more uniform than expected
        # Use Levene's test comparing to simulated natural variation
        natural_effects = np.random.normal(mean_effect, mean_effect * 0.4, size=len(observed_effects))
        _, p_value = stats.levene(observed_effects, natural_effects)
        
        interpretation = self._interpret_effect_distribution(
            mean_effect, cv, too_uniform, threshold_clustering,
            suspicious_alignment, bias_detected
        )
        
        result = BiasDetectionResult(
            test_name="Effect Size Distribution Test",
            bias_detected=bias_detected,
            severity=severity,
            p_value=p_value,
            effect_size=cv,
            evidence={
                'mean_effect': mean_effect,
                'std_effect': std_effect,
                'coefficient_of_variation': cv,
                'n_effects': len(observed_effects),
                'too_uniform': too_uniform,
                'threshold_clustering': threshold_clustering,
                'suspicious_alignment': suspicious_alignment
            },
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def test_temporal_precedence(
        self,
        narrative_timestamps: np.ndarray,
        outcome_timestamps: np.ndarray,
        narrative_changes: Optional[np.ndarray] = None,
        outcome_changes: Optional[np.ndarray] = None
    ) -> BiasDetectionResult:
        """
        Test 3: Temporal Precedence
        
        Does narrative actually precede outcomes, or are we explaining
        outcomes post-hoc with narratives constructed after the fact?
        
        Genuine causal patterns: Narrative → Outcome
        Reverse causation: Outcome → Narrative (we explain success with good stories)
        
        Parameters
        ----------
        narrative_timestamps : array-like
            When narratives were recorded/measured
        outcome_timestamps : array-like
            When outcomes occurred
        narrative_changes : array-like, optional
            Changes in narrative quality over time
        outcome_changes : array-like, optional
            Changes in outcomes over time
            
        Returns
        -------
        BiasDetectionResult
        """
        # Test 1: Basic temporal ordering
        precedence_violations = np.sum(narrative_timestamps > outcome_timestamps)
        violation_rate = precedence_violations / len(narrative_timestamps)
        
        # Test 2: If changes provided, test Granger causality direction
        granger_evidence = {}
        if narrative_changes is not None and outcome_changes is not None:
            # Simple cross-correlation test
            # Positive lag: narrative predicts outcome (good)
            # Negative lag: outcome predicts narrative (bad - reverse causation)
            correlations = []
            for lag in range(-5, 6):
                if lag < 0:
                    n = narrative_changes[-lag:]
                    o = outcome_changes[:lag]
                elif lag > 0:
                    n = narrative_changes[:-lag]
                    o = outcome_changes[lag:]
                else:
                    n = narrative_changes
                    o = outcome_changes
                
                if len(n) >= 2 and len(o) >= 2:
                    corr, _ = stats.pearsonr(n, o)
                    correlations.append((lag, corr))
            
            if correlations:
                best_lag, best_corr = max(correlations, key=lambda x: abs(x[1]))
                granger_evidence = {
                    'best_lag': best_lag,
                    'best_correlation': best_corr
                }
                
                # Reverse causation detected if best lag is negative
                reverse_causation = best_lag < 0 and abs(best_corr) > 0.3
            else:
                reverse_causation = False
        else:
            reverse_causation = False
        
        # Bias detection
        bias_detected = violation_rate > 0.2 or reverse_causation
        
        if bias_detected:
            if violation_rate > 0.5 or (reverse_causation and abs(granger_evidence.get('best_correlation', 0)) > 0.5):
                severity = 'severe'
            elif violation_rate > 0.3:
                severity = 'moderate'
            else:
                severity = 'mild'
        else:
            severity = 'none'
        
        p_value = violation_rate  # Simple: proportion of violations
        effect_size = 1 - violation_rate  # How well temporal order is maintained
        
        interpretation = self._interpret_temporal_precedence(
            violation_rate, precedence_violations, reverse_causation,
            granger_evidence, bias_detected
        )
        
        result = BiasDetectionResult(
            test_name="Temporal Precedence Test",
            bias_detected=bias_detected,
            severity=severity,
            p_value=p_value,
            effect_size=effect_size,
            evidence={
                'n_observations': len(narrative_timestamps),
                'precedence_violations': precedence_violations,
                'violation_rate': violation_rate,
                'reverse_causation_detected': reverse_causation,
                'granger_evidence': granger_evidence
            },
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def test_file_drawer_effect(
        self,
        reported_studies: List[Dict[str, Any]],
        expected_power: float = 0.8,
        alpha: float = 0.05
    ) -> BiasDetectionResult:
        """
        Test 4: File Drawer Effect (Publication Bias)
        
        Are null findings being hidden?
        
        If we always report significant findings but never report null results,
        it suggests cherry-picking or p-hacking. The file drawer test estimates
        how many null findings would need to exist to invalidate published findings.
        
        Parameters
        ----------
        reported_studies : List[Dict]
            Studies that were reported, each containing:
            - 'significant': bool
            - 'effect_size': float
            - 'n': int (sample size)
        expected_power : float
            Expected statistical power (typically 0.8)
        alpha : float
            Significance threshold
            
        Returns
        -------
        BiasDetectionResult
        """
        if not reported_studies:
            return BiasDetectionResult(
                test_name="File Drawer Effect Test",
                bias_detected=False,
                severity='none',
                p_value=1.0,
                effect_size=0.0,
                evidence={'error': 'No studies provided'},
                interpretation="Cannot assess - no studies to analyze"
            )
        
        n_significant = sum(1 for s in reported_studies if s.get('significant', False))
        n_total = len(reported_studies)
        
        if n_total < 3:
            return BiasDetectionResult(
                test_name="File Drawer Effect Test",
                bias_detected=False,
                severity='none',
                p_value=1.0,
                effect_size=0.0,
                evidence={'error': 'Insufficient studies (need >= 3)'},
                interpretation="Cannot assess - need more studies"
            )
        
        # Observed rate of significant findings
        observed_sig_rate = n_significant / n_total
        
        # Expected rate (with expected power and no bias)
        # If there's a true effect: ~80% should be significant
        # If there's no effect: ~5% should be significant (Type I errors)
        # Natural mix would be somewhere in between
        expected_sig_rate = expected_power * 0.5 + alpha * 0.5  # Rough estimate
        
        # Test if observed rate is suspiciously high
        # Use binomial test
        p_value = stats.binom_test(n_significant, n_total, expected_sig_rate, alternative='greater')
        
        # Estimate number of hidden null findings needed to bring rate to expected
        # (Rosenthal's fail-safe N)
        if n_significant > 0:
            failsafe_n = (n_significant / expected_sig_rate) - n_total
            failsafe_n = max(0, failsafe_n)
        else:
            failsafe_n = 0
        
        # Bias detection
        # If >90% of studies are significant, that's suspicious
        # Unless we have very high power and strong effects
        mean_effect = np.mean([s.get('effect_size', 0) for s in reported_studies])
        
        bias_detected = observed_sig_rate > 0.9 and p_value < 0.05
        
        if bias_detected:
            if observed_sig_rate > 0.95:
                severity = 'severe'
            else:
                severity = 'moderate'
        else:
            severity = 'none'
        
        effect_size = observed_sig_rate - expected_sig_rate
        
        interpretation = self._interpret_file_drawer(
            observed_sig_rate, expected_sig_rate, failsafe_n,
            n_total, bias_detected
        )
        
        result = BiasDetectionResult(
            test_name="File Drawer Effect Test",
            bias_detected=bias_detected,
            severity=severity,
            p_value=p_value,
            effect_size=effect_size,
            evidence={
                'n_studies': n_total,
                'n_significant': n_significant,
                'observed_sig_rate': observed_sig_rate,
                'expected_sig_rate': expected_sig_rate,
                'failsafe_n': failsafe_n,
                'mean_effect_size': mean_effect
            },
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result
    
    def compute_overall_bias_assessment(self) -> Dict[str, Any]:
        """
        Compute overall bias assessment from all tests.
        
        Returns
        -------
        Dict with overall assessment and recommendations
        """
        if not self.results:
            return {
                'bias_detected': False,
                'severity': 'none',
                'n_tests_failed': 0,
                'n_tests_passed': 0,
                'verdict': 'UNTESTED',
                'recommendations': ['Run bias detection tests']
            }
        
        n_bias_detected = sum(1 for r in self.results if r.bias_detected)
        n_total = len(self.results)
        
        # Severity levels
        severe_count = sum(1 for r in self.results if r.severity == 'severe')
        moderate_count = sum(1 for r in self.results if r.severity == 'moderate')
        
        # Overall severity
        if severe_count > 0:
            overall_severity = 'severe'
        elif moderate_count > 1:
            overall_severity = 'severe'
        elif moderate_count == 1:
            overall_severity = 'moderate'
        elif n_bias_detected > 0:
            overall_severity = 'mild'
        else:
            overall_severity = 'none'
        
        # Verdict
        if n_bias_detected == 0:
            verdict = "CLEAN - No significant bias detected"
        elif n_bias_detected == 1 and overall_severity == 'mild':
            verdict = "MOSTLY CLEAN - Minor concern"
        elif overall_severity == 'moderate':
            verdict = "QUESTIONABLE - Moderate bias concerns"
        else:
            verdict = "BIASED - Serious concerns about validity"
        
        recommendations = self._generate_bias_recommendations(overall_severity, self.results)
        
        return {
            'bias_detected': n_bias_detected > 0,
            'severity': overall_severity,
            'n_tests_failed': n_bias_detected,
            'n_tests_passed': n_total - n_bias_detected,
            'n_total_tests': n_total,
            'verdict': verdict,
            'test_results': self.results,
            'recommendations': recommendations
        }
    
    # Helper methods
    
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Compute evaluation metric."""
        if metric == 'accuracy':
            return np.mean(y_true == y_pred)
        elif metric == 'mse':
            return -np.mean((y_true - y_pred) ** 2)
        elif metric == 'correlation':
            return np.corrcoef(y_true, y_pred)[0, 1]
        else:
            return 0.0
    
    def _interpret_randomization(
        self, observed: float, mean_random: float, p_value: float,
        cohens_d: float, bias_detected: bool
    ) -> str:
        if bias_detected:
            return (
                f"Pattern does not survive randomization test. "
                f"Observed: {observed:.3f}, Random: {mean_random:.3f}, "
                f"p={p_value:.3f}, d={cohens_d:.2f}. "
                f"This suggests the 'pattern' may be noise, overfitting, or data leakage."
            )
        else:
            return (
                f"Pattern survives randomization. "
                f"Observed: {observed:.3f} significantly better than random: {mean_random:.3f} "
                f"(p={p_value:.4f}, d={cohens_d:.2f}). "
                f"This supports genuine signal detection."
            )
    
    def _interpret_effect_distribution(
        self, mean_effect: float, cv: float, too_uniform: bool,
        threshold_clustering: bool, suspicious_alignment: bool, bias_detected: bool
    ) -> str:
        issues = []
        if too_uniform:
            issues.append("effect sizes too uniform")
        if threshold_clustering:
            issues.append("clustering near significance threshold")
        if suspicious_alignment:
            issues.append("suspiciously close to theoretical predictions")
        
        if bias_detected:
            return (
                f"Suspicious effect size distribution: {', '.join(issues)}. "
                f"Mean: {mean_effect:.3f}, CV: {cv:.3f}. "
                f"This pattern suggests possible cherry-picking or p-hacking."
            )
        else:
            return (
                f"Natural effect size distribution. "
                f"Mean: {mean_effect:.3f}, CV: {cv:.3f}. "
                f"Variation is consistent with genuine research findings."
            )
    
    def _interpret_temporal_precedence(
        self, violation_rate: float, n_violations: int, reverse_causation: bool,
        granger_evidence: Dict, bias_detected: bool
    ) -> str:
        if bias_detected:
            issues = []
            if violation_rate > 0.2:
                issues.append(f"{violation_rate:.1%} temporal order violations")
            if reverse_causation:
                lag = granger_evidence.get('best_lag', 0)
                issues.append(f"reverse causation detected (outcome predicts narrative at lag {lag})")
            
            return (
                f"Temporal precedence violations: {', '.join(issues)}. "
                f"This suggests narratives may be constructed post-hoc to explain outcomes "
                f"rather than predicting them."
            )
        else:
            return (
                f"Temporal precedence maintained. "
                f"Only {violation_rate:.1%} violations, "
                f"consistent with genuine predictive relationship."
            )
    
    def _interpret_file_drawer(
        self, observed_rate: float, expected_rate: float, failsafe_n: float,
        n_total: int, bias_detected: bool
    ) -> str:
        if bias_detected:
            return (
                f"Suspicious publication pattern: {observed_rate:.1%} of studies significant "
                f"(expected: {expected_rate:.1%}). "
                f"Would need {failsafe_n:.0f} hidden null findings to explain this rate. "
                f"Suggests possible selective reporting or p-hacking."
            )
        else:
            return (
                f"No clear publication bias. "
                f"{observed_rate:.1%} significant findings close to expected {expected_rate:.1%}. "
                f"Mix of significant and null results suggests honest reporting."
            )
    
    def _generate_bias_recommendations(
        self, severity: str, results: List[BiasDetectionResult]
    ) -> List[str]:
        recommendations = []
        
        if severity == 'none':
            recommendations.append("No significant bias detected. Continue with current practices.")
            recommendations.append("Consider pre-registration for future studies.")
        
        elif severity == 'mild':
            recommendations.append("Minor concerns detected. Address specific issues:")
            for result in results:
                if result.bias_detected and result.severity == 'mild':
                    if "Randomization" in result.test_name:
                        recommendations.append("  - Strengthen signal/noise ratio or simplify model")
                    elif "Effect Size" in result.test_name:
                        recommendations.append("  - Report all analyses, not just significant ones")
        
        elif severity == 'moderate':
            recommendations.append("Moderate bias concerns. Take corrective action:")
            recommendations.append("- Pre-register hypotheses before analysis")
            recommendations.append("- Report all analyses conducted, including null findings")
            recommendations.append("- Use holdout test sets not touched during development")
        
        else:  # severe
            recommendations.append("SERIOUS BIAS CONCERNS. Results may be invalid.")
            recommendations.append("Required actions:")
            recommendations.append("  1. Collect independent validation data")
            recommendations.append("  2. Pre-register analysis plan before seeing new data")
            recommendations.append("  3. Have independent analyst replicate findings")
            recommendations.append("  4. Consider whether framework should be abandoned")
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate comprehensive text report of bias detection."""
        overall = self.compute_overall_bias_assessment()
        
        report = []
        report.append("=" * 80)
        report.append("CONFIRMATION BIAS DETECTION REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Verdict: {overall['verdict']}")
        report.append(f"Overall Severity: {overall['severity'].upper()}")
        report.append(f"Tests Failed: {overall['n_tests_failed']} / {overall['n_total_tests']}")
        report.append("")
        report.append("-" * 80)
        report.append("INDIVIDUAL TEST RESULTS")
        report.append("-" * 80)
        report.append("")
        
        for result in self.results:
            report.append(str(result))
            report.append(f"  {result.interpretation}")
            report.append("")
        
        report.append("-" * 80)
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("")
        
        for i, rec in enumerate(overall['recommendations'], 1):
            report.append(f"{i}. {rec}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

