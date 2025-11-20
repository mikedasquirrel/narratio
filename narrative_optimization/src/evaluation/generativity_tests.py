"""
Generativity Test Suite

Tests whether the narrative optimization framework is genuinely generative 
(produces novel insights and predictions) versus elaborate rationalization 
(finds patterns post-hoc without predictive power).

This module implements computational tests to assess the meta-question:
Is this framework discovering genuine patterns or creating elaborate 
descriptions that feel satisfying without being truly predictive?
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings


@dataclass
class GenerativityScore:
    """Results from a generativity test."""
    test_name: str
    score: float  # 0-1 scale, higher = more generative
    confidence: float  # 0-1 scale
    evidence: Dict[str, Any]
    interpretation: str
    passes_threshold: bool  # True if clearly generative
    
    def __repr__(self):
        status = "✓ GENERATIVE" if self.passes_threshold else "✗ QUESTIONABLE"
        return f"{status} | {self.test_name}: {self.score:.3f} (confidence: {self.confidence:.3f})"


class GenerativityTestSuite:
    """
    Comprehensive test suite to evaluate whether frameworks are generative.
    
    A framework is considered generative if it:
    1. Predicts novel outcomes better than baselines
    2. Converges across independent applications
    3. Is falsifiable with clear refutation criteria
    4. Compresses information rather than just relabeling
    5. Produces insights that external observers find compelling
    """
    
    def __init__(self, threshold: float = 0.6):
        """
        Parameters
        ----------
        threshold : float
            Minimum score (0-1) to pass generativity test
        """
        self.threshold = threshold
        self.results: List[GenerativityScore] = []
        
    def test_novel_prediction(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test_unseen_domain: np.ndarray,
        y_test_unseen_domain: np.ndarray,
        baseline_model: Any,
        metric: str = 'accuracy'
    ) -> GenerativityScore:
        """
        Test 1: Novel Prediction
        
        Does the framework predict outcomes in UNSEEN domains better than baselines?
        
        A truly generative framework should transfer to new domains because it
        captures genuine patterns, not just memorize training data patterns.
        
        Parameters
        ----------
        model : sklearn-compatible model
            The narrative framework model
        X_train, y_train : array-like
            Training data from SEEN domains
        X_test_unseen_domain, y_test_unseen_domain : array-like
            Test data from UNSEEN domain
        baseline_model : sklearn-compatible model
            Simple baseline (e.g., TF-IDF + LogisticRegression)
        metric : str
            'accuracy' for classification, 'r2' for regression
            
        Returns
        -------
        GenerativityScore
        """
        # Train both models on seen domains
        model.fit(X_train, y_train)
        baseline_model.fit(X_train, y_train)
        
        # Test on unseen domain
        narrative_pred = model.predict(X_test_unseen_domain)
        baseline_pred = baseline_model.predict(X_test_unseen_domain)
        
        if metric == 'accuracy':
            narrative_score = accuracy_score(y_test_unseen_domain, narrative_pred)
            baseline_score = accuracy_score(y_test_unseen_domain, baseline_pred)
        elif metric == 'r2':
            narrative_score = 1 - mean_squared_error(y_test_unseen_domain, narrative_pred)
            baseline_score = 1 - mean_squared_error(y_test_unseen_domain, baseline_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Improvement over baseline
        improvement = (narrative_score - baseline_score) / (1 - baseline_score + 1e-10)
        
        # Score: How much better than baseline?
        # > 0.1 improvement = generative
        # < 0.0 improvement = not generative
        generativity_score = np.clip(improvement * 5, 0, 1)
        
        # Confidence based on absolute performance
        confidence = narrative_score
        
        passes = generativity_score >= self.threshold and improvement > 0.05
        
        interpretation = self._interpret_novel_prediction(
            improvement, narrative_score, baseline_score, passes
        )
        
        result = GenerativityScore(
            test_name="Novel Prediction Test",
            score=generativity_score,
            confidence=confidence,
            evidence={
                'narrative_accuracy': narrative_score,
                'baseline_accuracy': baseline_score,
                'improvement': improvement,
                'absolute_gain': narrative_score - baseline_score
            },
            interpretation=interpretation,
            passes_threshold=passes
        )
        
        self.results.append(result)
        return result
    
    def test_convergence(
        self,
        analyses: List[Dict[str, Any]],
        key_metrics: List[str]
    ) -> GenerativityScore:
        """
        Test 2: Convergence Test
        
        Do independent analyses arrive at similar patterns?
        
        If multiple people analyze the same data with the framework and get
        very different results, it suggests the framework is flexible enough
        to find whatever pattern the analyst expects (bad). Convergence suggests
        the framework captures genuine structure (good).
        
        Parameters
        ----------
        analyses : List[Dict]
            Multiple independent analyses, each containing metrics
        key_metrics : List[str]
            Which metrics to check for convergence
            
        Returns
        -------
        GenerativityScore
        """
        convergence_scores = []
        
        for metric in key_metrics:
            values = [a[metric] for a in analyses if metric in a]
            
            if len(values) < 2:
                continue
                
            # Calculate coefficient of variation (lower = more convergence)
            cv = np.std(values) / (np.mean(values) + 1e-10)
            
            # Convert to convergence score (0-1, higher = more convergent)
            convergence = 1 / (1 + cv)
            convergence_scores.append(convergence)
        
        if not convergence_scores:
            return GenerativityScore(
                test_name="Convergence Test",
                score=0.0,
                confidence=0.0,
                evidence={'error': 'Insufficient data'},
                interpretation="Cannot assess - need multiple analyses",
                passes_threshold=False
            )
        
        avg_convergence = np.mean(convergence_scores)
        confidence = min(1.0, len(analyses) / 5)  # More analyses = more confidence
        
        passes = avg_convergence >= self.threshold
        
        interpretation = self._interpret_convergence(
            avg_convergence, len(analyses), convergence_scores, passes
        )
        
        result = GenerativityScore(
            test_name="Convergence Test",
            score=avg_convergence,
            confidence=confidence,
            evidence={
                'n_analyses': len(analyses),
                'convergence_by_metric': dict(zip(key_metrics, convergence_scores)),
                'mean_convergence': avg_convergence
            },
            interpretation=interpretation,
            passes_threshold=passes
        )
        
        self.results.append(result)
        return result
    
    def test_falsifiability(
        self,
        theoretical_claims: List[Dict[str, Any]],
        observed_data: Dict[str, Any]
    ) -> GenerativityScore:
        """
        Test 3: Falsifiability Test
        
        Does each claim specify what evidence would refute it?
        Does that evidence exist in the data?
        
        Unfalsifiable theories can never be wrong, which makes them
        scientifically useless even if they feel satisfying.
        
        Parameters
        ----------
        theoretical_claims : List[Dict]
            Each dict should have:
            - 'claim': str description
            - 'prediction': expected pattern
            - 'refutation_criterion': what would falsify this
            - 'test_function': callable to test the claim
        observed_data : Dict
            Data to test claims against
            
        Returns
        -------
        GenerativityScore
        """
        falsifiable_count = 0
        confirmed_count = 0
        refuted_count = 0
        unfalsifiable_count = 0
        
        claim_results = []
        
        for claim in theoretical_claims:
            if 'refutation_criterion' not in claim:
                unfalsifiable_count += 1
                claim_results.append({
                    'claim': claim.get('claim', 'Unknown'),
                    'status': 'unfalsifiable',
                    'reason': 'No refutation criterion specified'
                })
                continue
            
            falsifiable_count += 1
            
            # Test the claim
            if 'test_function' in claim and callable(claim['test_function']):
                try:
                    result = claim['test_function'](observed_data)
                    
                    if result['confirmed']:
                        confirmed_count += 1
                        status = 'confirmed'
                    else:
                        refuted_count += 1
                        status = 'refuted'
                    
                    claim_results.append({
                        'claim': claim.get('claim', 'Unknown'),
                        'status': status,
                        'evidence': result.get('evidence', {})
                    })
                except Exception as e:
                    claim_results.append({
                        'claim': claim.get('claim', 'Unknown'),
                        'status': 'untested',
                        'error': str(e)
                    })
        
        total_claims = len(theoretical_claims)
        
        if total_claims == 0:
            return GenerativityScore(
                test_name="Falsifiability Test",
                score=0.0,
                confidence=0.0,
                evidence={'error': 'No claims provided'},
                interpretation="Cannot assess - no theoretical claims",
                passes_threshold=False
            )
        
        # Score based on:
        # 1. Proportion of claims that are falsifiable (main factor)
        # 2. Bonus if some claims were actually refuted (shows honesty)
        falsifiability_ratio = falsifiable_count / total_claims
        refutation_bonus = min(0.2, refuted_count / total_claims) if refuted_count > 0 else 0
        
        score = min(1.0, falsifiability_ratio + refutation_bonus)
        confidence = min(1.0, total_claims / 10)
        
        passes = score >= self.threshold and falsifiability_ratio > 0.5
        
        interpretation = self._interpret_falsifiability(
            falsifiable_count, confirmed_count, refuted_count, 
            unfalsifiable_count, passes
        )
        
        result = GenerativityScore(
            test_name="Falsifiability Test",
            score=score,
            confidence=confidence,
            evidence={
                'total_claims': total_claims,
                'falsifiable': falsifiable_count,
                'confirmed': confirmed_count,
                'refuted': refuted_count,
                'unfalsifiable': unfalsifiable_count,
                'claim_results': claim_results
            },
            interpretation=interpretation,
            passes_threshold=passes
        )
        
        self.results.append(result)
        return result
    
    def test_compression(
        self,
        framework_features: np.ndarray,
        raw_features: np.ndarray,
        labels: np.ndarray,
        framework_model: Any,
        raw_model: Any
    ) -> GenerativityScore:
        """
        Test 4: Compression Test
        
        Does the framework compress information (good) or just relabel it (bad)?
        
        Good compression: Fewer features that predict as well or better.
        Bad compression: Many features that just restate the raw data differently.
        
        Parameters
        ----------
        framework_features : array-like
            Features extracted by narrative framework
        raw_features : array-like
            Raw features (e.g., TF-IDF)
        labels : array-like
            Target variable
        framework_model : sklearn model
            Model using framework features
        raw_model : sklearn model
            Model using raw features
            
        Returns
        -------
        GenerativityScore
        """
        n_framework_features = framework_features.shape[1]
        n_raw_features = raw_features.shape[1]
        
        # Cross-validate both approaches
        framework_scores = cross_val_score(
            framework_model, framework_features, labels, cv=5
        )
        raw_scores = cross_val_score(
            raw_model, raw_features, labels, cv=5
        )
        
        framework_acc = np.mean(framework_scores)
        raw_acc = np.mean(raw_scores)
        
        # Compression ratio
        compression_ratio = n_raw_features / (n_framework_features + 1e-10)
        
        # Quality retention
        quality_retention = framework_acc / (raw_acc + 1e-10)
        
        # Compression score: High compression + high quality = good
        # compression_ratio > 2 and quality_retention > 0.95 = excellent compression
        if compression_ratio > 2 and quality_retention > 0.95:
            score = 1.0
        elif compression_ratio > 1 and quality_retention > 0.90:
            score = 0.8
        elif quality_retention > 1.0:  # Better performance even without compression
            score = 0.9
        else:
            score = max(0, quality_retention - 0.5) * 2 * min(1, compression_ratio / 2)
        
        confidence = min(framework_acc, raw_acc)
        passes = score >= self.threshold
        
        interpretation = self._interpret_compression(
            compression_ratio, quality_retention, framework_acc, raw_acc, passes
        )
        
        result = GenerativityScore(
            test_name="Compression Test",
            score=score,
            confidence=confidence,
            evidence={
                'n_framework_features': n_framework_features,
                'n_raw_features': n_raw_features,
                'compression_ratio': compression_ratio,
                'framework_accuracy': framework_acc,
                'raw_accuracy': raw_acc,
                'quality_retention': quality_retention
            },
            interpretation=interpretation,
            passes_threshold=passes
        )
        
        self.results.append(result)
        return result
    
    def test_external_validation(
        self,
        insights: List[str],
        external_ratings: Dict[str, List[float]]
    ) -> GenerativityScore:
        """
        Test 5: External Validation Test
        
        Do people OUTSIDE the framework find patterns compelling and useful?
        
        If only the framework creator finds insights valuable, that suggests
        the framework is producing patterns that confirm their expectations
        rather than genuine discoveries.
        
        Parameters
        ----------
        insights : List[str]
            List of insights generated by framework
        external_ratings : Dict[str, List[float]]
            Ratings from external evaluators for each insight:
            {
                'novelty': [score1, score2, ...],
                'usefulness': [score1, score2, ...],
                'clarity': [score1, score2, ...]
            }
            Scores should be 0-1 scale
            
        Returns
        -------
        GenerativityScore
        """
        if not external_ratings:
            return GenerativityScore(
                test_name="External Validation Test",
                score=0.0,
                confidence=0.0,
                evidence={'error': 'No external ratings provided'},
                interpretation="Cannot assess - need external evaluators",
                passes_threshold=False
            )
        
        # Average across dimensions
        dimension_scores = {}
        for dimension, ratings in external_ratings.items():
            if ratings:
                dimension_scores[dimension] = np.mean(ratings)
        
        if not dimension_scores:
            return GenerativityScore(
                test_name="External Validation Test",
                score=0.0,
                confidence=0.0,
                evidence={'error': 'No valid ratings'},
                interpretation="Cannot assess - invalid rating data",
                passes_threshold=False
            )
        
        # Overall score is average across dimensions
        overall_score = np.mean(list(dimension_scores.values()))
        
        # Confidence based on number of raters
        n_raters = min([len(r) for r in external_ratings.values() if r])
        confidence = min(1.0, n_raters / 5)
        
        passes = overall_score >= self.threshold and n_raters >= 3
        
        interpretation = self._interpret_external_validation(
            dimension_scores, n_raters, overall_score, passes
        )
        
        result = GenerativityScore(
            test_name="External Validation Test",
            score=overall_score,
            confidence=confidence,
            evidence={
                'n_insights': len(insights),
                'n_raters': n_raters,
                'dimension_scores': dimension_scores,
                'overall_score': overall_score
            },
            interpretation=interpretation,
            passes_threshold=passes
        )
        
        self.results.append(result)
        return result
    
    def compute_overall_generativity(self) -> Dict[str, Any]:
        """
        Compute overall generativity assessment from all tests.
        
        Returns
        -------
        Dict with:
            - overall_score: weighted average of test scores
            - overall_confidence: minimum confidence across tests
            - verdict: "GENERATIVE", "QUESTIONABLE", or "NOT GENERATIVE"
            - passed_tests: number of tests passed
            - failed_tests: number of tests failed
            - recommendations: what to do next
        """
        if not self.results:
            return {
                'overall_score': 0.0,
                'overall_confidence': 0.0,
                'verdict': 'UNTESTED',
                'passed_tests': 0,
                'failed_tests': 0,
                'recommendations': ['Run generativity tests first']
            }
        
        # Weighted average (Novel Prediction and Compression are most important)
        weights = {
            'Novel Prediction Test': 0.3,
            'Compression Test': 0.3,
            'Falsifiability Test': 0.2,
            'Convergence Test': 0.1,
            'External Validation Test': 0.1
        }
        
        weighted_score = 0
        total_weight = 0
        min_confidence = 1.0
        
        for result in self.results:
            weight = weights.get(result.test_name, 0.1)
            weighted_score += result.score * weight
            total_weight += weight
            min_confidence = min(min_confidence, result.confidence)
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        passed_tests = sum(1 for r in self.results if r.passes_threshold)
        failed_tests = len(self.results) - passed_tests
        
        # Verdict
        if overall_score >= 0.7 and passed_tests >= 3:
            verdict = "GENERATIVE"
        elif overall_score >= 0.5 and passed_tests >= 2:
            verdict = "QUESTIONABLE"
        else:
            verdict = "NOT GENERATIVE"
        
        # Recommendations
        recommendations = self._generate_recommendations(verdict, self.results)
        
        return {
            'overall_score': overall_score,
            'overall_confidence': min_confidence,
            'verdict': verdict,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'total_tests': len(self.results),
            'test_results': self.results,
            'recommendations': recommendations
        }
    
    # Interpretation helpers
    
    def _interpret_novel_prediction(
        self, improvement: float, narrative_score: float, 
        baseline_score: float, passes: bool
    ) -> str:
        if passes:
            return (
                f"Framework generalizes well to unseen domains. "
                f"Achieved {narrative_score:.1%} accuracy vs {baseline_score:.1%} baseline "
                f"({improvement:+.1%} improvement). This suggests genuine pattern discovery."
            )
        elif improvement > 0:
            return (
                f"Framework shows modest improvement over baseline "
                f"({improvement:+.1%}), but not compelling enough to claim strong generativity. "
                f"May capture some genuine patterns mixed with noise."
            )
        else:
            return (
                f"Framework performs worse than baseline on unseen domains "
                f"({improvement:+.1%}). This suggests overfitting to training patterns "
                f"rather than discovering transferable insights."
            )
    
    def _interpret_convergence(
        self, avg_convergence: float, n_analyses: int,
        convergence_scores: List[float], passes: bool
    ) -> str:
        if passes:
            return (
                f"High convergence across {n_analyses} independent analyses "
                f"(score: {avg_convergence:.3f}). This suggests the framework captures "
                f"objective patterns rather than analyst-dependent interpretations."
            )
        else:
            return (
                f"Low convergence across {n_analyses} analyses (score: {avg_convergence:.3f}). "
                f"Different analysts find different patterns, suggesting the framework "
                f"may be flexible enough to confirm various expectations."
            )
    
    def _interpret_falsifiability(
        self, falsifiable: int, confirmed: int, refuted: int,
        unfalsifiable: int, passes: bool
    ) -> str:
        total = falsifiable + unfalsifiable
        
        if passes:
            refutation_note = ""
            if refuted > 0:
                refutation_note = (
                    f" Importantly, {refuted} claim(s) were refuted, "
                    f"demonstrating honest empirical testing."
                )
            return (
                f"{falsifiable}/{total} claims are falsifiable with clear refutation criteria. "
                f"{confirmed} confirmed, {refuted} refuted.{refutation_note} "
                f"This is good scientific practice."
            )
        else:
            return (
                f"Only {falsifiable}/{total} claims are falsifiable. "
                f"{unfalsifiable} claims lack clear refutation criteria, "
                f"making them scientifically weak even if they feel satisfying. "
                f"Framework needs more precise predictions."
            )
    
    def _interpret_compression(
        self, compression_ratio: float, quality_retention: float,
        framework_acc: float, raw_acc: float, passes: bool
    ) -> str:
        if passes:
            return (
                f"Excellent compression: {compression_ratio:.1f}x fewer features "
                f"with {quality_retention:.1%} quality retention. "
                f"Framework distills raw data into meaningful abstractions "
                f"rather than just relabeling it."
            )
        elif quality_retention < 0.9:
            return (
                f"Poor compression: Framework loses substantial information "
                f"(quality retention: {quality_retention:.1%}). "
                f"May be discarding signal along with noise."
            )
        else:
            return (
                f"Modest compression: {compression_ratio:.1f}x reduction "
                f"with {quality_retention:.1%} quality. "
                f"Framework provides some abstraction but may be redundant with raw features."
            )
    
    def _interpret_external_validation(
        self, dimension_scores: Dict, n_raters: int,
        overall_score: float, passes: bool
    ) -> str:
        if passes:
            top_dimensions = sorted(
                dimension_scores.items(), key=lambda x: x[1], reverse=True
            )[:2]
            return (
                f"Strong external validation from {n_raters} independent raters "
                f"(overall: {overall_score:.2f}). "
                f"Highest rated: {top_dimensions[0][0]} ({top_dimensions[0][1]:.2f}). "
                f"Framework insights resonate beyond the creator."
            )
        else:
            return (
                f"Weak external validation ({n_raters} raters, score: {overall_score:.2f}). "
                f"Insights may be more compelling to framework creator than to others, "
                f"suggesting possible confirmation bias."
            )
    
    def _generate_recommendations(
        self, verdict: str, results: List[GenerativityScore]
    ) -> List[str]:
        recommendations = []
        
        if verdict == "GENERATIVE":
            recommendations.append("Framework shows strong generativity. Continue development.")
            recommendations.append("Consider publication and external validation.")
            recommendations.append("Test on additional unseen domains to confirm generality.")
        
        elif verdict == "QUESTIONABLE":
            recommendations.append("Framework shows promise but needs strengthening.")
            
            for result in results:
                if not result.passes_threshold:
                    if "Novel Prediction" in result.test_name:
                        recommendations.append(
                            "Improve cross-domain transfer: simplify features or add domain adaptation."
                        )
                    elif "Compression" in result.test_name:
                        recommendations.append(
                            "Improve compression: reduce feature redundancy or increase predictive power."
                        )
                    elif "Falsifiability" in result.test_name:
                        recommendations.append(
                            "Make claims more falsifiable: specify precise refutation criteria."
                        )
        
        else:  # NOT GENERATIVE
            recommendations.append("Framework does not demonstrate generativity.")
            recommendations.append("Consider: Is this elaborate rationalization rather than discovery?")
            recommendations.append("Options: (1) Simplify drastically, (2) Abandon, (3) Reframe as exploratory only.")
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate a comprehensive text report of all generativity tests."""
        overall = self.compute_overall_generativity()
        
        report = []
        report.append("=" * 80)
        report.append("GENERATIVITY ASSESSMENT REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Overall Verdict: {overall['verdict']}")
        report.append(f"Overall Score: {overall['overall_score']:.3f} / 1.000")
        report.append(f"Confidence: {overall['overall_confidence']:.3f}")
        report.append(f"Tests Passed: {overall['passed_tests']} / {overall['total_tests']}")
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

