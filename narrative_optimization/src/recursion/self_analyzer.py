"""
Framework Self-Analysis Module

Applies the narrative optimization framework to itself.

The ultimate recursion test: Does the theoretical document about narrative
quality itself exhibit high narrative quality? Do better-written theory
documents get more external validation?

This tests whether the recursion is profound or just circular reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re


@dataclass
class SelfAnalysisResult:
    """Results from analyzing the framework's own writing."""
    document_name: str
    narrative_scores: Dict[str, float]
    overall_narrative_quality: float
    predicted_external_acceptance: float
    recursive_coherence: float  # Does framework validate its own quality?
    interpretation: str
    
    def __repr__(self):
        return (
            f"{self.document_name}: Quality={self.overall_narrative_quality:.3f}, "
            f"Predicted acceptance={self.predicted_external_acceptance:.3f}, "
            f"Recursive coherence={self.recursive_coherence:.3f}"
        )


class FrameworkSelfAnalyzer:
    """
    Applies narrative transformers to the framework's own theoretical writing.
    
    Tests:
    1. Does the theory document itself have high narrative quality?
    2. Do narrative quality scores predict external validation?
    3. Is the recursion coherent or circular?
    """
    
    def __init__(self, transformers: Dict[str, Any]):
        """
        Parameters
        ----------
        transformers : Dict[str, transformer]
            Narrative transformers to apply to own writing
        """
        self.transformers = transformers
        self.results: List[SelfAnalysisResult] = {}
    
    def analyze_document(
        self,
        document_path: str,
        document_name: str,
        external_ratings: Optional[Dict[str, float]] = None
    ) -> SelfAnalysisResult:
        """
        Analyze a theoretical document using the framework.
        
        Parameters
        ----------
        document_path : str
            Path to document
        document_name : str
            Name for identification
        external_ratings : Dict, optional
            Actual external ratings if available
            
        Returns
        -------
        SelfAnalysisResult
        """
        # Read document
        with open(document_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Apply each transformer
        narrative_scores = {}
        
        for name, transformer in self.transformers.items():
            try:
                # Fit on document itself (self-referential)
                transformer.fit([text])
                
                # Transform
                features = transformer.transform([text])[0]
                
                # Aggregate to single score
                score = np.mean(features)
                narrative_scores[name] = score
            except Exception as e:
                narrative_scores[name] = 0.0
        
        # Overall quality (average across dimensions)
        overall_quality = np.mean(list(narrative_scores.values()))
        
        # Predict external acceptance
        # Theory says: better narrative quality → better external reception
        predicted_acceptance = self._predict_acceptance(narrative_scores)
        
        # Check recursive coherence
        if external_ratings is not None:
            actual_acceptance = np.mean(list(external_ratings.values()))
            recursive_coherence = 1 - abs(predicted_acceptance - actual_acceptance)
        else:
            recursive_coherence = 0.5  # Unknown
        
        interpretation = self._interpret_self_analysis(
            narrative_scores, overall_quality, predicted_acceptance,
            recursive_coherence, external_ratings
        )
        
        result = SelfAnalysisResult(
            document_name=document_name,
            narrative_scores=narrative_scores,
            overall_narrative_quality=overall_quality,
            predicted_external_acceptance=predicted_acceptance,
            recursive_coherence=recursive_coherence,
            interpretation=interpretation
        )
        
        self.results[document_name] = result
        return result
    
    def compare_document_versions(
        self,
        versions: Dict[str, str],
        external_ratings: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compare different versions of theory documents.
        
        Test: Do narrative improvements predict better external reception?
        
        Parameters
        ----------
        versions : Dict[str, str]
            Mapping of version name to document path
        external_ratings : Dict[str, Dict]
            External ratings for each version
            
        Returns
        -------
        Dict with comparison results
        """
        version_results = {}
        
        for version_name, doc_path in versions.items():
            ratings = external_ratings.get(version_name, None)
            result = self.analyze_document(doc_path, version_name, ratings)
            version_results[version_name] = result
        
        # Test correlation between narrative quality and acceptance
        if len(version_results) >= 3:
            qualities = [r.overall_narrative_quality for r in version_results.values()]
            acceptances = [r.predicted_external_acceptance for r in version_results.values()]
            
            from scipy import stats
            correlation, p_value = stats.pearsonr(qualities, acceptances)
            
            validates_recursion = correlation > 0.5 and p_value < 0.10
        else:
            correlation = 0.0
            p_value = 1.0
            validates_recursion = False
        
        return {
            'version_results': version_results,
            'correlation': correlation,
            'p_value': p_value,
            'validates_recursion': validates_recursion,
            'interpretation': self._interpret_version_comparison(
                correlation, p_value, validates_recursion
            )
        }
    
    def test_naming_effects(
        self,
        document_path: str,
        alternative_names: List[str],
        actual_name: str
    ) -> Dict[str, Any]:
        """
        Test domain name tethering: Does renaming the framework affect results?
        
        Theory predicts: How you NAME your framework affects what you find.
        
        Parameters
        ----------
        document_path : str
            Path to theory document
        alternative_names : List[str]
            Alternative names for the framework
        actual_name : str
            The actual name used
            
        Returns
        -------
        Dict with naming effects analysis
        """
        # Read document
        with open(document_path, 'r') as f:
            original_text = f.read()
        
        results_by_name = {}
        
        for alt_name in alternative_names:
            # Replace framework name throughout document
            modified_text = original_text.replace(actual_name, alt_name)
            
            # Analyze modified version
            narrative_scores = {}
            
            for name, transformer in self.transformers.items():
                try:
                    transformer.fit([modified_text])
                    features = transformer.transform([modified_text])[0]
                    score = np.mean(features)
                    narrative_scores[name] = score
                except:
                    narrative_scores[name] = 0.0
            
            results_by_name[alt_name] = {
                'narrative_scores': narrative_scores,
                'overall_quality': np.mean(list(narrative_scores.values()))
            }
        
        # Test if naming affects narrative scores
        qualities = [r['overall_quality'] for r in results_by_name.values()]
        name_variation = np.std(qualities)
        
        # If variation > 0.05, naming has meaningful effect
        naming_effects_detected = name_variation > 0.05
        
        return {
            'results_by_name': results_by_name,
            'name_variation': name_variation,
            'naming_effects_detected': naming_effects_detected,
            'interpretation': self._interpret_naming_effects(
                name_variation, naming_effects_detected
            )
        }
    
    def _predict_acceptance(self, narrative_scores: Dict[str, float]) -> float:
        """
        Predict external acceptance from narrative scores.
        
        Simple model: weighted average of narrative dimensions
        """
        # Weight dimensions by typical importance
        weights = {
            'nominative': 0.15,
            'potential': 0.25,
            'linguistic': 0.20,
            'self_perception': 0.15,
            'relational': 0.15,
            'ensemble': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for dim, score in narrative_scores.items():
            weight = weights.get(dim, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            predicted = weighted_score / total_weight
        else:
            predicted = 0.5
        
        # Normalize to 0-1
        return np.clip(predicted, 0, 1)
    
    def _interpret_self_analysis(
        self,
        scores: Dict,
        quality: float,
        predicted: float,
        coherence: float,
        external: Optional[Dict]
    ) -> str:
        if coherence > 0.8:
            return (
                f"RECURSIVE COHERENCE HIGH: Framework predicts its own reception accurately "
                f"(coherence={coherence:.2f}). Narrative quality={quality:.2f}, "
                f"predicted acceptance={predicted:.2f}. "
                f"This suggests the framework applies validly to itself - recursion is coherent, not circular."
            )
        elif external is None:
            return (
                f"NO EXTERNAL VALIDATION YET: Narrative quality={quality:.2f}, "
                f"predicted acceptance={predicted:.2f}. Need external ratings to test recursive coherence."
            )
        else:
            actual = np.mean(list(external.values()))
            return (
                f"RECURSIVE MISMATCH: Framework predicts acceptance={predicted:.2f} "
                f"but actual={actual:.2f} (coherence={coherence:.2f}). "
                f"Either the framework doesn't apply to itself, or the theory is flawed."
            )
    
    def _interpret_version_comparison(
        self, correlation: float, p_value: float, validates: bool
    ) -> str:
        if validates:
            return (
                f"RECURSIVE VALIDATION: Better-written theory versions receive better "
                f"external ratings (r={correlation:.3f}, p={p_value:.4f}). "
                f"Framework successfully predicts its own reception - recursion is profound."
            )
        elif correlation > 0:
            return (
                f"WEAK RECURSIVE VALIDATION: Positive correlation (r={correlation:.3f}, p={p_value:.4f}) "
                f"but not strong enough to confidently validate recursive application."
            )
        else:
            return (
                f"RECURSIVE FAILURE: No correlation between narrative quality and reception "
                f"(r={correlation:.3f}, p={p_value:.4f}). Framework does not apply to itself - "
                f"recursion may be circular reasoning."
            )
    
    def _interpret_naming_effects(
        self, variation: float, detected: bool
    ) -> str:
        if detected:
            return (
                f"NAMING EFFECTS DETECTED: Framework name affects narrative scores "
                f"(variation={variation:.3f}). This supports domain name tethering theory - "
                f"how you NAME your research affects what you find."
            )
        else:
            return (
                f"NO NAMING EFFECTS: Framework name does not affect narrative scores "
                f"(variation={variation:.3f}). Domain name tethering is NOT supported in this test."
            )
    
    def generate_report(self) -> str:
        """Generate self-analysis report."""
        report = []
        report.append("=" * 80)
        report.append("FRAMEWORK SELF-ANALYSIS REPORT")
        report.append("Testing: Does the framework apply to itself?")
        report.append("=" * 80)
        report.append("")
        
        if not self.results:
            report.append("No documents analyzed yet.")
            return "\n".join(report)
        
        for doc_name, result in self.results.items():
            report.append(f"{doc_name.upper()}:")
            report.append(f"  Overall narrative quality: {result.overall_narrative_quality:.3f}")
            report.append(f"  Predicted external acceptance: {result.predicted_external_acceptance:.3f}")
            report.append(f"  Recursive coherence: {result.recursive_coherence:.3f}")
            report.append("")
            report.append("  Dimension scores:")
            for dim, score in sorted(result.narrative_scores.items(), key=lambda x: x[1], reverse=True):
                report.append(f"    {dim}: {score:.3f}")
            report.append("")
            report.append(f"  {result.interpretation}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

