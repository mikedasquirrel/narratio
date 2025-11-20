"""
Nominative-Narrative Integration Layer

Tests the theoretical entanglement between nominative and narrative dimensions.

Theory claims: Names (nominative) and stories (narrative) are entangled -
names are the first chapter of narratives, and both should interact
to predict outcomes better than either alone.

This module tests whether that entanglement is real or theoretical wishful thinking.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings


@dataclass
class EntanglementResult:
    """Results from testing nominative-narrative entanglement."""
    nominative_only_score: float
    narrative_only_score: float
    integrated_score: float
    improvement: float
    interaction_effects: Dict[str, float]
    entanglement_validated: bool
    interpretation: str
    
    def __repr__(self):
        status = "✓ ENTANGLED" if self.entanglement_validated else "✗ INDEPENDENT"
        return (
            f"{status} | Nom: {self.nominative_only_score:.3f}, "
            f"Narr: {self.narrative_only_score:.3f}, "
            f"Integrated: {self.integrated_score:.3f} "
            f"(+{self.improvement:+.3f})"
        )


class NominativeNarrativeBridge(BaseEstimator, TransformerMixin):
    """
    Integration layer that tests entanglement between nominative and narrative features.
    
    Combines:
    - Nominative features (from nominative_taxonomy transformers)
    - Narrative features (from 6 core narrative transformers)
    
    Tests:
    - Do they predict better together than separately?
    - Are there interaction effects?
    - Is the theory of entanglement validated?
    """
    
    def __init__(
        self,
        nominative_transformer: Any,
        narrative_transformer: Any,
        test_interactions: bool = True
    ):
        """
        Parameters
        ----------
        nominative_transformer : sklearn transformer
            Extracts nominative features (e.g., PhoneticFormulaTransformer)
        narrative_transformer : sklearn transformer
            Extracts narrative features (e.g., NarrativePotentialTransformer)
        test_interactions : bool
            Whether to create interaction features
        """
        self.nominative_transformer = nominative_transformer
        self.narrative_transformer = narrative_transformer
        self.test_interactions = test_interactions
        
        self.nominative_features_ = None
        self.narrative_features_ = None
        self.interaction_indices_ = None
    
    def fit(self, X, y=None):
        """Fit both transformers."""
        self.nominative_transformer.fit(X, y)
        self.narrative_transformer.fit(X, y)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Extract and combine nominative and narrative features.
        
        Returns
        -------
        np.ndarray with columns:
        - Nominative features
        - Narrative features
        - Interaction features (if test_interactions=True)
        """
        # Extract features from each transformer
        nom_features = self.nominative_transformer.transform(X)
        narr_features = self.narrative_transformer.transform(X)
        
        self.nominative_features_ = nom_features
        self.narrative_features_ = narr_features
        
        # Combine
        combined = np.hstack([nom_features, narr_features])
        
        # Add interaction features if requested
        if self.test_interactions:
            interactions = self._create_interactions(nom_features, narr_features)
            combined = np.hstack([combined, interactions])
        
        return combined
    
    def _create_interactions(
        self,
        nom_features: np.ndarray,
        narr_features: np.ndarray
    ) -> np.ndarray:
        """
        Create interaction features between nominative and narrative dimensions.
        
        Theory predicts certain interactions should matter:
        - Phonetic harshness × Narrative aggression
        - Semantic power × Self-perception confidence
        - Name length × Narrative complexity
        etc.
        """
        n_samples = nom_features.shape[0]
        
        # Sample key interactions (in practice, test all combinations)
        # Here we'll create multiplicative interactions for top features
        
        interactions = []
        
        # Create all pairwise products (expensive but comprehensive)
        n_nom = min(nom_features.shape[1], 10)  # Limit to top 10 nom features
        n_narr = min(narr_features.shape[1], 10)  # Limit to top 10 narr features
        
        for i in range(n_nom):
            for j in range(n_narr):
                interaction = nom_features[:, i] * narr_features[:, j]
                interactions.append(interaction)
        
        if interactions:
            self.interaction_indices_ = list(range(len(interactions)))
            return np.column_stack(interactions)
        else:
            return np.zeros((n_samples, 1))
    
    def test_entanglement(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Optional[Any] = None,
        cv: int = 5
    ) -> EntanglementResult:
        """
        Test if nominative and narrative features are truly entangled.
        
        Compares:
        1. Nominative only
        2. Narrative only
        3. Both combined (additive)
        4. Both with interactions (entangled)
        
        Parameters
        ----------
        X : array-like
            Text data
        y : array-like
            Labels
        model : sklearn model, optional
            Defaults to RandomForestClassifier
        cv : int
            Cross-validation folds
            
        Returns
        -------
        EntanglementResult
        """
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Fit transformers
        self.fit(X, y)
        
        # Test 1: Nominative only
        X_nom = self.nominative_transformer.transform(X)
        scores_nom = cross_val_score(model, X_nom, y, cv=cv)
        score_nom = np.mean(scores_nom)
        
        # Test 2: Narrative only
        X_narr = self.narrative_transformer.transform(X)
        scores_narr = cross_val_score(model, X_narr, y, cv=cv)
        score_narr = np.mean(scores_narr)
        
        # Test 3: Combined (without interactions)
        self.test_interactions = False
        X_combined = self.transform(X)
        scores_combined = cross_val_score(model, X_combined, y, cv=cv)
        score_combined = np.mean(scores_combined)
        
        # Test 4: Integrated (with interactions)
        self.test_interactions = True
        X_integrated = self.transform(X)
        scores_integrated = cross_val_score(model, X_integrated, y, cv=cv)
        score_integrated = np.mean(scores_integrated)
        
        # Analyze results
        best_individual = max(score_nom, score_narr)
        improvement_combined = score_combined - best_individual
        improvement_integrated = score_integrated - score_combined
        total_improvement = score_integrated - best_individual
        
        # Interaction effects
        interaction_effects = {
            'additive_gain': improvement_combined,
            'interaction_gain': improvement_integrated,
            'total_gain': total_improvement,
            'synergy_ratio': improvement_integrated / (improvement_combined + 1e-10)
        }
        
        # Validation criteria:
        # - Integrated > Combined > Max(Individual) by meaningful amount
        # - Interaction gain > 0 (shows true entanglement, not just addition)
        entanglement_validated = (
            total_improvement > 0.03 and  # >3% improvement
            improvement_integrated > 0.01  # >1% from interactions alone
        )
        
        interpretation = self._interpret_entanglement(
            score_nom, score_narr, score_combined, score_integrated,
            interaction_effects, entanglement_validated
        )
        
        result = EntanglementResult(
            nominative_only_score=score_nom,
            narrative_only_score=score_narr,
            integrated_score=score_integrated,
            improvement=total_improvement,
            interaction_effects=interaction_effects,
            entanglement_validated=entanglement_validated,
            interpretation=interpretation
        )
        
        return result
    
    def _interpret_entanglement(
        self,
        nom_score: float,
        narr_score: float,
        combined_score: float,
        integrated_score: float,
        effects: Dict,
        validated: bool
    ) -> str:
        if validated:
            return (
                f"ENTANGLEMENT VALIDATED: Nominative and narrative dimensions "
                f"interact synergistically. Integrated model ({integrated_score:.3f}) "
                f"exceeds both individual models (nom: {nom_score:.3f}, narr: {narr_score:.3f}) "
                f"and simple combination ({combined_score:.3f}). "
                f"Interaction effects contribute {effects['interaction_gain']:+.3f}, "
                f"supporting theory that names and narratives are entangled."
            )
        elif integrated_score > max(nom_score, narr_score):
            return (
                f"WEAK ENTANGLEMENT: Combined features improve prediction "
                f"({integrated_score:.3f} vs max individual {max(nom_score, narr_score):.3f}), "
                f"but interaction effects are small ({effects['interaction_gain']:+.3f}). "
                f"Nominative and narrative may be complementary but not strongly entangled."
            )
        else:
            return (
                f"NO ENTANGLEMENT: Integrated model ({integrated_score:.3f}) "
                f"does not improve over best individual model ({max(nom_score, narr_score):.3f}). "
                f"Nominative and narrative dimensions appear INDEPENDENT, contradicting theory. "
                f"Names and stories do not interact in this domain."
            )
    
    def get_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Optional[Any] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get feature importance for nominative, narrative, and interaction features.
        
        Returns
        -------
        Dict with 'nominative', 'narrative', 'interaction' importance arrays
        """
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Fit and transform
        self.test_interactions = True
        self.fit(X, y)
        X_transformed = self.transform(X)
        
        # Train model
        model.fit(X_transformed, y)
        
        # Get importances
        importances = model.feature_importances_
        
        n_nom = self.nominative_features_.shape[1]
        n_narr = self.narrative_features_.shape[1]
        
        return {
            'nominative': importances[:n_nom],
            'narrative': importances[n_nom:n_nom+n_narr],
            'interaction': importances[n_nom+n_narr:] if len(importances) > n_nom+n_narr else np.array([])
        }


def run_entanglement_experiment(
    domains: Dict[str, Tuple[np.ndarray, np.ndarray]],
    nominative_transformer: Any,
    narrative_transformer: Any
) -> Dict[str, EntanglementResult]:
    """
    Run entanglement test across multiple domains.
    
    Parameters
    ----------
    domains : Dict[str, Tuple]
        Mapping of domain name to (X, y) data
    nominative_transformer : transformer
        Nominative feature extractor
    narrative_transformer : transformer
        Narrative feature extractor
        
    Returns
    -------
    Dict mapping domain names to EntanglementResult
    """
    results = {}
    
    for domain_name, (X, y) in domains.items():
        print(f"Testing entanglement in {domain_name}...")
        
        bridge = NominativeNarrativeBridge(
            nominative_transformer=nominative_transformer,
            narrative_transformer=narrative_transformer,
            test_interactions=True
        )
        
        result = bridge.test_entanglement(X, y)
        results[domain_name] = result
        
        print(f"  {result}")
    
    return results


def generate_entanglement_report(results: Dict[str, EntanglementResult]) -> str:
    """Generate comprehensive entanglement test report."""
    report = []
    report.append("=" * 80)
    report.append("NOMINATIVE-NARRATIVE ENTANGLEMENT REPORT")
    report.append("=" * 80)
    report.append("")
    
    if not results:
        report.append("No domains tested.")
        return "\n".join(report)
    
    # Overall statistics
    n_validated = sum(1 for r in results.values() if r.entanglement_validated)
    validation_rate = n_validated / len(results)
    
    mean_improvement = np.mean([r.improvement for r in results.values()])
    mean_interaction = np.mean([
        r.interaction_effects['interaction_gain']
        for r in results.values()
    ])
    
    report.append("CROSS-DOMAIN SYNTHESIS:")
    report.append(f"  Domains tested: {len(results)}")
    report.append(f"  Domains with validated entanglement: {n_validated} ({validation_rate:.1%})")
    report.append(f"  Mean improvement from integration: {mean_improvement:+.3f}")
    report.append(f"  Mean interaction effect: {mean_interaction:+.3f}")
    report.append("")
    
    # Verdict
    if validation_rate >= 0.75:
        verdict = "THEORY VALIDATED - Nominative and narrative are entangled across domains"
    elif validation_rate >= 0.5:
        verdict = "PARTIAL VALIDATION - Entanglement exists in some domains"
    elif mean_improvement > 0:
        verdict = "WEAK EVIDENCE - Some integration benefit but not strong entanglement"
    else:
        verdict = "THEORY REFUTED - Nominative and narrative are independent"
    
    report.append(f"VERDICT: {verdict}")
    report.append("")
    
    # Domain-specific results
    report.append("DOMAIN-SPECIFIC RESULTS:")
    report.append("-" * 80)
    
    for domain_name, result in sorted(results.items(), key=lambda x: x[1].improvement, reverse=True):
        report.append(f"\n{domain_name.upper()}:")
        report.append(str(result))
        report.append(f"  {result.interpretation}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

