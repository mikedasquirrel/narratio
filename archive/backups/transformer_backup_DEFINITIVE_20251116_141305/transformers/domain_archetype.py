"""
Domain-Specific Archetype Transformer

Base class for domain-specific archetypal measurement.
Measures distance from domain-specific Ξ (golden narratio).

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
import re
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

from .base import NarrativeTransformer
from ..config.domain_config import DomainConfig


class DomainArchetypeTransformer(NarrativeTransformer):
    """
    Base class for domain-specific archetypal measurement.
    
    Measures distance from domain-specific Ξ (golden narratio):
    - Learns archetypal patterns from winners in THIS domain
    - Computes story quality as proximity to domain Ξ
    - Accounts for gravitational dilution (competing stories)
    
    This addresses the core problem: story quality should be measured
    relative to domain-specific archetypal perfection, not generic features.
    
    Features Extracted:
    - Individual archetype scores (mental_game, elite_skill, etc.)
    - archetype_distance: Euclidean distance from domain Ξ
    - archetype_similarity: Cosine similarity to domain Ξ
    - archetype_match_score: How many archetype patterns present
    - gravitational_dilution: Competing narrative interference
    - story_quality_composite: Overall Ξ proximity score (PRIMARY)
    
    Parameters
    ----------
    domain_config : DomainConfig
        Domain-specific configuration with archetype patterns
    
    Attributes
    ----------
    domain_config : DomainConfig
        Domain configuration
    archetype_patterns : dict
        Domain-specific patterns for each archetype dimension
    archetype_weights : dict
        Weights for each archetype dimension
    xi_vector : ndarray
        Learned domain Ξ (centroid of winner archetype space)
    
    Examples
    --------
    >>> from ..config.domain_config import DomainConfig
    >>> config = DomainConfig('golf')
    >>> transformer = DomainArchetypeTransformer(config)
    >>> features = transformer.fit_transform(texts, outcomes)
    >>> story_quality = features[:, -1]  # Last column is composite score
    """
    
    def __init__(self, domain_config: DomainConfig):
        super().__init__(
            narrative_id=f"domain_archetype_{domain_config.domain_name}",
            description=f"Domain-specific archetype measurement for {domain_config.domain_name}"
        )
        self.domain_config = domain_config
        self.archetype_patterns = domain_config.get_archetype_patterns()
        self.archetype_weights = domain_config.get_archetype_weights()
        self.xi_vector = None  # Learned from winners
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        """
        Learn domain-specific Ξ from winners.
        
        Parameters
        ----------
        X : list of str
            Training texts
        y : array-like, optional
            Outcomes (1=winner, 0=loser for binary; continuous for regression)
            If None, assumes all training data are winners
        
        Returns
        -------
        self
        """
        # Handle empty input
        if len(X) == 0:
            raise ValueError("Cannot fit on empty dataset")
        
        # Filter empty texts
        valid_indices = [i for i, text in enumerate(X) if text and len(str(text).strip()) > 0]
        if len(valid_indices) == 0:
            raise ValueError("No valid (non-empty) texts found")
        
        X_filtered = [X[i] for i in valid_indices]
        
        if y is None:
            y = np.ones(len(X_filtered))  # Assume all training data are winners
        else:
            y = np.array(y)[valid_indices]
        
        # Handle single sample
        if len(X_filtered) == 1:
            # Single sample - use it as Ξ
            archetype_features = self._extract_archetype_features(X_filtered)
            self.xi_vector = archetype_features[0]
            self.is_fitted_ = True
            return self
        
        # Extract archetype features for all texts
        archetype_features = self._extract_archetype_features(X_filtered)
        
        # Learn Ξ from winners (top 25%)
        if len(np.unique(y)) > 2:  # Continuous outcomes
            threshold = np.percentile(y, 75)
            winner_mask = y >= threshold
        else:  # Binary
            winner_mask = y == 1
        
        winners = archetype_features[winner_mask]
        
        # Handle edge cases
        if len(winners) == 0:
            # No winners identified - use top 10% by outcome
            if len(np.unique(y)) > 2:
                threshold = np.percentile(y, 90)
                winners = archetype_features[y >= threshold]
            else:
                # Binary but no 1s - use all data
                winners = archetype_features
        
        if len(winners) == 0:
            # Still no winners - use all data
            winners = archetype_features
        
        # Domain Ξ = centroid of winner archetype space
        self.xi_vector = np.mean(winners, axis=0)
        
        # Ensure xi_vector is valid
        if np.any(np.isnan(self.xi_vector)) or np.any(np.isinf(self.xi_vector)):
            # Fallback to mean of all features
            self.xi_vector = np.nanmean(archetype_features, axis=0)
            # Replace any remaining NaN/Inf with 0
            self.xi_vector = np.nan_to_num(self.xi_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.is_fitted_ = True
        return self
    
    def _extract_archetype_features(self, X) -> np.ndarray:
        """
        Extract domain-specific archetype features.
        
        Scores each text on domain-specific archetype dimensions
        (e.g., mental_game, elite_skill for Golf).
        
        Parameters
        ----------
        X : list of str
            Texts to extract features from
        
        Returns
        -------
        ndarray
            Shape (n_samples, n_archetypes) with weighted archetype scores
        """
        features = []
        
        for text in X:
            text_lower = text.lower()
            archetype_scores = {}
            
            # Score each archetype dimension
            for archetype_name, patterns in self.archetype_patterns.items():
                # Count matches, normalize by pattern count
                score = sum(1 for pattern in patterns if pattern.lower() in text_lower)
                score = score / len(patterns) if len(patterns) > 0 else 0.0
                archetype_scores[archetype_name] = score
            
            # Weight archetype scores
            weighted_scores = [
                archetype_scores[name] * self.archetype_weights.get(name, 0)
                for name in self.archetype_patterns.keys()
            ]
            
            features.append(weighted_scores)
        
        return np.array(features)
    
    def transform(self, X):
        """
        Compute distance from domain Ξ.
        
        Parameters
        ----------
        X : list of str
            Texts to transform
        
        Returns
        -------
        ndarray
            Features including archetype scores and Ξ distance measures
        """
        if not self.is_fitted_:
            raise RuntimeError("Transformer must be fitted before transform")
        
        archetype_features = self._extract_archetype_features(X)
        
        n_samples = len(X)
        n_archetypes = len(self.archetype_patterns)
        # archetypes + distance + similarity + match_score + dilution + story_quality
        n_features_total = n_archetypes + 5
        
        features = np.zeros((n_samples, n_features_total))
        
        for i, (text, arch_vec) in enumerate(zip(X, archetype_features)):
            # Distance from Ξ
            euclidean_dist = np.linalg.norm(arch_vec - self.xi_vector)
            
            # Cosine similarity (handle zero vectors)
            if np.linalg.norm(arch_vec) > 1e-10 and np.linalg.norm(self.xi_vector) > 1e-10:
                cosine_sim = cosine_similarity([arch_vec], [self.xi_vector])[0, 0]
            else:
                cosine_sim = 0.0
            
            # Individual archetype scores
            features[i, :n_archetypes] = arch_vec
            
            # Derived features
            features[i, n_archetypes] = euclidean_dist
            features[i, n_archetypes + 1] = cosine_sim
            features[i, n_archetypes + 2] = np.sum(arch_vec > 0) / n_archetypes  # Match score
            features[i, n_archetypes + 3] = self._compute_gravitational_dilution(text)
            
            # Story quality (PRIMARY MEASURE): proximity to Ξ
            # Lower distance = higher quality
            xi_norm = np.linalg.norm(self.xi_vector) + 1e-8
            features[i, n_archetypes + 4] = 1.0 - (euclidean_dist / xi_norm)
        
        return features
    
    def _compute_gravitational_dilution(self, text: str) -> float:
        """
        Measure competing narrative interference.
        
        Too many proper nouns = competing stories interfere
        Too few proper nouns = insufficient context
        Optimal range = clear single narrative
        
        Parameters
        ----------
        text : str
            Text to analyze
        
        Returns
        -------
        float
            Dilution score (0.2 = low, 0.8 = high)
        """
        # Count proper nouns as proxy for narrative complexity
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', text))
        required = self.domain_config.get_nominative_richness_requirement()
        
        # Dilution = too many (interference) or too few (insufficient context)
        if proper_nouns < required * 0.5:
            return 0.8  # High dilution from insufficient context
        elif proper_nouns > required * 2:
            return 0.6  # Moderate dilution from too many competing stories
        else:
            return 0.2  # Low dilution - optimal range
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names.
        
        Returns
        -------
        list
            Feature names
        """
        archetype_names = list(self.archetype_patterns.keys())
        derived_names = [
            'archetype_distance',
            'archetype_similarity',
            'archetype_match_score',
            'gravitational_dilution',
            'story_quality_composite'
        ]
        return archetype_names + derived_names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of the transformation."""
        n_archetypes = len(self.archetype_patterns)
        archetype_list = ', '.join(self.archetype_patterns.keys())
        
        return f"""Domain: {self.domain_config.domain_name}
Archetypes: {archetype_list}
Features extracted: {n_archetypes} archetype scores + 5 derived features
Primary measure: story_quality_composite (proximity to domain Ξ)
"""

