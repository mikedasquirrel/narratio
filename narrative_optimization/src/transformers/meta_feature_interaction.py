"""
Meta-Feature Interaction Transformer

Generates interaction features between existing features WITHOUT domain knowledge
about which interactions matter.

Philosophy: Let the learning system discover which feature combinations
predict outcomes. Don't assume multiplicative vs ratio vs polynomial.

Universal feature engineering approach.

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import entropy
from itertools import combinations
import warnings

from .base_transformer import FeatureNarrativeTransformer


class MetaFeatureInteractionTransformer(FeatureNarrativeTransformer):
    """
    Generate interaction features automatically.
    
    Discovers which feature combinations matter by generating candidates:
    - Multiplicative interactions (A × B)
    - Ratio features (A / B)
    - Polynomial features (A², A³, etc.)
    - Synergy scores (A + B > individual effects)
    - Antagonism scores (A cancels B)
    
    Example discoveries:
    - "nominative_richness × arc_steepness" predicts outcomes
    - "momentum / variance" is key ratio
    - "velocity²" matters more than velocity
    
    The learning system tests all candidates and discovers which matter.
    
    Features Extracted (dynamic, ~100+ typical):
    
    For N input features:
    - Multiplicative: N×(N-1)/2 pairwise products
    - Ratios: N×(N-1)/2 pairwise ratios
    - Polynomials: N×degree polynomial terms
    - Synergy: Selected high-correlation pairs
    - Antagonism: Selected anti-correlation pairs
    
    Feature selection based on:
    - Variance filtering (remove low-variance)
    - Correlation filtering (remove highly correlated with inputs)
    - Optional: Mutual information ranking
    
    Parameters
    ----------
    interaction_degree : int, default=2
        Maximum degree for polynomial features (1=linear, 2=quadratic, 3=cubic)
    include_ratios : bool, default=True
        Whether to include ratio features
    include_synergy : bool, default=True
        Whether to compute synergy scores
    max_features : int, default=200
        Maximum number of interaction features to generate
    variance_threshold : float, default=0.01
        Minimum variance for keeping features
    correlation_threshold : float, default=0.95
        Maximum correlation with input features
    
    Examples
    --------
    >>> transformer = MetaFeatureInteractionTransformer()
    >>> 
    >>> # Input: Genome features from other transformers
    >>> X = np.array([[0.5, 0.3, 0.8], [0.6, 0.4, 0.7], ...])
    >>> features = transformer.fit_transform(X)
    >>> 
    >>> # Or with feature names for interpretability
    >>> X = [
    ...     {
    ...         'genome_features': np.array([0.5, 0.3, 0.8]),
    ...         'feature_names': ['nom_richness', 'arc_quality', 'momentum']
    ...     },
    ...     ...
    ... ]
    >>> features = transformer.fit_transform(X)
    """
    
    def __init__(
        self,
        interaction_degree: int = 2,
        include_ratios: bool = True,
        include_synergy: bool = True,
        max_features: int = 200,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95
    ):
        super().__init__(
            narrative_id='meta_feature_interaction',
            description='Automatic interaction feature generation'
        )
        self.interaction_degree = interaction_degree
        self.include_ratios = include_ratios
        self.include_synergy = include_synergy
        self.max_features = max_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        
        # Will be populated during fit
        self.poly_features_ = None
        self.selected_interactions_ = None
        self.feature_names_in_ = None
        self.scaler_ = None
        self.base_feature_indices_ = None
        
    def fit(self, X, y=None):
        """
        Fit transformer to data.
        
        Learns which interaction features to generate based on variance
        and correlation with base features.
        
        Parameters
        ----------
        X : array-like or list of dict
            Training data (feature matrix or list with genome_features)
        y : array-like, optional
            Target values (used for interaction selection if provided)
            
        Returns
        -------
        self
        """
        self._validate_input(X)
        
        # Extract features and names
        X_matrix, feature_names = self._extract_features_and_names(X)
        self.feature_names_in_ = feature_names
        
        # Fit scaler for base features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_matrix)
        
        # Generate polynomial features
        self.poly_features_ = PolynomialFeatures(
            degree=self.interaction_degree,
            include_bias=False,
            interaction_only=False
        )
        X_poly = self.poly_features_.fit_transform(X_scaled)
        
        # Identify which columns are base features vs interactions
        n_base = X_matrix.shape[1]
        self.base_feature_indices_ = list(range(n_base))
        
        # Generate additional interactions
        all_interactions = [X_poly]
        
        if self.include_ratios:
            ratios = self._generate_ratio_features(X_scaled)
            if ratios.shape[1] > 0:
                all_interactions.append(ratios)
        
        if self.include_synergy:
            synergy = self._generate_synergy_features(X_scaled, y)
            if synergy.shape[1] > 0:
                all_interactions.append(synergy)
        
        # Concatenate all interactions
        X_all_interactions = np.hstack(all_interactions)
        
        # Select interactions based on variance and correlation
        selected_indices = self._select_interactions(
            X_scaled,
            X_all_interactions,
            y
        )
        
        self.selected_interactions_ = selected_indices
        
        # Store metadata
        self.metadata['n_samples'] = len(X_matrix)
        self.metadata['n_base_features'] = n_base
        self.metadata['n_interaction_features'] = len(selected_indices)
        self.metadata['n_features'] = len(selected_indices)
        self.metadata['feature_names'] = self._generate_interaction_names(selected_indices)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform data to interaction features.
        
        Parameters
        ----------
        X : array-like or list of dict
            Data to transform
            
        Returns
        -------
        features : ndarray, shape (n_samples, n_interaction_features)
            Interaction features
        """
        self._validate_fitted()
        self._validate_input(X)
        
        # Extract features
        X_matrix, _ = self._extract_features_and_names(X)
        
        # Scale
        X_scaled = self.scaler_.transform(X_matrix)
        
        # Generate polynomial features
        X_poly = self.poly_features_.transform(X_scaled)
        
        # Generate additional interactions
        all_interactions = [X_poly]
        
        if self.include_ratios:
            ratios = self._generate_ratio_features(X_scaled)
            if ratios.shape[1] > 0:
                all_interactions.append(ratios)
        
        if self.include_synergy:
            synergy = self._generate_synergy_features(X_scaled, y=None)
            if synergy.shape[1] > 0:
                all_interactions.append(synergy)
        
        # Concatenate and select
        X_all_interactions = np.hstack(all_interactions)
        
        return X_all_interactions[:, self.selected_interactions_]
    
    def _extract_features_and_names(
        self,
        X: Union[np.ndarray, List[Dict]]
    ) -> tuple:
        """Extract feature matrix and feature names."""
        if isinstance(X, np.ndarray):
            # Direct feature matrix
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            return X, feature_names
        
        elif isinstance(X, list) and len(X) > 0:
            if isinstance(X[0], dict):
                # List of dicts with genome_features
                features = []
                feature_names = None
                
                for item in X:
                    if 'genome_features' in item:
                        feat = np.array(item['genome_features'])
                        features.append(feat)
                        
                        if feature_names is None and 'feature_names' in item:
                            feature_names = item['feature_names']
                    else:
                        raise ValueError("Dict must contain 'genome_features'")
                
                X_matrix = np.array(features)
                
                if feature_names is None:
                    feature_names = [f'feature_{i}' for i in range(X_matrix.shape[1])]
                
                return X_matrix, feature_names
            else:
                # List of arrays
                X_matrix = np.array(X)
                feature_names = [f'feature_{i}' for i in range(X_matrix.shape[1])]
                return X_matrix, feature_names
        
        else:
            raise ValueError("Input must be numpy array or list of dicts/arrays")
    
    def _generate_ratio_features(self, X: np.ndarray) -> np.ndarray:
        """Generate ratio features (A/B for all pairs)."""
        n_features = X.shape[1]
        ratios = []
        
        for i, j in combinations(range(n_features), 2):
            # A / B
            ratio_ab = X[:, i] / (X[:, j] + 1e-10)
            ratios.append(ratio_ab)
            
            # B / A
            ratio_ba = X[:, j] / (X[:, i] + 1e-10)
            ratios.append(ratio_ba)
        
        if ratios:
            return np.column_stack(ratios)
        else:
            return np.array([]).reshape(X.shape[0], 0)
    
    def _generate_synergy_features(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate synergy/antagonism features."""
        n_features = X.shape[1]
        synergies = []
        
        for i, j in combinations(range(n_features), 2):
            A = X[:, i]
            B = X[:, j]
            
            # Synergy score (both high is better than sum)
            synergy = A * B - (A + B)
            synergies.append(synergy)
            
            # Antagonism score (one cancels the other)
            antagonism = np.abs(A) * np.abs(B) * np.sign(A * B) * -1
            synergies.append(antagonism)
            
            # Complementarity (high A, low B or vice versa)
            complementarity = np.abs(A - B) * (A + B)
            synergies.append(complementarity)
        
        if synergies:
            return np.column_stack(synergies)
        else:
            return np.array([]).reshape(X.shape[0], 0)
    
    def _select_interactions(
        self,
        X_base: np.ndarray,
        X_interactions: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Select interaction features based on variance and correlation.
        
        Parameters
        ----------
        X_base : ndarray
            Base features
        X_interactions : ndarray
            All generated interaction features
        y : ndarray, optional
            Target values for ranking
            
        Returns
        -------
        selected_indices : list of int
            Indices of selected features
        """
        n_total = X_interactions.shape[1]
        selected = []
        
        # 1. Variance filtering
        variances = np.var(X_interactions, axis=0)
        high_variance = variances > self.variance_threshold
        
        # 2. Correlation filtering (avoid features too correlated with base)
        low_correlation_with_base = np.ones(n_total, dtype=bool)
        
        for i in range(n_total):
            if not high_variance[i]:
                continue
            
            # Check correlation with all base features
            max_corr = 0.0
            for j in range(X_base.shape[1]):
                if np.std(X_interactions[:, i]) > 0 and np.std(X_base[:, j]) > 0:
                    corr = abs(np.corrcoef(X_interactions[:, i], X_base[:, j])[0, 1])
                    if not np.isnan(corr):
                        max_corr = max(max_corr, corr)
            
            if max_corr > self.correlation_threshold:
                low_correlation_with_base[i] = False
        
        # Combine filters
        keep_mask = high_variance & low_correlation_with_base
        candidate_indices = np.where(keep_mask)[0]
        
        # 3. Rank by correlation with outcome (if available)
        if y is not None and len(candidate_indices) > 0:
            correlations = []
            for idx in candidate_indices:
                if np.std(X_interactions[:, idx]) > 0 and np.std(y) > 0:
                    corr = abs(np.corrcoef(X_interactions[:, idx], y)[0, 1])
                    correlations.append(corr if not np.isnan(corr) else 0.0)
                else:
                    correlations.append(0.0)
            
            # Sort by correlation and take top max_features
            sorted_idx = np.argsort(correlations)[::-1]
            n_select = min(self.max_features, len(sorted_idx))
            selected = [candidate_indices[i] for i in sorted_idx[:n_select]]
        else:
            # No y provided - just take first max_features that pass filters
            selected = list(candidate_indices[:self.max_features])
        
        return selected
    
    def _generate_interaction_names(self, selected_indices: List[int]) -> List[str]:
        """Generate names for selected interaction features."""
        if self.feature_names_in_ is None:
            return [f'interaction_{i}' for i in range(len(selected_indices))]
        
        names = []
        poly_feature_names = self.poly_features_.get_feature_names_out(self.feature_names_in_)
        
        # Account for poly features, ratios, and synergies
        n_poly = len(poly_feature_names)
        n_base = len(self.feature_names_in_)
        
        # Number of ratio features
        if self.include_ratios:
            n_ratios = n_base * (n_base - 1)  # A/B and B/A for all pairs
        else:
            n_ratios = 0
        
        for idx in selected_indices:
            if idx < n_poly:
                # Polynomial feature
                names.append(poly_feature_names[idx])
            elif self.include_ratios and idx < n_poly + n_ratios:
                # Ratio feature
                ratio_idx = idx - n_poly
                pair_idx = ratio_idx // 2
                is_ab = ratio_idx % 2 == 0
                
                # Find which pair
                pairs = list(combinations(range(n_base), 2))
                if pair_idx < len(pairs):
                    i, j = pairs[pair_idx]
                    if is_ab:
                        names.append(f'{self.feature_names_in_[i]}_div_{self.feature_names_in_[j]}')
                    else:
                        names.append(f'{self.feature_names_in_[j]}_div_{self.feature_names_in_[i]}')
                else:
                    names.append(f'ratio_{ratio_idx}')
            else:
                # Synergy feature
                synergy_idx = idx - n_poly - n_ratios
                synergy_type = synergy_idx % 3
                pair_idx = synergy_idx // 3
                
                pairs = list(combinations(range(n_base), 2))
                if pair_idx < len(pairs):
                    i, j = pairs[pair_idx]
                    if synergy_type == 0:
                        names.append(f'{self.feature_names_in_[i]}_synergy_{self.feature_names_in_[j]}')
                    elif synergy_type == 1:
                        names.append(f'{self.feature_names_in_[i]}_antagonism_{self.feature_names_in_[j]}')
                    else:
                        names.append(f'{self.feature_names_in_[i]}_complementarity_{self.feature_names_in_[j]}')
                else:
                    names.append(f'synergy_{synergy_idx}')
        
        return names
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names."""
        if 'feature_names' in self.metadata:
            return self.metadata['feature_names']
        else:
            n_features = self.metadata.get('n_features', 0)
            return [f'interaction_{i}' for i in range(n_features)]
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of discovered patterns."""
        if not self.is_fitted_:
            return "Transformer not fitted yet."
        
        interpretation = f"""
Meta-Feature Interaction Analysis

Generated interaction features WITHOUT assuming which combinations matter.

Base Features: {self.metadata.get('n_base_features', 0)}
Generated Interactions: {self.metadata.get('n_interaction_features', 0)}
Samples Analyzed: {self.metadata.get('n_samples', 0)}

Interaction Types Generated:
1. Polynomial ({self.interaction_degree} degree): Feature powers and products
2. Ratios: A/B for all feature pairs{' (included)' if self.include_ratios else ''}
3. Synergy/Antagonism: Combined effects{' (included)' if self.include_synergy else ''}

Selection Criteria:
- Variance threshold: {self.variance_threshold}
- Correlation threshold: {self.correlation_threshold}
- Max features: {self.max_features}

These interactions enable DISCOVERY of which feature combinations
predict outcomes. The learning system tests all candidates and
identifies which matter without domain-specific assumptions.

Example: If "nominative_richness × narrative_velocity" emerges as
predictive, that's a discovered pattern, not an assumed one.
"""
        return interpretation.strip()

