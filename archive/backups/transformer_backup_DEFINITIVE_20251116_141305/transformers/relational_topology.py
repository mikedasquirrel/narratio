"""
Relational Topology Transformer

Captures structural relationships between entities in matchups/comparisons
WITHOUT assuming which relationships matter.

Philosophy: Extract TOPOLOGY of relationships, discover which configurations predict outcomes.

Universal across competitive domains: sports, business competition, elections, etc.

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.distance import euclidean, cosine, cityblock, mahalanobis
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

from .base_transformer import FeatureNarrativeTransformer


class RelationalTopologyTransformer(FeatureNarrativeTransformer):
    """
    Extract topological features of entity relationships.
    
    Analyzes the GEOMETRY of how two entities relate in feature space:
    - Distance metrics (how far apart)
    - Asymmetry measures (how unbalanced)
    - Complementarity patterns (opposing vs overlapping)
    - Dominance geometry (superset/subset relationships)
    - Interaction curvature (how features compose)
    
    Works across ALL competitive domains without semantic assumptions.
    
    Features Extracted (~35 total):
    
    Distance Metrics (7 features):
    - Euclidean distance
    - Cosine distance
    - Manhattan distance
    - Mahalanobis distance (if covariance available)
    - Relative distance (normalized by feature scales)
    - Feature-wise distance variance
    - Maximum single-feature distance
    
    Asymmetry Measures (8 features):
    - Overall asymmetry score
    - Dimension-wise asymmetry
    - Asymmetry in positive vs negative dimensions
    - Peak asymmetry (max difference dimension)
    - Asymmetry concentration (how many dims contribute)
    - Directional asymmetry (A>B vs B>A dims)
    - Asymmetry balance point
    - Asymmetry entropy
    
    Complementarity Patterns (7 features):
    - Opposition score (anti-correlation)
    - Overlap score (correlation)
    - Complementarity index (opposing strength in different dims)
    - Coverage (combined feature space coverage)
    - Redundancy (duplicate strengths)
    - Synergy potential (combined strength > individual)
    - Mutual exclusivity
    
    Dominance Geometry (6 features):
    - Dominance score (A strictly better than B)
    - Dominance ratio (% dimensions where A > B)
    - Dominance magnitude (how much better when better)
    - Dominance consistency (does dominance hold across related dims)
    - Pareto dominance (strict subset/superset)
    - Mixed dominance (trading off strengths)
    
    Interaction Topology (7 features):
    - Multiplicative synergy (A×B effects)
    - Additive synergy (A+B effects)
    - Antagonistic interaction (A cancels B)
    - Interaction curvature (nonlinear composition)
    - Coupling strength (how much do features co-vary)
    - Independence score (uncorrelated dimensions)
    - Emergence potential (whole > sum of parts)
    
    Parameters
    ----------
    normalize_features : bool, default=True
        Whether to normalize features before computing topology
    compute_pca_topology : bool, default=True
        Whether to compute topology in PCA space
    interaction_order : int, default=2
        Maximum order for interaction features (1=linear, 2=quadratic)
    
    Examples
    --------
    >>> transformer = RelationalTopologyTransformer()
    >>> 
    >>> # Matchup data format
    >>> X = [
    ...     {
    ...         'entity_a_features': np.array([0.7, 0.3, 0.9, ...]),
    ...         'entity_b_features': np.array([0.4, 0.8, 0.2, ...]),
    ...         'entity_a_text': "Fighter A narrative",
    ...         'entity_b_text': "Fighter B narrative"
    ...     },
    ...     ...
    ... ]
    >>> features = transformer.fit_transform(X)
    >>> 
    >>> # Features describe the RELATIONSHIP structure
    >>> asymmetry = features[:, 7]  # Overall asymmetry
    >>> dominance = features[:, 21]  # Dominance score
    """
    
    def __init__(
        self,
        normalize_features: bool = True,
        compute_pca_topology: bool = True,
        interaction_order: int = 2
    ):
        super().__init__(
            narrative_id='relational_topology',
            description='Structural topology of entity relationships'
        )
        self.normalize_features = normalize_features
        self.compute_pca_topology = compute_pca_topology
        self.interaction_order = interaction_order
        
        # Will be populated during fit
        self.scaler_ = None
        self.pca_ = None
        self.feature_dim_ = None
        self.covariance_ = None
    
    def _validate_input(self, X):
        """Override base validation - we accept list of dicts."""
        if X is None or len(X) == 0:
            raise ValueError("Input X cannot be None or empty")
        return True
        
    def fit(self, X, y=None):
        """
        Fit transformer to data.
        
        Learns normalization and PCA transformation for topology computation.
        
        Parameters
        ----------
        X : list of dict
            Training matchups with entity features
        y : array-like, optional
            Target values (not used, for sklearn compatibility)
            
        Returns
        -------
        self
        """
        self._validate_input(X)
        
        # Extract all entity features for fitting scaler/PCA
        all_features = []
        for item in X:
            a_feat, b_feat = self._extract_entity_features(item)
            all_features.append(a_feat)
            all_features.append(b_feat)
        
        all_features = np.array(all_features)
        
        # Store feature dimensionality
        self.feature_dim_ = all_features.shape[1] if len(all_features) > 0 else 0
        
        # Fit scaler
        if self.normalize_features and len(all_features) > 0:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(all_features)
        
        # Fit PCA for topological analysis in reduced space
        if self.compute_pca_topology and len(all_features) > 2:
            n_components = min(10, all_features.shape[1], len(all_features) // 2)
            if n_components > 0:
                self.pca_ = PCA(n_components=n_components)
                self.pca_.fit(all_features)
        
        # Compute covariance for Mahalanobis distance
        if len(all_features) > 1:
            try:
                self.covariance_ = np.cov(all_features.T)
            except:
                self.covariance_ = None
        
        # Store metadata
        self.metadata['n_samples'] = len(X)
        self.metadata['feature_dim'] = self.feature_dim_
        self.metadata['n_features'] = 35
        self.metadata['feature_names'] = self._get_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform matchups to relational topology features.
        
        Parameters
        ----------
        X : list of dict
            Matchups to transform
            
        Returns
        -------
        features : ndarray, shape (n_samples, 35)
            Relational topology features
        """
        self._validate_fitted()
        # Skip base class validation - we handle list of dicts
        
        features = []
        for item in X:
            # Extract entity features
            a_feat, b_feat = self._extract_entity_features(item)
            
            # Normalize if fitted
            if self.scaler_ is not None:
                a_feat = self.scaler_.transform([a_feat])[0]
                b_feat = self.scaler_.transform([b_feat])[0]
            
            # Extract relational topology
            feat_vector = self._extract_topology_features(a_feat, b_feat)
            
            features.append(feat_vector)
        
        return np.array(features)
    
    def _extract_entity_features(self, item: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature vectors for both entities.
        
        Parameters
        ----------
        item : dict
            Item with entity features
            
        Returns
        -------
        a_features, b_features : tuple of ndarrays
            Feature vectors for entities A and B
        """
        if not isinstance(item, dict):
            raise ValueError("Input must be dict with 'entity_a_features' and 'entity_b_features'")
        
        a_feat = item.get('entity_a_features')
        b_feat = item.get('entity_b_features')
        
        if a_feat is None or b_feat is None:
            raise ValueError("Dict must contain 'entity_a_features' and 'entity_b_features'")
        
        a_feat = np.array(a_feat)
        b_feat = np.array(b_feat)
        
        if len(a_feat) != len(b_feat):
            raise ValueError(f"Entity features must have same length (got {len(a_feat)} and {len(b_feat)})")
        
        return a_feat, b_feat
    
    def _extract_topology_features(self, a_feat: np.ndarray, b_feat: np.ndarray) -> np.ndarray:
        """
        Extract all topological features from entity pair.
        
        Parameters
        ----------
        a_feat, b_feat : ndarrays
            Feature vectors for entities A and B
            
        Returns
        -------
        features : ndarray, shape (35,)
            Topological features
        """
        features = []
        
        # 1. Distance Metrics (7)
        features.extend(self._compute_distance_metrics(a_feat, b_feat))
        
        # 2. Asymmetry Measures (8)
        features.extend(self._compute_asymmetry_metrics(a_feat, b_feat))
        
        # 3. Complementarity Patterns (7)
        features.extend(self._compute_complementarity_metrics(a_feat, b_feat))
        
        # 4. Dominance Geometry (6)
        features.extend(self._compute_dominance_metrics(a_feat, b_feat))
        
        # 5. Interaction Topology (7)
        features.extend(self._compute_interaction_metrics(a_feat, b_feat))
        
        return np.array(features)
    
    def _compute_distance_metrics(self, a: np.ndarray, b: np.ndarray) -> List[float]:
        """Compute distance metrics (7 features)."""
        features = []
        
        # Euclidean distance
        eucl_dist = euclidean(a, b)
        features.append(eucl_dist)
        
        # Cosine distance
        if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
            cos_dist = cosine(a, b)
            features.append(cos_dist)
        else:
            features.append(1.0)  # Maximum distance
        
        # Manhattan distance
        manh_dist = cityblock(a, b)
        features.append(manh_dist)
        
        # Mahalanobis distance (if covariance available)
        if self.covariance_ is not None and self.covariance_.shape[0] == len(a):
            try:
                # Check if covariance is invertible
                cov_inv = np.linalg.inv(self.covariance_ + np.eye(len(a)) * 1e-6)
                mahal_dist = mahalanobis(a, b, cov_inv)
                features.append(mahal_dist)
            except:
                features.append(eucl_dist)  # Fallback to Euclidean
        else:
            features.append(eucl_dist)
        
        # Relative distance (normalized by scales)
        scales = np.abs(a) + np.abs(b) + 1e-10
        relative_dist = np.mean(np.abs(a - b) / scales)
        features.append(relative_dist)
        
        # Feature-wise distance variance
        feature_dists = np.abs(a - b)
        dist_variance = np.var(feature_dists)
        features.append(dist_variance)
        
        # Maximum single-feature distance
        max_dist = np.max(feature_dists)
        features.append(max_dist)
        
        return features
    
    def _compute_asymmetry_metrics(self, a: np.ndarray, b: np.ndarray) -> List[float]:
        """Compute asymmetry metrics (8 features)."""
        features = []
        
        # Overall asymmetry score (mean absolute difference)
        asymmetry = np.mean(np.abs(a - b))
        features.append(asymmetry)
        
        # Dimension-wise asymmetry (std of differences)
        dim_asymmetry = np.std(a - b)
        features.append(dim_asymmetry)
        
        # Asymmetry in positive vs negative dimensions
        diff = a - b
        pos_asymmetry = np.mean(diff[diff > 0]) if np.any(diff > 0) else 0.0
        neg_asymmetry = np.mean(np.abs(diff[diff < 0])) if np.any(diff < 0) else 0.0
        features.append(pos_asymmetry)
        features.append(neg_asymmetry)
        
        # Peak asymmetry (maximum difference dimension)
        peak_asym = np.max(np.abs(diff))
        features.append(peak_asym)
        
        # Asymmetry concentration (how many dims contribute to asymmetry)
        total_asym = np.sum(np.abs(diff))
        if total_asym > 0:
            asym_concentration = 1.0 - entropy(np.abs(diff) / total_asym) / np.log(len(diff))
            features.append(asym_concentration)
        else:
            features.append(0.0)
        
        # Directional asymmetry (ratio of A>B to B>A)
        a_gt_b = np.sum(a > b)
        b_gt_a = np.sum(b > a)
        if a_gt_b + b_gt_a > 0:
            directional = abs(a_gt_b - b_gt_a) / (a_gt_b + b_gt_a)
            features.append(directional)
        else:
            features.append(0.0)
        
        # Asymmetry balance point (weighted average of where asymmetry occurs)
        weights = np.abs(diff)
        if np.sum(weights) > 0:
            balance = np.average(np.arange(len(diff)), weights=weights) / len(diff)
            features.append(balance)
        else:
            features.append(0.5)
        
        return features
    
    def _compute_complementarity_metrics(self, a: np.ndarray, b: np.ndarray) -> List[float]:
        """Compute complementarity metrics (7 features)."""
        features = []
        
        # Opposition score (negative correlation)
        if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0:
            corr = np.corrcoef(a, b)[0, 1]
            opposition = max(0, -corr)  # Only negative correlations
            overlap = max(0, corr)  # Only positive correlations
        else:
            opposition = 0.0
            overlap = 0.0
        
        features.append(opposition)
        features.append(overlap)
        
        # Complementarity index (strong in different dimensions)
        a_strong = a > np.median(a)
        b_strong = b > np.median(b)
        complementary = np.sum(np.logical_xor(a_strong, b_strong))
        complementarity_idx = complementary / len(a)
        features.append(complementarity_idx)
        
        # Coverage (combined feature space coverage)
        combined_max = np.maximum(a, b)
        coverage = np.mean(combined_max)
        features.append(coverage)
        
        # Redundancy (both strong in same dimensions)
        redundant = np.sum(np.logical_and(a_strong, b_strong))
        redundancy = redundant / len(a)
        features.append(redundancy)
        
        # Synergy potential (combined > individual)
        synergy = np.mean(combined_max > np.maximum(np.mean(a), np.mean(b)))
        features.append(synergy)
        
        # Mutual exclusivity (one strong where other weak)
        a_excl = np.sum(np.logical_and(a_strong, ~b_strong))
        b_excl = np.sum(np.logical_and(~a_strong, b_strong))
        mutual_excl = (a_excl + b_excl) / len(a)
        features.append(mutual_excl)
        
        return features
    
    def _compute_dominance_metrics(self, a: np.ndarray, b: np.ndarray) -> List[float]:
        """Compute dominance metrics (6 features)."""
        features = []
        
        # Dominance score (how many dimensions A > B)
        a_dominates = np.sum(a > b)
        dominance_ratio = a_dominates / len(a)
        features.append(dominance_ratio)
        
        # Net dominance (considering magnitude)
        net_dominance = np.sum(a - b) / len(a)
        features.append(net_dominance)
        
        # Dominance magnitude (when A is better, how much better?)
        a_better_mask = a > b
        if np.any(a_better_mask):
            dominance_mag = np.mean((a - b)[a_better_mask])
            features.append(dominance_mag)
        else:
            features.append(0.0)
        
        # Dominance consistency (does dominance hold in related dims?)
        if len(a) > 3:
            # Check if dominance clusters (consecutive dimensions)
            dom_changes = np.sum(np.abs(np.diff((a > b).astype(int))))
            consistency = 1.0 - (dom_changes / len(a))
            features.append(consistency)
        else:
            features.append(0.5)
        
        # Pareto dominance (A better in all or no dimensions)
        pareto = float(dominance_ratio == 1.0 or dominance_ratio == 0.0)
        features.append(pareto)
        
        # Mixed dominance (both have advantages)
        mixed = float(0.2 < dominance_ratio < 0.8)
        features.append(mixed)
        
        return features
    
    def _compute_interaction_metrics(self, a: np.ndarray, b: np.ndarray) -> List[float]:
        """Compute interaction topology metrics (7 features)."""
        features = []
        
        # Multiplicative synergy (A×B > A+B)
        product = a * b
        summed = a + b
        mult_synergy = np.mean(product > summed)
        features.append(mult_synergy)
        
        # Additive synergy magnitude
        add_synergy = np.mean(summed) / (np.mean(np.abs(a)) + np.mean(np.abs(b)) + 1e-10)
        features.append(add_synergy)
        
        # Antagonistic interaction (product less than expected)
        expected_product = np.abs(a) * np.abs(b)
        actual_product = np.abs(a * b)
        antagonism = np.mean(actual_product < expected_product * 0.5)
        features.append(antagonism)
        
        # Interaction curvature (nonlinearity)
        if len(a) > 2:
            linear_combo = 0.5 * a + 0.5 * b
            geometric_combo = np.sqrt(np.abs(a * b)) * np.sign(a * b)
            curvature = np.mean(np.abs(linear_combo - geometric_combo))
            features.append(curvature)
        else:
            features.append(0.0)
        
        # Coupling strength (feature covariation)
        if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0:
            coupling = abs(np.corrcoef(a, b)[0, 1])
            features.append(coupling)
        else:
            features.append(0.0)
        
        # Independence score (uncorrelated dimensions)
        # Count dimensions where changes are uncorrelated
        if len(a) > 5:
            a_changes = np.diff(a)
            b_changes = np.diff(b)
            if np.std(a_changes) > 0 and np.std(b_changes) > 0:
                independence = 1.0 - abs(np.corrcoef(a_changes, b_changes)[0, 1])
                features.append(independence)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Emergence potential (whole > sum of parts)
        # Compare combined variance to individual variances
        combined_var = np.var(a + b)
        individual_var = np.var(a) + np.var(b)
        if individual_var > 0:
            emergence = (combined_var - individual_var) / (individual_var + 1e-10)
            features.append(max(0, emergence))
        else:
            features.append(0.0)
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        
        # Distance metrics (7)
        names.extend([
            'distance_euclidean',
            'distance_cosine',
            'distance_manhattan',
            'distance_mahalanobis',
            'distance_relative',
            'distance_variance',
            'distance_max'
        ])
        
        # Asymmetry (8)
        names.extend([
            'asymmetry_overall',
            'asymmetry_dimensional',
            'asymmetry_positive',
            'asymmetry_negative',
            'asymmetry_peak',
            'asymmetry_concentration',
            'asymmetry_directional',
            'asymmetry_balance'
        ])
        
        # Complementarity (7)
        names.extend([
            'complementarity_opposition',
            'complementarity_overlap',
            'complementarity_index',
            'complementarity_coverage',
            'complementarity_redundancy',
            'complementarity_synergy',
            'complementarity_mutual_exclusivity'
        ])
        
        # Dominance (6)
        names.extend([
            'dominance_ratio',
            'dominance_net',
            'dominance_magnitude',
            'dominance_consistency',
            'dominance_pareto',
            'dominance_mixed'
        ])
        
        # Interaction (7)
        names.extend([
            'interaction_multiplicative',
            'interaction_additive',
            'interaction_antagonism',
            'interaction_curvature',
            'interaction_coupling',
            'interaction_independence',
            'interaction_emergence'
        ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of discovered patterns."""
        if not self.is_fitted_:
            return "Transformer not fitted yet."
        
        interpretation = f"""
Relational Topology Analysis

Extracted geometric/topological relationships between entities
WITHOUT assuming which relationships matter.

Features Extracted: {self.metadata['n_features']}
Matchups Analyzed: {self.metadata.get('n_samples', 'Unknown')}
Feature Dimensionality: {self.metadata.get('feature_dim', 'Unknown')}

Relationship Categories:
1. Distance Metrics (7): How far apart entities are in feature space
2. Asymmetry Measures (8): How unbalanced the matchup is
3. Complementarity (7): Opposing vs overlapping strengths
4. Dominance Geometry (6): Superiority patterns across dimensions
5. Interaction Topology (7): How features compose (synergy/antagonism)

These features enable DISCOVERY of which relationship geometries predict outcomes
across competitive domains without domain-specific assumptions.

Example discoveries:
- High asymmetry + high complementarity = competitive balance
- Low distance + high dominance = mismatch
- High interaction curvature = nonlinear dynamics matter
"""
        return interpretation.strip()

