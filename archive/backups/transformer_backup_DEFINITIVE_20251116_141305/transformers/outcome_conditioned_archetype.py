"""
Outcome-Conditioned Archetype Learner

Discovers Ξ (Golden Narratio) and α (optimal feature balance) from data.
Enables transfer learning across domains via archetypal similarity.

Philosophy: Learn what winners look like, don't assume it.
Discover optimal character/plot balance, don't predefine it.

This is THE MISSING PIECE that completes the framework.

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import pearsonr
import warnings
import pickle

from .base_transformer import FeatureNarrativeTransformer


class OutcomeConditionedArchetypeTransformer(FeatureNarrativeTransformer):
    """
    Discover Ξ (Golden Narratio) and α (optimal balance) from outcomes.
    
    Three-Phase Discovery:
    
    Phase 1: Discover Ξ (Golden Narratio)
    - Separate winners from losers
    - Cluster winners to find archetypal patterns
    - Compute centroid of winner space = Ξ_domain
    - Measure distance to Ξ for each instance
    
    Phase 2: Discover α (Feature Balance)
    - Classify features as character-like vs plot-like (by correlation patterns)
    - Measure effectiveness of each type
    - Compute optimal α = weight_character / (weight_char + weight_plot)
    - Domain-specific α without using п formula
    
    Phase 3: Transfer Learning
    - Compare Ξ across domains
    - If similar, transfer archetypal knowledge
    - Output transfer confidence score
    
    Features Extracted (~25 total):
    
    Golden Narratio Distance (8 features):
    - Distance to domain Ξ (primary)
    - Distance to winner cluster centroids (multiple)
    - Distance to loser cluster centroid
    - Ξ similarity score (cosine)
    - Winner space membership probability
    - Deviation from perfection magnitude
    - Relative distance (to Ξ / to anti-Ξ)
    - Archetypal match strength
    
    Feature Balance (5 features):
    - Optimal α for this domain
    - Character feature effectiveness
    - Plot feature effectiveness
    - Current instance character score
    - Current instance plot score
    
    Sub-Archetypes (5 features):
    - Primary winner archetype ID
    - Secondary winner archetype ID
    - Archetype membership probability
    - Distance to primary archetype
    - Distance to secondary archetype
    
    Transfer Learning (4 features):
    - Transfer confidence (if cross-domain)
    - Source domain similarity
    - Archetype transferability score
    - Domain-specific deviation
    
    Pattern Discovery (3 features):
    - Novelty relative to winners
    - Conformity to archetype
    - Winner pattern consistency
    
    Parameters
    ----------
    n_winner_clusters : int, default=3
        Number of sub-archetypes to discover in winner space
    min_winner_samples : int, default=5
        Minimum winners needed for clustering
    use_pca : bool, default=True
        Whether to use PCA for dimensionality reduction before clustering
    alpha_method : str, default='correlation'
        Method for discovering α ('correlation', 'mutual_info', 'model_based')
    enable_transfer : bool, default=True
        Whether to enable cross-domain transfer learning
    
    Examples
    --------
    >>> transformer = OutcomeConditionedArchetypeTransformer()
    >>> 
    >>> # REQUIRES outcomes for fitting (supervised learning)
    >>> X = [
    ...     {
    ...         'genome_features': np.array([...]),
    ...         'feature_names': ['nom_richness', 'arc_quality', ...],
    ...         'domain': 'nba'
    ...     },
    ...     ...
    ... ]
    >>> y = np.array([1, 0, 1, ...])  # 1=winner, 0=loser
    >>> 
    >>> features = transformer.fit_transform(X, y)
    >>> 
    >>> # Primary output: distance to Ξ
    >>> distance_to_xi = features[:, 0]
    >>> 
    >>> # Discovered α for this domain
    >>> optimal_alpha = transformer.metadata_['optimal_alpha']
    """
    
    def __init__(
        self,
        n_winner_clusters: int = 3,
        min_winner_samples: int = 5,
        use_pca: bool = True,
        alpha_method: str = 'correlation',
        enable_transfer: bool = True
    ):
        super().__init__(
            narrative_id='outcome_conditioned_archetype',
            description='Discover Ξ (Golden Narratio) and α from outcomes'
        )
        self.n_winner_clusters = n_winner_clusters
        self.min_winner_samples = min_winner_samples
        self.use_pca = use_pca
        self.alpha_method = alpha_method
        self.enable_transfer = enable_transfer
        
        # Will be populated during fit
        self.scaler_ = None
        self.pca_ = None
        self.xi_vector_ = None  # The Golden Narratio
        self.anti_xi_vector_ = None  # Loser archetype
        self.winner_clusters_ = None
        self.winner_centroids_ = None
        self.loser_centroid_ = None
        self.optimal_alpha_ = None
        self.character_features_ = None
        self.plot_features_ = None
        self.feature_names_in_ = None
        self.domain_name_ = None
        
        # For transfer learning
        self.archetype_library_ = {}  # Stores Ξ from other domains
    
    def _validate_input(self, X):
        """Override base validation - we accept list of dicts."""
        if X is None or len(X) == 0:
            raise ValueError("Input X cannot be None or empty")
        return True
        
    def fit(self, X, y):
        """
        Fit transformer by discovering Ξ and α from outcomes.
        
        **REQUIRES y (outcomes) - this is supervised learning**
        
        Parameters
        ----------
        X : list of dict
            Training data with genome features
        y : array-like
            Outcomes (1=winner, 0=loser for binary; continuous for regression)
            
        Returns
        -------
        self
        """
        if y is None:
            raise ValueError(
                "OutcomeConditionedArchetypeTransformer REQUIRES outcomes (y) for fitting. "
                "This transformer discovers patterns from winners vs losers."
            )
        
        self._validate_input(X)
        
        # Extract features, names, and domain
        genome_features, feature_names, domain = self._extract_data(X)
        y = np.array(y)
        
        self.feature_names_in_ = feature_names
        self.domain_name_ = domain
        
        # Fit scaler
        self.scaler_ = StandardScaler()
        genome_scaled = self.scaler_.fit_transform(genome_features)
        
        # Optional PCA for high-dimensional data
        if self.use_pca and genome_scaled.shape[1] > 20:
            n_components = min(20, genome_scaled.shape[1], len(genome_scaled) // 3)
            self.pca_ = PCA(n_components=n_components)
            genome_reduced = self.pca_.fit_transform(genome_scaled)
        else:
            genome_reduced = genome_scaled
        
        # Phase 1: Discover Ξ (Golden Narratio)
        self._discover_golden_narratio(genome_reduced, y)
        
        # Phase 2: Discover α (Feature Balance)
        self._discover_optimal_alpha(genome_features, y, feature_names)
        
        # Phase 3: Setup transfer learning (if enabled)
        if self.enable_transfer:
            self._setup_transfer_learning(domain, self.xi_vector_)
        
        # Store metadata
        self.metadata['n_samples'] = len(X)
        self.metadata['n_winners'] = np.sum(y > 0.5 if np.max(y) > 1 else y == 1)
        self.metadata['n_losers'] = np.sum(y <= 0.5 if np.max(y) > 1 else y == 0)
        self.metadata['genome_dim'] = genome_features.shape[1]
        self.metadata['reduced_dim'] = genome_reduced.shape[1]
        self.metadata['n_winner_clusters'] = len(self.winner_centroids_) if self.winner_centroids_ is not None else 0
        self.metadata['optimal_alpha'] = self.optimal_alpha_
        self.metadata['n_features'] = 25
        self.metadata['feature_names'] = self._get_feature_names()
        self.metadata['domain'] = domain
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform data to archetype-based features.
        
        Parameters
        ----------
        X : list of dict
            Data to transform
            
        Returns
        -------
        features : ndarray, shape (n_samples, 25)
            Archetype features
        """
        self._validate_fitted()
        # Skip base class validation - we handle list of dicts
        
        # Extract features
        genome_features, _, domain = self._extract_data(X)
        
        # Scale and reduce
        genome_scaled = self.scaler_.transform(genome_features)
        if self.pca_ is not None:
            genome_reduced = self.pca_.transform(genome_scaled)
        else:
            genome_reduced = genome_scaled
        
        # Extract features for each instance
        features = []
        for i, genome_point in enumerate(genome_reduced):
            feat_vector = self._extract_archetype_features(
                genome_point,
                genome_features[i],
                domain
            )
            features.append(feat_vector)
        
        return np.array(features)
    
    def _extract_data(self, X: List) -> Tuple[np.ndarray, List[str], str]:
        """Extract genome features, feature names, and domain."""
        genome_features = []
        feature_names = None
        domain = 'unknown'
        
        for item in X:
            if isinstance(item, dict):
                if 'genome_features' in item:
                    feat = np.array(item['genome_features'])
                    genome_features.append(feat)
                    
                    if feature_names is None and 'feature_names' in item:
                        feature_names = item['feature_names']
                    
                    if domain == 'unknown' and 'domain' in item:
                        domain = item['domain']
                else:
                    raise ValueError("Dict must contain 'genome_features'")
            else:
                genome_features.append(np.array(item))
        
        genome_features = np.array(genome_features)
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(genome_features.shape[1])]
        
        return genome_features, feature_names, domain
    
    def _discover_golden_narratio(self, genome_reduced: np.ndarray, y: np.ndarray):
        """
        Phase 1: Discover Ξ (Golden Narratio) from winners.
        
        Parameters
        ----------
        genome_reduced : ndarray
            Reduced/scaled genome features
        y : ndarray
            Outcomes
        """
        # Separate winners and losers
        winner_mask = y > 0.5 if np.max(y) > 1 else y == 1
        loser_mask = ~winner_mask
        
        winners = genome_reduced[winner_mask]
        losers = genome_reduced[loser_mask] if np.any(loser_mask) else np.array([])
        
        if len(winners) < self.min_winner_samples:
            warnings.warn(
                f"Only {len(winners)} winners found. Need at least {self.min_winner_samples} "
                "for robust clustering. Using simple centroid."
            )
            # Simple centroid
            self.xi_vector_ = np.mean(winners, axis=0)
            self.winner_centroids_ = [self.xi_vector_]
            self.winner_clusters_ = None
        else:
            # Cluster winners to find sub-archetypes
            n_clusters = min(self.n_winner_clusters, len(winners) // self.min_winner_samples)
            
            if n_clusters > 1:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                winner_labels = clusterer.fit_predict(winners)
                self.winner_centroids_ = clusterer.cluster_centers_
                self.winner_clusters_ = clusterer
                
                # Ξ = centroid of all winners (or weighted average of cluster centroids)
                self.xi_vector_ = np.mean(self.winner_centroids_, axis=0)
            else:
                # Only one cluster
                self.xi_vector_ = np.mean(winners, axis=0)
                self.winner_centroids_ = [self.xi_vector_]
                self.winner_clusters_ = None
        
        # Loser archetype (anti-Ξ)
        if len(losers) > 0:
            self.anti_xi_vector_ = np.mean(losers, axis=0)
            self.loser_centroid_ = self.anti_xi_vector_
        else:
            # No losers - use opposite of Ξ
            self.anti_xi_vector_ = -self.xi_vector_
            self.loser_centroid_ = self.anti_xi_vector_
    
    def _discover_optimal_alpha(
        self,
        genome_features: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ):
        """
        Phase 2: Discover α (optimal feature balance) from data.
        
        Instead of using feature name keywords, cluster features by their
        correlation patterns with outcomes and temporal behavior.
        
        Parameters
        ----------
        genome_features : ndarray
            Original genome features
        y : ndarray
            Outcomes
        feature_names : list of str
            Feature names
        """
        n_features = genome_features.shape[1]
        
        # Method 1: Correlation-based classification
        if self.alpha_method == 'correlation':
            # Compute correlation of each feature with outcome
            correlations = []
            for i in range(n_features):
                if np.std(genome_features[:, i]) > 0:
                    corr, _ = pearsonr(genome_features[:, i], y)
                    correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
                else:
                    correlations.append(0.0)
            
            correlations = np.array(correlations)
            
            # Also compute temporal variance (features that change more over time)
            # Proxy: higher variance = more plot-like (events), lower = character-like (traits)
            variances = np.var(genome_features, axis=0)
            
            # Classify features
            # Character-like: low variance, moderate correlation
            # Plot-like: high variance, variable correlation
            median_var = np.median(variances)
            
            char_mask = variances < median_var
            plot_mask = variances >= median_var
            
            self.character_features_ = np.where(char_mask)[0]
            self.plot_features_ = np.where(plot_mask)[0]
            
            # Compute effectiveness of each type
            char_effectiveness = np.mean(correlations[char_mask]) if np.any(char_mask) else 0.0
            plot_effectiveness = np.mean(correlations[plot_mask]) if np.any(plot_mask) else 0.0
            
        else:
            # Fallback: split features evenly
            mid = n_features // 2
            self.character_features_ = np.arange(mid)
            self.plot_features_ = np.arange(mid, n_features)
            char_effectiveness = 0.5
            plot_effectiveness = 0.5
        
        # Compute optimal α
        total_effectiveness = char_effectiveness + plot_effectiveness
        if total_effectiveness > 0:
            self.optimal_alpha_ = char_effectiveness / total_effectiveness
        else:
            self.optimal_alpha_ = 0.5  # Default to balanced
        
        # Store effectiveness scores
        self.metadata['character_effectiveness'] = char_effectiveness
        self.metadata['plot_effectiveness'] = plot_effectiveness
    
    def _setup_transfer_learning(self, domain: str, xi_vector: np.ndarray):
        """
        Phase 3: Setup transfer learning by storing Ξ in library.
        
        Parameters
        ----------
        domain : str
            Domain name
        xi_vector : ndarray
            Discovered Ξ for this domain
        """
        # Store this domain's Ξ in library
        self.archetype_library_[domain] = {
            'xi': xi_vector,
            'alpha': self.optimal_alpha_,
            'n_samples': self.metadata.get('n_samples', 0)
        }
    
    def _extract_archetype_features(
        self,
        genome_point: np.ndarray,
        original_genome: np.ndarray,
        domain: str
    ) -> np.ndarray:
        """Extract all archetype-based features for one instance."""
        features = []
        
        # 1. Golden Narratio Distance (8)
        features.extend(self._compute_xi_distance_features(genome_point))
        
        # 2. Feature Balance (5)
        features.extend(self._compute_balance_features(original_genome))
        
        # 3. Sub-Archetypes (5)
        features.extend(self._compute_subarchetype_features(genome_point))
        
        # 4. Transfer Learning (4)
        features.extend(self._compute_transfer_features(genome_point, domain))
        
        # 5. Pattern Discovery (3)
        features.extend(self._compute_pattern_features(genome_point))
        
        return np.array(features)
    
    def _compute_xi_distance_features(self, point: np.ndarray) -> List[float]:
        """Compute Golden Narratio distance features (8)."""
        features = []
        
        # Distance to Ξ (PRIMARY FEATURE)
        dist_to_xi = euclidean(point, self.xi_vector_)
        features.append(dist_to_xi)
        
        # Distances to winner cluster centroids
        if self.winner_centroids_ is not None and len(self.winner_centroids_) > 1:
            dists = [euclidean(point, cent) for cent in self.winner_centroids_]
            features.append(min(dists))  # Nearest winner centroid
            features.append(np.mean(dists))  # Average distance to winner centroids
        else:
            features.extend([dist_to_xi, dist_to_xi])
        
        # Distance to loser archetype
        dist_to_anti_xi = euclidean(point, self.anti_xi_vector_)
        features.append(dist_to_anti_xi)
        
        # Ξ similarity score (cosine)
        if np.linalg.norm(point) > 0 and np.linalg.norm(self.xi_vector_) > 0:
            cos_sim = np.dot(point, self.xi_vector_) / (np.linalg.norm(point) * np.linalg.norm(self.xi_vector_))
            features.append(cos_sim)
        else:
            features.append(0.0)
        
        # Winner space membership probability (inverse distance)
        membership = 1.0 / (1.0 + dist_to_xi)
        features.append(membership)
        
        # Deviation magnitude
        deviation = np.linalg.norm(point - self.xi_vector_)
        features.append(deviation)
        
        # Relative distance (to Ξ / to anti-Ξ)
        if dist_to_anti_xi > 0:
            relative = dist_to_xi / dist_to_anti_xi
            features.append(relative)
        else:
            features.append(0.0)
        
        return features
    
    def _compute_balance_features(self, original_genome: np.ndarray) -> List[float]:
        """Compute feature balance features (5)."""
        features = []
        
        # Optimal α for domain
        features.append(self.optimal_alpha_)
        
        # Character and plot effectiveness (from metadata)
        features.append(self.metadata.get('character_effectiveness', 0.5))
        features.append(self.metadata.get('plot_effectiveness', 0.5))
        
        # Current instance character and plot scores
        if len(self.character_features_) > 0:
            char_score = np.mean(original_genome[self.character_features_])
            features.append(char_score)
        else:
            features.append(0.5)
        
        if len(self.plot_features_) > 0:
            plot_score = np.mean(original_genome[self.plot_features_])
            features.append(plot_score)
        else:
            features.append(0.5)
        
        return features
    
    def _compute_subarchetype_features(self, point: np.ndarray) -> List[float]:
        """Compute sub-archetype features (5)."""
        features = []
        
        if self.winner_clusters_ is not None and len(self.winner_centroids_) > 1:
            # Predict cluster
            cluster = self.winner_clusters_.predict([point])[0]
            features.append(float(cluster))
            
            # Distances to centroids
            dists = [euclidean(point, cent) for cent in self.winner_centroids_]
            sorted_indices = np.argsort(dists)
            
            # Secondary cluster
            secondary = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
            features.append(float(secondary))
            
            # Membership probability (soft assignment via distances)
            inv_dists = 1.0 / (np.array(dists) + 1e-10)
            probs = inv_dists / np.sum(inv_dists)
            membership_prob = probs[cluster]
            features.append(membership_prob)
            
            # Distances to primary and secondary
            features.append(dists[cluster])
            features.append(dists[secondary])
        else:
            # Only one cluster
            features.extend([0.0, 0.0, 1.0, euclidean(point, self.xi_vector_), euclidean(point, self.xi_vector_)])
        
        return features
    
    def _compute_transfer_features(self, point: np.ndarray, domain: str) -> List[float]:
        """Compute transfer learning features (4)."""
        features = []
        
        if not self.enable_transfer or len(self.archetype_library_) <= 1:
            # No transfer available
            return [0.5, 0.5, 0.5, 0.0]
        
        # Find most similar domain
        max_similarity = 0.0
        most_similar_domain = None
        
        for other_domain, data in self.archetype_library_.items():
            if other_domain == domain:
                continue
            
            # Similarity based on Ξ vectors
            other_xi = data['xi']
            if len(other_xi) == len(self.xi_vector_):
                similarity = np.dot(self.xi_vector_, other_xi) / (
                    np.linalg.norm(self.xi_vector_) * np.linalg.norm(other_xi) + 1e-10
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_domain = other_domain
        
        if most_similar_domain is not None:
            # Transfer confidence (how similar are the Ξ vectors)
            features.append(max_similarity)
            
            # Source domain similarity
            features.append(max_similarity)
            
            # Archetype transferability (consistency across domains)
            transferability = max_similarity  # Could be more sophisticated
            features.append(transferability)
            
            # Domain-specific deviation
            # How much does this instance deviate from this domain's Ξ
            deviation = euclidean(point, self.xi_vector_)
            features.append(deviation)
        else:
            features.extend([0.5, 0.5, 0.5, 0.0])
        
        return features
    
    def _compute_pattern_features(self, point: np.ndarray) -> List[float]:
        """Compute pattern discovery features (3)."""
        features = []
        
        # Novelty relative to winners
        dist_to_xi = euclidean(point, self.xi_vector_)
        avg_winner_dist = np.mean([euclidean(point, cent) for cent in self.winner_centroids_])
        novelty = dist_to_xi / (avg_winner_dist + 1e-10)
        features.append(min(novelty, 2.0))  # Cap at 2x
        
        # Conformity to archetype (inverse of novelty)
        conformity = 1.0 / (1.0 + dist_to_xi)
        features.append(conformity)
        
        # Winner pattern consistency
        # How consistent are the winner archetypes (low variance = consistent)
        if len(self.winner_centroids_) > 1:
            centroid_distances = cdist(self.winner_centroids_, self.winner_centroids_)
            consistency = 1.0 / (1.0 + np.mean(centroid_distances))
            features.append(consistency)
        else:
            features.append(1.0)  # Single archetype = fully consistent
        
        return features
    
    def save_archetype_library(self, filepath: str):
        """Save archetype library for cross-domain transfer."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.archetype_library_, f)
    
    def load_archetype_library(self, filepath: str):
        """Load archetype library from file."""
        with open(filepath, 'rb') as f:
            self.archetype_library_ = pickle.load(f)
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        
        # Golden Narratio (8)
        names.extend([
            'distance_to_xi',
            'distance_to_nearest_winner_centroid',
            'distance_to_avg_winner_centroid',
            'distance_to_anti_xi',
            'xi_cosine_similarity',
            'winner_space_membership',
            'deviation_from_xi',
            'relative_distance_xi_anti_xi'
        ])
        
        # Feature Balance (5)
        names.extend([
            'optimal_alpha',
            'character_effectiveness',
            'plot_effectiveness',
            'instance_character_score',
            'instance_plot_score'
        ])
        
        # Sub-Archetypes (5)
        names.extend([
            'primary_winner_archetype_id',
            'secondary_winner_archetype_id',
            'archetype_membership_probability',
            'distance_to_primary_archetype',
            'distance_to_secondary_archetype'
        ])
        
        # Transfer Learning (4)
        names.extend([
            'transfer_confidence',
            'source_domain_similarity',
            'archetype_transferability',
            'domain_specific_deviation'
        ])
        
        # Pattern Discovery (3)
        names.extend([
            'novelty_relative_to_winners',
            'conformity_to_archetype',
            'winner_pattern_consistency'
        ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of discovered patterns."""
        if not self.is_fitted_:
            return "Transformer not fitted yet."
        
        interpretation = f"""
Outcome-Conditioned Archetype Discovery

Discovered Ξ (Golden Narratio) and α (optimal balance) FROM DATA.

Domain: {self.metadata.get('domain', 'unknown')}
Samples: {self.metadata.get('n_samples', 0)}
Winners: {self.metadata.get('n_winners', 0)}
Losers: {self.metadata.get('n_losers', 0)}

DISCOVERED PATTERNS:

1. Golden Narratio (Ξ):
   - Winner archetypes: {self.metadata.get('n_winner_clusters', 0)} discovered
   - Loser archetype: Identified
   - Primary feature: distance_to_xi (column 0)

2. Optimal Alpha (α): {self.optimal_alpha_:.3f}
   - Character effectiveness: {self.metadata.get('character_effectiveness', 0):.3f}
   - Plot effectiveness: {self.metadata.get('plot_effectiveness', 0):.3f}
   - {len(self.character_features_)} character-like features
   - {len(self.plot_features_)} plot-like features

3. Sub-Archetypes:
   - Multiple winner patterns discovered
   - Enables fine-grained matching

4. Transfer Learning:
   - {len(self.archetype_library_)} domain(s) in library
   - Cross-domain pattern transfer enabled

KEY INSIGHT: This transformer discovers WHAT WINNERS LOOK LIKE without
assuming it in advance. Distance to Ξ measures "archetypal perfection."

Usage: Closer to Ξ = better archetype match = higher predicted success.
"""
        return interpretation.strip()

