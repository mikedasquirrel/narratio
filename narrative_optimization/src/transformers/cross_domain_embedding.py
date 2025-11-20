"""
Cross-Domain Embedding Transformer

Maps all narratives into universal embedding space where structurally similar
narratives cluster REGARDLESS OF DOMAIN.

Philosophy: Discover universal patterns through embedding space clustering.
Enable transfer learning across domains via structural isomorphism.

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import warnings

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("UMAP not available. Using PCA for dimensionality reduction.")

from .base_transformer import FeatureNarrativeTransformer


class CrossDomainEmbeddingTransformer(FeatureNarrativeTransformer):
    """
    Project narratives into universal embedding space.
    
    Discovers STRUCTURAL ISOMORPHISM across domains:
    - NFL Playoff game ≈ Tennis Grand Slam final (elimination pressure)
    - NBA Rivalry game ≈ Political debate (historic conflict)
    - Startup pitch ≈ Movie trailer (anticipation building)
    
    Works by:
    1. Taking ж features from other transformers
    2. Projecting into universal low-dimensional space
    3. Clustering to find archetypal patterns
    4. Computing distances to cluster centroids (proxy for Ξ)
    5. Enabling transfer learning via cluster similarity
    
    Features Extracted (~30 total):
    
    Cluster Membership (5 features):
    - Primary cluster ID
    - Cluster membership probability
    - Secondary cluster ID (if mixed)
    - Cluster coherence score
    - Distance to cluster boundary
    
    Embedding Coordinates (10 features):
    - Top 10 dimensions of embedded space
    - (Captures position in universal narrative space)
    
    Archetypal Distances (8 features):
    - Distance to nearest cluster centroid
    - Distance to farthest cluster centroid
    - Average distance to all centroids
    - Distance to winner archetype (if known)
    - Distance to loser archetype (if known)
    - Distance variance across clusters
    - Relative distance (to nearest / to farthest)
    - Isolation score
    
    Inter-Cluster Features (4 features):
    - Between-cluster distance (nearest two)
    - Cluster separation score
    - Cluster density at position
    - Cross-cluster contamination
    
    Domain Transfer Features (3 features):
    - Cross-domain similarity (if domain labels available)
    - Transfer confidence score
    - Domain-specific deviation
    
    Parameters
    ----------
    n_embedding_dims : int, default=10
        Number of dimensions in embedded space
    n_clusters : int, default=8
        Number of archetypal clusters to discover
    embedding_method : str, default='umap'
        Method for dimensionality reduction ('umap', 'pca', 'auto')
    clustering_method : str, default='kmeans'
        Method for clustering ('kmeans', 'gmm')
    track_domains : bool, default=True
        Whether to track domain labels for transfer learning
    
    Examples
    --------
    >>> transformer = CrossDomainEmbeddingTransformer()
    >>> 
    >>> # Input: genome features from other transformers
    >>> X = [
    ...     {
    ...         'genome_features': np.array([...]),  # ж from other transformers
    ...         'domain': 'nba',
    ...         'text': "Narrative text..."
    ...     },
    ...     ...
    ... ]
    >>> features = transformer.fit_transform(X)
    >>> 
    >>> # Discover: Are NBA games in same cluster as NFL games?
    >>> nba_clusters = features[nba_mask, 0]  # Cluster IDs
    >>> nfl_clusters = features[nfl_mask, 0]
    >>> 
    >>> # Enable transfer: Learn from similar cross-domain instances
    >>> transfer_confidence = features[:, -1]
    """
    
    def __init__(
        self,
        n_embedding_dims: int = 10,
        n_clusters: int = 8,
        embedding_method: str = 'auto',
        clustering_method: str = 'kmeans',
        track_domains: bool = True
    ):
        super().__init__(
            narrative_id='cross_domain_embedding',
            description='Universal narrative embedding space projection'
        )
        self.n_embedding_dims = n_embedding_dims
        self.n_clusters = n_clusters
        self.embedding_method = embedding_method
        self.clustering_method = clustering_method
        self.track_domains = track_domains
        
        # Will be populated during fit
        self.scaler_ = None
        self.embedder_ = None
        self.clusterer_ = None
        self.cluster_centroids_ = None
        self.domain_labels_ = None
        self.domain_cluster_map_ = None  # Which clusters dominant in which domains
    
    def _validate_input(self, X):
        """Override base validation - we accept list of dicts."""
        if X is None or len(X) == 0:
            raise ValueError("Input X cannot be None or empty")
        return True
        
    def fit(self, X, y=None):
        """
        Fit transformer to data.
        
        Learns:
        1. Feature normalization
        2. Embedding projection
        3. Archetypal clusters
        4. Domain-cluster relationships (for transfer learning)
        
        Parameters
        ----------
        X : list of dict
            Training data with genome features
        y : array-like, optional
            Outcomes (used to identify winner/loser archetypes)
            
        Returns
        -------
        self
        """
        self._validate_input(X)
        
        # Extract genome features and domain labels
        genome_features = []
        domain_labels = []
        
        for item in X:
            feat = self._extract_genome_features(item)
            genome_features.append(feat)
            
            if self.track_domains and isinstance(item, dict):
                domain = item.get('domain', 'unknown')
                domain_labels.append(domain)
            else:
                domain_labels.append('unknown')
        
        genome_features = np.array(genome_features)
        self.domain_labels_ = domain_labels if self.track_domains else None
        
        # 1. Fit scaler
        self.scaler_ = StandardScaler()
        genome_scaled = self.scaler_.fit_transform(genome_features)
        
        # 2. Fit embedder (dimensionality reduction)
        method = self.embedding_method
        if method == 'auto':
            method = 'umap' if HAS_UMAP else 'pca'
        
        if method == 'umap' and HAS_UMAP:
            n_dims = min(self.n_embedding_dims, genome_scaled.shape[1] - 1)
            n_neighbors = min(15, len(genome_scaled) // 2)
            self.embedder_ = umap.UMAP(
                n_components=n_dims,
                n_neighbors=n_neighbors,
                metric='cosine',
                random_state=42
            )
        else:
            # Fallback to PCA
            n_dims = min(self.n_embedding_dims, genome_scaled.shape[1], len(genome_scaled) // 2)
            self.embedder_ = PCA(n_components=n_dims, random_state=42)
        
        embedded = self.embedder_.fit_transform(genome_scaled)
        
        # 3. Fit clusterer (discover archetypes)
        n_clust = min(self.n_clusters, len(embedded) // 3, embedded.shape[1] * 2)
        
        if self.clustering_method == 'gmm':
            self.clusterer_ = GaussianMixture(
                n_components=n_clust,
                random_state=42,
                covariance_type='full'
            )
            self.clusterer_.fit(embedded)
            self.cluster_centroids_ = self.clusterer_.means_
        else:
            # KMeans
            self.clusterer_ = KMeans(
                n_clusters=n_clust,
                random_state=42,
                n_init=10
            )
            self.clusterer_.fit(embedded)
            self.cluster_centroids_ = self.clusterer_.cluster_centers_
        
        # 4. Learn domain-cluster mapping (for transfer learning)
        if self.track_domains and self.domain_labels_ is not None:
            self._learn_domain_cluster_mapping(embedded, self.domain_labels_)
        
        # 5. Identify winner/loser archetypes if outcomes provided
        if y is not None:
            self._identify_outcome_archetypes(embedded, y)
        
        # Store metadata
        self.metadata['n_samples'] = len(X)
        self.metadata['genome_dim'] = genome_features.shape[1]
        self.metadata['embedding_dim'] = embedded.shape[1]
        self.metadata['n_clusters'] = n_clust
        self.metadata['n_features'] = 30
        self.metadata['feature_names'] = self._get_feature_names()
        self.metadata['embedding_method'] = method
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform data to cross-domain embedding features.
        
        Parameters
        ----------
        X : list of dict
            Data to transform
            
        Returns
        -------
        features : ndarray, shape (n_samples, 30)
            Cross-domain embedding features
        """
        self._validate_fitted()
        # Skip base class validation - we handle list of dicts
        
        # Extract genome features
        genome_features = []
        domain_labels = []
        
        for item in X:
            feat = self._extract_genome_features(item)
            genome_features.append(feat)
            
            if self.track_domains and isinstance(item, dict):
                domain = item.get('domain', 'unknown')
                domain_labels.append(domain)
            else:
                domain_labels.append('unknown')
        
        genome_features = np.array(genome_features)
        
        # Scale and embed
        genome_scaled = self.scaler_.transform(genome_features)
        embedded = self.embedder_.transform(genome_scaled)
        
        # Extract features
        features = []
        for i, emb in enumerate(embedded):
            feat_vector = self._extract_embedding_features(
                emb,
                domain=domain_labels[i] if self.track_domains else None
            )
            features.append(feat_vector)
        
        return np.array(features)
    
    def _extract_genome_features(self, item) -> np.ndarray:
        """Extract genome features from item."""
        if isinstance(item, dict):
            if 'genome_features' in item:
                return np.array(item['genome_features'])
            else:
                raise ValueError("Dict must contain 'genome_features'")
        elif isinstance(item, np.ndarray):
            return item
        else:
            raise ValueError("Input must be dict with 'genome_features' or ndarray")
    
    def _learn_domain_cluster_mapping(self, embedded: np.ndarray, domain_labels: List[str]):
        """Learn which clusters are dominant in which domains."""
        # Predict clusters
        if self.clustering_method == 'gmm':
            clusters = self.clusterer_.predict(embedded)
        else:
            clusters = self.clusterer_.labels_
        
        # Build domain -> cluster distribution map
        self.domain_cluster_map_ = {}
        
        unique_domains = set(domain_labels)
        for domain in unique_domains:
            domain_mask = np.array([d == domain for d in domain_labels])
            domain_clusters = clusters[domain_mask]
            
            if len(domain_clusters) > 0:
                # Distribution of clusters for this domain
                cluster_counts = np.bincount(domain_clusters, minlength=self.n_clusters)
                cluster_probs = cluster_counts / len(domain_clusters)
                self.domain_cluster_map_[domain] = cluster_probs
    
    def _identify_outcome_archetypes(self, embedded: np.ndarray, y: np.ndarray):
        """Identify winner and loser archetypal clusters."""
        y = np.array(y)
        
        # Get cluster assignments
        if self.clustering_method == 'gmm':
            clusters = self.clusterer_.predict(embedded)
        else:
            clusters = self.clusterer_.labels_
        
        # For each cluster, compute win rate
        cluster_win_rates = {}
        for c in range(self.n_clusters):
            cluster_mask = clusters == c
            if np.sum(cluster_mask) > 0:
                win_rate = np.mean(y[cluster_mask])
                cluster_win_rates[c] = win_rate
        
        # Store in metadata
        self.metadata['cluster_win_rates'] = cluster_win_rates
        
        # Identify winner/loser archetype clusters
        if cluster_win_rates:
            winner_cluster = max(cluster_win_rates.items(), key=lambda x: x[1])[0]
            loser_cluster = min(cluster_win_rates.items(), key=lambda x: x[1])[0]
            
            self.metadata['winner_archetype_cluster'] = winner_cluster
            self.metadata['loser_archetype_cluster'] = loser_cluster
            self.metadata['winner_centroid'] = self.cluster_centroids_[winner_cluster]
            self.metadata['loser_centroid'] = self.cluster_centroids_[loser_cluster]
    
    def _extract_embedding_features(
        self,
        embedded_point: np.ndarray,
        domain: Optional[str] = None
    ) -> np.ndarray:
        """Extract all features from embedded point."""
        features = []
        
        # 1. Cluster Membership (5)
        features.extend(self._compute_cluster_membership(embedded_point))
        
        # 2. Embedding Coordinates (10) - pad if needed
        coords = list(embedded_point[:10])
        coords.extend([0.0] * (10 - len(coords)))  # Pad to 10
        features.extend(coords)
        
        # 3. Archetypal Distances (8)
        features.extend(self._compute_archetypal_distances(embedded_point))
        
        # 4. Inter-Cluster Features (4)
        features.extend(self._compute_intercluster_features(embedded_point))
        
        # 5. Domain Transfer Features (3)
        features.extend(self._compute_transfer_features(embedded_point, domain))
        
        return np.array(features)
    
    def _compute_cluster_membership(self, point: np.ndarray) -> List[float]:
        """Compute cluster membership features (5)."""
        features = []
        
        # Predict cluster
        point_reshaped = point.reshape(1, -1)
        
        if self.clustering_method == 'gmm':
            cluster = self.clusterer_.predict(point_reshaped)[0]
            probs = self.clusterer_.predict_proba(point_reshaped)[0]
            primary_prob = probs[cluster]
            
            # Secondary cluster (second highest probability)
            probs_sorted = np.argsort(probs)[::-1]
            secondary_cluster = probs_sorted[1] if len(probs_sorted) > 1 else cluster
            
        else:  # KMeans
            cluster = self.clusterer_.predict(point_reshaped)[0]
            
            # Compute soft probabilities based on distances
            distances = cdist(point_reshaped, self.cluster_centroids_)[0]
            # Convert distances to probabilities (inverse distance weighting)
            inv_distances = 1.0 / (distances + 1e-10)
            probs = inv_distances / np.sum(inv_distances)
            primary_prob = probs[cluster]
            
            probs_sorted = np.argsort(probs)[::-1]
            secondary_cluster = probs_sorted[1] if len(probs_sorted) > 1 else cluster
        
        features.append(float(cluster))
        features.append(primary_prob)
        features.append(float(secondary_cluster))
        
        # Cluster coherence (how well does it fit in primary cluster)
        coherence = primary_prob
        features.append(coherence)
        
        # Distance to cluster boundary (minimum distance to another cluster)
        dist_to_primary = np.linalg.norm(point - self.cluster_centroids_[cluster])
        distances_to_others = [
            np.linalg.norm(point - cent)
            for i, cent in enumerate(self.cluster_centroids_)
            if i != cluster
        ]
        if distances_to_others:
            min_dist_to_other = min(distances_to_others)
            boundary_dist = min_dist_to_other - dist_to_primary
            features.append(boundary_dist)
        else:
            features.append(dist_to_primary)
        
        return features
    
    def _compute_archetypal_distances(self, point: np.ndarray) -> List[float]:
        """Compute distances to archetypal centroids (8)."""
        features = []
        
        # Distances to all centroids
        distances = [np.linalg.norm(point - cent) for cent in self.cluster_centroids_]
        
        # Nearest and farthest
        nearest_dist = min(distances)
        farthest_dist = max(distances)
        avg_dist = np.mean(distances)
        
        features.append(nearest_dist)
        features.append(farthest_dist)
        features.append(avg_dist)
        
        # Distance to winner/loser archetypes (if known)
        if 'winner_centroid' in self.metadata:
            winner_dist = np.linalg.norm(point - self.metadata['winner_centroid'])
            features.append(winner_dist)
        else:
            features.append(avg_dist)
        
        if 'loser_centroid' in self.metadata:
            loser_dist = np.linalg.norm(point - self.metadata['loser_centroid'])
            features.append(loser_dist)
        else:
            features.append(avg_dist)
        
        # Distance variance
        dist_variance = np.var(distances)
        features.append(dist_variance)
        
        # Relative distance (nearest / farthest)
        relative_dist = nearest_dist / (farthest_dist + 1e-10)
        features.append(relative_dist)
        
        # Isolation score (how far from all clusters)
        isolation = avg_dist / (nearest_dist + 1e-10)
        features.append(isolation)
        
        return features
    
    def _compute_intercluster_features(self, point: np.ndarray) -> List[float]:
        """Compute inter-cluster features (4)."""
        features = []
        
        # Find nearest two clusters
        distances = [np.linalg.norm(point - cent) for cent in self.cluster_centroids_]
        sorted_idx = np.argsort(distances)
        
        # Between-cluster distance (distance between nearest two centroids)
        if len(sorted_idx) > 1:
            nearest_two = [self.cluster_centroids_[sorted_idx[0]], 
                          self.cluster_centroids_[sorted_idx[1]]]
            between_dist = np.linalg.norm(nearest_two[0] - nearest_two[1])
            features.append(between_dist)
        else:
            features.append(0.0)
        
        # Cluster separation score
        if len(distances) > 1:
            separation = (distances[sorted_idx[1]] - distances[sorted_idx[0]]) / (np.mean(distances) + 1e-10)
            features.append(separation)
        else:
            features.append(0.0)
        
        # Cluster density at position (inverse of average distance)
        density = 1.0 / (np.mean(distances) + 1e-10)
        features.append(density)
        
        # Cross-cluster contamination (how mixed is the neighborhood)
        # Use distances to estimate mixing
        if len(distances) > 2:
            contamination = np.std(distances) / (np.mean(distances) + 1e-10)
            features.append(contamination)
        else:
            features.append(0.0)
        
        return features
    
    def _compute_transfer_features(self, point: np.ndarray, domain: Optional[str]) -> List[float]:
        """Compute domain transfer features (3)."""
        features = []
        
        if self.domain_cluster_map_ is not None and domain is not None and domain in self.domain_cluster_map_:
            # Get cluster probabilities for this domain
            domain_cluster_probs = self.domain_cluster_map_[domain]
            
            # Get cluster for this point
            point_reshaped = point.reshape(1, -1)
            if self.clustering_method == 'gmm':
                cluster = self.clusterer_.predict(point_reshaped)[0]
            else:
                cluster = self.clusterer_.predict(point_reshaped)[0]
            
            # Cross-domain similarity (how typical is this cluster for this domain)
            cross_domain_sim = domain_cluster_probs[cluster]
            features.append(cross_domain_sim)
            
            # Transfer confidence (entropy of domain's cluster distribution)
            # Low entropy = domain concentrated in few clusters = high transfer confidence
            domain_entropy = -np.sum(domain_cluster_probs * np.log(domain_cluster_probs + 1e-10))
            max_entropy = np.log(len(domain_cluster_probs))
            transfer_confidence = 1.0 - (domain_entropy / max_entropy)
            features.append(transfer_confidence)
            
            # Domain-specific deviation (distance from domain's typical pattern)
            # Compute expected position for this domain
            expected_position = np.average(
                self.cluster_centroids_,
                axis=0,
                weights=domain_cluster_probs
            )
            deviation = np.linalg.norm(point - expected_position)
            features.append(deviation)
        else:
            # No domain information
            features.extend([0.5, 0.5, 0.0])
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        
        # Cluster membership (5)
        names.extend([
            'cluster_id',
            'cluster_probability',
            'cluster_secondary_id',
            'cluster_coherence',
            'cluster_boundary_distance'
        ])
        
        # Embedding coordinates (10)
        names.extend([f'embedding_dim_{i}' for i in range(10)])
        
        # Archetypal distances (8)
        names.extend([
            'distance_nearest_centroid',
            'distance_farthest_centroid',
            'distance_avg_centroid',
            'distance_winner_archetype',
            'distance_loser_archetype',
            'distance_variance',
            'distance_relative',
            'isolation_score'
        ])
        
        # Inter-cluster (4)
        names.extend([
            'intercluster_distance',
            'cluster_separation',
            'cluster_density',
            'cross_cluster_contamination'
        ])
        
        # Domain transfer (3)
        names.extend([
            'cross_domain_similarity',
            'transfer_confidence',
            'domain_deviation'
        ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of discovered patterns."""
        if not self.is_fitted_:
            return "Transformer not fitted yet."
        
        interpretation = f"""
Cross-Domain Embedding Analysis

Projected narratives into UNIVERSAL embedding space to discover
structural isomorphism across domains.

Embedding: {self.metadata.get('embedding_method', 'unknown')}
Dimensions: {self.metadata.get('embedding_dim', 0)}
Clusters Discovered: {self.metadata.get('n_clusters', 0)}
Samples Analyzed: {self.metadata.get('n_samples', 0)}

Discovered Archetypal Clusters:
- {self.metadata.get('n_clusters', 0)} universal narrative patterns
- Clusters span multiple domains (structural similarity)
- Enable transfer learning via cluster membership

Features Enable:
1. Discovering which narratives are structurally similar across domains
2. Transfer learning from domain A to domain B via shared clusters
3. Measuring distance to "golden narratio" (Ξ) via cluster centroids
4. Identifying universal vs domain-specific patterns

Example Use: If NBA Playoff game and NFL Playoff game fall in same cluster,
they share structural narrative patterns despite different domains.
"""
        
        # Add cluster win rates if available
        if 'cluster_win_rates' in self.metadata:
            interpretation += "\n\nCluster Win Rates (if outcomes provided):\n"
            for cluster, rate in self.metadata['cluster_win_rates'].items():
                interpretation += f"  Cluster {cluster}: {rate:.1%}\n"
        
        return interpretation.strip()

