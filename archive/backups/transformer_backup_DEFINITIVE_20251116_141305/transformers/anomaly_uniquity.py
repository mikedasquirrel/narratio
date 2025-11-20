"""
Anomaly Uniquity Transformer

Measures how unusual/novel a narrative is WITHOUT defining what "unusual" means.

Philosophy: Discover which level of uniqueness predicts outcomes.
Sometimes novelty wins (disruption), sometimes it loses (weird fails).
Let the learning system discover which.

Universal across all domains.

Author: Narrative Optimization Framework
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from scipy.spatial.distance import cdist, mahalanobis
from scipy.stats import zscore
import warnings

from .base_transformer import FeatureNarrativeTransformer


class AnomalyUniquityTransformer(FeatureNarrativeTransformer):
    """
    Measure narrative uniqueness and anomaly scores.
    
    Discovers how unusual an instance is WITHOUT predefining "unusual":
    - Statistical outliers (z-scores, Mahalanobis distance)
    - Multivariate uniqueness (IsolationForest, LOF)
    - Historical precedent (distance to nearest neighbors)
    - Novelty decay (if temporal data available)
    
    Enables discovery of uniqueness effects:
    - Does novelty predict success in startups? (hypothesis: yes)
    - Does novelty predict failure in sports? (hypothesis: yes, risky)
    - Does being unusual help or hurt? (domain-dependent)
    
    Features Extracted (~20 total):
    
    Statistical Outliers (5 features):
    - Mean z-score (averaged across dimensions)
    - Max z-score (most extreme dimension)
    - Number of dimensions with |z| > 2
    - Multivariate outlier score
    - Mahalanobis distance
    
    Isolation Metrics (4 features):
    - Isolation forest anomaly score
    - Local outlier factor (LOF)
    - Elliptic envelope outlier score
    - Combined isolation score
    
    Historical Precedent (5 features):
    - Distance to nearest neighbor
    - Distance to 5th nearest neighbor
    - Average distance to 10 nearest neighbors
    - Nearest neighbor distance variance
    - Historical similarity score (inverse distance)
    
    Cluster Analysis (3 features):
    - Distance to nearest cluster center
    - Cluster membership ambiguity
    - Inter-cluster position score
    
    Novelty Metrics (3 features):
    - Overall novelty score (composite)
    - Uniqueness rank (percentile)
    - Conformity score (inverse of novelty)
    
    Parameters
    ----------
    contamination : float, default=0.1
        Expected proportion of outliers (for IsolationForest/LOF)
    n_neighbors : int, default=10
        Number of neighbors for precedent analysis
    use_mahalanobis : bool, default=True
        Whether to compute Mahalanobis distance (requires invertible covariance)
    
    Examples
    --------
    >>> transformer = AnomalyUniquityTransformer()
    >>> 
    >>> # Standard feature matrix input
    >>> X = np.array([[0.5, 0.3], [0.6, 0.4], [0.1, 0.9], ...])
    >>> features = transformer.fit_transform(X)
    >>> 
    >>> # Primary outputs
    >>> novelty = features[:, -3]  # Overall novelty score
    >>> precedent = features[:, 10]  # Distance to nearest neighbor
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_neighbors: int = 10,
        use_mahalanobis: bool = True
    ):
        super().__init__(
            narrative_id='anomaly_uniquity',
            description='Novelty and uniqueness detection'
        )
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.use_mahalanobis = use_mahalanobis
        
        # Will be populated during fit
        self.scaler_ = None
        self.isolation_forest_ = None
        self.lof_ = None
        self.elliptic_envelope_ = None
        self.training_data_ = None  # For nearest neighbor search
        self.covariance_inv_ = None  # For Mahalanobis
        self.mean_ = None
        
    def fit(self, X, y=None):
        """
        Fit transformer to data.
        
        Learns the distribution and anomaly detectors.
        
        Parameters
        ----------
        X : array-like
            Training data (feature matrix)
        y : array-like, optional
            Target values (not used, for sklearn compatibility)
            
        Returns
        -------
        self
        """
        X = self._convert_to_array(X)
        self._validate_input(X)
        
        # Fit scaler
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Store training data for nearest neighbor comparisons
        self.training_data_ = X_scaled
        self.mean_ = np.mean(X_scaled, axis=0)
        
        # Fit anomaly detectors
        if len(X_scaled) > 10:  # Need sufficient data
            # Isolation Forest
            self.isolation_forest_ = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.isolation_forest_.fit(X_scaled)
            
            # Local Outlier Factor (need to refit on each transform, store params)
            self.lof_ = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, len(X_scaled) - 1),
                contamination=self.contamination,
                novelty=True  # Enable predict for new data
            )
            self.lof_.fit(X_scaled)
            
            # Elliptic Envelope (assumes Gaussian distribution)
            if len(X_scaled) > X_scaled.shape[1] + 1:
                try:
                    self.elliptic_envelope_ = EllipticEnvelope(
                        contamination=self.contamination,
                        random_state=42
                    )
                    self.elliptic_envelope_.fit(X_scaled)
                except:
                    self.elliptic_envelope_ = None
        
        # Compute covariance for Mahalanobis distance
        if self.use_mahalanobis and len(X_scaled) > X_scaled.shape[1]:
            try:
                cov = np.cov(X_scaled.T)
                self.covariance_inv_ = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
            except:
                self.covariance_inv_ = None
        
        # Store metadata
        self.metadata['n_samples'] = len(X)
        self.metadata['n_dimensions'] = X.shape[1]
        self.metadata['n_features'] = 20
        self.metadata['feature_names'] = self._get_feature_names()
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform data to uniqueness/anomaly features.
        
        Parameters
        ----------
        X : array-like
            Data to transform
            
        Returns
        -------
        features : ndarray, shape (n_samples, 20)
            Uniqueness/anomaly features
        """
        self._validate_fitted()
        X = self._convert_to_array(X)
        self._validate_input(X)
        
        # Scale
        X_scaled = self.scaler_.transform(X)
        
        # Extract features for each instance
        features = []
        for point in X_scaled:
            feat_vector = self._extract_uniqueness_features(point, X_scaled)
            features.append(feat_vector)
        
        return np.array(features)
    
    def _convert_to_array(self, X) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, list):
            if len(X) > 0 and isinstance(X[0], dict):
                # Extract genome_features from dicts
                features = []
                for item in X:
                    if 'genome_features' in item:
                        features.append(np.array(item['genome_features']))
                    else:
                        raise ValueError("Dict must contain 'genome_features'")
                return np.array(features)
            else:
                return np.array(X)
        else:
            return np.array(X)
    
    def _extract_uniqueness_features(self, point: np.ndarray, X_context: np.ndarray) -> np.ndarray:
        """Extract all uniqueness features for one instance."""
        features = []
        
        # 1. Statistical Outliers (5)
        features.extend(self._compute_statistical_outliers(point))
        
        # 2. Isolation Metrics (4)
        features.extend(self._compute_isolation_metrics(point))
        
        # 3. Historical Precedent (5)
        features.extend(self._compute_precedent_features(point))
        
        # 4. Cluster Analysis (3)
        features.extend(self._compute_cluster_features(point))
        
        # 5. Novelty Metrics (3)
        features.extend(self._compute_novelty_metrics(point, features))
        
        return np.array(features)
    
    def _compute_statistical_outliers(self, point: np.ndarray) -> List[float]:
        """Compute statistical outlier scores (5)."""
        features = []
        
        # Z-scores
        if np.any(np.std(self.training_data_, axis=0) > 0):
            z_scores = (point - self.mean_) / (np.std(self.training_data_, axis=0) + 1e-10)
            
            # Mean z-score
            mean_z = np.mean(np.abs(z_scores))
            features.append(mean_z)
            
            # Max z-score
            max_z = np.max(np.abs(z_scores))
            features.append(max_z)
            
            # Number of extreme dimensions (|z| > 2)
            n_extreme = np.sum(np.abs(z_scores) > 2)
            features.append(n_extreme)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Multivariate outlier score (distance from mean in stds)
        distance_from_mean = np.linalg.norm(point - self.mean_)
        avg_std = np.mean(np.std(self.training_data_, axis=0))
        if avg_std > 0:
            multivariate_outlier = distance_from_mean / avg_std
            features.append(multivariate_outlier)
        else:
            features.append(0.0)
        
        # Mahalanobis distance
        if self.covariance_inv_ is not None:
            try:
                mahal_dist = mahalanobis(point, self.mean_, self.covariance_inv_)
                features.append(mahal_dist)
            except:
                features.append(distance_from_mean)
        else:
            features.append(distance_from_mean)
        
        return features
    
    def _compute_isolation_metrics(self, point: np.ndarray) -> List[float]:
        """Compute isolation-based metrics (4)."""
        features = []
        
        point_reshaped = point.reshape(1, -1)
        
        # Isolation Forest anomaly score
        if self.isolation_forest_ is not None:
            iso_score = self.isolation_forest_.score_samples(point_reshaped)[0]
            # Convert to [0, 1] where 1 = more anomalous
            iso_anomaly = 1.0 / (1.0 + np.exp(iso_score))
            features.append(iso_anomaly)
        else:
            features.append(0.5)
        
        # Local Outlier Factor
        if self.lof_ is not None:
            try:
                lof_score = self.lof_.score_samples(point_reshaped)[0]
                # Convert to anomaly score
                lof_anomaly = max(0, -lof_score)
                features.append(lof_anomaly)
            except:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Elliptic Envelope
        if self.elliptic_envelope_ is not None:
            try:
                ee_score = self.elliptic_envelope_.score_samples(point_reshaped)[0]
                ee_anomaly = 1.0 / (1.0 + np.exp(ee_score))
                features.append(ee_anomaly)
            except:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Combined isolation score (average of available metrics)
        available = [f for f in features if f is not None]
        combined = np.mean(available) if available else 0.5
        features.append(combined)
        
        return features
    
    def _compute_precedent_features(self, point: np.ndarray) -> List[float]:
        """Compute historical precedent features (5)."""
        features = []
        
        # Compute distances to all training points
        distances = cdist([point], self.training_data_, metric='euclidean')[0]
        sorted_distances = np.sort(distances)
        
        # Distance to nearest neighbor
        nearest = sorted_distances[1] if len(sorted_distances) > 1 else sorted_distances[0]
        features.append(nearest)
        
        # Distance to 5th nearest neighbor
        fifth_nearest = sorted_distances[min(5, len(sorted_distances) - 1)]
        features.append(fifth_nearest)
        
        # Average distance to 10 nearest neighbors
        k = min(self.n_neighbors, len(sorted_distances))
        avg_k_nearest = np.mean(sorted_distances[:k])
        features.append(avg_k_nearest)
        
        # Nearest neighbor distance variance
        k_nearest_var = np.var(sorted_distances[:k])
        features.append(k_nearest_var)
        
        # Historical similarity score (inverse distance)
        similarity = 1.0 / (1.0 + nearest)
        features.append(similarity)
        
        return features
    
    def _compute_cluster_features(self, point: np.ndarray) -> List[float]:
        """Compute cluster-based features (3)."""
        features = []
        
        # Simple k-means clustering of training data
        from sklearn.cluster import KMeans
        
        n_clusters = min(5, len(self.training_data_) // 10)
        if n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(self.training_data_)
            
            # Distance to nearest cluster center
            distances_to_centers = cdist([point], kmeans.cluster_centers_)[0]
            nearest_cluster_dist = np.min(distances_to_centers)
            features.append(nearest_cluster_dist)
            
            # Cluster membership ambiguity (ratio of distances to nearest 2 clusters)
            sorted_cluster_dists = np.sort(distances_to_centers)
            if len(sorted_cluster_dists) > 1:
                ambiguity = sorted_cluster_dists[0] / (sorted_cluster_dists[1] + 1e-10)
                features.append(ambiguity)
            else:
                features.append(1.0)
            
            # Inter-cluster position (how far from all clusters)
            avg_cluster_dist = np.mean(distances_to_centers)
            inter_cluster_score = avg_cluster_dist / (nearest_cluster_dist + 1e-10)
            features.append(inter_cluster_score)
        else:
            features.extend([0.0, 1.0, 1.0])
        
        return features
    
    def _compute_novelty_metrics(self, point: np.ndarray, previous_features: List[float]) -> List[float]:
        """Compute overall novelty metrics (3)."""
        features = []
        
        # Overall novelty score (composite of multiple signals)
        # Weight different aspects
        if len(previous_features) >= 17:
            statistical = previous_features[0]  # Mean z-score
            isolation = previous_features[8]  # Combined isolation
            precedent_dist = previous_features[10]  # Distance to nearest neighbor
            
            # Normalize and combine
            novelty = (statistical * 0.3 + isolation * 0.4 + min(precedent_dist, 5.0) / 5.0 * 0.3)
            features.append(novelty)
        else:
            features.append(0.5)
        
        # Uniqueness rank (percentile in training distribution)
        # Compute percentile of distance from mean
        distances_from_mean = np.linalg.norm(self.training_data_ - self.mean_, axis=1)
        point_dist = np.linalg.norm(point - self.mean_)
        percentile = np.mean(distances_from_mean < point_dist)
        features.append(percentile)
        
        # Conformity score (inverse of novelty)
        conformity = 1.0 - features[0]
        features.append(conformity)
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names."""
        names = []
        
        # Statistical outliers (5)
        names.extend([
            'outlier_mean_z_score',
            'outlier_max_z_score',
            'outlier_n_extreme_dimensions',
            'outlier_multivariate_score',
            'outlier_mahalanobis_distance'
        ])
        
        # Isolation metrics (4)
        names.extend([
            'isolation_forest_score',
            'local_outlier_factor',
            'elliptic_envelope_score',
            'isolation_combined_score'
        ])
        
        # Historical precedent (5)
        names.extend([
            'precedent_nearest_neighbor_distance',
            'precedent_5th_nearest_distance',
            'precedent_avg_10_nearest_distance',
            'precedent_distance_variance',
            'precedent_similarity_score'
        ])
        
        # Cluster analysis (3)
        names.extend([
            'cluster_nearest_center_distance',
            'cluster_membership_ambiguity',
            'cluster_inter_cluster_position'
        ])
        
        # Novelty metrics (3)
        names.extend([
            'novelty_overall_score',
            'novelty_uniqueness_percentile',
            'novelty_conformity_score'
        ])
        
        return names
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of discovered patterns."""
        if not self.is_fitted_:
            return "Transformer not fitted yet."
        
        interpretation = f"""
Anomaly Uniquity Analysis

Measured novelty and uniqueness WITHOUT predefining "unusual".

Samples Analyzed: {self.metadata.get('n_samples', 0)}
Dimensions: {self.metadata.get('n_dimensions', 0)}
Contamination: {self.contamination}

Feature Categories:
1. Statistical Outliers (5): Z-scores, Mahalanobis distance
2. Isolation Metrics (4): IsolationForest, LOF, Elliptic Envelope
3. Historical Precedent (5): Nearest neighbor distances
4. Cluster Analysis (3): Position relative to clusters
5. Novelty Metrics (3): Overall uniqueness scores

These features enable DISCOVERY of uniqueness effects:
- Does novelty predict success? (startups, innovation)
- Does novelty predict failure? (risky strategies)
- Is conformity better? (established domains)
- What level of uniqueness is optimal?

The learning system discovers which level of uniqueness
predicts outcomes in each domain without assumptions.

Key Features:
- novelty_overall_score (column 17): Composite novelty measure
- precedent_nearest_neighbor_distance (column 10): Historical similarity
- isolation_combined_score (column 8): Anomaly detection
"""
        return interpretation.strip()

