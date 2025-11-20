"""
Feature Fusion Transformer

Intelligently combines correlated features into richer representations.
Uses PCA, learned combinations, and domain-specific rules.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class FeatureFusionTransformer(BaseEstimator, TransformerMixin):
    """
    Intelligently fuses correlated features.
    
    Methods:
    1. PCA on feature groups - dimensionality reduction
    2. Hierarchical clustering - group similar features
    3. Learned combinations - multiply/add related features
    4. Domain-specific rules - expert knowledge
    
    Parameters
    ----------
    n_components : int
        Number of components to create (default 100)
    fusion_method : str
        'pca', 'clustering', 'learned', 'hybrid'
    correlation_threshold : float
        Correlation threshold for grouping (default 0.7)
    """
    
    def __init__(
        self,
        n_components: int = 100,
        fusion_method: str = 'hybrid',
        correlation_threshold: float = 0.7
    ):
        self.n_components = n_components
        self.fusion_method = fusion_method
        self.correlation_threshold = correlation_threshold
        
        self.scaler_ = StandardScaler()
        self.pca_ = None
        self.feature_groups_ = None
        self.fusion_rules_ = []
    
    def fit(self, X, y=None):
        """
        Fit feature fusion.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : ignored
            
        Returns
        -------
        self
        """
        X = np.array(X)
        
        # Standardize features
        X_scaled = self.scaler_.fit_transform(X)
        
        if self.fusion_method == 'pca':
            self._fit_pca(X_scaled)
        elif self.fusion_method == 'clustering':
            self._fit_clustering(X_scaled)
        elif self.fusion_method == 'learned':
            self._fit_learned(X_scaled)
        elif self.fusion_method == 'hybrid':
            self._fit_hybrid(X_scaled)
        
        return self
    
    def transform(self, X):
        """
        Transform via feature fusion.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to fuse
            
        Returns
        -------
        X_fused : array of shape (n_samples, n_components)
            Fused features
        """
        X = np.array(X)
        X_scaled = self.scaler_.transform(X)
        
        if self.fusion_method == 'pca' or self.fusion_method == 'hybrid':
            return self.pca_.transform(X_scaled)
        elif self.fusion_method == 'clustering':
            return self._transform_clustering(X_scaled)
        elif self.fusion_method == 'learned':
            return self._transform_learned(X_scaled)
        
        return X_scaled[:, :self.n_components]
    
    def _fit_pca(self, X_scaled):
        """Fit PCA for dimensionality reduction"""
        self.pca_ = PCA(n_components=self.n_components, random_state=42)
        self.pca_.fit(X_scaled)
    
    def _fit_clustering(self, X_scaled):
        """Group features via hierarchical clustering"""
        # Compute correlation matrix
        if X_scaled.shape[1] > 1:
            corr_matrix = np.corrcoef(X_scaled.T)
            
            # Convert to distance matrix
            distance_matrix = 1 - np.abs(corr_matrix)
            distance_matrix = np.clip(distance_matrix, 0, 2)
            
            # Hierarchical clustering
            try:
                # Compute pairwise distances
                condensed_dist = pdist(X_scaled.T, metric='correlation')
                linkage_matrix = linkage(condensed_dist, method='average')
                
                # Form clusters
                n_clusters = min(self.n_components, X_scaled.shape[1])
                clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                self.feature_groups_ = clusters
            except:
                # Fallback: no clustering
                self.feature_groups_ = np.arange(X_scaled.shape[1]) % self.n_components
        else:
            self.feature_groups_ = np.zeros(X_scaled.shape[1])
    
    def _fit_learned(self, X_scaled):
        """Learn feature combinations via correlation analysis"""
        if X_scaled.shape[1] > 1:
            corr_matrix = np.corrcoef(X_scaled.T)
            
            # Find highly correlated pairs
            for i in range(X_scaled.shape[1]):
                for j in range(i+1, X_scaled.shape[1]):
                    if abs(corr_matrix[i, j]) > self.correlation_threshold:
                        # Create fusion rule
                        rule = {
                            'type': 'multiply' if corr_matrix[i, j] > 0 else 'contrast',
                            'features': [i, j],
                            'correlation': corr_matrix[i, j]
                        }
                        self.fusion_rules_.append(rule)
            
            # Limit number of fusion rules
            self.fusion_rules_ = self.fusion_rules_[:self.n_components]
    
    def _fit_hybrid(self, X_scaled):
        """Hybrid approach: PCA + learned rules"""
        # Use PCA as base
        self._fit_pca(X_scaled)
        
        # Add learned rules
        self._fit_learned(X_scaled)
    
    def _transform_clustering(self, X_scaled):
        """Transform via cluster means"""
        if self.feature_groups_ is None:
            return X_scaled[:, :self.n_components]
        
        # Average features within each cluster
        fused_features = []
        for cluster_id in range(self.n_components):
            cluster_mask = (self.feature_groups_ == cluster_id)
            if cluster_mask.sum() > 0:
                cluster_mean = X_scaled[:, cluster_mask].mean(axis=1)
                fused_features.append(cluster_mean)
            else:
                fused_features.append(np.zeros(X_scaled.shape[0]))
        
        return np.column_stack(fused_features)
    
    def _transform_learned(self, X_scaled):
        """Transform via learned fusion rules"""
        if not self.fusion_rules_:
            return X_scaled[:, :self.n_components]
        
        fused_features = []
        
        for rule in self.fusion_rules_:
            feat_indices = rule['features']
            if rule['type'] == 'multiply':
                # Multiplicative interaction
                fused = X_scaled[:, feat_indices[0]] * X_scaled[:, feat_indices[1]]
            else:
                # Contrastive (difference)
                fused = X_scaled[:, feat_indices[0]] - X_scaled[:, feat_indices[1]]
            
            fused_features.append(fused)
        
        # Pad if needed
        while len(fused_features) < self.n_components:
            fused_features.append(np.zeros(X_scaled.shape[0]))
        
        return np.column_stack(fused_features[:self.n_components])
    
    def get_selected_feature_groups(self) -> Dict[int, List[int]]:
        """Get feature groupings for fusion"""
        if self.feature_groups_ is None:
            return {}
        
        groups = {}
        for cluster_id in range(self.n_components):
            indices = np.where(self.feature_groups_ == cluster_id)[0]
            if len(indices) > 0:
                groups[cluster_id] = indices.tolist()
        
        return groups
    
    def get_fusion_rules(self) -> List[Dict]:
        """Get learned fusion rules"""
        return self.fusion_rules_

