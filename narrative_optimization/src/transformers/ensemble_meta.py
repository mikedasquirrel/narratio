"""
Ensemble Meta-Transformer

Learns optimal transformer weights per domain.
Uses meta-learning and stacking for intelligent ensembling.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster


class EnsembleMetaTransformer(BaseEstimator, TransformerMixin):
    """
    Meta-learns optimal transformer combination.
    
    Approach:
    1. Each transformer produces features independently
    2. Learn weights for each transformer's contribution
    3. Combine via weighted stacking
    4. Domain-adaptive weighting
    
    Parameters
    ----------
    transformers : list of (name, transformer) tuples
        Base transformers to ensemble
    meta_learner : str
        'ridge', 'logistic', 'voting'
    weighting : str
        'uniform', 'performance', 'learned'
    """
    
    def __init__(
        self,
        transformers: Optional[List[Tuple[str, Any]]] = None,
        meta_learner: str = 'ridge',
        weighting: str = 'learned',
        n_components: int = 32,
        fusion_mode: str = 'pca',
        correlation_threshold: float = 0.85
    ):
        self.transformers = transformers or []
        self.meta_learner = meta_learner
        self.weighting = weighting
        self.n_components = max(1, n_components)
        self.fusion_mode = fusion_mode
        self.correlation_threshold = correlation_threshold
        
        self.transformer_weights_ = None
        self.meta_model_ = None
        self.scaler_ = StandardScaler()
        self.transformer_performances_ = None
        self.precomputed_blocks_: Optional[List[Tuple[str, np.ndarray]]] = None
        self.cached_features_: Optional[List[np.ndarray]] = None
        self.transformer_names_: List[str] = []
        self.pca_: Optional[PCA] = None
        self.feature_groups_ = None
        self.fusion_rules_ = None
        self.output_dim_: int = 0
    
    def set_precomputed_blocks(self, blocks: List[Tuple[str, np.ndarray]]):
        """
        Provide precomputed feature blocks instead of raw transformers.
        
        Parameters
        ----------
        blocks : list of (name, ndarray)
            Each ndarray must be shape (n_samples, n_features).
        """
        cleaned_blocks: List[Tuple[str, np.ndarray]] = []
        for name, block in blocks:
            arr = np.asarray(block)
            if arr.ndim != 2:
                raise ValueError(f"Precomputed block for {name} must be 2D array.")
            cleaned_blocks.append((name, arr))
        self.precomputed_blocks_ = cleaned_blocks

    def fit(self, X, y):
        """
        Fit ensemble meta-learner.
        
        Parameters
        ----------
        X : array-like of strings
            Training texts
        y : array-like
            Target variable
            
        Returns
        -------
        self
        """
        use_precomputed = self.precomputed_blocks_ is not None
        if not self.transformers and not use_precomputed:
            raise ValueError("No transformers or precomputed feature blocks provided")
        
        transformer_features: List[np.ndarray] = []
        transformer_scores: List[float] = []
        
        if use_precomputed:
            print(f"[EnsembleMeta] Using {len(self.precomputed_blocks_)} precomputed feature blocks...")
            for name, block in self.precomputed_blocks_:
                transformer_features.append(block)
        else:
            print(f"[EnsembleMeta] Fitting {len(self.transformers)} transformers...")
            for name, transformer in self.transformers:
                print(f"  - Fitting {name}...")
                transformer.fit(X, y)
                X_trans = transformer.transform(X)
                transformer_features.append(X_trans)
        
        # Validate sample counts across blocks
        sample_counts = {block.shape[0] for block in transformer_features}
        if len(sample_counts) != 1:
            raise ValueError("EnsembleMeta requires all feature blocks to have identical sample counts.")
        
        self.transformer_names_ = (
            [name for name, _ in self.precomputed_blocks_]
            if use_precomputed else
            [name for name, _ in self.transformers]
        )
        self.cached_features_ = transformer_features
        self.pca_ = None  # Reset PCA when refitting
        
        # Evaluate individual block performance if requested
        if self.weighting == 'performance':
            transformer_scores = []
            for name, block in zip(self.transformer_names_, transformer_features):
                try:
                    if self.meta_learner == 'logistic':
                        model = LogisticRegression(max_iter=1000)
                    else:
                        model = Ridge()
                    scores = cross_val_score(model, block, y, cv=3, n_jobs=-1)
                    performance = np.mean(scores)
                    transformer_scores.append(performance)
                    print(f"    Performance ({name}): {performance:.3f}")
                except Exception:
                    transformer_scores.append(0.5)
        
        # Concatenate all features
        X_all = np.hstack(transformer_features)
        
        # Scale
        X_scaled = self.scaler_.fit_transform(X_all)
        
        # Learn meta-model
        if self.meta_learner == 'logistic':
            self.meta_model_ = LogisticRegression(max_iter=1000)
        else:
            self.meta_model_ = Ridge(alpha=1.0)
        
        self.meta_model_.fit(X_scaled, y)
        
        # Compute transformer weights
        if self.weighting == 'uniform':
            self.transformer_weights_ = np.ones(len(self.transformer_names_)) / len(self.transformer_names_)
        elif self.weighting == 'performance':
            # Weight by cross-validation performance
            weights = np.array(transformer_scores)
            weights = weights / weights.sum()
            self.transformer_weights_ = weights
            self.transformer_performances_ = transformer_scores
        elif self.weighting == 'learned':
            # Extract weights from meta-model coefficients
            coef = self.meta_model_.coef_
            if len(coef.shape) > 1:
                coef = coef[0]
            
            # Assign weights based on coefficient magnitudes per transformer
            start_idx = 0
            weights = []
            for X_trans in transformer_features:
                end_idx = start_idx + X_trans.shape[1]
                # Average absolute coefficient for this transformer's features
                transformer_weight = np.mean(np.abs(coef[start_idx:end_idx]))
                weights.append(transformer_weight)
                start_idx = end_idx
            
            weights = np.array(weights)
            self.transformer_weights_ = weights / weights.sum()
        
        print(f"[EnsembleMeta] Transformer weights:")
        for i, name in enumerate(self.transformer_names_):
            print(f"  {name}: {self.transformer_weights_[i]:.3f}")
        
        return self
    
    def transform(self, X=None):
        """
        Transform via ensemble.
        
        Parameters
        ----------
        X : array-like of strings
            Texts to transform
            
        Returns
        -------
        X_ensemble : array
            Ensemble features
        """
        if self.precomputed_blocks_:
            transformer_features = [block for _, block in self.precomputed_blocks_]
        else:
            if not self.transformers:
                raise ValueError("No transformers provided")
            if X is None:
                raise ValueError("Input texts are required when transformers are provided.")
            transformer_features = []
            for name, transformer in self.transformers:
                X_trans = transformer.transform(X)
                transformer_features.append(X_trans)
        
        # Concatenate
        X_all = np.hstack(transformer_features)
        
        # Scale
        X_scaled = self.scaler_.transform(X_all)
        
        # Apply meta-model
        # Return scaled features weighted by transformer importance
        start_idx = 0
        weighted_features = []
        
        for i, X_trans in enumerate(transformer_features):
            end_idx = start_idx + X_trans.shape[1]
            
            # Weight this transformer's features
            weighted = X_scaled[:, start_idx:end_idx] * self.transformer_weights_[i]
            weighted_features.append(weighted)
            
            start_idx = end_idx
        
        # Concatenate weighted features
        X_weighted = np.hstack(weighted_features)
        
        # Reduce to n_components via PCA if needed
        if X_weighted.shape[1] > self.n_components:
            if self.pca_ is None:
                self.pca_ = PCA(n_components=self.n_components, random_state=42)
                self.pca_.fit(X_weighted)
            
            transformed = self.pca_.transform(X_weighted)
            self.output_dim_ = transformed.shape[1]
            return transformed
        
        self.output_dim_ = X_weighted.shape[1]
        return X_weighted
    
    def get_feature_names(self) -> List[str]:
        """
        Return human-readable feature names for downstream pipelines.
        """
        if self.output_dim_ <= 0:
            return []
        return [f'ensemble_meta_{i}' for i in range(self.output_dim_)]
    
    def _fit_pca(self, X_scaled):
        """Fit PCA for fusion"""
        self.pca_ = PCA(n_components=self.n_components, random_state=42)
        self.pca_.fit(X_scaled)
    
    def _fit_clustering(self, X_scaled):
        """Fit hierarchical clustering"""
        # Compute correlation-based distance
        corr_matrix = np.corrcoef(X_scaled.T)
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Hierarchical clustering
        try:
            condensed_dist = pdist(X_scaled.T, metric='correlation')
            linkage_matrix = linkage(condensed_dist, method='average')
            clusters = fcluster(linkage_matrix, self.n_components, criterion='maxclust')
            self.feature_groups_ = clusters
        except:
            self.feature_groups_ = np.arange(X_scaled.shape[1]) % self.n_components
    
    def _fit_learned(self, X_scaled):
        """Learn feature combinations via correlation"""
        corr_matrix = np.corrcoef(X_scaled.T)
        
        # Find high-correlation groups
        self.fusion_rules_ = []
        for i in range(X_scaled.shape[1]):
            high_corr = np.where(np.abs(corr_matrix[i]) > self.correlation_threshold)[0]
            if len(high_corr) > 1:
                self.fusion_rules_.append({
                    'features': high_corr.tolist(),
                    'operation': 'mean'
                })
    
    def _fit_hybrid(self, X_scaled):
        """Hybrid: PCA + performance weighting"""
        self._fit_pca(X_scaled)
        # Weights computed in main fit() method
    
    def _transform_clustering(self, X_scaled):
        """Transform via cluster aggregation"""
        if self.feature_groups_ is None:
            return X_scaled[:, :self.n_components]
        
        fused = []
        for cluster_id in range(self.n_components):
            mask = (self.feature_groups_ == cluster_id)
            if mask.sum() > 0:
                cluster_features = X_scaled[:, mask].mean(axis=1)
                fused.append(cluster_features)
            else:
                fused.append(np.zeros(X_scaled.shape[0]))
        
        return np.column_stack(fused)
    
    def _transform_learned(self, X_scaled):
        """Transform via learned rules"""
        if not self.fusion_rules_:
            return X_scaled[:, :self.n_components]
        
        fused = []
        for rule in self.fusion_rules_[:self.n_components]:
            feat_indices = rule['features']
            if rule['operation'] == 'mean':
                fused_feat = X_scaled[:, feat_indices].mean(axis=1)
            elif rule['operation'] == 'multiply':
                fused_feat = np.prod(X_scaled[:, feat_indices], axis=1)
            else:
                fused_feat = X_scaled[:, feat_indices[0]]
            
            fused.append(fused_feat)
        
        return np.column_stack(fused)
    
    def get_transformer_weights(self) -> Dict[str, float]:
        """Get learned transformer weights"""
        if self.transformer_weights_ is None:
            return {}
        
        weights = {}
        for i, (name, _) in enumerate(self.transformers):
            weights[name] = self.transformer_weights_[i]
        
        return weights
    
    def get_transformer_performances(self) -> Optional[Dict[str, float]]:
        """Get individual transformer performances"""
        if self.transformer_performances_ is None:
            return None
        
        perfs = {}
        for i, (name, _) in enumerate(self.transformers):
            perfs[name] = self.transformer_performances_[i]
        
        return perfs

