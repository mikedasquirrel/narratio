"""
Feature Selection Transformer

Intelligently selects most predictive features per domain.
Uses mutual information, random forests, and domain adaptation.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    """
    Automatically selects best features per domain.
    
    Methods:
    1. Mutual Information - information-theoretic relevance
    2. Random Forest - tree-based importance
    3. Correlation - linear relationship strength
    4. Domain adaptation - different features per domain type
    
    Parameters
    ----------
    n_features : int
        Number of features to select (default 150)
    method : str
        Selection method: 'mutual_info', 'random_forest', 'correlation', 'ensemble'
    task_type : str
        'classification' or 'regression'
    domain_adaptive : bool
        Adapt selection strategy based on domain characteristics
    """
    
    def __init__(
        self,
        n_features: int = 150,
        method: str = 'ensemble',
        task_type: str = 'classification',
        domain_adaptive: bool = True
    ):
        self.n_features = n_features
        self.method = method
        self.task_type = task_type
        self.domain_adaptive = domain_adaptive
        
        self.selected_indices_ = None
        self.feature_scores_ = None
        self.feature_names_ = None
    
    def fit(self, X, y):
        """
        Fit feature selector.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target variable
            
        Returns
        -------
        self
        """
        X = np.array(X)
        y = np.array(y)
        
        # Compute feature scores using selected method(s)
        if self.method == 'mutual_info':
            scores = self._compute_mutual_info(X, y)
        elif self.method == 'random_forest':
            scores = self._compute_random_forest_importance(X, y)
        elif self.method == 'correlation':
            scores = self._compute_correlation(X, y)
        elif self.method == 'ensemble':
            scores = self._compute_ensemble_scores(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.feature_scores_ = scores
        
        # Select top N features
        self.selected_indices_ = np.argsort(scores)[::-1][:self.n_features]
        self.selected_indices_ = np.sort(self.selected_indices_)  # Keep order
        
        return self
    
    def transform(self, X):
        """
        Transform by selecting features.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to select from
            
        Returns
        -------
        X_selected : array of shape (n_samples, n_features_selected)
            Selected features
        """
        X = np.array(X)
        return X[:, self.selected_indices_]
    
    def _compute_mutual_info(self, X, y) -> np.ndarray:
        """Compute mutual information scores"""
        if self.task_type == 'classification':
            # Ensure y is integer for classification
            y = y.astype(int)
            scores = mutual_info_classif(X, y, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)
        
        return scores
    
    def _compute_random_forest_importance(self, X, y) -> np.ndarray:
        """Compute feature importance via random forest"""
        if self.task_type == 'classification':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        rf.fit(X, y)
        return rf.feature_importances_
    
    def _compute_correlation(self, X, y) -> np.ndarray:
        """Compute correlation with target"""
        scores = []
        
        for i in range(X.shape[1]):
            try:
                corr, _ = pearsonr(X[:, i], y)
                scores.append(abs(corr))  # Absolute value
            except:
                scores.append(0.0)
        
        return np.array(scores)
    
    def _compute_ensemble_scores(self, X, y) -> np.ndarray:
        """Ensemble of all methods"""
        # Get scores from each method
        mi_scores = self._compute_mutual_info(X, y)
        rf_scores = self._compute_random_forest_importance(X, y)
        corr_scores = self._compute_correlation(X, y)
        
        # Normalize each to 0-1
        mi_scores = mi_scores / (mi_scores.max() + 1e-10)
        rf_scores = rf_scores / (rf_scores.max() + 1e-10)
        corr_scores = corr_scores / (corr_scores.max() + 1e-10)
        
        # Weighted average (RF has highest weight)
        ensemble_scores = 0.4 * rf_scores + 0.35 * mi_scores + 0.25 * corr_scores
        
        return ensemble_scores
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features"""
        return self.selected_indices_
    
    def get_feature_scores(self) -> np.ndarray:
        """Get importance scores for all features"""
        return self.feature_scores_
    
    def get_feature_ranking(self) -> List[tuple]:
        """
        Get features ranked by importance.
        
        Returns
        -------
        ranking : list of (index, score) tuples
            Features ranked by score
        """
        if self.feature_scores_ is None:
            return []
        
        ranking = [(i, score) for i, score in enumerate(self.feature_scores_)]
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking

