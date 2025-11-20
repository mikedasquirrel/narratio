"""
Weighted Feature Integration

Learn optimal weights for combining multiple narrative transformers.
Addresses the problem that simple concatenation underperforms.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import warnings


class WeightedFeatureUnion(BaseEstimator, TransformerMixin):
    """
    Learn weights for each transformer in the union.
    
    Instead of simple concatenation, learns optimal weights for each
    transformer's features based on their predictive value.
    
    Parameters
    ----------
    transformers : list of (name, transformer) tuples
        Transformers to combine
    weight_learning : str
        Method for learning weights: 'ridge', 'lasso', 'uniform', or 'performance'
    alpha : float
        Regularization strength for ridge/lasso
    
    Attributes
    ----------
    weights_ : array
        Learned weights for each transformer's features
    transformer_ranges_ : list
        Feature index ranges for each transformer
    """
    
    def __init__(
        self,
        transformers: List[Tuple[str, BaseEstimator]],
        weight_learning: str = 'ridge',
        alpha: float = 1.0
    ):
        self.transformers = transformers
        self.weight_learning = weight_learning
        self.alpha = alpha
        
        self.weights_ = None
        self.transformer_ranges_ = []
        self.fitted_transformers_ = []
    
    def fit(self, X, y):
        """
        Fit all transformers and learn optimal weights.
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like
            Target values
        
        Returns
        -------
        self
        """
        # Fit each transformer and collect features
        all_features = []
        self.transformer_ranges_ = []
        self.fitted_transformers_ = []
        
        current_idx = 0
        
        for name, transformer in self.transformers:
            # Fit transformer
            transformer.fit(X, y)
            self.fitted_transformers_.append((name, transformer))
            
            # Transform training data
            X_transformed = transformer.transform(X)
            
            # Handle sparse matrices
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            
            # Track feature ranges
            n_features = X_transformed.shape[1]
            self.transformer_ranges_.append((current_idx, current_idx + n_features, name))
            current_idx += n_features
            
            all_features.append(X_transformed)
        
        # Concatenate all features
        X_all = np.hstack(all_features)
        
        # Learn weights
        if self.weight_learning == 'uniform':
            # Equal weights
            self.weights_ = np.ones(X_all.shape[1])
            
        elif self.weight_learning == 'performance':
            # Weight by transformer's individual performance
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LogisticRegression
            
            transformer_scores = []
            for name, transformer in self.fitted_transformers_:
                # Quick CV to estimate performance
                try:
                    scores = cross_val_score(
                        LogisticRegression(max_iter=100),
                        transformer.transform(X),
                        y,
                        cv=3,
                        scoring='accuracy'
                    )
                    transformer_scores.append(scores.mean())
                except:
                    transformer_scores.append(0.1)  # Low weight if fails
            
            # Assign weights proportional to performance
            self.weights_ = np.ones(X_all.shape[1])
            for i, (start, end, name) in enumerate(self.transformer_ranges_):
                self.weights_[start:end] = transformer_scores[i]
                
        elif self.weight_learning in ['ridge', 'lasso']:
            # Learn weights via regularized regression
            WeightModel = Ridge if self.weight_learning == 'ridge' else Lasso
            
            weight_learner = WeightModel(alpha=self.alpha, fit_intercept=False)
            weight_learner.fit(X_all, y)
            
            # Use absolute coefficients as weights
            self.weights_ = np.abs(weight_learner.coef_).flatten()
            
            # Normalize to prevent scale issues
            self.weights_ = self.weights_ / (self.weights_.mean() + 1e-10)
        
        else:
            raise ValueError(f"Unknown weight_learning: {self.weight_learning}")
        
        return self
    
    def transform(self, X):
        """
        Transform data with learned weights.
        
        Parameters
        ----------
        X : array-like
            Data to transform
        
        Returns
        -------
        X_weighted : array
            Weighted features
        """
        if self.weights_ is None:
            raise ValueError("Must fit before transform")
        
        # Transform with each transformer
        all_features = []
        
        for name, transformer in self.fitted_transformers_:
            X_transformed = transformer.transform(X)
            
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            
            all_features.append(X_transformed)
        
        # Concatenate
        X_all = np.hstack(all_features)
        
        # Apply weights
        X_weighted = X_all * self.weights_
        
        return X_weighted
    
    def get_transformer_weights(self) -> Dict[str, float]:
        """Get average weight for each transformer."""
        if self.weights_ is None:
            raise ValueError("Must fit first")
        
        transformer_weights = {}
        for start, end, name in self.transformer_ranges_:
            avg_weight = self.weights_[start:end].mean()
            transformer_weights[name] = float(avg_weight)
        
        return transformer_weights


class StackedNarrativeModel(BaseEstimator):
    """
    Two-stage stacked model for narrative prediction.
    
    Stage 1: Base transformers (e.g., Statistical) get initial predictions
    Stage 2: Advanced transformers + Stage 1 predictions → final prediction
    
    Parameters
    ----------
    base_pipeline : Pipeline
        First stage pipeline
    meta_transformers : list
        Additional transformers for second stage
    meta_classifier
        Final classifier for stage 2
    """
    
    def __init__(self, base_pipeline, meta_transformers, meta_classifier):
        self.base_pipeline = base_pipeline
        self.meta_transformers = meta_transformers
        self.meta_classifier = meta_classifier
        
        self.fitted_meta_transformers_ = []
    
    def fit(self, X, y):
        """Fit both stages."""
        # Stage 1: Fit base pipeline
        self.base_pipeline.fit(X, y)
        
        # Get stage 1 predictions (probabilities)
        if hasattr(self.base_pipeline, 'predict_proba'):
            stage1_preds = self.base_pipeline.predict_proba(X)
        else:
            stage1_preds = self.base_pipeline.predict(X).reshape(-1, 1)
        
        # Stage 2: Fit meta transformers
        meta_features = [stage1_preds]
        
        for name, transformer in self.meta_transformers:
            transformer.fit(X, y)
            self.fitted_meta_transformers_.append((name, transformer))
            
            X_meta = transformer.transform(X)
            if hasattr(X_meta, 'toarray'):
                X_meta = X_meta.toarray()
            
            meta_features.append(X_meta)
        
        # Combine stage 1 predictions + meta features
        X_meta_all = np.hstack(meta_features)
        
        # Fit meta classifier
        self.meta_classifier.fit(X_meta_all, y)
        
        return self
    
    def predict(self, X):
        """Predict using stacked model."""
        # Stage 1 predictions
        if hasattr(self.base_pipeline, 'predict_proba'):
            stage1_preds = self.base_pipeline.predict_proba(X)
        else:
            stage1_preds = self.base_pipeline.predict(X).reshape(-1, 1)
        
        # Stage 2 features
        meta_features = [stage1_preds]
        
        for name, transformer in self.fitted_meta_transformers_:
            X_meta = transformer.transform(X)
            if hasattr(X_meta, 'toarray'):
                X_meta = X_meta.toarray()
            meta_features.append(X_meta)
        
        X_meta_all = np.hstack(meta_features)
        
        # Final prediction
        return self.meta_classifier.predict(X_meta_all)
    
    def predict_proba(self, X):
        """Predict probabilities if meta classifier supports it."""
        if not hasattr(self.meta_classifier, 'predict_proba'):
            raise AttributeError("Meta classifier doesn't support predict_proba")
        
        # Stage 1
        if hasattr(self.base_pipeline, 'predict_proba'):
            stage1_preds = self.base_pipeline.predict_proba(X)
        else:
            stage1_preds = self.base_pipeline.predict(X).reshape(-1, 1)
        
        # Stage 2 features
        meta_features = [stage1_preds]
        
        for name, transformer in self.fitted_meta_transformers_:
            X_meta = transformer.transform(X)
            if hasattr(X_meta, 'toarray'):
                X_meta = X_meta.toarray()
            meta_features.append(X_meta)
        
        X_meta_all = np.hstack(meta_features)
        
        return self.meta_classifier.predict_proba(X_meta_all)

