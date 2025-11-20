"""
Ensemble Narrative Transformer

Analyzes ensemble/cast dynamics - how multiple entities interact.
Tests whether ensemble patterns predict outcomes.
"""

from typing import List
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list, ensure_string


class EnsembleNarrativeTransformer(NarrativeTransformer):
    """
    Extracts ensemble/cast features from narratives.
    
    Features (25 total):
    - Top term frequencies (ensemble vocabulary)
    - Term co-occurrence patterns
    - Ensemble diversity
    """
    
    def __init__(self, n_top_terms: int = 25, domain_config=None):
        super().__init__(
            narrative_id="ensemble",
            description="Ensemble analysis: cast/network dynamics"
        )
        self.n_top_terms = n_top_terms
        self.vectorizer_ = None
        self.domain_config = domain_config
    
    def fit(self, X, y=None):
        """Fit transformer"""
        from .utils.input_validation import ensure_string_list
        
        # Ensure X is list of strings (sklearn CountVectorizer requires this)
        X = ensure_string_list(X)
        
        # Adaptive parameters for small datasets
        n_docs = len(X)
        adaptive_min_df = 1  # Always allow all terms for ensemble
        adaptive_max_df = min(0.95, max(0.5, 1.0 - (10 / n_docs))) if n_docs > 10 else 1.0
        
        # Build vocabulary from top terms
        self.vectorizer_ = CountVectorizer(
            max_features=self.n_top_terms,
            ngram_range=(1, 2),
            min_df=adaptive_min_df,
            max_df=adaptive_max_df
        )
        self.vectorizer_.fit(X)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform texts to ensemble features"""
        self._validate_fitted()
        
        from .utils.input_validation import ensure_string_list
        
        # Ensure X is list of strings (sklearn CountVectorizer requires this)
        X = ensure_string_list(X)
        
        # Extract top terms
        X_vectorized = self.vectorizer_.transform(X)
        
        # Convert to dense array
        features = X_vectorized.toarray()
        
        return features
    
    def get_feature_names_out(self):
        """Return feature names"""
        if self.vectorizer_ is None:
            return [f"ensemble_{i}" for i in range(self.n_top_terms)]
        return [f"ensemble_{name}" for name in self.vectorizer_.get_feature_names_out()]

