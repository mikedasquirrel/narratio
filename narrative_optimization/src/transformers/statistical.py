"""
Statistical baseline transformer using TF-IDF.

No domain narrative - pure frequency-based statistical features.
This serves as the baseline to test whether narrative approaches add value.
"""

from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import NarrativeTransformer
from .utils.input_validation import ensure_string_list


class StatisticalTransformer(NarrativeTransformer):
    """
    Statistical baseline using TF-IDF vectorization.
    
    This transformer has no domain narrative - it treats text as bags of words
    and uses pure statistical frequency features. Serves as the baseline to
    demonstrate what can be achieved without narrative-driven feature engineering.
    
    Parameters
    ----------
    max_features : int, optional
        Maximum number of TF-IDF features to extract
    ngram_range : tuple, optional
        Range of n-grams to consider (default: (1, 2) for unigrams and bigrams)
    min_df : int or float, optional
        Minimum document frequency for term inclusion
    max_df : float, optional
        Maximum document frequency (filter out very common terms)
    
    Attributes
    ----------
    vectorizer_ : TfidfVectorizer
        Fitted vectorizer
    """
    
    def __init__(
        self,
        max_features: Optional[int] = 1000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.8
    ):
        super().__init__(
            narrative_id="statistical_baseline",
            description="Pure statistical approach with TF-IDF features, no domain narrative"
        )
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer_ = None
    
    def fit(self, X, y=None):
        """
        Fit TF-IDF vectorizer to text data.
        
        Parameters
        ----------
        X : list or array of str
            Text documents
        y : ignored
        
        Returns
        -------
        self : StatisticalTransformer
        """
        # Ensure X is list of strings (sklearn TfidfVectorizer requires this)
        X = ensure_string_list(X)
        
        # Adaptive min_df/max_df to handle small datasets
        n_docs = len(X)
        adaptive_min_df = min(self.min_df, max(1, n_docs // 10))  # At least 1, at most 10% of docs
        adaptive_max_df = min(self.max_df, 0.95)  # Cap at 95%
        
        # Ensure max_df (as absolute count) > min_df
        if adaptive_max_df < 1.0:
            max_df_absolute = int(adaptive_max_df * n_docs)
            if max_df_absolute < adaptive_min_df:
                adaptive_min_df = 1  # Fall back to minimum
        
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=adaptive_min_df,
            max_df=adaptive_max_df,
            stop_words='english'
        )
        
        self.vectorizer_.fit(X)
        
        # Store metadata
        self.metadata['num_features'] = len(self.vectorizer_.get_feature_names_out())
        self.metadata['vocabulary_size'] = len(self.vectorizer_.vocabulary_)
        self.metadata['approach'] = 'statistical'
        self.metadata['adaptive_min_df'] = adaptive_min_df
        self.metadata['adaptive_max_df'] = adaptive_max_df
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform text to TF-IDF features.
        
        Parameters
        ----------
        X : list or array of str
            Text documents
        
        Returns
        -------
        X_transformed : sparse matrix
            TF-IDF feature matrix
        """
        self._validate_fitted()
        
        # Ensure X is list of strings (sklearn TfidfVectorizer requires this)
        X = ensure_string_list(X)
        
        return self.vectorizer_.transform(X)
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of the statistical baseline."""
        num_features = self.metadata.get('num_features', 0)
        
        interpretation = (
            f"Statistical Baseline: Pure TF-IDF approach with {num_features} features. "
            f"This transformer uses word frequencies and inverse document frequencies "
            f"with no domain-specific narrative. It treats text as bags of words, "
            f"capturing statistical patterns but not semantic or structural meaning. "
            f"This serves as our baseline - if narrative approaches don't beat this, "
            f"they're not adding value beyond basic statistics."
        )
        
        return interpretation
    
    def get_feature_names_out(self):
        """Get feature names from the vectorizer."""
        self._validate_fitted()
        return self.vectorizer_.get_feature_names_out()

