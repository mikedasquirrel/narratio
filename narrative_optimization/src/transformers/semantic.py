"""
Semantic narrative transformer using embeddings and semantic features.

Captures semantic meaning through embeddings, clustering, and context patterns.
Tests the hypothesis that semantic understanding improves predictions.
"""

from typing import Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from .base import NarrativeTransformer


class SemanticNarrativeTransformer(NarrativeTransformer):
    """
    Semantic narrative using embeddings and semantic structure.
    
    Narrative Hypothesis: Understanding semantic meaning and document structure
    provides better features than pure word frequencies. This transformer captures:
    - Semantic embeddings (LSA/SVD of TF-IDF as a lightweight alternative to BERT)
    - Document clustering (semantic groupings)
    - Semantic density (how focused vs scattered the content is)
    
    Parameters
    ----------
    n_components : int
        Number of semantic dimensions (LSA components)
    n_clusters : int
        Number of semantic clusters to identify
    max_features : int
        Maximum vocabulary size for initial TF-IDF
    
    Attributes
    ----------
    vectorizer_ : TfidfVectorizer
        Initial text vectorizer
    svd_ : TruncatedSVD
        Semantic embedding model
    clusterer_ : KMeans
        Semantic clustering model
    """
    
    def __init__(
        self,
        n_components: int = 50,
        n_clusters: int = 10,
        max_features: int = 5000
    ):
        super().__init__(
            narrative_id="semantic_narrative",
            description="Semantic narrative: embeddings + clustering reveal deeper meaning"
        )
        
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.max_features = max_features
        
        self.vectorizer_ = None
        self.svd_ = None
        self.clusterer_ = None
    
    def fit(self, X, y=None):
        """
        Fit semantic models to text data.
        
        Parameters
        ----------
        X : list or array of str
            Text documents
        y : ignored
        
        Returns
        -------
        self : SemanticNarrativeTransformer
        """
        # Step 1: Vectorize text
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        X_tfidf = self.vectorizer_.fit_transform(X)
        
        # Step 2: Semantic embeddings via LSA
        self.svd_ = TruncatedSVD(
            n_components=min(self.n_components, X_tfidf.shape[1] - 1),
            random_state=42
        )
        X_semantic = self.svd_.fit_transform(X_tfidf)
        
        # Step 3: Semantic clustering
        self.clusterer_ = KMeans(
            n_clusters=min(self.n_clusters, len(X)),
            random_state=42,
            n_init=10
        )
        self.clusterer_.fit(X_semantic)
        
        # Store metadata
        self.metadata['n_semantic_dimensions'] = self.svd_.n_components
        self.metadata['explained_variance_ratio'] = float(
            np.sum(self.svd_.explained_variance_ratio_)
        )
        self.metadata['n_clusters'] = self.clusterer_.n_clusters
        self.metadata['approach'] = 'semantic'
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform text to semantic features.
        
        Creates features capturing:
        - Semantic embeddings (dense representation of meaning)
        - Cluster membership (semantic category)
        - Cluster distances (semantic positioning)
        - Semantic coherence (how focused the document is)
        
        Parameters
        ----------
        X : list or array of str
            Text documents
        
        Returns
        -------
        X_transformed : array
            Semantic feature matrix
        """
        self._validate_fitted()
        
        # Get TF-IDF representation
        X_tfidf = self.vectorizer_.transform(X)
        
        # Get semantic embeddings
        X_semantic = self.svd_.transform(X_tfidf)
        
        # Get cluster assignments
        cluster_labels = self.clusterer_.predict(X_semantic)
        
        # Get distances to all cluster centers
        cluster_distances = self.clusterer_.transform(X_semantic)
        
        # Compute semantic coherence (inverse of spread across dimensions)
        semantic_coherence = np.abs(X_semantic).std(axis=1).reshape(-1, 1)
        
        # Compute semantic density (how concentrated in semantic space)
        semantic_density = np.linalg.norm(X_semantic, axis=1).reshape(-1, 1)
        
        # One-hot encode cluster membership
        cluster_one_hot = np.zeros((len(X), self.clusterer_.n_clusters))
        cluster_one_hot[np.arange(len(X)), cluster_labels] = 1
        
        # Combine all semantic features
        features = np.hstack([
            X_semantic,                 # Dense semantic representation
            cluster_one_hot,            # Which semantic group
            cluster_distances,          # Distance to each semantic group
            semantic_coherence,         # How coherent the semantics are
            semantic_density            # How semantically dense
        ])
        
        return features
    
    def _generate_interpretation(self) -> str:
        """Generate interpretation of the semantic narrative."""
        n_dims = self.metadata.get('n_semantic_dimensions', 0)
        variance_explained = self.metadata.get('explained_variance_ratio', 0)
        n_clusters = self.metadata.get('n_clusters', 0)
        
        interpretation = (
            f"Semantic Narrative: Captures meaning through {n_dims} semantic dimensions "
            f"(explaining {variance_explained:.1%} of variance) and {n_clusters} semantic clusters. "
            f"This narrative assumes that understanding deeper semantic structure—not just "
            f"word frequencies—is key to prediction. Features capture semantic embeddings, "
            f"cluster membership (semantic categories), cluster distances (semantic positioning), "
            f"and coherence metrics (how focused vs scattered the semantic content is). "
            f"If this outperforms the statistical baseline, it suggests semantic understanding matters."
        )
        
        return interpretation

