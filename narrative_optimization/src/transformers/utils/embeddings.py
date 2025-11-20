"""
Embedding Manager

Centralized management of sentence embeddings for intelligent feature extraction.
Provides caching, fallback mechanisms, and unified interface.
"""

import numpy as np
from pathlib import Path
import pickle
from typing import List, Union, Optional
import warnings


class EmbeddingManager:
    """
    Manages sentence embeddings with caching and fallback.
    
    Supports:
    - sentence-transformers (preferred)
    - LSA/SVD fallback (if transformers not available)
    - Embedding caching for performance
    """
    
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        cache_dir=None,
        use_cache=True
    ):
        """
        Initialize embedding manager.
        
        Parameters
        ----------
        model_name : str
            Sentence-transformer model name
        cache_dir : str, optional
            Directory for caching embeddings
        use_cache : bool
            Whether to use caching
        """
        self.model_name = model_name
        self.use_cache = use_cache
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.narrative_cache' / 'embeddings'
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load sentence-transformers
        self.model = None
        self.mode = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.mode = 'transformer'
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            warnings.warn("sentence-transformers not available, using LSA fallback")
            self.mode = 'lsa'
            self.embedding_dim = 100  # Default for LSA
            self._setup_lsa_fallback()
    
    def _setup_lsa_fallback(self):
        """Setup LSA as fallback when transformers not available"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.svd = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
        self._lsa_max_components = self.embedding_dim
        self.fitted_fallback = False

    def _adjust_lsa_components(self, n_features: int):
        """Ensure SVD components respect available feature dimensionality."""
        if not hasattr(self, 'svd'):
            return
        from sklearn.decomposition import TruncatedSVD
        max_allowed = max(1, n_features - 1)
        desired = min(self._lsa_max_components, max_allowed)
        if desired != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=desired, random_state=42)
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Parameters
        ----------
        texts : str or list of str
            Text(s) to encode
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        embeddings : ndarray
            Shape (n_texts, embedding_dim)
        """
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key(texts)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Encode
        if self.mode == 'transformer':
            embeddings = self.model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
        else:
            # LSA fallback
            embeddings = self._encode_lsa(texts)
        
        # Cache
        if self.use_cache:
            self._save_to_cache(cache_key, embeddings)
        
        return embeddings
    
    def _encode_lsa(self, texts):
        """Encode using LSA fallback"""
        if not self.fitted_fallback:
            # Need to fit on these texts
            X_tfidf = self.vectorizer.fit_transform(texts)
            self._adjust_lsa_components(X_tfidf.shape[1])
            X_svd = self.svd.fit_transform(X_tfidf)
            self.fitted_fallback = True
            return X_svd
        else:
            X_tfidf = self.vectorizer.transform(texts)
            self._adjust_lsa_components(X_tfidf.shape[1])
            X_svd = self.svd.transform(X_tfidf)
            return X_svd
    
    def semantic_similarity(
        self,
        texts: List[str],
        anchor: Union[str, np.ndarray],
        method='cosine'
    ) -> np.ndarray:
        """
        Compute similarity between texts and anchor.
        
        Parameters
        ----------
        texts : list of str
            Texts to compare
        anchor : str or ndarray
            Anchor text or embedding
        method : str
            Similarity method ('cosine', 'dot', 'euclidean')
            
        Returns
        -------
        similarities : ndarray
            Similarity scores
        """
        # Encode texts
        text_embeddings = self.encode(texts)
        
        # Encode anchor if needed
        if isinstance(anchor, str):
            anchor_embedding = self.encode([anchor])[0]
        else:
            anchor_embedding = anchor
        
        # Compute similarity
        if method == 'cosine':
            # Normalize
            text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            anchor_norm = np.linalg.norm(anchor_embedding)
            
            # Cosine similarity
            similarities = (text_embeddings @ anchor_embedding) / (text_norms.flatten() * anchor_norm + 1e-8)
        
        elif method == 'dot':
            similarities = text_embeddings @ anchor_embedding
        
        elif method == 'euclidean':
            distances = np.linalg.norm(text_embeddings - anchor_embedding, axis=1)
            similarities = 1 / (1 + distances)  # Convert distance to similarity
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return similarities
    
    def get_anchor_embedding(self, concept: str) -> np.ndarray:
        """
        Get embedding for a concept.
        
        Parameters
        ----------
        concept : str
            Concept description (e.g., "joy and happiness")
            
        Returns
        -------
        embedding : ndarray
            Concept embedding
        """
        return self.encode([concept])[0]
    
    def _get_cache_key(self, texts):
        """Generate cache key from texts"""
        # Simple hash of texts
        text_hash = hash(tuple(texts[:10]))  # Use first 10 for key
        return f"{self.model_name}_{text_hash}.pkl"
    
    def _load_from_cache(self, cache_key):
        """Load embeddings from cache"""
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def _save_to_cache(self, cache_key, embeddings):
        """Save embeddings to cache"""
        cache_path = self.cache_dir / cache_key
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
        except:
            pass  # Silent fail on cache errors


# Global singleton instance
_global_embedder = None

def get_default_embedder():
    """Get or create default embedding manager"""
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = EmbeddingManager()
    return _global_embedder

