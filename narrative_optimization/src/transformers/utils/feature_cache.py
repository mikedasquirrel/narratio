"""
Feature Caching System

Caches transformer features for repeated analyses.
Provides near-instant feature extraction for previously seen texts.

Author: Narrative Integration System
Date: November 2025
"""

import hashlib
import pickle
import json
from pathlib import Path
from typing import Any, Optional, Dict
import numpy as np
from datetime import datetime
import threading

# Global cache instance
_cache_lock = threading.Lock()
_memory_cache: Dict[str, Dict[str, Any]] = {}
_cache_hits = 0
_cache_misses = 0


class FeatureCache:
    """
    Thread-safe feature cache with memory and disk persistence.
    
    Features:
    - Content-based hashing (MD5)
    - Memory cache for speed
    - Optional disk persistence
    - Automatic size management
    - Cache statistics
    
    Usage:
    ------
    cache = FeatureCache(cache_dir="./feature_cache")
    
    # Try to get cached features
    features = cache.get(transformer_name, text)
    if features is None:
        # Compute features
        features = transformer.transform([text])[0]
        # Cache for next time
        cache.set(transformer_name, text, features)
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_items: int = 10000,
        enable_disk_cache: bool = True
    ):
        """
        Initialize feature cache.
        
        Parameters
        ----------
        cache_dir : str, optional
            Directory for disk cache. If None, memory-only cache.
        max_memory_items : int
            Maximum items in memory cache before eviction
        enable_disk_cache : bool
            Whether to persist cache to disk
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items
        self.enable_disk_cache = enable_disk_cache
        
        if self.enable_disk_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.meta_file = self.cache_dir / "cache_meta.json"
            self._load_metadata()
        else:
            self.meta_file = None
            self.metadata = {'created': str(datetime.now()), 'hits': 0, 'misses': 0}
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.meta_file and self.meta_file.exists():
            with open(self.meta_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'created': str(datetime.now()), 'hits': 0, 'misses': 0}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        if self.meta_file:
            with open(self.meta_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
    
    def _compute_key(self, transformer_name: str, text: str) -> str:
        """
        Compute cache key from transformer name and text content.
        
        Parameters
        ----------
        transformer_name : str
            Name of transformer
        text : str
            Input text
            
        Returns
        -------
        key : str
            MD5 hash of transformer + text
        """
        content = f"{transformer_name}:{text}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def get(self, transformer_name: str, text: str) -> Optional[np.ndarray]:
        """
        Get cached features if available.
        
        Parameters
        ----------
        transformer_name : str
            Name of transformer
        text : str
            Input text
            
        Returns
        -------
        features : ndarray or None
            Cached features if found, None otherwise
        """
        global _memory_cache, _cache_hits, _cache_misses
        
        cache_key = self._compute_key(transformer_name, text)
        
        with _cache_lock:
            # Try memory cache first
            if cache_key in _memory_cache:
                _cache_hits += 1
                self.metadata['hits'] += 1
                return _memory_cache[cache_key]['features']
            
            # Try disk cache
            if self.enable_disk_cache and self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        
                        # Load into memory cache
                        _memory_cache[cache_key] = cached_data
                        
                        _cache_hits += 1
                        self.metadata['hits'] += 1
                        return cached_data['features']
                    except Exception:
                        # Corrupted cache file
                        cache_file.unlink()
            
            # Cache miss
            _cache_misses += 1
            self.metadata['misses'] += 1
            return None
    
    def set(self, transformer_name: str, text: str, features: np.ndarray):
        """
        Cache features for transformer and text.
        
        Parameters
        ----------
        transformer_name : str
            Name of transformer
        text : str
            Input text
        features : ndarray
            Computed features to cache
        """
        global _memory_cache
        
        cache_key = self._compute_key(transformer_name, text)
        
        cached_data = {
            'transformer': transformer_name,
            'features': features,
            'timestamp': str(datetime.now()),
            'text_length': len(text)
        }
        
        with _cache_lock:
            # Add to memory cache
            _memory_cache[cache_key] = cached_data
            
            # Evict oldest if over limit
            if len(_memory_cache) > self.max_memory_items:
                # Simple FIFO eviction
                oldest_key = next(iter(_memory_cache))
                del _memory_cache[oldest_key]
            
            # Save to disk
            if self.enable_disk_cache and self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cached_data, f)
                except Exception:
                    pass  # Fail silently on disk cache errors
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        stats : dict
            Cache hit rate, size, etc.
        """
        global _cache_hits, _cache_misses
        
        total_requests = _cache_hits + _cache_misses
        hit_rate = _cache_hits / total_requests if total_requests > 0 else 0.0
        
        disk_files = 0
        if self.enable_disk_cache and self.cache_dir:
            disk_files = len(list(self.cache_dir.glob("*.pkl")))
        
        return {
            'memory_items': len(_memory_cache),
            'disk_items': disk_files,
            'cache_hits': _cache_hits,
            'cache_misses': _cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def clear_memory(self):
        """Clear memory cache."""
        global _memory_cache
        with _cache_lock:
            _memory_cache.clear()
    
    def clear_disk(self):
        """Clear disk cache."""
        if self.enable_disk_cache and self.cache_dir:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            if self.meta_file and self.meta_file.exists():
                self.meta_file.unlink()
    
    def clear_all(self):
        """Clear both memory and disk cache."""
        self.clear_memory()
        self.clear_disk()


# Global default cache instance
_default_cache = None


def get_default_cache() -> FeatureCache:
    """Get or create default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = FeatureCache(
            cache_dir="./.narrative_cache",
            max_memory_items=10000,
            enable_disk_cache=True
        )
    return _default_cache


def use_feature_cache(cache: Optional[FeatureCache] = None):
    """
    Decorator to add caching to transformer.
    
    Usage:
    ------
    @use_feature_cache()
    class MyTransformer(BaseEstimator, TransformerMixin):
        def transform(self, X):
            # Automatically cached
            ...
    """
    if cache is None:
        cache = get_default_cache()
    
    def decorator(transformer_class):
        original_transform = transformer_class.transform
        
        def new_transform(self, X):
            # Get transformer name
            transformer_name = self.__class__.__name__
            
            # For single text, try cache
            if isinstance(X, str):
                cached = cache.get(transformer_name, X)
                if cached is not None:
                    return cached.reshape(1, -1)
                
                # Compute
                result = original_transform(self, [X])
                cache.set(transformer_name, X, result[0])
                return result
            
            # For list, cache each
            elif isinstance(X, (list, tuple)):
                results = []
                for text in X:
                    cached = cache.get(transformer_name, text)
                    if cached is not None:
                        results.append(cached)
                    else:
                        # Compute single item
                        result = original_transform(self, [text])[0]
                        cache.set(transformer_name, text, result)
                        results.append(result)
                return np.array(results)
            
            # Default: no caching
            return original_transform(self, X)
        
        transformer_class.transform = new_transform
        return transformer_class
    
    return decorator
