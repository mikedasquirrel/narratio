"""
Cache Management System

Caches computed results for fast retrieval.

Author: Narrative Integration System
Date: November 2025
"""

import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps
import time


class CacheManager:
    """
    Manages caching for expensive computations.
    
    Features:
    - Disk-based caching
    - LRU eviction
    - TTL (time-to-live)
    - Cache statistics
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_mb: int = 1000,
        default_ttl: Optional[int] = None
    ):
        if cache_dir is None:
            cache_dir = Path.home() / '.narrative_optimization' / 'cache'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl  # seconds
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def _get_meta_path(self, key: str) -> Path:
        """Get metadata file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta.json"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Parameters
        ----------
        key : str
            Cache key
        
        Returns
        -------
        Any or None
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        if not cache_path.exists():
            self.misses += 1
            return None
        
        # Check TTL
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            
            if 'expires_at' in meta:
                if time.time() > meta['expires_at']:
                    # Expired
                    self._remove(key)
                    self.misses += 1
                    return None
        
        # Load value
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            self.hits += 1
            return value
        except Exception:
            self.misses += 1
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Store value in cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        ttl : int, optional
            Time-to-live in seconds
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        # Save value
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f)
        
        # Save metadata
        meta = {
            'key': key,
            'created_at': time.time()
        }
        
        ttl = ttl or self.default_ttl
        if ttl:
            meta['expires_at'] = time.time() + ttl
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
        
        # Check size
        self._enforce_size_limit()
    
    def _remove(self, key: str):
        """Remove key from cache."""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        if cache_path.exists():
            cache_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
    
    def _enforce_size_limit(self):
        """Enforce cache size limit using LRU."""
        # Get total size
        total_size = sum(
            f.stat().st_size
            for f in self.cache_dir.glob('*.pkl')
        )
        
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Get files sorted by access time
            files = list(self.cache_dir.glob('*.pkl'))
            files.sort(key=lambda f: f.stat().st_atime)
            
            # Remove oldest files until under limit
            for f in files:
                if total_size <= max_size_bytes:
                    break
                
                size = f.stat().st_size
                f.unlink()
                
                # Remove metadata
                meta_path = f.with_suffix('.meta.json')
                if meta_path.exists():
                    meta_path.unlink()
                
                total_size -= size
    
    def clear(self):
        """Clear all cache."""
        for f in self.cache_dir.glob('*'):
            f.unlink()
        
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        cache_size = sum(
            f.stat().st_size
            for f in self.cache_dir.glob('*.pkl')
        )
        cache_size_mb = cache_size / (1024 * 1024)
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size_mb': cache_size_mb,
            'n_items': len(list(self.cache_dir.glob('*.pkl')))
        }


def cached(cache_manager: CacheManager, ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Parameters
    ----------
    cache_manager : CacheManager
        Cache manager instance
    ttl : int, optional
        Time-to-live in seconds
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = "_".join(key_parts)
            
            # Try cache
            cached_value = cache_manager.get(key)
            if cached_value is not None:
                return cached_value
            
            # Compute
            result = func(*args, **kwargs)
            
            # Cache
            cache_manager.set(key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache instance
_global_cache = None

def get_global_cache() -> CacheManager:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache

