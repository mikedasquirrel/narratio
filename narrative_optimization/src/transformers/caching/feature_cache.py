"""
Feature Cache for Narrative Transformers

Disk-based caching system with version tracking and automatic invalidation.
Enables iterative transformer development without re-extracting features.

Author: Narrative Integration System
Date: November 2025
"""

import hashlib
import inspect
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import joblib


class FeatureCache:
    """
    Intelligent caching system for transformer outputs.
    
    Features:
    - Disk-based persistence using joblib
    - Version tracking based on transformer code hash
    - Automatic invalidation when transformer changes
    - Partial cache updates (add new transformers without re-running old ones)
    - Cache statistics tracking
    
    Cache Key Format:
    (domain_name, transformer_id, data_hash, transformer_version)
    
    Parameters
    ----------
    cache_dir : str or Path
        Directory to store cached features
    verbose : bool, default=True
        Whether to print cache hit/miss information
    
    Examples
    --------
    >>> cache = FeatureCache(cache_dir='data/features/cache')
    >>> 
    >>> # Try to get cached features
    >>> features = cache.get('nba', 'nominative', texts_hash, transformer_obj)
    >>> 
    >>> if features is None:
    >>>     # Cache miss - compute features
    >>>     features = transformer.fit_transform(texts)
    >>>     cache.set('nba', 'nominative', texts_hash, transformer_obj, features)
    """
    
    def __init__(self, cache_dir: str = 'data/features/cache', verbose: bool = True):
        """Initialize feature cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'invalidations': 0
        }
        
        # Metadata file
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'created': datetime.now().isoformat(),
            'entries': {},
            'statistics': {
                'total_hits': 0,
                'total_misses': 0,
                'total_sets': 0
            }
        }
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_transformer_version(self, transformer: Any) -> str:
        """
        Get version hash of transformer based on its source code.
        
        Parameters
        ----------
        transformer : object
            Transformer instance
        
        Returns
        -------
        version : str
            MD5 hash of transformer's source code
        """
        try:
            # Get transformer class source code
            source = inspect.getsource(transformer.__class__)
            # Hash it
            version = hashlib.md5(source.encode()).hexdigest()[:8]
            return version
        except (TypeError, OSError):
            # Fallback: use class name + module
            return f"{transformer.__class__.__module__}.{transformer.__class__.__name__}"
    
    def _get_data_hash(self, data: Any) -> str:
        """
        Get hash of input data.
        
        Parameters
        ----------
        data : list or array
            Input data (texts or features)
        
        Returns
        -------
        hash : str
            MD5 hash of data
        """
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, list):
            # Convert list to string and hash
            data_str = ''.join(str(item) for item in data)
            data_bytes = data_str.encode()
        else:
            data_bytes = str(data).encode()
        
        return hashlib.md5(data_bytes).hexdigest()[:8]
    
    def _get_cache_key(
        self, 
        domain: str, 
        transformer_id: str, 
        data_hash: str,
        transformer_version: str
    ) -> str:
        """
        Generate cache key.
        
        Parameters
        ----------
        domain : str
            Domain name (e.g., 'nba', 'movies')
        transformer_id : str
            Transformer identifier (e.g., 'nominative')
        data_hash : str
            Hash of input data
        transformer_version : str
            Version hash of transformer code
        
        Returns
        -------
        key : str
            Cache key
        """
        return f"{domain}_{transformer_id}_{data_hash}_v{transformer_version}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(
        self,
        domain: str,
        transformer_id: str,
        data: Any,
        transformer: Any
    ) -> Optional[np.ndarray]:
        """
        Retrieve cached features.
        
        Parameters
        ----------
        domain : str
            Domain name
        transformer_id : str
            Transformer identifier
        data : list or array
            Input data (for hashing)
        transformer : object
            Transformer instance (for version tracking)
        
        Returns
        -------
        features : np.ndarray or None
            Cached features if found, None otherwise
        """
        data_hash = self._get_data_hash(data)
        transformer_version = self._get_transformer_version(transformer)
        cache_key = self._get_cache_key(domain, transformer_id, data_hash, transformer_version)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                features = joblib.load(cache_path)
                self.stats['hits'] += 1
                self.metadata['statistics']['total_hits'] += 1
                
                if self.verbose:
                    print(f"✓ Cache HIT: {domain}/{transformer_id} (v{transformer_version})")
                
                return features
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Cache read error: {e}")
                return None
        else:
            self.stats['misses'] += 1
            self.metadata['statistics']['total_misses'] += 1
            
            if self.verbose:
                print(f"✗ Cache MISS: {domain}/{transformer_id} (v{transformer_version})")
            
            return None
    
    def set(
        self,
        domain: str,
        transformer_id: str,
        data: Any,
        transformer: Any,
        features: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """
        Store features in cache.
        
        Parameters
        ----------
        domain : str
            Domain name
        transformer_id : str
            Transformer identifier
        data : list or array
            Input data (for hashing)
        transformer : object
            Transformer instance (for version tracking)
        features : np.ndarray
            Features to cache
        metadata : dict, optional
            Additional metadata to store
        """
        data_hash = self._get_data_hash(data)
        transformer_version = self._get_transformer_version(transformer)
        cache_key = self._get_cache_key(domain, transformer_id, data_hash, transformer_version)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Save features
            joblib.dump(features, cache_path, compress=3)
            
            # Update metadata
            entry = {
                'domain': domain,
                'transformer_id': transformer_id,
                'transformer_version': transformer_version,
                'data_hash': data_hash,
                'feature_shape': features.shape,
                'cached_at': datetime.now().isoformat(),
                'cache_key': cache_key
            }
            
            if metadata:
                entry['metadata'] = metadata
            
            self.metadata['entries'][cache_key] = entry
            self._save_metadata()
            
            self.stats['sets'] += 1
            self.metadata['statistics']['total_sets'] += 1
            
            if self.verbose:
                print(f"💾 Cached: {domain}/{transformer_id} → {features.shape}")
        
        except Exception as e:
            if self.verbose:
                print(f"⚠ Cache write error: {e}")
    
    def invalidate(self, domain: Optional[str] = None, transformer_id: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Parameters
        ----------
        domain : str, optional
            If provided, invalidate all entries for this domain
        transformer_id : str, optional
            If provided, invalidate all entries for this transformer
        """
        entries_to_remove = []
        
        for cache_key, entry in self.metadata['entries'].items():
            should_remove = False
            
            if domain and entry['domain'] == domain:
                should_remove = True
            
            if transformer_id and entry['transformer_id'] == transformer_id:
                should_remove = True
            
            if should_remove:
                entries_to_remove.append(cache_key)
        
        # Remove entries
        for cache_key in entries_to_remove:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
            
            del self.metadata['entries'][cache_key]
            self.stats['invalidations'] += 1
        
        self._save_metadata()
        
        if self.verbose and entries_to_remove:
            print(f"🗑 Invalidated {len(entries_to_remove)} cache entries")
    
    def clear_all(self):
        """Clear entire cache."""
        for cache_file in self.cache_dir.glob('*.pkl'):
            cache_file.unlink()
        
        self.metadata['entries'] = {}
        self._save_metadata()
        
        if self.verbose:
            print("🗑 Cleared all cache")
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns
        -------
        stats : dict
            Cache hit/miss statistics
        """
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0
        
        return {
            'session': self.stats,
            'hit_rate': f"{hit_rate:.1f}%",
            'total_entries': len(self.metadata['entries']),
            'all_time': self.metadata['statistics']
        }
    
    def list_entries(self, domain: Optional[str] = None) -> list:
        """
        List cache entries.
        
        Parameters
        ----------
        domain : str, optional
            Filter by domain
        
        Returns
        -------
        entries : list
            List of cache entries
        """
        entries = []
        for cache_key, entry in self.metadata['entries'].items():
            if domain is None or entry['domain'] == domain:
                entries.append(entry)
        
        return sorted(entries, key=lambda x: x['cached_at'], reverse=True)

