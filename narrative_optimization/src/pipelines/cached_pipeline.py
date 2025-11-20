"""
Cached Pipeline for Transformer Execution

Wrapper that checks cache before calling transformer.fit_transform().
Aggregates cached features from multiple transformers efficiently.

Author: Narrative Integration System
Date: November 2025
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# Handle imports for different execution contexts
try:
    # Try absolute import first (when running as module)
    from src.transformers.caching.feature_cache import FeatureCache
except ImportError:
    try:
        # Try relative import (when running as package)
        from ..transformers.caching.feature_cache import FeatureCache
    except ImportError:
        try:
            # Try without src prefix
            from transformers.caching.feature_cache import FeatureCache
        except ImportError:
            # Last resort: add path and try again
            import sys
            from pathlib import Path
            # Add narrative_optimization to path
            nar_opt_path = Path(__file__).parent.parent.parent
            if str(nar_opt_path) not in sys.path:
                sys.path.insert(0, str(nar_opt_path))
            try:
                from src.transformers.caching.feature_cache import FeatureCache
            except ImportError:
                # If still fails, create a minimal stub
                print("⚠ Warning: FeatureCache not found, using minimal stub")
                class FeatureCache:
                    def __init__(self, *args, **kwargs):
                        pass
                    def get(self, *args, **kwargs):
                        return None
                    def set(self, *args, **kwargs):
                        pass


class CachedTransformerPipeline:
    """
    Pipeline that caches transformer outputs for fast re-runs.
    
    Features:
    - Checks cache before executing transformers
    - Aggregates features from multiple transformers
    - Tracks cache hit/miss statistics
    - Supports partial updates (add new transformers without re-running old)
    
    Parameters
    ----------
    cache_dir : str or Path, default='data/features/cache'
        Directory for cache storage
    verbose : bool, default=True
        Whether to print progress
    
    Examples
    --------
    >>> pipeline = CachedTransformerPipeline()
    >>> 
    >>> # Execute transformers with caching
    >>> features, stats = pipeline.execute_transformers(
    ...     domain='nba',
    ...     transformers=[nominative, phonetic, emotional],
    ...     data=texts
    ... )
    >>> 
    >>> print(f"Cache hit rate: {stats['hit_rate']}")
    """
    
    def __init__(
        self, 
        cache_dir: str = 'data/features/cache',
        verbose: bool = True
    ):
        """Initialize cached pipeline."""
        self.cache = FeatureCache(cache_dir=cache_dir, verbose=verbose)
        self.verbose = verbose
        
        # Execution statistics
        self.execution_stats = {
            'transformers_run': 0,
            'transformers_cached': 0,
            'total_time': 0.0,
            'cache_time_saved': 0.0
        }
    
    def execute_transformer(
        self,
        domain: str,
        transformer: Any,
        data: Any,
        y: Optional[Any] = None,
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Execute single transformer with caching.
        
        Parameters
        ----------
        domain : str
            Domain name
        transformer : object
            Transformer instance
        data : list or array
            Input data
        y : array, optional
            Target labels
        force_recompute : bool, default=False
            If True, ignore cache and recompute
        
        Returns
        -------
        features : np.ndarray
            Extracted features
        stats : dict
            Execution statistics
        """
        transformer_id = transformer.narrative_id if hasattr(transformer, 'narrative_id') else transformer.__class__.__name__
        
        start_time = time.time()
        
        # Try cache first
        if not force_recompute:
            features = self.cache.get(domain, transformer_id, data, transformer)
            
            if features is not None:
                elapsed = time.time() - start_time
                self.execution_stats['transformers_cached'] += 1
                
                return features, {
                    'cached': True,
                    'time': elapsed,
                    'transformer': transformer_id,
                    'shape': features.shape
                }
        
        # Cache miss - compute features
        if self.verbose:
            print(f"  ⚙️  Computing {transformer_id}...")
        
        try:
            features = transformer.fit_transform(data, y)
            
            # Ensure 2D array
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            
            elapsed = time.time() - start_time
            
            # Cache the result
            self.cache.set(
                domain=domain,
                transformer_id=transformer_id,
                data=data,
                transformer=transformer,
                features=features,
                metadata={
                    'computation_time': elapsed,
                    'n_samples': features.shape[0],
                    'n_features': features.shape[1]
                }
            )
            
            self.execution_stats['transformers_run'] += 1
            self.execution_stats['total_time'] += elapsed
            
            return features, {
                'cached': False,
                'time': elapsed,
                'transformer': transformer_id,
                'shape': features.shape
            }
        
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  Error in {transformer_id}: {e}")
            
            # Return empty features on error
            return np.zeros((len(data), 0)), {
                'cached': False,
                'time': time.time() - start_time,
                'transformer': transformer_id,
                'error': str(e),
                'shape': (len(data), 0)
            }
    
    def execute_transformers(
        self,
        domain: str,
        transformers: List[Any],
        data: Any,
        y: Optional[Any] = None,
        force_recompute: bool = False,
        skip_on_error: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Execute multiple transformers and aggregate features.
        
        Parameters
        ----------
        domain : str
            Domain name
        transformers : list
            List of transformer instances
        data : list or array
            Input data
        y : array, optional
            Target labels
        force_recompute : bool, default=False
            If True, ignore cache and recompute all
        skip_on_error : bool, default=True
            If True, skip transformers that error instead of failing
        
        Returns
        -------
        features : np.ndarray
            Aggregated feature matrix (n_samples, total_features)
        stats : dict
            Execution statistics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Executing {len(transformers)} transformers on {domain}")
            print(f"{'='*60}\n")
        
        all_features = []
        transformer_stats = []
        feature_names = []
        
        start_time = time.time()
        
        for i, transformer in enumerate(transformers, 1):
            transformer_id = transformer.narrative_id if hasattr(transformer, 'narrative_id') else transformer.__class__.__name__
            
            if self.verbose:
                print(f"[{i}/{len(transformers)}] {transformer_id}")
            
            features, stats = self.execute_transformer(
                domain=domain,
                transformer=transformer,
                data=data,
                y=y,
                force_recompute=force_recompute
            )
            
            if 'error' in stats and not skip_on_error:
                raise RuntimeError(f"Transformer {transformer_id} failed: {stats['error']}")
            
            if features.shape[1] > 0:  # Only include if features were generated
                all_features.append(features)
                transformer_stats.append(stats)
                
                # Get feature names if available
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        names = transformer.get_feature_names_out()
                        feature_names.extend(names)
                    except:
                        feature_names.extend([f"{transformer_id}_{j}" for j in range(features.shape[1])])
                else:
                    feature_names.extend([f"{transformer_id}_{j}" for j in range(features.shape[1])])
        
        # Aggregate features
        if all_features:
            aggregated_features = np.hstack(all_features)
        else:
            aggregated_features = np.zeros((len(data), 0))
        
        total_time = time.time() - start_time
        
        # Compute statistics
        n_cached = sum(1 for s in transformer_stats if s.get('cached', False))
        n_computed = len(transformer_stats) - n_cached
        cache_hit_rate = (n_cached / len(transformer_stats) * 100) if transformer_stats else 0
        
        total_computation_time = sum(s['time'] for s in transformer_stats if not s.get('cached', False))
        estimated_time_without_cache = sum(s['time'] for s in transformer_stats)
        time_saved = estimated_time_without_cache - total_time if n_cached > 0 else 0
        
        stats = {
            'domain': domain,
            'n_transformers': len(transformers),
            'n_cached': n_cached,
            'n_computed': n_computed,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'total_features': aggregated_features.shape[1],
            'n_samples': aggregated_features.shape[0],
            'total_time': total_time,
            'computation_time': total_computation_time,
            'time_saved': time_saved,
            'speedup': f"{estimated_time_without_cache / total_time:.1f}x" if total_time > 0 else "N/A",
            'transformer_stats': transformer_stats,
            'feature_names': feature_names,
            'completed_at': datetime.now().isoformat()
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ Completed: {domain}")
            print(f"   • Features: {aggregated_features.shape}")
            print(f"   • Cache hits: {n_cached}/{len(transformer_stats)} ({cache_hit_rate:.1f}%)")
            print(f"   • Time: {total_time:.1f}s (saved {time_saved:.1f}s)")
            if time_saved > 0:
                print(f"   • Speedup: {estimated_time_without_cache / total_time:.1f}x")
            print(f"{'='*60}\n")
        
        return aggregated_features, stats
    
    def save_features(
        self,
        features: np.ndarray,
        stats: Dict,
        output_path: Path,
        feature_names: Optional[List[str]] = None
    ):
        """
        Save aggregated features to disk.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix
        stats : dict
            Execution statistics
        output_path : Path
            Output file path (.npz format)
        feature_names : list, optional
            Feature names
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as compressed npz
        save_dict = {
            'features': features,
            'stats': np.array([stats], dtype=object),
        }
        
        if feature_names:
            save_dict['feature_names'] = np.array(feature_names, dtype=object)
        
        np.savez_compressed(output_path, **save_dict)
        
        if self.verbose:
            print(f"💾 Saved features: {output_path}")
    
    def load_features(self, input_path: Path) -> Tuple[np.ndarray, Dict, Optional[List[str]]]:
        """
        Load saved features from disk.
        
        Parameters
        ----------
        input_path : Path
            Input file path (.npz format)
        
        Returns
        -------
        features : np.ndarray
            Feature matrix
        stats : dict
            Execution statistics
        feature_names : list or None
            Feature names if available
        """
        data = np.load(input_path, allow_pickle=True)
        
        features = data['features']
        stats = data['stats'].item() if 'stats' in data else {}
        feature_names = data['feature_names'].tolist() if 'feature_names' in data else None
        
        return features, stats, feature_names
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            'cache': cache_stats,
            'execution': self.execution_stats
        }
    
    def clear_cache(self, domain: Optional[str] = None):
        """
        Clear cache.
        
        Parameters
        ----------
        domain : str, optional
            If provided, clear only this domain's cache
        """
        if domain:
            self.cache.invalidate(domain=domain)
        else:
            self.cache.clear_all()
