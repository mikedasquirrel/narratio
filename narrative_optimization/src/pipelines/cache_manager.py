"""
Cache Management Utilities

Centralized tools for managing pipeline and transformer caches.
Provides cache inspection, clearing, and statistics.
"""

from pathlib import Path
from typing import Dict, List, Optional
import shutil
from datetime import datetime


class CacheManager:
    """
    Manages all narrative pipeline caches.
    
    Provides utilities to:
    - Inspect cache sizes and contents
    - Clear caches selectively or globally
    - Get cache statistics
    - Monitor cache performance
    
    Examples
    --------
    >>> manager = CacheManager()
    >>> 
    >>> # Get cache statistics
    >>> stats = manager.get_cache_stats()
    >>> print(f"Total cache size: {stats['total_size_mb']:.1f} MB")
    >>> 
    >>> # Clear old caches
    >>> manager.clear_old_caches(days=30)
    >>> 
    >>> # Clear specific pipeline
    >>> manager.clear_pipeline_cache('my_narrative')
    """
    
    def __init__(self, base_cache_dir: Optional[str] = None):
        """
        Initialize cache manager.
        
        Parameters
        ----------
        base_cache_dir : str, optional
            Base directory for all caches.
            Defaults to ~/.narrative_cache
        """
        if base_cache_dir is None:
            self.base_cache_dir = Path.home() / '.narrative_cache'
        else:
            self.base_cache_dir = Path(base_cache_dir)
    
    def get_cache_stats(self) -> Dict:
        """
        Get comprehensive cache statistics.
        
        Returns
        -------
        stats : dict
            Cache statistics including:
            - total_size_mb: Total cache size in MB
            - pipeline_count: Number of cached pipelines
            - oldest_cache: Date of oldest cache
            - newest_cache: Date of newest cache
            - pipelines: List of cached pipeline names with sizes
        """
        stats = {
            'total_size_mb': 0.0,
            'pipeline_count': 0,
            'oldest_cache': None,
            'newest_cache': None,
            'pipelines': []
        }
        
        if not self.base_cache_dir.exists():
            return stats
        
        # Scan pipeline caches
        pipeline_dir = self.base_cache_dir / 'pipelines'
        if pipeline_dir.exists():
            for pipeline_cache in pipeline_dir.iterdir():
                if pipeline_cache.is_dir():
                    size_mb = self._get_dir_size(pipeline_cache) / (1024 * 1024)
                    mtime = datetime.fromtimestamp(pipeline_cache.stat().st_mtime)
                    
                    stats['pipelines'].append({
                        'name': pipeline_cache.name,
                        'size_mb': size_mb,
                        'modified': mtime
                    })
                    
                    stats['total_size_mb'] += size_mb
                    stats['pipeline_count'] += 1
                    
                    if stats['oldest_cache'] is None or mtime < stats['oldest_cache']:
                        stats['oldest_cache'] = mtime
                    if stats['newest_cache'] is None or mtime > stats['newest_cache']:
                        stats['newest_cache'] = mtime
        
        # Add other cache types (features, embeddings, etc.)
        for cache_type in ['features', 'embeddings']:
            cache_path = self.base_cache_dir / cache_type
            if cache_path.exists():
                size_mb = self._get_dir_size(cache_path) / (1024 * 1024)
                stats['total_size_mb'] += size_mb
        
        # Sort pipelines by size
        stats['pipelines'].sort(key=lambda x: x['size_mb'], reverse=True)
        
        return stats
    
    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except Exception:
            pass
        return total
    
    def print_cache_report(self):
        """Print formatted cache report."""
        stats = self.get_cache_stats()
        
        print("\n" + "="*70)
        print("NARRATIVE PIPELINE CACHE REPORT")
        print("="*70)
        print(f"\nCache Location: {self.base_cache_dir}")
        print(f"Total Size: {stats['total_size_mb']:.1f} MB")
        print(f"Pipeline Caches: {stats['pipeline_count']}")
        
        if stats['oldest_cache']:
            print(f"Oldest Cache: {stats['oldest_cache'].strftime('%Y-%m-%d %H:%M')}")
        if stats['newest_cache']:
            print(f"Newest Cache: {stats['newest_cache'].strftime('%Y-%m-%d %H:%M')}")
        
        if stats['pipelines']:
            print("\n" + "-"*70)
            print("CACHED PIPELINES (sorted by size):")
            print("-"*70)
            print(f"{'Pipeline Name':<40} {'Size (MB)':<12} {'Last Modified'}")
            print("-"*70)
            
            for pipeline in stats['pipelines']:
                print(f"{pipeline['name']:<40} {pipeline['size_mb']:>10.2f}  "
                      f"{pipeline['modified'].strftime('%Y-%m-%d %H:%M')}")
        
        print("="*70 + "\n")
    
    def clear_pipeline_cache(self, pipeline_name: str) -> bool:
        """
        Clear cache for a specific pipeline.
        
        Parameters
        ----------
        pipeline_name : str
            Name of the pipeline to clear
        
        Returns
        -------
        success : bool
            True if cache was cleared successfully
        """
        cache_path = self.base_cache_dir / 'pipelines' / pipeline_name
        
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                print(f"✓ Cleared cache for pipeline: {pipeline_name}")
                return True
            except Exception as e:
                print(f"✗ Failed to clear cache for {pipeline_name}: {e}")
                return False
        else:
            print(f"No cache found for pipeline: {pipeline_name}")
            return False
    
    def clear_old_caches(self, days: int = 30) -> int:
        """
        Clear caches older than specified days.
        
        Parameters
        ----------
        days : int
            Clear caches older than this many days
        
        Returns
        -------
        cleared_count : int
            Number of caches cleared
        """
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        cleared = 0
        
        pipeline_dir = self.base_cache_dir / 'pipelines'
        if pipeline_dir.exists():
            for pipeline_cache in pipeline_dir.iterdir():
                if pipeline_cache.is_dir():
                    mtime = datetime.fromtimestamp(pipeline_cache.stat().st_mtime)
                    if mtime < cutoff:
                        try:
                            shutil.rmtree(pipeline_cache)
                            print(f"✓ Cleared old cache: {pipeline_cache.name} "
                                  f"(last modified {mtime.strftime('%Y-%m-%d')})")
                            cleared += 1
                        except Exception as e:
                            print(f"✗ Failed to clear {pipeline_cache.name}: {e}")
        
        print(f"\nCleared {cleared} cache(s) older than {days} days")
        return cleared
    
    def clear_all_caches(self, confirm: bool = False):
        """
        Clear ALL narrative caches.
        
        **WARNING: This deletes all cached data!**
        
        Parameters
        ----------
        confirm : bool
            Must be True to actually clear caches (safety check)
        """
        if not confirm:
            print("\n⚠️  WARNING: This will delete ALL cached data!")
            print("   Call with confirm=True to proceed.")
            return
        
        if self.base_cache_dir.exists():
            try:
                shutil.rmtree(self.base_cache_dir)
                print(f"✓ Cleared all caches at {self.base_cache_dir}")
            except Exception as e:
                print(f"✗ Failed to clear caches: {e}")
        else:
            print("No caches found to clear.")
    
    def get_pipeline_cache_path(self, pipeline_name: str) -> Path:
        """
        Get the cache path for a specific pipeline.
        
        Parameters
        ----------
        pipeline_name : str
            Name of the pipeline
        
        Returns
        -------
        cache_path : Path
            Path to the pipeline's cache directory
        """
        return self.base_cache_dir / 'pipelines' / pipeline_name
    
    def cache_exists(self, pipeline_name: str) -> bool:
        """
        Check if cache exists for a pipeline.
        
        Parameters
        ----------
        pipeline_name : str
            Name of the pipeline
        
        Returns
        -------
        exists : bool
            True if cache exists
        """
        cache_path = self.get_pipeline_cache_path(pipeline_name)
        return cache_path.exists() and any(cache_path.iterdir())


# Global cache manager instance
_global_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get or create global cache manager."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager

