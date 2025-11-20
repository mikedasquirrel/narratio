"""
Performance Profiling

Profiles code performance to identify bottlenecks.

Author: Narrative Integration System
Date: November 2025
"""

import time
import functools
from typing import Callable, Dict, List
from collections import defaultdict
import numpy as np


class PerformanceProfiler:
    """
    Profile performance of functions and code blocks.
    
    Features:
    - Function timing
    - Call counts
    - Memory usage tracking
    - Bottleneck identification
    """
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.memory_usage = defaultdict(list)
        
    def time_function(self, name: str):
        """
        Decorator to time a function.
        
        Parameters
        ----------
        name : str
            Function name for profiling
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                
                elapsed = end - start
                self.timings[name].append(elapsed)
                self.call_counts[name] += 1
                
                return result
            return wrapper
        return decorator
    
    def profile_block(self, name: str):
        """
        Context manager for profiling code blocks.
        
        Parameters
        ----------
        name : str
            Block name
        
        Examples
        --------
        >>> with profiler.profile_block('my_computation'):
        ...     # expensive computation
        ...     pass
        """
        class ProfileContext:
            def __init__(self, profiler, block_name):
                self.profiler = profiler
                self.block_name = block_name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.perf_counter()
                elapsed = end_time - self.start_time
                self.profiler.timings[self.block_name].append(elapsed)
                self.profiler.call_counts[self.block_name] += 1
        
        return ProfileContext(self, name)
    
    def get_summary(self) -> Dict[str, Dict]:
        """
        Get profiling summary.
        
        Returns
        -------
        dict
            name -> statistics
        """
        summary = {}
        
        for name in self.timings.keys():
            times = self.timings[name]
            
            summary[name] = {
                'count': self.call_counts[name],
                'total_time': sum(times),
                'mean_time': np.mean(times),
                'median_time': np.median(times),
                'std_time': np.std(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        return summary
    
    def get_bottlenecks(self, n: int = 5) -> List[tuple]:
        """
        Identify top bottlenecks.
        
        Parameters
        ----------
        n : int
            Number of bottlenecks to return
        
        Returns
        -------
        list of (name, total_time)
            Top bottlenecks
        """
        bottlenecks = []
        
        for name in self.timings.keys():
            total_time = sum(self.timings[name])
            bottlenecks.append((name, total_time))
        
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks[:n]
    
    def print_report(self):
        """Print profiling report."""
        print("\n" + "="*80)
        print("PERFORMANCE PROFILING REPORT")
        print("="*80 + "\n")
        
        summary = self.get_summary()
        
        # Sort by total time
        sorted_items = sorted(
            summary.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        
        for name, stats in sorted_items:
            print(f"{name}:")
            print(f"  Calls: {stats['count']}")
            print(f"  Total: {stats['total_time']:.4f}s")
            print(f"  Mean: {stats['mean_time']:.4f}s")
            print(f"  Median: {stats['median_time']:.4f}s")
            print(f"  Std: {stats['std_time']:.4f}s")
            print(f"  Range: [{stats['min_time']:.4f}s, {stats['max_time']:.4f}s]")
            print()
        
        print("\nTop Bottlenecks:")
        for i, (name, total_time) in enumerate(self.get_bottlenecks(5), 1):
            print(f"  {i}. {name}: {total_time:.4f}s")
    
    def reset(self):
        """Reset profiling data."""
        self.timings.clear()
        self.call_counts.clear()
        self.memory_usage.clear()


# Global profiler instance
_global_profiler = None

def get_global_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile(name: str):
    """
    Decorator using global profiler.
    
    Parameters
    ----------
    name : str
        Function name for profiling
    """
    profiler = get_global_profiler()
    return profiler.time_function(name)

