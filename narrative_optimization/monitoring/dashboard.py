"""
Monitoring Dashboard

Real-time monitoring dashboard for system status.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import time
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import get_domain_registry
from src.optimization import get_global_cache, get_global_profiler


class MonitoringDashboard:
    """
    Real-time system monitoring dashboard.
    
    Displays:
    - Domain count and status
    - Cache performance
    - Processing speed
    - Pattern quality
    - Learning progress
    """
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        
    def collect_snapshot(self) -> dict:
        """Collect current metrics snapshot."""
        registry = get_domain_registry()
        cache = get_global_cache()
        profiler = get_global_profiler()
        
        stats = registry.get_statistics()
        cache_stats = cache.get_stats()
        
        snapshot = {
            'timestamp': time.time(),
            'domains': stats.get('total_domains', 0),
            'avg_pi': stats.get('avg_pi', 0),
            'cache_hit_rate': cache_stats.get('hit_rate', 0),
            'cache_size_mb': cache_stats.get('cache_size_mb', 0)
        }
        
        self.metrics_history.append(snapshot)
        
        return snapshot
    
    def display_dashboard(self):
        """Display ASCII dashboard."""
        snapshot = self.collect_snapshot()
        
        # Clear screen (simplified)
        print("\033[2J\033[H")
        
        print("="*80)
        print("NARRATIVE OPTIMIZATION MONITORING DASHBOARD")
        print("="*80)
        print(f"\nLast Update: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Domain stats
        print("DOMAIN REGISTRY")
        print("-"*40)
        print(f"  Total Domains: {snapshot['domains']}")
        print(f"  Average π: {snapshot['avg_pi']:.3f}")
        print()
        
        # Cache stats
        print("CACHE PERFORMANCE")
        print("-"*40)
        print(f"  Hit Rate: {snapshot['cache_hit_rate']:.1%}")
        print(f"  Size: {snapshot['cache_size_mb']:.1f}MB / 1000MB")
        
        # Progress bar for cache
        cache_pct = min(100, snapshot['cache_size_mb'] / 10)
        bar_filled = int(cache_pct / 2)
        bar = "█" * bar_filled + "░" * (50 - bar_filled)
        print(f"  [{bar}] {cache_pct:.0f}%")
        print()
        
        # History
        if len(self.metrics_history) > 1:
            print("TREND (last 10 snapshots)")
            print("-"*40)
            
            recent = list(self.metrics_history)[-10:]
            
            # Cache hit rate trend
            hit_rates = [s['cache_hit_rate'] for s in recent]
            trend = "↗" if hit_rates[-1] > hit_rates[0] else "↘" if hit_rates[-1] < hit_rates[0] else "→"
            print(f"  Cache hit rate: {trend} {hit_rates[-1]:.1%}")
            
            # Domain count trend
            domains = [s['domains'] for s in recent]
            trend = "↗" if domains[-1] > domains[0] else "→"
            print(f"  Domains: {trend} {domains[-1]}")
        
        print()
        print("="*80)
        print("[Ctrl+C to exit]")
    
    def run_dashboard(self, refresh_interval: int = 5):
        """
        Run live dashboard.
        
        Parameters
        ----------
        refresh_interval : int
            Refresh interval in seconds
        """
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nDashboard stopped")


def run_monitoring():
    """Run monitoring dashboard."""
    dashboard = MonitoringDashboard()
    
    import argparse
    parser = argparse.ArgumentParser(description='Monitoring dashboard')
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval (seconds)')
    parser.add_argument('--once', action='store_true', help='Display once and exit')
    
    args = parser.parse_args()
    
    if args.once:
        dashboard.display_dashboard()
    else:
        dashboard.run_dashboard(refresh_interval=args.interval)


if __name__ == '__main__':
    run_monitoring()

