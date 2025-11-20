"""
Performance Monitoring

Monitors system performance and generates alerts.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import get_domain_registry
from src.optimization import get_global_profiler, get_global_cache


class PerformanceMonitor:
    """
    Monitor system performance over time.
    
    Tracks:
    - Domain performance (R², Д)
    - Learning improvements
    - Cache efficiency
    - Processing speed
    - Pattern quality
    """
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        
    def collect_metrics(self) -> Dict:
        """Collect current performance metrics."""
        registry = get_domain_registry()
        cache = get_global_cache()
        profiler = get_global_profiler()
        
        # Registry metrics
        stats = registry.get_statistics()
        
        # Cache metrics
        cache_stats = cache.get_stats()
        
        # Profiler metrics
        profiler_summary = profiler.get_summary()
        bottlenecks = profiler.get_bottlenecks(5)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'registry': {
                'total_domains': stats.get('total_domains', 0),
                'avg_pi': stats.get('avg_pi', 0),
                'high_performers': stats.get('high_performers', 0)
            },
            'cache': {
                'hit_rate': cache_stats.get('hit_rate', 0),
                'size_mb': cache_stats.get('cache_size_mb', 0),
                'n_items': cache_stats.get('n_items', 0)
            },
            'profiler': {
                'total_functions': len(profiler_summary),
                'bottlenecks': [{'name': name, 'time': time} for name, time in bottlenecks]
            }
        }
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def check_for_issues(self, metrics: Dict) -> List[str]:
        """
        Check for performance issues.
        
        Parameters
        ----------
        metrics : dict
            Current metrics
        
        Returns
        -------
        list
            List of issues/alerts
        """
        alerts = []
        
        # Cache hit rate too low
        cache_hit_rate = metrics['cache'].get('hit_rate', 0)
        if cache_hit_rate < 0.5:
            alerts.append(f"LOW_CACHE_HIT_RATE: {cache_hit_rate:.1%} < 50%")
        
        # Cache size too large
        cache_size = metrics['cache'].get('size_mb', 0)
        if cache_size > 900:  # Near limit
            alerts.append(f"CACHE_SIZE_HIGH: {cache_size:.0f}MB")
        
        # Check bottlenecks
        bottlenecks = metrics['profiler'].get('bottlenecks', [])
        if len(bottlenecks) > 0:
            slowest = bottlenecks[0]
            if slowest['time'] > 10.0:
                alerts.append(f"SLOW_FUNCTION: {slowest['name']} takes {slowest['time']:.1f}s")
        
        return alerts
    
    def generate_report(self) -> str:
        """Generate performance monitoring report."""
        report = "# Performance Monitoring Report\n\n"
        report += f"**Generated**: {datetime.now().isoformat()}\n\n"
        report += "---\n\n"
        
        if len(self.metrics_history) == 0:
            report += "No metrics collected yet.\n"
            return report
        
        latest = self.metrics_history[-1]
        
        # Registry status
        report += "## Registry Status\n\n"
        report += f"- Total domains: {latest['registry']['total_domains']}\n"
        report += f"- Average π: {latest['registry']['avg_pi']:.3f}\n"
        report += f"- High performers: {latest['registry']['high_performers']}\n\n"
        
        # Cache status
        report += "## Cache Performance\n\n"
        report += f"- Hit rate: {latest['cache']['hit_rate']:.1%}\n"
        report += f"- Size: {latest['cache']['size_mb']:.1f}MB\n"
        report += f"- Items: {latest['cache']['n_items']}\n\n"
        
        # Bottlenecks
        report += "## Top Bottlenecks\n\n"
        bottlenecks = latest['profiler'].get('bottlenecks', [])
        for i, bottleneck in enumerate(bottlenecks[:5], 1):
            report += f"{i}. {bottleneck['name']}: {bottleneck['time']:.2f}s\n"
        report += "\n"
        
        # Alerts
        if len(self.alerts) > 0:
            report += "## Alerts\n\n"
            for alert in self.alerts[-10:]:  # Last 10
                report += f"- {alert}\n"
        else:
            report += "## Alerts\n\nNo active alerts.\n"
        
        return report
    
    def save_report(self, path: Path = None):
        """Save monitoring report."""
        if path is None:
            path = Path(__file__).parent.parent / 'PERFORMANCE_REPORT.md'
        
        report = self.generate_report()
        
        with open(path, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved: {path}")


def monitor_once():
    """Run monitoring once."""
    monitor = PerformanceMonitor()
    
    print("="*80)
    print("PERFORMANCE MONITORING")
    print("="*80)
    
    print("\nCollecting metrics...")
    metrics = monitor.collect_metrics()
    
    print("\nChecking for issues...")
    alerts = monitor.check_for_issues(metrics)
    
    if len(alerts) > 0:
        print(f"\n⚠ {len(alerts)} alerts:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("\n✓ No issues detected")
    
    print("\nGenerating report...")
    monitor.save_report()
    
    print("\n✓ Monitoring complete")


if __name__ == '__main__':
    monitor_once()

