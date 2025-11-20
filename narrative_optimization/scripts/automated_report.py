"""
Automated Reporting

Generates comprehensive system reports automatically.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import get_domain_registry
from src.optimization import get_global_cache, get_global_profiler


def generate_system_report() -> str:
    """Generate complete system status report."""
    report = "# Narrative Optimization System Report\n\n"
    report += f"**Generated**: {datetime.now().isoformat()}\n\n"
    report += "---\n\n"
    
    # Registry status
    registry = get_domain_registry()
    stats = registry.get_statistics()
    
    report += "## Domain Registry\n\n"
    report += f"- **Total Domains**: {stats.get('total_domains', 0)}\n"
    report += f"- **Average π**: {stats.get('avg_pi', 0):.3f}\n"
    report += f"- **High Performers**: {stats.get('high_performers', 0)} (R² ≥ 70%)\n"
    report += f"- **Total Samples**: {stats.get('total_samples', 0):,}\n\n"
    
    # By type
    report += "### By Type\n\n"
    by_type = stats.get('by_type', {})
    for domain_type, count in sorted(by_type.items()):
        report += f"- {domain_type}: {count}\n"
    report += "\n"
    
    # Cache performance
    cache = get_global_cache()
    cache_stats = cache.get_stats()
    
    report += "## Cache Performance\n\n"
    report += f"- **Hit Rate**: {cache_stats.get('hit_rate', 0):.1%}\n"
    report += f"- **Size**: {cache_stats.get('cache_size_mb', 0):.1f}MB / 1000MB\n"
    report += f"- **Items**: {cache_stats.get('n_items', 0)}\n"
    report += f"- **Hits**: {cache_stats.get('hits', 0):,}\n"
    report += f"- **Misses**: {cache_stats.get('misses', 0):,}\n\n"
    
    # Profiler summary
    profiler = get_global_profiler()
    bottlenecks = profiler.get_bottlenecks(5)
    
    if len(bottlenecks) > 0:
        report += "## Performance Bottlenecks\n\n"
        for i, (name, total_time) in enumerate(bottlenecks, 1):
            report += f"{i}. {name}: {total_time:.3f}s total\n"
        report += "\n"
    
    # Domain performance
    report += "## Top Performing Domains\n\n"
    
    top_domains = registry.get_high_performers(min_r_squared=0.7)
    top_domains_sorted = sorted(top_domains, key=lambda d: d.r_squared if d.r_squared else 0, reverse=True)
    
    for domain in top_domains_sorted[:10]:
        r2 = domain.r_squared if domain.r_squared else 0
        delta = domain.delta if domain.delta else 0
        report += f"- **{domain.name}**: R²={r2:.1%}, Д={delta:.3f}, π={domain.pi:.2f}\n"
    
    report += "\n"
    
    # Learning status
    pipeline_state = Path(__file__).parent.parent / 'pipeline_state.json'
    
    if pipeline_state.exists():
        with open(pipeline_state) as f:
            state = json.load(f)
        
        report += "## Learning System Status\n\n"
        report += f"- **Iteration**: {state.get('iteration', 0)}\n"
        report += f"- **Domains Learned**: {len(state.get('domains', []))}\n"
        report += f"- **Last Update**: {state.get('timestamp', 'Unknown')}\n\n"
    
    # System health
    report += "## System Health\n\n"
    
    health_issues = []
    
    # Check cache hit rate
    if cache_stats.get('hit_rate', 0) < 0.4:
        health_issues.append("Low cache hit rate (< 40%)")
    
    # Check for errors in domains
    error_domains = [d for d in registry.get_all_domains() if d.status == 'error']
    if len(error_domains) > 0:
        health_issues.append(f"{len(error_domains)} domains with errors")
    
    if len(health_issues) == 0:
        report += "✓ System is healthy\n"
    else:
        report += "**Issues Detected**:\n\n"
        for issue in health_issues:
            report += f"- {issue}\n"
    
    report += "\n---\n\n"
    report += "*Auto-generated report*\n"
    
    return report


def generate_and_save_report():
    """Generate and save system report."""
    print("="*80)
    print("GENERATING SYSTEM REPORT")
    print("="*80)
    
    print("\nCollecting metrics...")
    report = generate_system_report()
    
    # Save report
    output_path = Path(__file__).parent.parent / 'SYSTEM_REPORT.md'
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report saved: {output_path}")
    
    # Also print to console
    print("\n" + "="*80)
    print(report)


if __name__ == '__main__':
    generate_and_save_report()

