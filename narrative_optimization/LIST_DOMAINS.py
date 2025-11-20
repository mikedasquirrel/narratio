"""
Domain Listing Tool

Lists all registered domains with their characteristics.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.registry import get_domain_registry


def list_all_domains():
    """List all registered domains."""
    registry = get_domain_registry()
    
    print("="*80)
    print("REGISTERED DOMAINS")
    print("="*80)
    
    stats = registry.get_statistics()
    
    print(f"\nTotal Domains: {stats.get('total_domains', 0)}")
    print(f"Average π: {stats.get('avg_pi', 0):.3f}")
    print(f"High Performers (R² ≥ 70%): {stats.get('high_performers', 0)}")
    
    if 'total_samples' in stats:
        print(f"Total Samples: {stats['total_samples']:,}")
    
    print(f"\n{'='*80}")
    print("BY TYPE")
    print(f"{'='*80}\n")
    
    by_type = stats.get('by_type', {})
    for domain_type, count in sorted(by_type.items()):
        print(f"  {domain_type}: {count} domains")
    
    print(f"\n{'='*80}")
    print("ALL DOMAINS (sorted by π)")
    print(f"{'='*80}\n")
    
    domains = sorted(registry.get_all_domains(), key=lambda d: d.pi, reverse=True)
    
    for domain in domains:
        r2_str = f"R²={domain.r_squared:.1%}" if domain.r_squared else "R²=N/A"
        d_str = f"Д={domain.delta:.3f}" if domain.delta else "Д=N/A"
        n_str = f"n={domain.n_samples:,}" if domain.n_samples else "n=N/A"
        
        print(f"  {domain.name:20s} π={domain.pi:.2f}  {r2_str:12s} {d_str:10s} {n_str:12s} [{domain.status}]")
        
        if domain.similar_domains:
            print(f"    Similar to: {', '.join(domain.similar_domains[:3])}")
    
    print(f"\n{'='*80}")


def search_domains(query: str):
    """Search for domains matching query."""
    registry = get_domain_registry()
    
    query_lower = query.lower()
    matches = []
    
    for domain in registry.get_all_domains():
        if (query_lower in domain.name.lower() or
            query_lower in domain.domain_type.lower()):
            matches.append(domain)
    
    print(f"\nFound {len(matches)} matches for '{query}':\n")
    
    for domain in matches:
        print(f"  - {domain.name} (π={domain.pi:.2f}, type={domain.domain_type})")


def compare_domains(domain1: str, domain2: str):
    """Compare two domains."""
    registry = get_domain_registry()
    
    d1 = registry.get_domain(domain1)
    d2 = registry.get_domain(domain2)
    
    if not d1 or not d2:
        print(f"Domain not found: {domain1 if not d1 else domain2}")
        return
    
    print(f"\n{'='*80}")
    print(f"COMPARING: {domain1} vs {domain2}")
    print(f"{'='*80}\n")
    
    print(f"{'Characteristic':<25} {domain1:<20} {domain2:<20}")
    print("-" * 70)
    print(f"{'π (Narrativity)':<25} {d1.pi:<20.3f} {d2.pi:<20.3f}")
    print(f"{'Domain Type':<25} {d1.domain_type:<20} {d2.domain_type:<20}")
    
    if d1.r_squared and d2.r_squared:
        print(f"{'R²':<25} {d1.r_squared:<20.1%} {d2.r_squared:<20.1%}")
    
    if d1.delta and d2.delta:
        print(f"{'Д (Bridge)':<25} {d1.delta:<20.3f} {d2.delta:<20.3f}")
    
    if d1.n_samples and d2.n_samples:
        print(f"{'Samples':<25} {d1.n_samples:<20,} {d2.n_samples:<20,}")
    
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Domain registry tools')
    parser.add_argument('--list', action='store_true', help='List all domains')
    parser.add_argument('--search', type=str, help='Search for domains')
    parser.add_argument('--compare', nargs=2, metavar=('DOMAIN1', 'DOMAIN2'), help='Compare two domains')
    
    args = parser.parse_args()
    
    if args.list or not any([args.search, args.compare]):
        list_all_domains()
    
    if args.search:
        search_domains(args.search)
    
    if args.compare:
        compare_domains(args.compare[0], args.compare[1])

