"""
Integrate Existing Domains

Batch integrates all existing domain work into the new system.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import register_domain, get_domain_registry
from src.data import DataLoader


def scan_existing_domains():
    """Scan for existing domain work."""
    domains_dir = Path(__file__).parent.parent / 'narrative_optimization' / 'domains'
    
    existing_domains = []
    
    for domain_dir in domains_dir.iterdir():
        if domain_dir.is_dir() and not domain_dir.name.startswith('_'):
            existing_domains.append(domain_dir.name)
    
    return existing_domains


def integrate_domain(domain_name: str):
    """Integrate a single existing domain."""
    domains_dir = Path(__file__).parent.parent / 'narrative_optimization' / 'domains'
    domain_dir = domains_dir / domain_name
    
    if not domain_dir.exists():
        return False
    
    # Look for results files
    result_files = [
        'migrated_results.json',
        'integration_results.json',
        f'{domain_name}_results.json',
        'framework_results.json'
    ]
    
    for result_file in result_files:
        result_path = domain_dir / result_file
        
        if result_path.exists():
            try:
                with open(result_path) as f:
                    results = json.load(f)
                
                # Extract metadata
                pi = results.get('narrativity', results.get('pi', results.get('п', 0.5)))
                r_squared = results.get('r_squared', results.get('r2', None))
                delta = results.get('delta', results.get('Д', None))
                
                # Register in registry
                register_domain(
                    name=domain_name,
                    pi=pi,
                    domain_type='unknown',  # Would need to infer
                    status='integrated',
                    r_squared=r_squared,
                    delta=delta
                )
                
                return True
                
            except Exception as e:
                print(f"  ⚠ {domain_name}: Error reading {result_file} - {e}")
                continue
    
    return False


def integrate_all_existing():
    """Integrate all existing domain work."""
    print("="*80)
    print("INTEGRATING EXISTING DOMAIN WORK")
    print("="*80)
    
    existing = scan_existing_domains()
    print(f"\nFound {len(existing)} existing domains\n")
    
    integrated = 0
    skipped = 0
    
    for domain in existing:
        print(f"Integrating {domain}...", end=" ")
        
        if integrate_domain(domain):
            print("✓")
            integrated += 1
        else:
            print("⊙ No results found")
            skipped += 1
    
    print(f"\n{'='*80}")
    print(f"INTEGRATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Integrated: {integrated}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(existing)}")
    
    # Show registry
    print(f"\n{'='*80}")
    print("UPDATED REGISTRY")
    print(f"{'='*80}\n")
    
    registry = get_domain_registry()
    stats = registry.get_statistics()
    
    print(f"Total domains: {stats.get('total_domains', 0)}")
    print(f"Average π: {stats.get('avg_pi', 0):.3f}")
    
    if stats.get('total_domains', 0) > 0:
        print("\nRegistered domains:")
        for domain in sorted(registry.get_all_domains(), key=lambda d: d.pi, reverse=True)[:10]:
            r2_str = f"R²={domain.r_squared:.1%}" if domain.r_squared else ""
            print(f"  {domain.name:20s} π={domain.pi:.2f}  {r2_str}")


if __name__ == '__main__':
    integrate_all_existing()

