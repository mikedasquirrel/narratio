"""
Cache Management Tool

Command-line utility for managing the feature extraction cache.

Commands:
- list: Show cache status
- stats: Display cache statistics
- clear DOMAIN: Clear specific domain cache
- clear-all: Clear entire cache
- rebuild TRANSFORMER: Rebuild transformer across all domains
- info: Show cache directory info

Usage:
    python narrative_optimization/scripts/manage_cache.py --list
    python narrative_optimization/scripts/manage_cache.py --clear nba
    python narrative_optimization/scripts/manage_cache.py --stats

Author: Narrative Integration System
Date: November 2025
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.src.transformers.caching.feature_cache import FeatureCache


def format_size(bytes_size):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def list_cache(cache: FeatureCache):
    """List all cache entries."""
    print("\n" + "="*80)
    print("CACHE ENTRIES")
    print("="*80 + "\n")
    
    entries = cache.list_entries()
    
    if not entries:
        print("  No cache entries found.")
        return
    
    # Group by domain
    by_domain = {}
    for entry in entries:
        domain = entry['domain']
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(entry)
    
    print(f"Total entries: {len(entries)}\n")
    print(f"Domains: {len(by_domain)}\n")
    
    for domain, domain_entries in sorted(by_domain.items()):
        print(f"  {domain} ({len(domain_entries)} transformers):")
        
        for entry in domain_entries[:5]:  # Show first 5
            transformer = entry['transformer_id']
            shape = entry['feature_shape']
            cached_at = entry['cached_at'][:19]  # Trim to date+time
            
            print(f"    • {transformer:30s} {shape} @ {cached_at}")
        
        if len(domain_entries) > 5:
            print(f"    ... and {len(domain_entries) - 5} more")
        
        print()


def show_stats(cache: FeatureCache):
    """Show cache statistics."""
    print("\n" + "="*80)
    print("CACHE STATISTICS")
    print("="*80 + "\n")
    
    stats = cache.get_stats()
    
    print("Session Stats:")
    print(f"  • Hits: {stats['session']['hits']}")
    print(f"  • Misses: {stats['session']['misses']}")
    print(f"  • Hit Rate: {stats['hit_rate']}")
    print(f"  • Sets: {stats['session']['sets']}")
    
    print("\nAll-Time Stats:")
    print(f"  • Total Hits: {stats['all_time']['total_hits']}")
    print(f"  • Total Misses: {stats['all_time']['total_misses']}")
    print(f"  • Total Sets: {stats['all_time']['total_sets']}")
    
    print("\nCache Entries:")
    print(f"  • Total entries: {stats['total_entries']}")
    
    # Calculate cache size
    cache_dir = cache.cache_dir
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.glob('*.pkl'))
        print(f"  • Disk usage: {format_size(total_size)}")
    
    # Group by domain
    entries = cache.list_entries()
    by_domain = {}
    for entry in entries:
        domain = entry['domain']
        by_domain[domain] = by_domain.get(domain, 0) + 1
    
    print(f"\nDomains in cache: {len(by_domain)}")
    for domain, count in sorted(by_domain.items()):
        print(f"  • {domain:20s}: {count:3d} transformers")


def show_info(cache: FeatureCache):
    """Show cache directory information."""
    print("\n" + "="*80)
    print("CACHE INFORMATION")
    print("="*80 + "\n")
    
    cache_dir = cache.cache_dir
    
    print(f"Cache Directory: {cache_dir}")
    print(f"Exists: {cache_dir.exists()}")
    
    if cache_dir.exists():
        # Count files
        pkl_files = list(cache_dir.glob('*.pkl'))
        json_files = list(cache_dir.glob('*.json'))
        
        print(f"\nFiles:")
        print(f"  • Cached features (.pkl): {len(pkl_files)}")
        print(f"  • Metadata (.json): {len(json_files)}")
        
        # Total size
        total_size = sum(f.stat().st_size for f in cache_dir.glob('*'))
        print(f"  • Total size: {format_size(total_size)}")
        
        # Oldest and newest
        if pkl_files:
            oldest = min(pkl_files, key=lambda f: f.stat().st_mtime)
            newest = max(pkl_files, key=lambda f: f.stat().st_mtime)
            
            oldest_time = datetime.fromtimestamp(oldest.stat().st_mtime)
            newest_time = datetime.fromtimestamp(newest.stat().st_mtime)
            
            print(f"\nTimestamps:")
            print(f"  • Oldest entry: {oldest_time}")
            print(f"  • Newest entry: {newest_time}")


def clear_domain(cache: FeatureCache, domain: str):
    """Clear cache for specific domain."""
    print(f"\nClearing cache for domain: {domain}")
    
    # Get count before
    entries_before = len([e for e in cache.list_entries() if e['domain'] == domain])
    
    if entries_before == 0:
        print(f"  No entries found for {domain}")
        return
    
    # Confirm
    response = input(f"  Delete {entries_before} cache entries for {domain}? (y/N): ")
    
    if response.lower() == 'y':
        cache.invalidate(domain=domain)
        print(f"  ✅ Cleared {entries_before} entries")
    else:
        print("  Cancelled")


def clear_all(cache: FeatureCache):
    """Clear entire cache."""
    print("\n⚠️  WARNING: This will delete ALL cache entries!")
    
    entries = cache.list_entries()
    print(f"   Total entries to delete: {len(entries)}")
    
    cache_dir = cache.cache_dir
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.glob('*.pkl'))
        print(f"   Disk space to be freed: {format_size(total_size)}")
    
    response = input("\n   Type 'DELETE' to confirm: ")
    
    if response == 'DELETE':
        cache.clear_all()
        print("   ✅ Cache cleared")
    else:
        print("   Cancelled")


def rebuild_transformer(cache: FeatureCache, transformer_name: str):
    """Rebuild specific transformer across all domains."""
    print(f"\nRebuilding transformer: {transformer_name}")
    
    # Get affected entries
    entries = [e for e in cache.list_entries() if transformer_name.lower() in e['transformer_id'].lower()]
    
    if not entries:
        print(f"  No entries found for transformer: {transformer_name}")
        return
    
    print(f"  Found {len(entries)} entries across {len(set(e['domain'] for e in entries))} domains")
    
    response = input(f"  Invalidate these entries? (y/N): ")
    
    if response.lower() == 'y':
        cache.invalidate(transformer_id=transformer_name)
        print(f"  ✅ Invalidated {len(entries)} entries")
        print(f"  💡 Run run_all_transformers.py to rebuild")
    else:
        print("  Cancelled")


def main():
    """Main CLI execution."""
    parser = argparse.ArgumentParser(
        description='Manage narrative transformer feature cache',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_cache.py --list          # List all cache entries
  python manage_cache.py --stats         # Show cache statistics
  python manage_cache.py --info          # Show cache directory info
  python manage_cache.py --clear nba     # Clear NBA domain cache
  python manage_cache.py --clear-all     # Clear entire cache
  python manage_cache.py --rebuild phonetic  # Rebuild phonetic transformer
        """
    )
    
    parser.add_argument('--list', action='store_true', help='List all cache entries')
    parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--info', action='store_true', help='Show cache directory info')
    parser.add_argument('--clear', metavar='DOMAIN', help='Clear cache for specific domain')
    parser.add_argument('--clear-all', action='store_true', help='Clear entire cache')
    parser.add_argument('--rebuild', metavar='TRANSFORMER', help='Rebuild transformer across all domains')
    parser.add_argument('--cache-dir', default='narrative_optimization/data/features/cache',
                       help='Cache directory (default: narrative_optimization/data/features/cache)')
    
    args = parser.parse_args()
    
    # Initialize cache
    cache_dir = project_root / args.cache_dir
    cache = FeatureCache(cache_dir=str(cache_dir), verbose=False)
    
    # Execute command
    if args.list:
        list_cache(cache)
    elif args.stats:
        show_stats(cache)
    elif args.info:
        show_info(cache)
    elif args.clear:
        clear_domain(cache, args.clear)
    elif args.clear_all:
        clear_all(cache)
    elif args.rebuild:
        rebuild_transformer(cache, args.rebuild)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

