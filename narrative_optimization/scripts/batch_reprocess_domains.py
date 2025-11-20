"""
Batch Re-Process All Domains with Phase 7 Transformers

Efficiently re-processes all 16 domains with new transformers (θ, λ, α, Ξ).
Uses caching to avoid re-computing existing transformers.

Usage:
    python narrative_optimization/scripts/batch_reprocess_domains.py

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.scripts.run_all_transformers import (
    DOMAIN_CONFIGS,
    load_domain_data,
    instantiate_transformers
)
from narrative_optimization.src.pipelines.cached_pipeline import CachedTransformerPipeline


def reprocess_domain(domain_name, config, cache_pipeline, project_root):
    """
    Re-process single domain with new transformers.
    
    Strategy:
    - Use cached features for existing 29 transformers
    - Only compute θ and λ (new transformers)
    - Combine all features
    """
    print(f"\n{'='*80}")
    print(f"RE-PROCESSING: {domain_name.upper()}")
    print(f"{'='*80}")
    
    # Load data
    data, outcomes = load_domain_data(domain_name, config, project_root)
    
    if data is None:
        print(f"⚠️  Skipping {domain_name} - data not available")
        return None
    
    # Get transformers (includes new Phase 7 transformers)
    transformers = instantiate_transformers(config['pi'])
    
    print(f"\nDomain π: {config['pi']:.2f}")
    print(f"Samples: {len(data)}")
    print(f"Transformers: {len(transformers)} total (29 existing + 2 new)")
    
    # Execute all transformers with caching
    print(f"\nExecuting transformers (cached ones will be instant)...")
    start_time = time.time()
    
    features, stats = cache_pipeline.execute_transformers(
        domain=domain_name,
        transformers=transformers,
        data=data,
        y=outcomes,
        force_recompute=False  # Use cache
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Completed in {elapsed:.2f}s")
    print(f"  Features: {features.shape}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Transformers cached: {stats['transformers_cached']}")
    print(f"  Transformers computed: {stats['transformers_computed']}")
    
    # Save features
    output_path = project_root / 'narrative_optimization' / 'data' / 'features' / f'{domain_name}_complete_features.npz'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        features=features,
        outcomes=outcomes,
        domain=domain_name,
        narrativity=config['pi'],
        n_transformers=len(transformers)
    )
    
    print(f"✓ Saved to: {output_path}")
    
    return {
        'domain': domain_name,
        'narrativity': config['pi'],
        'samples': len(data),
        'features': features.shape,
        'cache_hit_rate': stats['cache_hit_rate'],
        'processing_time': elapsed
    }


def main():
    """Re-process all domains"""
    print("="*80)
    print("BATCH RE-PROCESSING ALL DOMAINS")
    print("="*80)
    print(f"\nStrategy:")
    print(f"  - Use cached features for 29 existing transformers (instant)")
    print(f"  - Compute θ and λ only (new transformers, ~5s)")
    print(f"  - Total time per domain: ~5-10s")
    
    # Create cache pipeline
    cache_dir = project_root / 'narrative_optimization' / 'data' / 'features' / 'cache'
    cache_pipeline = CachedTransformerPipeline(cache_dir=str(cache_dir), verbose=True)
    
    # Process all domains
    results = []
    total_start = time.time()
    
    for domain_name, config in DOMAIN_CONFIGS.items():
        try:
            result = reprocess_domain(domain_name, config, cache_pipeline, project_root)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing {domain_name}: {e}")
            continue
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    
    print(f"\nProcessed: {len(results)} domains")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Average time per domain: {total_elapsed/len(results):.1f}s")
    
    # Cache efficiency
    avg_cache_hit_rate = np.mean([r['cache_hit_rate'] for r in results])
    print(f"\nCache efficiency:")
    print(f"  Average hit rate: {avg_cache_hit_rate:.1%}")
    print(f"  Time saved: ~{total_elapsed * avg_cache_hit_rate / 60:.1f} minutes")
    
    # Save summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'domains_processed': len(results),
        'total_time_seconds': total_elapsed,
        'avg_cache_hit_rate': avg_cache_hit_rate,
        'results': results
    }
    
    summary_path = project_root / 'narrative_optimization' / 'data' / 'batch_reprocessing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved summary to: {summary_path}")
    print("\n" + "="*80)
    print("✓ ALL DOMAINS RE-PROCESSED WITH PHASE 7 TRANSFORMERS")
    print("="*80)


if __name__ == '__main__':
    main()

