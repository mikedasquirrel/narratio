"""
Safe Batch Re-Process All Domains with Phase 7 Transformers

Memory-efficient version that processes domains one at a time with progress tracking.
Uses caching and handles large datasets safely.

Usage:
    python narrative_optimization/scripts/batch_reprocess_domains_safe.py

Author: Narrative Integration System
Date: November 2025
"""

import sys
import json
import numpy as np
from pathlib import Path
import time
import gc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from narrative_optimization.src.transformers import (
    # Phase 7 - NEW
    AwarenessResistanceTransformer,
    FundamentalConstraintsTransformer,
    # Core for validation
    NominativeAnalysisTransformer,
    StatisticalTransformer
)

# Domain configurations (simplified - only active domains)
ACTIVE_DOMAINS = [
    ('nba', 0.49, 'data/domains/nba_games_with_nominative.json'),
    ('nfl', 0.57, 'data/domains/nfl_complete_dataset.json'),
    ('tennis', 0.75, 'data/domains/tennis_matches.json'),
    ('golf', 0.70, 'data/domains/golf_tournaments.json'),
    ('ufc', 0.722, 'data/domains/ufc_fights.json'),
    ('mental_health', 0.55, 'data/domains/mental_health_complete_200.json'),
    ('startups', 0.76, 'data/domains/startups_with_outcomes.json'),
    ('movies', 0.65, 'data/domains/imdb_movies.json'),
    ('music', 0.702, 'data/domains/spotify_tracks.json'),
]


def process_domain_incremental(domain_name, pi, data_path, project_root):
    """
    Process single domain with only new transformers.
    
    Strategy: Only compute θ and λ (new), skip existing transformers.
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {domain_name.upper()}")
    print(f"{'='*80}")
    
    full_path = project_root / 'narrative_optimization' / data_path
    
    if not full_path.exists():
        print(f"⚠️  Data not found: {full_path}")
        return None
    
    try:
        # Load data
        print(f"Loading data...")
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            narratives = [item.get('narrative', '') or item.get('text', '') for item in data]
            outcomes = [item.get('outcome', 0) for item in data]
        else:
            print("⚠️  Unexpected data format")
            return None
        
        # Limit to first 5000 for memory efficiency
        if len(narratives) > 5000:
            print(f"  Limiting to 5000 samples (out of {len(narratives)})")
            narratives = narratives[:5000]
            outcomes = outcomes[:5000]
        
        print(f"✓ Loaded {len(narratives)} samples")
        
        # Extract NEW features only (θ and λ)
        print(f"\nExtracting Phase 7 features...")
        
        # θ (Awareness Resistance)
        print(f"  [1/2] Computing θ (Awareness Resistance)...")
        theta_transformer = AwarenessResistanceTransformer()
        theta_features = theta_transformer.fit_transform(narratives)
        theta_values = theta_features[:, 14]
        
        print(f"    ✓ θ: Mean={theta_values.mean():.3f}, Std={theta_values.std():.3f}")
        
        # λ (Fundamental Constraints)
        print(f"  [2/2] Computing λ (Fundamental Constraints)...")
        lambda_transformer = FundamentalConstraintsTransformer()
        lambda_features = lambda_transformer.fit_transform(narratives)
        lambda_values = lambda_features[:, 11]
        
        print(f"    ✓ λ: Mean={lambda_values.mean():.3f}, Std={lambda_values.std():.3f}")
        
        # Combine new features
        new_features = np.hstack([theta_features, lambda_features])
        
        print(f"\n✓ Extracted {new_features.shape[1]} new features from Phase 7")
        
        # Save ONLY new features (existing features already cached separately)
        output_dir = project_root / 'narrative_optimization' / 'data' / 'features' / 'phase7'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f'{domain_name}_phase7_features.npz'
        np.savez_compressed(
            output_path,
            theta_features=theta_features,
            lambda_features=lambda_features,
            theta_values=theta_values,
            lambda_values=lambda_values,
            domain=domain_name,
            narrativity=pi,
            n_samples=len(narratives)
        )
        
        print(f"✓ Saved Phase 7 features to: {output_path}")
        
        # Clean up memory
        del theta_features, lambda_features, narratives
        gc.collect()
        
        return {
            'domain': domain_name,
            'narrativity': pi,
            'samples': len(outcomes),
            'theta_mean': float(theta_values.mean()),
            'lambda_mean': float(lambda_values.mean()),
            'success': True
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Process all active domains safely"""
    print("="*80)
    print("SAFE BATCH RE-PROCESSING (PHASE 7 TRANSFORMERS ONLY)")
    print("="*80)
    print(f"\nStrategy:")
    print(f"  - Extract ONLY θ and λ (new Phase 7 transformers)")
    print(f"  - Save separately for later combination")
    print(f"  - Process one domain at a time")
    print(f"  - Limit to 5000 samples per domain for memory efficiency")
    print(f"\nProcessing {len(ACTIVE_DOMAINS)} domains...")
    
    results = []
    total_start = time.time()
    
    for i, (domain_name, pi, data_path) in enumerate(ACTIVE_DOMAINS, 1):
        print(f"\n[{i}/{len(ACTIVE_DOMAINS)}] Processing {domain_name}...")
        
        result = process_domain_incremental(domain_name, pi, data_path, project_root)
        if result:
            results.append(result)
        
        # Force garbage collection between domains
        gc.collect()
        time.sleep(0.5)
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    
    print(f"\nProcessed: {len(results)}/{len(ACTIVE_DOMAINS)} domains")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    
    if results:
        print(f"Average time per domain: {total_elapsed/len(results):.1f}s")
        
        # Force statistics
        print(f"\nForce Statistics Across Domains:")
        print(f"{'Domain':<20} π     θ(mean) λ(mean)")
        print("-" * 50)
        for r in results:
            print(f"{r['domain']:<20} {r['narrativity']:.2f}  {r['theta_mean']:.3f}   {r['lambda_mean']:.3f}")
        
        # Save summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'domains_processed': len(results),
            'total_time_seconds': total_elapsed,
            'results': results
        }
        
        summary_path = project_root / 'narrative_optimization' / 'data' / 'phase7_processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Saved summary to: {summary_path}")
    
    print("\n" + "="*80)
    print("✓ PHASE 7 FEATURES EXTRACTED FOR ALL DOMAINS")
    print("="*80)
    print("\nNext step: Combine Phase 7 features with existing cached features")
    print("Location: data/features/phase7/{domain}_phase7_features.npz")


if __name__ == '__main__':
    main()

