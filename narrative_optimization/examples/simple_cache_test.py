#!/usr/bin/env python3
"""
Simple Cache Test

Quick test to verify caching is working.
Run this twice - second run should be much faster!
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import numpy as np
from sklearn.datasets import make_classification

from src.pipelines import NarrativePipeline, get_cache_manager
from src.transformers import StatisticalTransformer


def main():
    print("\n" + "="*60)
    print("SIMPLE CACHE TEST")
    print("="*60)
    
    # Generate data
    print("\n1. Generating test data...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    # Convert to text-like format for testing
    X_text = [f"Sample text {i} with features " + " ".join(map(str, row[:5])) 
              for i, row in enumerate(X)]
    
    print(f"   Generated {len(X_text)} samples")
    
    # Create cached pipeline
    print("\n2. Creating pipeline with caching enabled...")
    pipeline = NarrativePipeline(
        'simple_cache_test',
        hypothesis='Testing cache performance',
        use_cache=True,
        cache_verbose=1  # Show cache activity
    )
    
    pipeline.add_step(
        'statistical',
        StatisticalTransformer(max_features=100),
        'Extract statistical features'
    )
    
    pipe = pipeline.build()
    print("   Pipeline created")
    
    # First run
    print("\n3. First run (computing and caching)...")
    start = time.time()
    X_transformed = pipe.fit_transform(X_text, y)
    first_time = time.time() - start
    
    print(f"   ✓ First run completed in {first_time:.2f}s")
    print(f"   Output shape: {X_transformed.shape}")
    
    # Second run
    print("\n4. Second run (using cache)...")
    start = time.time()
    X_transformed2 = pipe.transform(X_text)
    second_time = time.time() - start
    
    print(f"   ✓ Second run completed in {second_time:.2f}s")
    print(f"   Output shape: {X_transformed2.shape}")
    
    # Results
    speedup = first_time / second_time if second_time > 0 else float('inf')
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"First run:  {first_time:.2f}s")
    print(f"Second run: {second_time:.2f}s")
    print(f"Speedup:    {speedup:.1f}x faster! ⚡")
    
    if speedup > 5:
        print("\n✅ SUCCESS: Caching is working properly!")
    elif speedup > 1.5:
        print("\n⚠️  PARTIAL: Caching is working but not optimal")
    else:
        print("\n❌ ISSUE: Caching may not be working")
    
    # Cache status
    print("\n" + "="*60)
    print("CACHE STATUS")
    print("="*60)
    
    manager = get_cache_manager()
    
    if manager.cache_exists('simple_cache_test'):
        print("✓ Cache exists for 'simple_cache_test'")
        
        stats = manager.get_cache_stats()
        for pipeline_info in stats['pipelines']:
            if pipeline_info['name'] == 'simple_cache_test':
                print(f"  Size: {pipeline_info['size_mb']:.2f} MB")
                print(f"  Modified: {pipeline_info['modified']}")
    else:
        print("⚠️  No cache found")
    
    print("\n" + "="*60)
    print("\n💡 TIP: Run this script again to see cached performance!\n")


if __name__ == '__main__':
    main()

