"""
Caching Demo

Demonstrates the performance benefits of joblib caching in narrative pipelines.
Shows before/after comparisons and cache management.
"""

import numpy as np
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipelines import NarrativePipeline, CacheManager, get_cache_manager
from src.transformers import (
    NominativeAnalysisTransformer,
    LinguisticPatternsTransformer,
    StatisticalTransformer
)


def generate_sample_data(n_samples=1000):
    """Generate sample text data"""
    categories = ['alt.atheism', 'comp.graphics', 'sci.space']
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Use subset
    X = newsgroups.data[:n_samples]
    y = newsgroups.target[:n_samples]
    
    return X, y


def time_pipeline_run(pipeline, X, y, label="Run"):
    """Time a pipeline execution"""
    start = time.time()
    
    # Fit
    fit_start = time.time()
    pipeline.fit(X, y)
    fit_time = time.time() - fit_start
    
    # Transform
    transform_start = time.time()
    X_transformed = pipeline.transform(X)
    transform_time = time.time() - transform_start
    
    total_time = time.time() - start
    
    print(f"\n{label}:")
    print(f"  Fit:       {fit_time:.2f}s")
    print(f"  Transform: {transform_time:.2f}s")
    print(f"  Total:     {total_time:.2f}s")
    print(f"  Output shape: {X_transformed.shape}")
    
    return total_time


def demo_basic_caching():
    """Demonstrate basic caching benefits"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Caching Benefits")
    print("="*70)
    
    # Generate data
    print("\nGenerating sample data...")
    X, y = generate_sample_data(n_samples=500)
    print(f"Generated {len(X)} samples")
    
    # Create pipeline WITH caching
    print("\n--- WITH CACHING (cache_verbose=1) ---")
    cached_pipeline = NarrativePipeline(
        'demo_cached',
        hypothesis='Text patterns predict categories',
        use_cache=True,
        cache_verbose=1
    )
    
    # Add transformers
    cached_pipeline.add_step(
        'linguistic',
        LinguisticPatternsTransformer(),
        'Extract linguistic patterns'
    )
    cached_pipeline.add_step(
        'statistical',
        StatisticalTransformer(max_features=100),
        'Statistical baseline features'
    )
    
    # Build
    pipe_cached = cached_pipeline.build()
    
    # First run (no cache)
    print("\n🔄 First run (computing and caching)...")
    time1 = time_pipeline_run(pipe_cached, X, y, "First Run")
    
    # Second run (cached)
    print("\n⚡ Second run (using cache)...")
    time2 = time_pipeline_run(pipe_cached, X, y, "Second Run (Cached)")
    
    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"\n✨ SPEEDUP: {speedup:.1f}x faster with cache!")
    
    # Compare with non-cached
    print("\n\n--- WITHOUT CACHING ---")
    uncached_pipeline = NarrativePipeline(
        'demo_uncached',
        hypothesis='Text patterns predict categories',
        use_cache=False
    )
    
    uncached_pipeline.add_step(
        'linguistic',
        LinguisticPatternsTransformer(),
        'Extract linguistic patterns'
    )
    uncached_pipeline.add_step(
        'statistical',
        StatisticalTransformer(max_features=100),
        'Statistical baseline features'
    )
    
    pipe_uncached = uncached_pipeline.build()
    
    print("\n🔄 First run...")
    time3 = time_pipeline_run(pipe_uncached, X, y, "Uncached Run 1")
    
    print("\n🔄 Second run (still no cache)...")
    time4 = time_pipeline_run(pipe_uncached, X, y, "Uncached Run 2")
    
    print(f"\n📊 Uncached runs are similar: {time3:.2f}s vs {time4:.2f}s")


def demo_cache_management():
    """Demonstrate cache management tools"""
    print("\n" + "="*70)
    print("DEMO 2: Cache Management")
    print("="*70)
    
    # Get cache manager
    manager = get_cache_manager()
    
    # Print cache report
    manager.print_cache_report()
    
    # Show how to clear specific cache
    print("\nCache Management Commands:")
    print("-" * 70)
    print("# Clear specific pipeline:")
    print("  manager.clear_pipeline_cache('demo_cached')")
    print("\n# Clear caches older than 30 days:")
    print("  manager.clear_old_caches(days=30)")
    print("\n# Clear ALL caches (with confirmation):")
    print("  manager.clear_all_caches(confirm=True)")
    print("\n# Check if cache exists:")
    print("  manager.cache_exists('demo_cached')")
    

def demo_cache_invalidation():
    """Demonstrate automatic cache invalidation"""
    print("\n" + "="*70)
    print("DEMO 3: Automatic Cache Invalidation")
    print("="*70)
    
    print("\nJoblib automatically invalidates cache when:")
    print("  ✓ Transformer code changes")
    print("  ✓ Transformer parameters change")
    print("  ✓ Input data changes")
    print("\nThis ensures you never get stale results!")
    
    # Generate data
    X, y = generate_sample_data(n_samples=300)
    
    # Create pipeline with one config
    print("\n--- Pipeline with max_features=50 ---")
    pipeline1 = NarrativePipeline(
        'demo_invalidation',
        use_cache=True,
        cache_verbose=1
    )
    pipeline1.add_step(
        'statistical',
        StatisticalTransformer(max_features=50),
        'Statistical features'
    )
    pipe1 = pipeline1.build()
    
    time1 = time_pipeline_run(pipe1, X, y, "Config 1 (max_features=50)")
    
    # Create pipeline with different config
    print("\n--- Pipeline with max_features=100 (different config) ---")
    pipeline2 = NarrativePipeline(
        'demo_invalidation',
        use_cache=True,
        cache_verbose=1
    )
    pipeline2.add_step(
        'statistical',
        StatisticalTransformer(max_features=100),  # Different parameter!
        'Statistical features'
    )
    pipe2 = pipeline2.build()
    
    print("\n🔄 Cache is invalidated because parameter changed...")
    time2 = time_pipeline_run(pipe2, X, y, "Config 2 (max_features=100)")
    
    print("\n✓ Cache was recomputed with new configuration")


def demo_multiple_pipelines():
    """Demonstrate caching with multiple pipelines"""
    print("\n" + "="*70)
    print("DEMO 4: Multiple Independent Pipelines")
    print("="*70)
    
    print("\nEach pipeline gets its own cache namespace")
    
    X, y = generate_sample_data(n_samples=300)
    
    # Pipeline 1: Linguistic focus
    print("\n--- Pipeline 1: Linguistic Focus ---")
    pipeline1 = NarrativePipeline(
        'linguistic_narrative',
        hypothesis='Linguistic patterns matter',
        use_cache=True
    )
    pipeline1.add_step('linguistic', LinguisticPatternsTransformer(), 'Linguistic features')
    pipe1 = pipeline1.build()
    
    time_pipeline_run(pipe1, X, y, "Linguistic Pipeline")
    
    # Pipeline 2: Statistical focus
    print("\n--- Pipeline 2: Statistical Focus ---")
    pipeline2 = NarrativePipeline(
        'statistical_narrative',
        hypothesis='Statistical patterns matter',
        use_cache=True
    )
    pipeline2.add_step('statistical', StatisticalTransformer(max_features=100), 'Statistical features')
    pipe2 = pipeline2.build()
    
    time_pipeline_run(pipe2, X, y, "Statistical Pipeline")
    
    # Show caches
    manager = get_cache_manager()
    stats = manager.get_cache_stats()
    
    print("\n📊 Cache Status:")
    print(f"  Total pipelines cached: {stats['pipeline_count']}")
    print(f"  Total cache size: {stats['total_size_mb']:.1f} MB")
    for pipeline in stats['pipelines']:
        print(f"    • {pipeline['name']}: {pipeline['size_mb']:.2f} MB")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("NARRATIVE PIPELINE CACHING DEMO")
    print("="*70)
    print("\nThis demo shows how joblib caching dramatically improves")
    print("performance by memoizing expensive pipeline operations.")
    
    # Run demos
    demo_basic_caching()
    demo_cache_management()
    demo_cache_invalidation()
    demo_multiple_pipelines()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✨ Key Benefits:")
    print("  • 10-100x faster for repeated runs")
    print("  • Automatic invalidation on code/data changes")
    print("  • Separate caches per pipeline")
    print("  • Easy cache management")
    print("\n💡 Best Practices:")
    print("  • Use cache_verbose=1 for development")
    print("  • Name pipelines descriptively")
    print("  • Clear old caches periodically")
    print("  • Disable caching for one-off experiments")
    
    # Final cache report
    manager = get_cache_manager()
    manager.print_cache_report()
    
    print("\n✓ Demo complete!\n")

