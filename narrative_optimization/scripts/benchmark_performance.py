"""
Performance Benchmarking

Benchmarks system performance for optimization.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer
from src.learning import UniversalArchetypeLearner, DomainSpecificLearner
from src.data import DataLoader
from src.config import CompleteGenomeExtractor
from src.optimization import get_global_profiler


def benchmark_genome_extraction(n_samples: int = 100):
    """Benchmark genome extraction speed."""
    print("\n[1/5] Benchmarking genome extraction...")
    
    # Generate synthetic texts
    texts = [f"Sample narrative text {i} with various patterns" for i in range(n_samples)]
    outcomes = np.random.binomial(1, 0.5, n_samples)
    
    # Create extractor
    extractor = CompleteGenomeExtractor()
    extractor.fit(texts, outcomes)
    
    # Benchmark
    start = time.perf_counter()
    genomes = extractor.transform_batch(texts)
    end = time.perf_counter()
    
    elapsed = end - start
    per_sample = elapsed / n_samples * 1000  # ms
    
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per sample: {per_sample:.2f}ms")
    print(f"  Throughput: {n_samples/elapsed:.0f} samples/s")
    
    return per_sample


def benchmark_pattern_discovery(n_samples: int = 100):
    """Benchmark pattern discovery speed."""
    print("\n[2/5] Benchmarking pattern discovery...")
    
    texts = [
        f"Underdog entity {i} shows remarkable skill and determination"
        if i % 2 == 0
        else f"Dominant entity {i} continues winning streak"
        for i in range(n_samples)
    ]
    outcomes = np.random.binomial(1, 0.6, n_samples)
    
    # Universal discovery
    learner = UniversalArchetypeLearner()
    
    start = time.perf_counter()
    patterns = learner.discover_patterns(texts, outcomes, n_patterns=5)
    end = time.perf_counter()
    
    elapsed = end - start
    
    print(f"  Discovery time: {elapsed:.3f}s")
    print(f"  Patterns found: {len(patterns)}")
    print(f"  Time per pattern: {elapsed/len(patterns) if len(patterns) > 0 else 0:.3f}s")
    
    return elapsed


def benchmark_full_analysis(n_samples: int = 100):
    """Benchmark complete domain analysis."""
    print("\n[3/5] Benchmarking full analysis...")
    
    texts = [f"Complex narrative {i} with multiple patterns and entities" for i in range(n_samples)]
    outcomes = np.random.binomial(1, 0.5, n_samples)
    
    analyzer = DomainSpecificAnalyzer('golf')  # Use existing domain
    
    start = time.perf_counter()
    results = analyzer.analyze_complete(texts, outcomes)
    end = time.perf_counter()
    
    elapsed = end - start
    
    print(f"  Analysis time: {elapsed:.3f}s")
    print(f"  Per sample: {elapsed/n_samples*1000:.2f}ms")
    print(f"  R²: {results['r_squared']:.3f}")
    
    return elapsed


def benchmark_data_loading():
    """Benchmark data loading speed."""
    print("\n[4/5] Benchmarking data loading...")
    
    # Create temp data
    import tempfile
    import json
    
    test_data = {
        'texts': [f"Text {i}" for i in range(1000)],
        'outcomes': np.random.binomial(1, 0.5, 1000).tolist()
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)
    
    loader = DataLoader()
    
    start = time.perf_counter()
    data = loader.load(temp_path)
    end = time.perf_counter()
    
    elapsed = end - start
    
    # Cleanup
    temp_path.unlink()
    
    print(f"  Load time: {elapsed:.3f}s")
    print(f"  Samples loaded: {len(data['texts'])}")
    print(f"  Load rate: {len(data['texts'])/elapsed:.0f} samples/s")
    
    return elapsed


def benchmark_cache():
    """Benchmark cache operations."""
    print("\n[5/5] Benchmarking cache...")
    
    from src.optimization import get_global_cache
    
    cache = get_global_cache()
    
    # Benchmark writes
    start = time.perf_counter()
    for i in range(100):
        cache.set(f"key_{i}", {"data": f"value_{i}"})
    end = time.perf_counter()
    
    write_time = end - start
    
    # Benchmark reads
    start = time.perf_counter()
    for i in range(100):
        cache.get(f"key_{i}")
    end = time.perf_counter()
    
    read_time = end - start
    
    print(f"  Write time (100): {write_time:.3f}s ({write_time/100*1000:.2f}ms each)")
    print(f"  Read time (100): {read_time:.3f}s ({read_time/100*1000:.2f}ms each)")
    
    stats = cache.get_stats()
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    
    return write_time, read_time


def run_benchmark(n_samples: int = 100):
    """Run complete benchmark suite."""
    print("="*80)
    print("PERFORMANCE BENCHMARKING")
    print("="*80)
    print(f"\nSample size: {n_samples}")
    
    # Run benchmarks
    genome_time = benchmark_genome_extraction(n_samples)
    discovery_time = benchmark_pattern_discovery(n_samples)
    analysis_time = benchmark_full_analysis(n_samples)
    loading_time = benchmark_data_loading()
    write_time, read_time = benchmark_cache()
    
    # Summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Genome extraction: {genome_time:.2f}ms per sample")
    print(f"Pattern discovery: {discovery_time:.3f}s total")
    print(f"Full analysis: {analysis_time/n_samples*1000:.2f}ms per sample")
    print(f"Data loading: {loading_time:.3f}s for 1000 samples")
    print(f"Cache write: {write_time/100*1000:.2f}ms per item")
    print(f"Cache read: {read_time/100*1000:.2f}ms per item")
    
    # Performance assessment
    print(f"\n{'='*80}")
    print("ASSESSMENT")
    print(f"{'='*80}\n")
    
    if genome_time < 10:
        print("✓ Genome extraction: Excellent (< 10ms)")
    elif genome_time < 50:
        print("⚠ Genome extraction: Acceptable (< 50ms)")
    else:
        print("✗ Genome extraction: Slow (> 50ms) - needs optimization")
    
    if analysis_time / n_samples < 0.1:
        print("✓ Full analysis: Excellent (< 100ms per sample)")
    elif analysis_time / n_samples < 0.5:
        print("⚠ Full analysis: Acceptable (< 500ms per sample)")
    else:
        print("✗ Full analysis: Slow - needs optimization")
    
    print("\n✓ Benchmark complete")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance benchmarking')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples for benchmark')
    
    args = parser.parse_args()
    
    run_benchmark(args.samples)

