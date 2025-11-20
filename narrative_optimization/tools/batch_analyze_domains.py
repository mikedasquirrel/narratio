"""
Batch Domain Analysis

Analyzes multiple domains in batch mode.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader
from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer
from src.registry import get_domain_registry, register_domain


def analyze_single_domain(domain_name: str, data_path: Path) -> dict:
    """Analyze a single domain."""
    try:
        # Load data
        loader = DataLoader()
        data = loader.load(data_path)
        
        if not loader.validate_data(data):
            return {'domain': domain_name, 'status': 'invalid_data'}
        
        # Analyze
        analyzer = DomainSpecificAnalyzer(domain_name)
        results = analyzer.analyze_complete(
            texts=data['texts'],
            outcomes=data['outcomes'],
            names=data.get('names')
        )
        
        return {
            'domain': domain_name,
            'status': 'success',
            'r_squared': results['r_squared'],
            'delta': results['delta'],
            'efficiency': results['efficiency'],
            'n_samples': len(data['texts']),
            'narrativity': results['narrativity']
        }
        
    except Exception as e:
        return {
            'domain': domain_name,
            'status': 'error',
            'error': str(e)
        }


def batch_analyze(
    domains: list = None,
    parallel: bool = True,
    max_workers: int = 4
):
    """
    Batch analyze multiple domains.
    
    Parameters
    ----------
    domains : list, optional
        List of domain names (None = all in registry)
    parallel : bool
        Use parallel processing
    max_workers : int
        Max parallel workers
    """
    print("="*80)
    print("BATCH DOMAIN ANALYSIS")
    print("="*80)
    
    # Get domains to analyze
    if domains is None:
        registry = get_domain_registry()
        domains = [d.name for d in registry.get_all_domains()]
    
    print(f"\nAnalyzing {len(domains)} domains...")
    print(f"Parallel: {parallel} (workers={max_workers if parallel else 1})\n")
    
    # Find data files
    from src.pipeline_config import get_config
    config = get_config()
    
    domain_data_pairs = []
    for domain in domains:
        data_path = config.get_domain_data_path(domain)
        if data_path and data_path.exists():
            domain_data_pairs.append((domain, data_path))
    
    print(f"Found data for {len(domain_data_pairs)} domains\n")
    
    # Analyze
    results = {}
    
    if parallel and len(domain_data_pairs) > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(analyze_single_domain, domain, path): domain
                for domain, path in domain_data_pairs
            }
            
            for future in as_completed(futures):
                result = future.result()
                domain = result['domain']
                results[domain] = result
                
                if result['status'] == 'success':
                    print(f"  ✓ {domain:20s} R²={result['r_squared']:.1%}, Д={result['delta']:.3f}")
                else:
                    print(f"  ✗ {domain:20s} {result.get('error', result['status'])}")
    else:
        # Sequential processing
        for domain, data_path in domain_data_pairs:
            result = analyze_single_domain(domain, data_path)
            results[domain] = result
            
            if result['status'] == 'success':
                print(f"  ✓ {domain:20s} R²={result['r_squared']:.1%}, Д={result['delta']:.3f}")
            else:
                print(f"  ✗ {domain:20s} {result.get('error', result['status'])}")
    
    # Update registry
    print(f"\nUpdating registry...")
    for domain, result in results.items():
        if result['status'] == 'success':
            register_domain(
                name=domain,
                pi=result['narrativity'],
                domain_type='analyzed',
                status='validated',
                r_squared=result['r_squared'],
                delta=result['delta'],
                n_samples=result['n_samples']
            )
    
    # Save results
    output_path = Path(__file__).parent.parent / 'batch_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    success = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"  Success: {success}/{len(results)}")
    print(f"  Results saved: {output_path}")
    
    # Performance summary
    if success > 0:
        successful_results = [r for r in results.values() if r['status'] == 'success']
        avg_r2 = np.mean([r['r_squared'] for r in successful_results])
        avg_delta = np.mean([r['delta'] for r in successful_results])
        
        print(f"\nPerformance:")
        print(f"  Average R²: {avg_r2:.1%}")
        print(f"  Average Д: {avg_delta:.3f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch analyze domains')
    parser.add_argument('--domains', nargs='+', help='Specific domains to analyze')
    parser.add_argument('--sequential', action='store_true', help='Sequential (not parallel)')
    parser.add_argument('--workers', type=int, default=4, help='Max workers')
    
    args = parser.parse_args()
    
    batch_analyze(
        domains=args.domains,
        parallel=not args.sequential,
        max_workers=args.workers
    )

