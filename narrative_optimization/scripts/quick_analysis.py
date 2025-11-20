"""
Quick Analysis Script

Fast analysis of a single domain for rapid iteration.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer
from src.data import DataLoader


def quick_analyze(domain_name: str, data_path: str = None):
    """
    Quick analysis with minimal output.
    
    Parameters
    ----------
    domain_name : str
        Domain name
    data_path : str, optional
        Data file path (auto-discovers if None)
    """
    print(f"Analyzing {domain_name}...", end=" ", flush=True)
    
    try:
        # Load data
        loader = DataLoader()
        
        if data_path:
            data = loader.load(Path(data_path))
        else:
            data = loader.load_domain(domain_name)
        
        # Analyze
        analyzer = DomainSpecificAnalyzer(domain_name)
        results = analyzer.analyze_complete(
            texts=data['texts'],
            outcomes=data['outcomes']
        )
        
        # Print results
        print(f"✓")
        print(f"  R²: {results['r_squared']:.1%}")
        print(f"  Д: {results['delta']:.3f}")
        print(f"  π: {results['narrativity']:.3f}")
        print(f"  n: {len(data['texts'])}")
        
        if results['passes_threshold']:
            print(f"  Status: ✓ PASSES threshold")
        else:
            print(f"  Status: ⚠ Below threshold")
        
        return results
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def batch_quick_analyze(domains: List[str]):
    """Quick analyze multiple domains."""
    print(f"\nQuick analyzing {len(domains)} domains:\n")
    
    results = {}
    
    for domain in domains:
        result = quick_analyze(domain)
        results[domain] = result
        print()
    
    # Summary
    success = sum(1 for r in results.values() if r is not None)
    print(f"{'='*80}")
    print(f"Quick Analysis Complete: {success}/{len(domains)} successful")
    print(f"{'='*80}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick domain analysis')
    parser.add_argument('domain', help='Domain name')
    parser.add_argument('--file', help='Data file path')
    parser.add_argument('--batch', nargs='+', help='Analyze multiple domains')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_quick_analyze(args.batch)
    else:
        quick_analyze(args.domain, args.file)

