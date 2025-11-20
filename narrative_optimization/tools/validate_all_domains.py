"""
Validate All Domains

Runs validation on all registered domains to ensure quality.

Author: Narrative Integration System
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import get_domain_registry
from src.data import DataLoader
from src.analysis.domain_specific_analyzer import DomainSpecificAnalyzer


def validate_domain(domain_name: str) -> dict:
    """Validate a single domain."""
    try:
        # Load data
        loader = DataLoader()
        data = loader.load_domain(domain_name)
        
        # Run analysis
        analyzer = DomainSpecificAnalyzer(domain_name)
        results = analyzer.analyze_complete(
            texts=data['texts'],
            outcomes=data['outcomes']
        )
        
        return {
            'status': 'validated',
            'r_squared': results['r_squared'],
            'delta': results['delta'],
            'n_samples': len(data['texts'])
        }
        
    except FileNotFoundError:
        return {'status': 'no_data', 'error': 'Data file not found'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def validate_all():
    """Validate all registered domains."""
    print("="*80)
    print("VALIDATING ALL REGISTERED DOMAINS")
    print("="*80)
    
    registry = get_domain_registry()
    domains = registry.get_all_domains()
    
    print(f"\nValidating {len(domains)} domains...\n")
    
    results = {}
    validated = 0
    errors = 0
    
    for domain in domains:
        print(f"{domain.name:20s} ", end="", flush=True)
        
        result = validate_domain(domain.name)
        results[domain.name] = result
        
        if result['status'] == 'validated':
            print(f"✓ R²={result['r_squared']:.1%}, Д={result['delta']:.3f}")
            validated += 1
            
            # Update registry
            domain.r_squared = result['r_squared']
            domain.delta = result['delta']
            domain.n_samples = result['n_samples']
        else:
            print(f"✗ {result.get('error', 'Unknown error')}")
            errors += 1
    
    # Save updated registry
    registry.save()
    
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Validated: {validated}")
    print(f"  Errors: {errors}")
    print(f"  Total: {len(domains)}")
    
    return results


if __name__ == '__main__':
    validate_all()

