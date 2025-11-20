"""
Register All Validated Domains

Registers all completed domain validations in the CrossDomainValidator.
This enables the cross-domain learning gate and tracks validation history.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.cross_domain_validator import get_cross_domain_validator
from datetime import datetime

def main():
    """Register all validated domains"""
    
    print("="*80)
    print("REGISTERING ALL VALIDATED DOMAINS")
    print("="*80)
    
    validator = get_cross_domain_validator()
    
    # Clear any existing (for clean slate)
    print("\nClearing existing registry for fresh registration...")
    validator.clear_all(confirm=True)
    
    # Register all validated domains
    today = datetime.now().strftime('%Y-%m-%d')
    
    domains_to_register = [
        {
            'domain_name': 'Coin Flips',
            'narrativity': 0.12,
            'efficiency': 0.04,
            'passes': False,
            'transformer_count': 2,
            'sample_size': 100,
            'date_validated': today
        },
        {
            'domain_name': 'Math Problems',
            'narrativity': 0.15,
            'efficiency': 0.05,
            'passes': False,
            'transformer_count': 2,
            'sample_size': 100,
            'date_validated': today
        },
        {
            'domain_name': 'Hurricanes',
            'narrativity': 0.30,
            'efficiency': 0.12,  # Estimated from п*r*κ
            'passes': False,
            'transformer_count': 5,
            'sample_size': 236,
            'date_validated': today
        },
        {
            'domain_name': 'NCAA Basketball',
            'narrativity': 0.44,
            'efficiency': -0.11,
            'passes': False,
            'transformer_count': 3,
            'sample_size': 500,
            'date_validated': today
        },
        {
            'domain_name': 'NBA',
            'narrativity': 0.49,
            'efficiency': -0.03,
            'passes': False,
            'transformer_count': 12,
            'sample_size': 11979,
            'date_validated': today
        },
        {
            'domain_name': 'Mental Health',
            'narrativity': 0.55,
            'efficiency': 0.12,  # Estimated from п*r*κ
            'passes': False,
            'transformer_count': 4,
            'sample_size': 200,
            'date_validated': today
        },
        {
            'domain_name': 'Movies (IMDB)',
            'narrativity': 0.65,
            'efficiency': 0.04,
            'passes': False,
            'transformer_count': 8,
            'sample_size': 6047,
            'date_validated': today
        },
        {
            'domain_name': 'Startups',
            'narrativity': 0.76,
            'efficiency': 0.29,
            'passes': False,
            'transformer_count': 8,
            'sample_size': 269,
            'date_validated': today
        },
        {
            'domain_name': 'Character',
            'narrativity': 0.85,
            'efficiency': 0.73,
            'passes': True,  # ✓ PASSES!
            'transformer_count': 7,
            'sample_size': 200,
            'date_validated': today
        },
        {
            'domain_name': 'Self-Rated',
            'narrativity': 0.95,
            'efficiency': 0.59,
            'passes': True,  # ✓ PASSES!
            'transformer_count': 6,
            'sample_size': 50,
            'date_validated': today
        }
    ]
    
    print(f"\nRegistering {len(domains_to_register)} validated domains...")
    print("")
    
    for domain in domains_to_register:
        validator.register_validation(**domain)
        status = "✓ PASS" if domain['passes'] else "❌ FAIL"
        print(f"  {status} {domain['domain_name']:25s} - п={domain['narrativity']:.2f}, eff={domain['efficiency']:.2f}")
    
    print("\n" + "="*80)
    print("REGISTRATION COMPLETE")
    print("="*80)
    
    # Print validation report
    validator.print_validation_report()
    
    # Check if ready for cross-domain learning
    print("\n" + "="*80)
    print("CROSS-DOMAIN LEARNING READINESS")
    print("="*80)
    
    if validator.can_learn_cross_domain(min_domains=3, min_passing=2):
        print("\n✅ READY for cross-domain pattern learning!")
        print(f"  • {len(domains_to_register)} domains validated")
        print(f"  • {len(validator.get_passing_domains())} domains pass threshold")
        print("\n  Can now analyze:")
        print("    - Which features transfer across domains")
        print("    - What predicts п (narrativity)")
        print("    - Universal vs domain-specific patterns")
    else:
        print("\n⚠️  NOT READY for cross-domain learning yet")
        print("  Need more domain validations")
    
    print("\n✓ All domains registered in validation system!")
    
    return validator


if __name__ == '__main__':
    main()

