"""
Cross-Domain Temporal Validation

Systematically tests whether "better stories win over time" holds across multiple domains.

Tests on:
- Sports (NBA, potentially hurricanes)
- Products (Crypto)
- Profiles (Mental health stigma)
- Relationships (Marriage)

This is the critical test: Does temporal pattern replicate or is it domain-specific?
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.temporal_validator import TemporalValidator
from src.evaluation.better_stories_validator import BetterStoriesValidator


class CrossDomainTemporalValidation:
    """
    Runs systematic temporal validation across all available domains.
    
    Reports honest findings: which domains validate, which refute, which are inconclusive.
    """
    
    def __init__(self):
        self.temporal_validator = TemporalValidator()
        self.better_stories_validator = BetterStoriesValidator()
        self.results = {}
    
    def run_all_domains(self) -> Dict[str, Any]:
        """
        Run validation on all available domains.
        
        Returns
        -------
        Dict with complete cross-domain results
        """
        # Note: In actual implementation, this would load real data
        # Here we document the structure for when real data is available
        
        domains_to_test = {
            'nba_games': {
                'description': 'NBA game predictions over seasons',
                'status': 'data_available',
                'temporal_structure': 'games_to_season'
            },
            'crypto_markets': {
                'description': 'Cryptocurrency performance over time',
                'status': 'data_available',
                'temporal_structure': 'daily_to_yearly'
            },
            'mental_health_outcomes': {
                'description': 'Disorder name effects on treatment seeking',
                'status': 'cross_sectional',  # No temporal data
                'temporal_structure': 'none'
            },
            'marriage_stability': {
                'description': 'Couple compatibility over years',
                'status': 'limited_data',
                'temporal_structure': 'initial_to_longterm'
            }
        }
        
        results = {
            'domains_tested': len([d for d in domains_to_test.values() if d['status'] == 'data_available']),
            'domains_with_temporal_data': len([d for d in domains_to_test.values() if d['temporal_structure'] != 'none']),
            'domain_details': domains_to_test,
            'validation_summary': self._generate_validation_summary(domains_to_test)
        }
        
        self.results = results
        return results
    
    def _generate_validation_summary(self, domains: Dict) -> Dict[str, Any]:
        """Generate summary of what can/cannot be tested."""
        
        can_test_temporal = []
        cannot_test_temporal = []
        
        for domain_name, details in domains.items():
            if details['temporal_structure'] != 'none' and details['status'] == 'data_available':
                can_test_temporal.append(domain_name)
            else:
                cannot_test_temporal.append(domain_name)
        
        return {
            'can_test': can_test_temporal,
            'cannot_test': cannot_test_temporal,
            'coverage': len(can_test_temporal) / len(domains) if domains else 0,
            'recommendation': self._generate_recommendation(len(can_test_temporal))
        }
    
    def _generate_recommendation(self, n_testable: int) -> str:
        if n_testable >= 4:
            return "Sufficient domains for cross-domain validation. Proceed with testing."
        elif n_testable >= 2:
            return "Limited domains. Results will be suggestive but not definitive."
        else:
            return "Insufficient temporal data. Need to collect more longitudinal datasets."
    
    def generate_report(self) -> str:
        """Generate cross-domain validation report."""
        report = []
        report.append("=" * 80)
        report.append("CROSS-DOMAIN TEMPORAL VALIDATION")
        report.append("Testing: Does 'better stories win over time' replicate across domains?")
        report.append("=" * 80)
        report.append("")
        
        if not self.results:
            report.append("No validation run yet. Call run_all_domains() first.")
            return "\n".join(report)
        
        summary = self.results['validation_summary']
        
        report.append("DOMAIN COVERAGE:")
        report.append(f"  Total domains: {self.results['domains_tested']}")
        report.append(f"  With temporal data: {self.results['domains_with_temporal_data']}")
        report.append(f"  Can test: {len(summary['can_test'])}")
        report.append(f"  Cannot test: {len(summary['cannot_test'])}")
        report.append(f"  Coverage: {summary['coverage']:.1%}")
        report.append("")
        
        report.append("TESTABLE DOMAINS:")
        for domain in summary['can_test']:
            details = self.results['domain_details'][domain]
            report.append(f"  ✓ {domain}: {details['description']}")
            report.append(f"    Temporal structure: {details['temporal_structure']}")
        report.append("")
        
        report.append("NON-TESTABLE DOMAINS:")
        for domain in summary['cannot_test']:
            details = self.results['domain_details'][domain]
            report.append(f"  ✗ {domain}: {details['description']}")
            report.append(f"    Reason: {details['status']}, {details['temporal_structure']}")
        report.append("")
        
        report.append("RECOMMENDATION:")
        report.append(f"  {summary['recommendation']}")
        report.append("")
        
        report.append("NEXT STEPS:")
        report.append("  1. Load actual temporal data for testable domains")
        report.append("  2. Run TemporalValidator.validate_domain() for each")
        report.append("  3. Check if accuracy increases with time horizon")
        report.append("  4. Compute cross_domain_validation() to test replication")
        report.append("  5. Report null findings if patterns don't replicate")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Run cross-domain validation."""
    print("Cross-Domain Temporal Validation")
    print("=" * 80)
    
    validator = CrossDomainTemporalValidation()
    results = validator.run_all_domains()
    
    print(validator.generate_report())
    
    print("\nHONEST ASSESSMENT:")
    print("-" * 80)
    print("Current status: Framework for validation exists.")
    print("Actual validation: Requires running on real temporal data.")
    print("Evidence so far: NBA shows suggestive temporal patterns.")
    print("Verdict: INCOMPLETE - Need systematic cross-domain testing.")
    print("")
    print("What's needed:")
    print("  - Collect longitudinal data for 3+ additional domains")
    print("  - Run pre-registered temporal analyses")
    print("  - Report all findings (including null results)")
    print("  - Only then can we claim cross-domain validation")


if __name__ == "__main__":
    main()

