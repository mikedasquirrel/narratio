"""
Comprehensive Reanalysis Script

Master script for reanalyzing all domains with:
- Full November 2025 renovation transformer suite
- Rich feature extraction (800-2000 features per domain)
- Unsupervised pattern discovery
- Universal law testing
- 7-force model validation

Usage:
------
# Run single domain
python run_comprehensive_reanalysis.py --domain nba --sample 2000

# Run batch
python run_comprehensive_reanalysis.py --batch sports_priority

# Run all domains
python run_comprehensive_reanalysis.py --all --max-per-domain 1000

Author: Narrative Optimization Framework
Date: November 2025
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_domain_processor import UniversalDomainProcessor
from domain_registry import DOMAINS, list_available_domains


# Domain batches as defined in validation report
BATCH_DEFINITIONS = {
    'sports_priority': {
        'name': 'High-Priority Sports (Betting)',
        'domains': ['nba', 'nfl', 'tennis'],
        'sample_size': 2000,
        'rationale': 'Sports with betting odds data - highest value'
    },
    'sports_other': {
        'name': 'Other Sports',
        'domains': ['golf', 'ufc', 'mlb', 'boxing', 'poker'],
        'sample_size': 1500,
        'rationale': 'Complete sports coverage'
    },
    'entertainment': {
        'name': 'Entertainment',
        'domains': ['movies', 'oscars', 'music', 'wwe'],  # Novels needs consolidation
        'sample_size': 1500,
        'rationale': 'Narrative art forms'
    },
    'nominative': {
        'name': 'Nominative Showcase',
        'domains': ['housing', 'aviation', 'ships', 'meta_nominative'],
        'sample_size': 1000,
        'rationale': 'Pure nominative effects demonstration'
    },
    'specialized': {
        'name': 'Specialized Domains',
        'domains': ['startups', 'crypto', 'hurricanes', 'mental_health', 
                   'dinosaurs', 'mythology', 'bible', 'conspiracy_theories'],
        'sample_size': 1000,
        'rationale': 'Diverse domain validation'
    },
    'benchmarks': {
        'name': 'Benchmarks & Social',
        'domains': ['coin_flips', 'math_problems', 'humor', 
                   'immigration', 'marriage'],
        'sample_size': 1000,
        'rationale': 'Control domains and social dynamics'
    }
}


class ComprehensiveReanalysis:
    """
    Orchestrates comprehensive reanalysis across all domains.
    """
    
    def __init__(
        self,
        results_dir='narrative_optimization/results/comprehensive_reanalysis',
        fast_mode=False
    ):
        """Initialize reanalysis system."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor with transformers enabled
        self.processor = UniversalDomainProcessor(
            results_dir=str(self.results_dir / 'domains'),
            use_transformers=True,
            fast_mode=fast_mode
        )
        
        self.fast_mode = fast_mode
        
        # Track results
        self.batch_results = {}
        self.all_results = {}
        
        # Master log
        self.master_log = self.results_dir / 'MASTER_LOG.md'
        self._init_master_log()
    
    def _init_master_log(self):
        """Initialize master log file."""
        with open(self.master_log, 'w') as f:
            f.write(f"# Comprehensive Reanalysis Master Log\n\n")
            f.write(f"**Started**: {datetime.now().isoformat()}\n")
            f.write(f"**Framework**: November 2025 Renovation\n")
            f.write(f"**Transformers**: Full suite with renovation features\n")
            f.write(f"**Mode**: {'Fast' if self.fast_mode else 'Full'}\n\n")
            f.write(f"---\n\n")
    
    def _log(self, message: str):
        """Log to master log."""
        with open(self.master_log, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    def run_single_domain(
        self,
        domain_name: str,
        sample_size: int = 1000
    ) -> Dict:
        """
        Run comprehensive reanalysis on single domain.
        
        Parameters
        ----------
        domain_name : str
            Domain to analyze
        sample_size : int
            Number of narratives to process
        
        Returns
        -------
        result : dict
            Complete analysis result
        """
        self._log(f"\n## Processing Domain: {domain_name}")
        self._log(f"Sample size: {sample_size:,}")
        self._log(f"Timestamp: {datetime.now().isoformat()}\n")
        
        # Auto-detect domain type
        domain_type = self._detect_domain_type(domain_name)
        
        # Process domain
        result = self.processor.process_domain(
            domain_name=domain_name,
            sample_size=sample_size,
            save_results=True,
            domain_type=domain_type
        )
        
        # Log results
        if 'error' not in result:
            self._log(f"✓ Complete: {result['metadata']['n_clusters']} patterns discovered")
            self._log(f"  Significant: {len([c for c in result['outcome_correlations'] if c.get('significant')])}")
            if 'extraction_info' in result:
                self._log(f"  Features: {result['extraction_info']['successful']} transformers")
        else:
            self._log(f"✗ Failed: {result['error']}")
        
        self.all_results[domain_name] = result
        return result
    
    def run_batch(
        self,
        batch_name: str,
        override_sample_size: int = None
    ) -> Dict[str, Dict]:
        """
        Run batch of domains.
        
        Parameters
        ----------
        batch_name : str
            Batch identifier from BATCH_DEFINITIONS
        override_sample_size : int, optional
            Override default batch sample size
        
        Returns
        -------
        results : dict
            {domain_name: result}
        """
        if batch_name not in BATCH_DEFINITIONS:
            available = ', '.join(BATCH_DEFINITIONS.keys())
            raise ValueError(f"Unknown batch: {batch_name}. Available: {available}")
        
        batch_def = BATCH_DEFINITIONS[batch_name]
        sample_size = override_sample_size or batch_def['sample_size']
        
        self._log(f"\n{'='*80}")
        self._log(f"BATCH: {batch_def['name']}")
        self._log(f"{'='*80}")
        self._log(f"Domains: {len(batch_def['domains'])}")
        self._log(f"Sample size per domain: {sample_size:,}")
        self._log(f"Rationale: {batch_def['rationale']}")
        self._log(f"Started: {datetime.now().isoformat()}\n")
        
        batch_results = {}
        
        for i, domain in enumerate(batch_def['domains'], 1):
            self._log(f"\n[{i}/{len(batch_def['domains'])}] {domain}")
            self._log("-" * 80)
            
            try:
                result = self.run_single_domain(domain, sample_size)
                batch_results[domain] = result
            except Exception as e:
                self._log(f"✗ Error: {e}")
                batch_results[domain] = {'error': str(e)}
                continue
        
        # Batch summary
        self._print_batch_summary(batch_name, batch_results)
        self.batch_results[batch_name] = batch_results
        
        # Save batch results
        self._save_batch_results(batch_name, batch_results)
        
        return batch_results
    
    def run_all_batches(
        self,
        exclude_batches: List[str] = None
    ):
        """
        Run all defined batches in sequence.
        
        Parameters
        ----------
        exclude_batches : list of str, optional
            Batches to skip
        """
        exclude_batches = exclude_batches or []
        
        self._log(f"\n{'='*80}")
        self._log("COMPREHENSIVE REANALYSIS: ALL BATCHES")
        self._log(f"{'='*80}")
        self._log(f"Total batches: {len(BATCH_DEFINITIONS)}")
        self._log(f"Excluded: {exclude_batches if exclude_batches else 'None'}")
        self._log(f"Started: {datetime.now().isoformat()}\n")
        
        for batch_name in BATCH_DEFINITIONS.keys():
            if batch_name in exclude_batches:
                self._log(f"\n⊘ Skipping batch: {batch_name}")
                continue
            
            try:
                self.run_batch(batch_name)
            except Exception as e:
                self._log(f"\n✗ Batch {batch_name} failed: {e}")
                continue
        
        # Final summary
        self._print_final_summary()
    
    def _detect_domain_type(self, domain_name: str) -> str:
        """Auto-detect domain type from name."""
        sports = ['nba', 'nfl', 'tennis', 'golf', 'ufc', 'mlb', 'boxing', 'poker', 'wwe']
        entertainment = ['movies', 'imdb', 'oscars', 'music', 'novels']
        business = ['startups', 'crypto']
        nominative = ['housing', 'aviation', 'ships', 'meta_nominative']
        
        if domain_name.lower() in sports:
            return 'sports'
        elif domain_name.lower() in entertainment:
            return 'entertainment'
        elif domain_name.lower() in business:
            return 'business'
        elif domain_name.lower() in nominative:
            return 'nominative'
        else:
            return 'specialized'
    
    def _print_batch_summary(self, batch_name: str, results: Dict):
        """Print batch summary."""
        successful = len([r for r in results.values() if 'error' not in r])
        failed = len(results) - successful
        
        self._log(f"\n{'='*80}")
        self._log(f"BATCH SUMMARY: {BATCH_DEFINITIONS[batch_name]['name']}")
        self._log(f"{'='*80}")
        self._log(f"Total: {len(results)}")
        self._log(f"✓ Successful: {successful}")
        self._log(f"✗ Failed: {failed}\n")
        
        for domain, result in results.items():
            if 'error' in result:
                self._log(f"  ✗ {domain}: {result['error']}")
            else:
                n_patterns = result['metadata']['n_clusters']
                n_sig = len([c for c in result['outcome_correlations'] if c.get('significant')])
                pi = result['domain_info']['estimated_pi']
                self._log(f"  ✓ {domain}: {n_patterns} patterns, {n_sig} significant (π={pi:.2f})")
        
        self._log(f"{'='*80}\n")
    
    def _print_final_summary(self):
        """Print final summary of all batches."""
        total_domains = sum(len(r) for r in self.batch_results.values())
        total_successful = sum(
            len([d for d in r.values() if 'error' not in d])
            for r in self.batch_results.values()
        )
        
        self._log(f"\n{'='*80}")
        self._log("COMPREHENSIVE REANALYSIS COMPLETE")
        self._log(f"{'='*80}")
        self._log(f"Batches processed: {len(self.batch_results)}")
        self._log(f"Total domains: {total_domains}")
        self._log(f"✓ Successful: {total_successful}")
        self._log(f"✗ Failed: {total_domains - total_successful}")
        self._log(f"Completed: {datetime.now().isoformat()}")
        self._log(f"{'='*80}\n")
        
        # Print detailed summary
        self._log("\nDetailed Results by Batch:\n")
        for batch_name, results in self.batch_results.items():
            successful = len([r for r in results.values() if 'error' not in r])
            self._log(f"  {batch_name}: {successful}/{len(results)} successful")
    
    def _save_batch_results(self, batch_name: str, results: Dict):
        """Save batch results to JSON."""
        output_file = self.results_dir / f'{batch_name}_results.json'
        
        # Make serializable
        serializable = {}
        for domain, result in results.items():
            if 'error' in result:
                serializable[domain] = result
            else:
                serializable[domain] = {
                    'domain': domain,
                    'n_patterns': result['metadata']['n_clusters'],
                    'significant_patterns': len([c for c in result['outcome_correlations'] if c.get('significant')]),
                    'pi': result['domain_info']['estimated_pi'],
                    'sample_size': result['domain_info']['sample_size'],
                    'used_transformers': result['domain_info'].get('used_transformers', True),
                    'fast_mode': result['domain_info'].get('fast_mode', False),
                    'timestamp': datetime.now().isoformat()
                }
                
                if 'extraction_info' in result:
                    serializable[domain]['extraction_info'] = {
                        'successful_transformers': result['extraction_info']['successful'],
                        'failed_transformers': result['extraction_info']['failed'],
                        'total_features': len(result['extraction_info'].get('feature_provenance', {}))
                    }
        
        with open(output_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        self._log(f"✓ Saved batch results: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive reanalysis with full transformer suite'
    )
    parser.add_argument(
        '--domain',
        type=str,
        help='Single domain to process'
    )
    parser.add_argument(
        '--batch',
        type=str,
        choices=list(BATCH_DEFINITIONS.keys()),
        help='Batch to process'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all batches'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size (overrides batch default)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use fast mode (subset of transformers)'
    )
    parser.add_argument(
        '--list-batches',
        action='store_true',
        help='List available batches'
    )
    
    args = parser.parse_args()
    
    # List batches
    if args.list_batches:
        print("\nAvailable Batches:\n")
        for name, definition in BATCH_DEFINITIONS.items():
            print(f"  {name}:")
            print(f"    Name: {definition['name']}")
            print(f"    Domains: {', '.join(definition['domains'])}")
            print(f"    Default sample size: {definition['sample_size']:,}")
            print(f"    Rationale: {definition['rationale']}\n")
        return
    
    # Create reanalysis system
    reanalysis = ComprehensiveReanalysis(fast_mode=args.fast)
    
    # Process based on arguments
    if args.domain:
        sample_size = args.sample or 1000
        reanalysis.run_single_domain(args.domain, sample_size)
    
    elif args.batch:
        reanalysis.run_batch(args.batch, args.sample)
    
    elif args.all:
        reanalysis.run_all_batches()
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python run_comprehensive_reanalysis.py --domain nba --sample 2000")
        print("  python run_comprehensive_reanalysis.py --batch sports_priority")
        print("  python run_comprehensive_reanalysis.py --all --fast")
        print("  python run_comprehensive_reanalysis.py --list-batches")


if __name__ == '__main__':
    main()

