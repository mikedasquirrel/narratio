"""
Universal Domain Processor

One processor for ALL domains.
Automatically handles any registered domain.

Usage:
------
# Process ANY domain
processor.process_domain('movies', sample_size=2000)
processor.process_domain('nba', sample_size=1000)
processor.process_domain('new_domain', sample_size=500)

# Process batch of domains
processor.process_batch(['movies', 'nba', 'nfl'])

# Process all available domains
processor.process_all_domains(max_per_domain=2000)

Author: Narrative Optimization Framework
Date: November 2025
"""

# FIX TENSORFLOW MUTEX DEADLOCK ON MACOS
# Must be set BEFORE any TensorFlow/transformers imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops
os.environ['OMP_NUM_THREADS'] = '1'  # Single thread for OpenMP
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism

import argparse
import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Optional, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = Path(__file__).parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.unsupervised_narrative_discovery import UnsupervisedNarrativeDiscovery
from domain_registry import DOMAINS, load_domain_safe, list_available_domains
from transformers.transformer_selector import TransformerSelector
# LAZY IMPORT - Don't import FeatureExtractionPipeline here to avoid transformer imports
# It will be imported only when use_transformers=True and process_domain is called
# from pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

# NEW: Cross-domain learning components (lazy import to avoid dependency issues)
_meta_learner = None
_cross_domain_patterns = None
_universal_pattern_cache = None


class UniversalDomainProcessor:
    """
    Process ANY domain with same pipeline.
    
    Handles:
    - Loading (domain-specific extraction)
    - Discovery (unsupervised patterns)
    - Validation (safeguards)
    - Saving (standardized format)
    - Logging (track everything)
    """
    
    def __init__(
        self,
        results_dir='narrative_optimization/results/domains',
        use_transformers=True,
        fast_mode=False,
        enable_cross_domain=True
    ):
        """
        Initialize universal processor.
        
        Parameters
        ----------
        results_dir : str
            Directory for saving results
        use_transformers : bool
            Apply transformers for feature extraction (default: True)
            If False, uses raw embeddings only
        fast_mode : bool
            Use fast subset of transformers (default: False)
        enable_cross_domain : bool
            Enable cross-domain learning and pattern transfer (default: True)
            When True, domains learn from each other's patterns
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.discoverer = UnsupervisedNarrativeDiscovery()
        self.transformer_selector = TransformerSelector()
        
        self.use_transformers = use_transformers
        self.fast_mode = fast_mode
        self.enable_cross_domain = enable_cross_domain
        
        # Track all processed
        self.processed = {}
        
        # Cross-domain learning (lazy initialization)
        self.meta_learner = None
        self.cross_domain_analyzer = None
        self.universal_patterns = {}
        
        # Master log
        self.log_file = self.results_dir / 'processing_log.md'
        self.cross_domain_log = self.results_dir / 'cross_domain_insights.json'
    
    def process_domain(
        self,
        domain_name: str,
        sample_size: Optional[int] = None,
        min_cluster_size: Optional[int] = None,
        save_results: bool = True,
        domain_type: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process single domain with unsupervised discovery.
        
        Parameters
        ----------
        domain_name : str
            Domain name from registry
        sample_size : int, optional
            Number of narratives to analyze (None = all)
        min_cluster_size : int, optional
            Minimum cluster size (None = auto)
        save_results : bool
            Save results to disk
        domain_type : str, optional
            Domain category (sports, entertainment, etc.)
            If None, auto-detected from domain_name
        progress_callback : callable, optional
            Callback for progress updates: callback(step, percent, message)
            
        Returns
        -------
        results : dict
            Complete analysis results
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING DOMAIN: {domain_name.upper()}")
        print(f"{'='*80}\n")
        
        # Helper for progress updates
        def update_progress(step: str, percent: float, message: str):
            if progress_callback:
                progress_callback(step, percent, message)
        
        update_progress('loading', 0.05, f'Initializing processing for {domain_name}')
        
        # Load domain
        narratives, outcomes, config = load_domain_safe(domain_name)
        
        update_progress('loading', 0.10, f'Loaded {len(narratives) if narratives else 0} narratives')
        
        if narratives is None:
            return {'error': 'Failed to load domain'}
        
        # Determine sample size
        if sample_size is None:
            sample_size = len(narratives)
        else:
            sample_size = min(sample_size, len(narratives))
        
        # Auto min_cluster_size if not specified
        if min_cluster_size is None:
            min_cluster_size = max(30, sample_size // 40)
        adaptive_cap = max(2, sample_size // 2) if sample_size > 10 else max(2, sample_size - 1)
        min_cluster_size = min(min_cluster_size, max(5, adaptive_cap))
        min_cluster_size = max(2, min_cluster_size)
        
        print(f"Sample size: {sample_size:,}")
        print(f"Min cluster size: {min_cluster_size}")
        print(f"Expected patterns: {10 + 15 * config.estimated_pi:.0f} (based on π={config.estimated_pi:.2f})")
        print(f"Mode: {'Transformer-based' if self.use_transformers else 'Raw embeddings'}\n")
        
        update_progress('loading', 0.15, f'Configuration complete: {sample_size} samples')
        
        # PHASE 1: Feature Extraction (NEW!)
        features_for_discovery = None
        extraction_report = None
        
        if self.use_transformers:
            print(f"\n{'='*80}")
            print("PHASE 1: TRANSFORMER-BASED FEATURE EXTRACTION")
            print(f"{'='*80}\n")
            
            update_progress('feature_extraction', 0.20, 'Selecting transformers for domain')
            
            # Select appropriate transformers
            transformer_names = self.transformer_selector.select_transformers(
                domain_name=domain_name,
                pi_value=config.estimated_pi,
                domain_type=domain_type,
                data_sample=narratives[:100],
                include_renovation=True,
                include_expensive=not self.fast_mode
            )
            
            if self.fast_mode:
                transformer_names = self.transformer_selector.get_fast_subset(
                    transformer_names, max_count=20
                )
            
            print(f"Selected {len(transformer_names)} transformers for π={config.estimated_pi:.2f}")
            print(f"Fast mode: {self.fast_mode}\n")
            
            # Extract features
            try:
                from pipelines.feature_extraction_pipeline import FeatureExtractionPipeline
                
                pipeline = FeatureExtractionPipeline(
                    transformer_names=transformer_names,
                    domain_name=domain_name,
                    enable_caching=True,
                    verbose=True
                )
                
                features_for_discovery = pipeline.fit_transform(
                    narratives[:sample_size]
                )
                
                extraction_report = pipeline.get_extraction_report()
                
                print(f"\n✓ Feature extraction complete: {features_for_discovery.shape[1]:,} features")
                
            except Exception as e:
                print(f"\n✗ Feature extraction failed: {e}")
                print("Falling back to raw embeddings...\n")
                features_for_discovery = None
        
        # PHASE 2: Pattern Discovery
        print(f"\n{'='*80}")
        print("PHASE 2: UNSUPERVISED PATTERN DISCOVERY")
        print(f"{'='*80}\n")
        
        if features_for_discovery is not None:
            # Use rich features
            print("Using rich transformer features for pattern discovery")
            result = self.discoverer.discover_patterns(
                    narratives=narratives[:sample_size],
                    outcomes=outcomes[:sample_size],
                    min_cluster_size=min_cluster_size,
                    n_latent_dimensions=50,
                    features=features_for_discovery
            )
        else:
            # Fall back to raw embeddings
            print("Using raw embeddings for pattern discovery")
            result = self.discoverer.discover_patterns(
                narratives=narratives[:sample_size],
                outcomes=outcomes[:sample_size],
                min_cluster_size=min_cluster_size,
                n_latent_dimensions=50
            )
        
        # Add domain info
        result['domain_info'] = {
            'name': domain_name,
            'estimated_pi': config.estimated_pi,
            'sample_size': sample_size,
            'total_available': len(narratives),
            'outcome_type': config.outcome_type,
            'used_transformers': self.use_transformers,
            'fast_mode': self.fast_mode
        }
        
        # Add extraction info if transformers were used
        if extraction_report:
            result['extraction_info'] = extraction_report
        
        # Validate
        print(f"\n[Validating] Results...")
        validation = self._validate_result(result, config.estimated_pi)
        result['validation'] = validation
        
        if validation['safe_to_proceed']:
            print(f"✓ Validation PASSED\n")
        else:
            print(f"⚠ Validation FAILED")
            print(f"  Issues: {validation['issues']}\n")
        
        # Save if requested
        if save_results:
            self._save_result(domain_name, sample_size, result)
        
        # Log
        self._log_processing(domain_name, sample_size, result)
        
        # Track
        self.processed[domain_name] = result
        
        print(f"✓ {domain_name} complete: {result['metadata']['n_clusters']} patterns, {len([c for c in result['outcome_correlations'] if c.get('significant')])} significant\n")
        
        return result
    
    def process_batch(
        self,
        domain_names: List[str],
        sample_size_per_domain: int = 1000
    ) -> Dict[str, Dict]:
        """
        Process multiple domains in sequence.
        
        Parameters
        ----------
        domain_names : list of str
            Domains to process
        sample_size_per_domain : int
            Sample size for each domain
            
        Returns
        -------
        results : dict
            {domain_name: result_dict}
        """
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING: {len(domain_names)} DOMAINS")
        print(f"{'='*80}\n")
        print(f"Domains: {', '.join(domain_names)}")
        print(f"Sample size per domain: {sample_size_per_domain:,}\n")
        
        results = {}
        
        for i, domain_name in enumerate(domain_names):
            print(f"\n[{i+1}/{len(domain_names)}] Processing {domain_name}...")
            
            try:
                result = self.process_domain(
                    domain_name,
                    sample_size=sample_size_per_domain,
                    save_results=True
                )
                results[domain_name] = result
            
            except Exception as e:
                print(f"✗ Error processing {domain_name}: {e}")
                continue
        
        # Batch summary
        self._print_batch_summary(results)
        
        return results
    
    def process_all_available(
        self,
        max_per_domain: int = 2000,
        pi_min: float = 0.0,
        pi_max: float = 1.0
    ) -> Dict[str, Dict]:
        """
        Process all available domains in registry.
        
        Parameters
        ----------
        max_per_domain : int
            Maximum samples per domain
        pi_min, pi_max : float
            Only process domains in this π range
            
        Returns
        -------
        results : dict
            Results for each processed domain
        """
        print(f"\n{'='*80}")
        print("PROCESS ALL AVAILABLE DOMAINS")
        print(f"{'='*80}\n")
        
        # Get available domains
        available = []
        for name, config in DOMAINS.items():
            if config.data_path.exists() and pi_min <= config.estimated_pi <= pi_max:
                available.append(name)
        
        available = sorted(available, key=lambda n: DOMAINS[n].estimated_pi)
        
        print(f"Available domains in π range [{pi_min:.2f}, {pi_max:.2f}]: {len(available)}")
        print(f"Max per domain: {max_per_domain:,}")
        print(f"\nProcessing order (by π):\n")
        
        for name in available:
            pi = DOMAINS[name].estimated_pi
            print(f"  {name:<15s} π={pi:.2f}")
        
        print()
        
        # Process all
        return self.process_batch(available, sample_size_per_domain=max_per_domain)
    
    def _validate_result(self, result, estimated_pi):
        """Validate discovery result."""
        validation = {
            'safe_to_proceed': True,
            'issues': [],
            'warnings': []
        }
        
        n_patterns = result['metadata']['n_clusters']
        correlations = result['outcome_correlations']
        cluster_corrs = [c for c in correlations if c.get('correlation_type') == 'cluster_membership']
        significant = [
            c for c in cluster_corrs
            if c.get('significant_fdr', c.get('significant', False))
        ]
        raw_sig = sum(1 for c in cluster_corrs if c.get('significant'))
        result.setdefault('validation_stats', {})['cluster_significant_raw'] = raw_sig
        result['validation_stats']['cluster_significant_fdr'] = len(significant)
        effect_sizes = [abs(c.get('effect_size', 0.0)) for c in significant]
        median_effect = float(np.median(effect_sizes)) if effect_sizes else 0.0
        result['validation_stats']['cluster_median_effect'] = median_effect
        
        # Check pattern count
        if n_patterns < 5:
            validation['issues'].append(f"Too few patterns ({n_patterns})")
            validation['safe_to_proceed'] = False
        elif n_patterns > 40:
            validation['issues'].append(f"Too many patterns ({n_patterns})")
            validation['safe_to_proceed'] = False
        
        # Check significance
        sig_rate = len(significant) / max(len(cluster_corrs), 1)
        if sig_rate < 0.15:
            validation['warnings'].append(f"Low significance rate ({sig_rate*100:.0f}%)")
        elif sig_rate > 0.95:
            if median_effect < 0.05:
                validation['issues'].append(f"Suspiciously high significance ({sig_rate*100:.0f}%)")
                validation['safe_to_proceed'] = False
            else:
                validation['warnings'].append(
                    f"High significance rate ({sig_rate*100:.0f}%) with strong effects (median {median_effect:.2f})"
                )
        
        # Check π prediction
        predicted = 10 + 15 * estimated_pi
        error = abs(n_patterns - predicted)
        if error > 10:
            validation['warnings'].append(f"Pattern count differs from π prediction (exp {predicted:.0f}, got {n_patterns})")
        
        return validation
    
    def _save_result(self, domain_name, sample_size, result):
        """Save result with standard naming."""
        domain_dir = self.results_dir / domain_name
        domain_dir.mkdir(exist_ok=True)
        
        filename = domain_dir / f'n{sample_size}_analysis.json'
        
        # Serialize (remove numpy)
        serializable = {
            'domain': domain_name,
            'sample_size': sample_size,
            'timestamp': datetime.now().isoformat(),
            'domain_info': result.get('domain_info', {}),
            'n_patterns': result['metadata']['n_clusters'],
            'patterns': [{k: v for k, v in p.items() if not isinstance(v, np.ndarray)} 
                        for p in result['patterns']],
            'significant_correlations': [c for c in result['outcome_correlations'] if c.get('significant')],
            'all_correlations': result['outcome_correlations'],
            'validation': result.get('validation', {})
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2, cls=NumpyEncoder)
        
        print(f"  ✓ Saved: {filename}")
    
    def _log_processing(self, domain_name, sample_size, result):
        """Log to master log."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M')} - {domain_name} (n={sample_size:,})\n\n")
            f.write(f"- Patterns: {result['metadata']['n_clusters']}\n")
            f.write(f"- Significant: {len([c for c in result['outcome_correlations'] if c.get('significant')])}\n")
            f.write(f"- Validation: {result['validation']['safe_to_proceed']}\n")
            f.write(f"\n")
    
    def _print_batch_summary(self, results: Dict[str, Dict]):
        """Print summary of batch processing."""
        print(f"\n{'='*80}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*80}\n")
        
        for domain, result in results.items():
            if 'error' in result:
                print(f"✗ {domain:<15s}: {result['error']}")
            else:
                n_pat = result['metadata']['n_clusters']
                n_sig = len([c for c in result['outcome_correlations'] if c.get('significant')])
                pi = result['domain_info']['estimated_pi']
                
                print(f"✓ {domain:<15s}: {n_pat} patterns, {n_sig} significant (π={pi:.2f})")
        
        print(f"\n{'='*80}\n")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ==============================================================================
# EASY ADDITION EXAMPLE
# ==============================================================================

def example_add_new_domain():
    """
    Example: How to add a new domain in 5 lines.
    """
    from domain_registry import add_new_domain
    
    # Add chess domain (example)
    add_new_domain(
        name='chess',
        data_path='data/domains/chess_games.json',
        narrative_field='game_narrative',
        outcome_field='won',
        estimated_pi=0.40,
        description='Chess games with move narratives',
        outcome_type='binary'
    )
    
    # Now process it like any other domain
    processor = UniversalDomainProcessor()
    processor.process_domain('chess', sample_size=1000)
    
    # That's it! New domain integrated.


def _build_cli_parser() -> argparse.ArgumentParser:
    """Define CLI for the universal processor."""
    parser = argparse.ArgumentParser(
        description="Universal Domain Processor CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--domain',
        help='Single domain to process (e.g., hurricanes, nba)',
        default=None,
    )
    parser.add_argument(
        '--batch',
        nargs='+',
        help='Process a batch of domains sequentially',
    )
    parser.add_argument(
        '--process_all',
        action='store_true',
        help='Process every registered domain with available data',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        help='Sample size per domain (defaults to full dataset when omitted)',
    )
    parser.add_argument(
        '--min_cluster_size',
        type=int,
        help='Override automatic min cluster size',
    )
    parser.add_argument(
        '--results_dir',
        default='narrative_optimization/results/domains',
        help='Directory for saving per-domain analysis outputs',
    )
    parser.add_argument(
        '--domain_type',
        help='Override domain type hint (sports, entertainment, etc.)',
    )
    parser.add_argument(
        '--fast_mode',
        action='store_true',
        help='Use reduced transformer subset for faster runs',
    )
    parser.add_argument(
        '--use_transformers',
        dest='use_transformers',
        action='store_true',
        help='Force-enable transformer feature extraction',
    )
    parser.add_argument(
        '--no_use_transformers',
        dest='use_transformers',
        action='store_false',
        help='Force-disable transformer feature extraction',
    )
    parser.set_defaults(use_transformers=None)
    parser.add_argument(
        '--save_results',
        dest='save_results',
        action='store_true',
        help='Persist run artifacts to the results directory',
    )
    parser.add_argument(
        '--no_save_results',
        dest='save_results',
        action='store_false',
        help='Skip writing result JSON files',
    )
    parser.set_defaults(save_results=None)
    parser.add_argument(
        '--list_domains',
        action='store_true',
        help='Print the registry summary before executing',
    )
    parser.add_argument(
        '--pi_min',
        type=float,
        default=0.0,
        help='Minimum π threshold when using --process_all',
    )
    parser.add_argument(
        '--pi_max',
        type=float,
        default=1.0,
        help='Maximum π threshold when using --process_all',
    )
    return parser


def _should_print_registry(args) -> bool:
    """Determine whether to show the domain registry overview."""
    if args.list_domains:
        return True
    if not any([args.domain, args.batch, args.process_all]):
        # No action requested; mimic legacy behavior (just show registry)
        return True
    return False


if __name__ == '__main__':
    parser = _build_cli_parser()
    cli_args = parser.parse_args()

    if _should_print_registry(cli_args):
        list_available_domains()
        if not any([cli_args.domain, cli_args.batch, cli_args.process_all]):
            sys.exit(0)

    resolved_use_transformers = (
        True if cli_args.use_transformers is None else cli_args.use_transformers
    )
    resolved_save_results = (
        True if cli_args.save_results is None else cli_args.save_results
    )

    processor = UniversalDomainProcessor(
        results_dir=cli_args.results_dir,
        use_transformers=resolved_use_transformers,
        fast_mode=cli_args.fast_mode,
    )

    if cli_args.domain:
        processor.process_domain(
            cli_args.domain,
            sample_size=cli_args.sample_size,
            min_cluster_size=cli_args.min_cluster_size,
            save_results=resolved_save_results,
            domain_type=cli_args.domain_type,
        )
    elif cli_args.batch:
        processor.process_batch(
            cli_args.batch,
            sample_size_per_domain=cli_args.sample_size or 1000,
        )
    elif cli_args.process_all:
        processor.process_all_available(
            max_per_domain=cli_args.sample_size or 2000,
            pi_min=cli_args.pi_min,
            pi_max=cli_args.pi_max,
        )

