"""
Run Domain Pipeline with Clear Progress Tracking

Provides detailed, real-time progress updates during pipeline execution.
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.pipeline_composer import PipelineComposer
from src.pipelines.domain_config import DomainConfig


def print_progress(stage: str, message: str, level: str = "info"):
    """Print formatted progress message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbols = {
        'info': '▶',
        'success': '✓',
        'warning': '⚠',
        'error': '✗',
        'step': '➤'
    }
    symbol = symbols.get(level, '•')
    print(f"[{timestamp}] {symbol} [{stage}] {message}")


def run_domain_pipeline(domain_name: str, force: bool = False):
    """
    Run pipeline for a single domain with progress tracking.
    
    Parameters
    ----------
    domain_name : str
        Name of domain to process
    force : bool
        Force rerun even if results exist
    """
    project_root = Path(__file__).parent.parent
    domain_dir = project_root / 'domains' / domain_name
    config_path = domain_dir / 'config.yaml'
    results_path = domain_dir / f'{domain_name}_results.json'
    
    print("\n" + "=" * 100)
    print(f"  PROCESSING DOMAIN: {domain_name.upper()}")
    print("=" * 100 + "\n")
    
    # Check if results exist
    if results_path.exists() and not force:
        print_progress("CHECK", f"Results already exist: {results_path}", "warning")
        print_progress("CHECK", "Use --force to regenerate", "info")
        return None
    
    # Load config
    print_progress("CONFIG", f"Loading configuration from {config_path.name}", "step")
    try:
        config = DomainConfig.from_yaml(config_path)
        print_progress("CONFIG", f"Loaded: п={config.pi:.3f}, type={config.type.value}", "success")
    except Exception as e:
        print_progress("CONFIG", f"Failed to load config: {e}", "error")
        return None
    
    # Find data file
    print_progress("DATA", "Searching for data files...", "step")
    data_dirs = [
        project_root.parent / 'data' / 'domains',
        project_root / 'data' / 'domains'
    ]
    
    data_patterns = [
        f'{domain_name}_complete_dataset.json',
        f'{domain_name}_enhanced_narratives.json',
        f'{domain_name}_with_narratives.json',
        f'{domain_name}_dataset.json',
        f'{domain_name}_data.json',
    ]
    
    data_path = None
    for data_dir in data_dirs:
        for pattern in data_patterns:
            test_path = data_dir / pattern
            if test_path.exists():
                data_path = test_path
                break
        if data_path:
            break
    
    if data_path is None:
        print_progress("DATA", "No data file found", "error")
        return None
    
    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    print_progress("DATA", f"Found: {data_path.name} ({file_size_mb:.1f} MB)", "success")
    
    # Initialize composer
    print_progress("SETUP", "Initializing pipeline composer...", "step")
    composer = PipelineComposer(project_root)
    print_progress("SETUP", "Composer ready", "success")
    
    # Run pipeline
    print_progress("PIPELINE", "Starting unified pipeline execution...", "step")
    print()
    
    start_time = time.time()
    
    try:
        results = composer.run_pipeline(
            config,
            data_path=data_path,
            target_feature_count=300,
            use_cache=True
        )
        
        elapsed = time.time() - start_time
        print()
        print_progress("PIPELINE", f"Completed in {elapsed:.1f}s", "success")
        
        # Save results
        print_progress("SAVE", f"Saving results to {results_path.name}...", "step")
        
        # Format results
        formatted_results = {
            'domain': domain_name,
            'pi': config.pi,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'type': config.type.value,
                'perspectives': config.perspectives,
                'quality_methods': config.quality_methods,
                'scales': config.scales
            },
            'analysis': results.get('analysis', {}),
            'comprehensive_ю': results.get('comprehensive_ю', {}),
            'pipeline_info': {
                'transformers': results.get('pipeline_info', {}).get('transformers', []),
                'total_features': results.get('features', {}).get('n_features', 0)
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(formatted_results, f, indent=2, default=str)
        
        print_progress("SAVE", f"Results saved ({results_path.stat().st_size / 1024:.1f} KB)", "success")
        
        # Print summary
        print("\n" + "-" * 100)
        print("  SUMMARY")
        print("-" * 100)
        analysis = results.get('analysis', {})
        print(f"  Narrativity (п):      {config.pi:.3f}")
        print(f"  R-narrative (r):      {analysis.get('r_narrative', 0):.3f}")
        print(f"  Narrative Agency (Д): {analysis.get('Д', 0):.3f}")
        print(f"  Efficiency (Д/п):     {analysis.get('efficiency', 0):.3f}")
        print(f"  Features extracted:   {results.get('features', {}).get('shape', [0, 0])[1]}")
        
        comp_ю = results.get('comprehensive_ю', {})
        if comp_ю:
            print(f"  Perspectives:         {len(comp_ю.get('ю_perspectives', {}))}")
            print(f"  Methods:              {len(comp_ю.get('ю_methods', {}))}")
            print(f"  Scales:               {len(comp_ю.get('ю_scales', {}))}")
        
        print("-" * 100 + "\n")
        
        return results
        
    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print_progress("PIPELINE", f"Failed after {elapsed:.1f}s: {e}", "error")
        import traceback
        print("\n" + "=" * 100)
        print("ERROR DETAILS:")
        print("=" * 100)
        traceback.print_exc()
        print("=" * 100 + "\n")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run domain pipeline with progress tracking')
    parser.add_argument('domain', help='Domain name (e.g., golf, tennis, nba)')
    parser.add_argument('--force', action='store_true', help='Force rerun even if results exist')
    
    args = parser.parse_args()
    
    result = run_domain_pipeline(args.domain, args.force)
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)

