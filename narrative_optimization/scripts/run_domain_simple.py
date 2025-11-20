"""
Simple Domain Runner - Process one domain with clear progress

This version processes the full dataset but with better progress tracking.
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.pipeline_composer import PipelineComposer
from src.pipelines.domain_config import DomainConfig


def print_status(msg, level="info"):
    """Print status with timestamp"""
    ts = datetime.now().strftime("%H:%M:%S")
    icons = {"info": "▶", "success": "✓", "error": "✗", "warn": "⚠"}
    icon = icons.get(level, "•")
    print(f"[{ts}] {icon} {msg}", flush=True)


def run_domain_simple(domain_name: str, max_samples: int = 1000, force: bool = False):
    """Run pipeline with progress tracking"""
    
    project_root = Path(__file__).parent.parent
    domain_dir = project_root / 'domains' / domain_name
    config_path = domain_dir / 'config.yaml'
    results_path = domain_dir / f'{domain_name}_results.json'
    
    print("\n" + "="*80)
    print(f"  DOMAIN: {domain_name.upper()}")
    print("="*80 + "\n")
    
    if results_path.exists() and not force:
        print_status("Results exist. Use --force to regenerate.", "warn")
        return None
    
    # Load config
    print_status(f"Loading config...", "info")
    try:
        config = DomainConfig.from_yaml(config_path)
        print_status(f"Config loaded: п={config.pi:.3f}", "success")
    except Exception as e:
        print_status(f"Config error: {e}", "error")
        return None
    
    # Find data
    print_status("Finding data file...", "info")
    data_dirs = [
        project_root.parent / 'data' / 'domains',
        project_root / 'data' / 'domains'
    ]
    
    data_path = None
    for data_dir in data_dirs:
        for pattern in [f'{domain_name}_enhanced_narratives.json',
                        f'{domain_name}_complete_dataset.json',
                        f'{domain_name}_with_narratives.json']:
            test_path = data_dir / pattern
            if test_path.exists():
                data_path = test_path
                break
        if data_path:
            break
    
    if not data_path:
        print_status("No data file found", "error")
        return None
    
    size_mb = data_path.stat().st_size / (1024 * 1024)
    print_status(f"Data file: {data_path.name} ({size_mb:.1f} MB)", "success")
    
    # Initialize composer
    print_status("Initializing pipeline composer...", "info")
    composer = PipelineComposer(project_root)
    print_status("Composer ready", "success")
    
    # Run pipeline
    print_status(f"Starting pipeline (max {max_samples} samples)...", "info")
    print()
    
    start_time = time.time()
    
    try:
        # Temporarily set sample size
        original_sample_size = config.sample_size
        config.sample_size = max_samples
        
        results = composer.run_pipeline(
            config,
            data_path=data_path,
            target_feature_count=300,
            use_cache=True
        )
        
        elapsed = time.time() - start_time
        print()
        print_status(f"Pipeline completed in {elapsed:.1f}s", "success")
        
        # Save results
        print_status("Saving results...", "info")
        formatted = {
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
            'pipeline_info': results.get('pipeline_info', {}),
            'features': results.get('features', {})
        }
        
        with open(results_path, 'w') as f:
            json.dump(formatted, f, indent=2, default=str)
        
        size_kb = results_path.stat().st_size / 1024
        print_status(f"Results saved ({size_kb:.1f} KB)", "success")
        
        # Summary
        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)
        analysis = results.get('analysis', {})
        print(f"  Narrativity (п):  {config.pi:.3f}")
        print(f"  R-narrative (r):  {analysis.get('r_narrative', 0):.3f}")
        print(f"  Features:          {results.get('features', {}).get('n_features', 0)}")
        print(f"  Time:             {elapsed:.1f}s")
        print("-"*80 + "\n")
        
        return formatted
        
    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print_status(f"Pipeline failed after {elapsed:.1f}s: {e}", "error")
        import traceback
        traceback.print_exc()
        return None
    finally:
        config.sample_size = original_sample_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', help='Domain name')
    parser.add_argument('--max-samples', type=int, default=1000, help='Max samples to process')
    parser.add_argument('--force', action='store_true', help='Force rerun')
    
    args = parser.parse_args()
    
    result = run_domain_simple(args.domain, args.max_samples, args.force)
    sys.exit(0 if result else 1)

