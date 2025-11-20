"""
Run Domain Pipeline in Chunks with Clear Progress

Processes data in manageable chunks with progress tracking and checkpointing.
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import time
import numpy as np

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
    print(f"[{timestamp}] {symbol} [{stage}] {message}", flush=True)


def run_domain_chunked(domain_name: str, chunk_size: int = 500, force: bool = False):
    """
    Run pipeline for a domain in chunks.
    
    Parameters
    ----------
    domain_name : str
        Name of domain to process
    chunk_size : int
        Number of records per chunk
    force : bool
        Force rerun even if results exist
    """
    project_root = Path(__file__).parent.parent
    domain_dir = project_root / 'domains' / domain_name
    config_path = domain_dir / 'config.yaml'
    results_path = domain_dir / f'{domain_name}_results.json'
    
    print("\n" + "=" * 100)
    print(f"  PROCESSING DOMAIN: {domain_name.upper()} (CHUNKED MODE)")
    print("=" * 100 + "\n")
    
    # Check if results exist
    if results_path.exists() and not force:
        print_progress("CHECK", f"Results already exist: {results_path}", "warning")
        print_progress("CHECK", "Use --force to regenerate", "info")
        return None
    
    # Load config
    print_progress("CONFIG", f"Loading configuration...", "step")
    try:
        config = DomainConfig.from_yaml(config_path)
        print_progress("CONFIG", f"Loaded: п={config.pi:.3f}, type={config.type.value}", "success")
    except Exception as e:
        print_progress("CONFIG", f"Failed: {e}", "error")
        return None
    
    # Find and load data
    print_progress("DATA", "Searching for data files...", "step")
    data_dirs = [
        project_root.parent / 'data' / 'domains',
        project_root / 'data' / 'domains'
    ]
    
    data_patterns = [
        f'{domain_name}_enhanced_narratives.json',
        f'{domain_name}_complete_dataset.json',
        f'{domain_name}_with_narratives.json',
        f'{domain_name}_dataset.json',
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
    
    # Load data
    print_progress("DATA", "Loading JSON data...", "step")
    start_load = time.time()
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    load_time = time.time() - start_load
    print_progress("DATA", f"Loaded {len(all_data)} records in {load_time:.1f}s", "success")
    
    # Sample if too large
    max_samples = config.sample_size if config.sample_size else 2000
    if len(all_data) > max_samples:
        import random
        random.seed(42)
        all_data = random.sample(all_data, max_samples)
        print_progress("DATA", f"Sampled to {len(all_data)} records", "warning")
    
    # Initialize composer
    print_progress("SETUP", "Initializing pipeline composer...", "step")
    composer = PipelineComposer(project_root)
    print_progress("SETUP", "Composer ready", "success")
    
    # Process in chunks
    n_chunks = (len(all_data) + chunk_size - 1) // chunk_size
    print_progress("PIPELINE", f"Processing {len(all_data)} records in {n_chunks} chunks of {chunk_size}", "info")
    print()
    
    all_features = []
    all_outcomes = []
    all_names = []
    all_texts = []
    
    total_start = time.time()
    
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, len(all_data))
        chunk_data = all_data[chunk_start:chunk_end]
        
        print_progress("CHUNK", f"Processing chunk {chunk_idx + 1}/{n_chunks} ({len(chunk_data)} records)...", "step")
        chunk_time_start = time.time()
        
        try:
            # Extract texts, outcomes, names from chunk
            texts = []
            outcomes = []
            names = []
            
            if config.domain == 'golf':
                for record in chunk_data:
                    narrative = ''
                    if 'narrative' in record and record['narrative']:
                        narrative = str(record['narrative'])
                    elif 'enhanced_narrative' in record and record['enhanced_narrative']:
                        narrative = str(record['enhanced_narrative'])
                    texts.append(narrative if narrative else 'placeholder')
                    
                    outcome = 0
                    if 'won_tournament' in record:
                        outcome = int(record['won_tournament'])
                    outcomes.append(outcome)
                    
                    if 'player_name' in record and 'tournament_name' in record:
                        names.append(f"{record['player_name']}_{record['tournament_name']}")
                    else:
                        names.append(f"golf_{len(names)}")
            else:
                # Generic extraction
                for record in chunk_data:
                    text_parts = []
                    for field in config.data.text_fields:
                        if field in record and record[field]:
                            text_parts.append(str(record[field]))
                    texts.append(' '.join(text_parts) if text_parts else 'placeholder')
                    
                    if config.data.outcome_field in record:
                        outcomes.append(record[config.data.outcome_field])
                    else:
                        outcomes.append(0)
                    
                    if config.data.name_field and config.data.name_field in record:
                        names.append(str(record[config.data.name_field]))
                    else:
                        names.append(f"entity_{len(names)}")
            
            outcomes = np.array(outcomes)
            if config.outcome_type.value == 'binary':
                if outcomes.dtype != bool and outcomes.dtype != int:
                    unique_values = np.unique(outcomes)
                    if len(unique_values) == 2:
                        outcomes = (outcomes == unique_values[1]).astype(int)
                    else:
                        median = np.median(outcomes)
                        outcomes = (outcomes > median).astype(int)
            
            # Run pipeline on chunk
            # Create temporary config with smaller sample
            chunk_config = config
            results = composer.run_pipeline(
                chunk_config,
                data_path=None,  # We'll pass data directly
                target_feature_count=300,
                use_cache=True
            )
            
            # Extract features from results
            if 'features' in results and 'data' in results['features']:
                features = results['features']['data']
                all_features.append(features)
                all_outcomes.extend(outcomes)
                all_names.extend(names)
                all_texts.extend(texts)
                
                chunk_time = time.time() - chunk_time_start
                print_progress("CHUNK", f"Chunk {chunk_idx + 1} complete: {features.shape} features in {chunk_time:.1f}s", "success")
            else:
                print_progress("CHUNK", f"Chunk {chunk_idx + 1} failed: no features in results", "error")
                
        except Exception as e:
            chunk_time = time.time() - chunk_time_start
            print_progress("CHUNK", f"Chunk {chunk_idx + 1} failed after {chunk_time:.1f}s: {e}", "error")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_features:
        print_progress("PIPELINE", "No features extracted from any chunks", "error")
        return None
    
    # Combine all chunks
    print_progress("COMBINE", f"Combining {len(all_features)} chunks...", "step")
    combined_features = np.vstack(all_features)
    combined_outcomes = np.array(all_outcomes)
    
    print_progress("COMBINE", f"Final shape: {combined_features.shape}", "success")
    
    # Run analysis on combined data
    print_progress("ANALYSIS", "Running analysis on combined features...", "step")
    analyzer = composer._create_analyzer(config)
    
    analysis_results = analyzer.analyze_complete(
        texts=all_texts,
        outcomes=combined_outcomes,
        names=all_names,
        genome=combined_features,
        feature_names=[f"feature_{i}" for i in range(combined_features.shape[1])],
        masses=None,
        context_features=None
    )
    
    # Save results
    print_progress("SAVE", f"Saving results...", "step")
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
        'analysis': analysis_results.get('analysis', {}),
        'comprehensive_ю': analysis_results.get('comprehensive_ю', {}),
        'pipeline_info': {
            'transformers': results.get('pipeline_info', {}).get('transformers', []),
            'total_features': combined_features.shape[1]
        },
        'features': {
            'shape': combined_features.shape,
            'n_samples': len(all_texts)
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(formatted_results, f, indent=2, default=str)
    
    total_time = time.time() - total_start
    file_size_kb = results_path.stat().st_size / 1024
    
    print_progress("SAVE", f"Results saved ({file_size_kb:.1f} KB)", "success")
    
    # Print summary
    print("\n" + "-" * 100)
    print("  SUMMARY")
    print("-" * 100)
    print(f"  Domain:              {domain_name}")
    print(f"  Narrativity (п):      {config.pi:.3f}")
    print(f"  Samples processed:   {len(all_texts)}")
    print(f"  Features extracted:   {combined_features.shape[1]}")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"  Avg time/chunk:       {total_time/n_chunks:.1f}s")
    print("-" * 100 + "\n")
    
    return formatted_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run domain pipeline in chunks')
    parser.add_argument('domain', help='Domain name')
    parser.add_argument('--chunk-size', type=int, default=500, help='Records per chunk')
    parser.add_argument('--force', action='store_true', help='Force rerun')
    
    args = parser.parse_args()
    
    result = run_domain_chunked(args.domain, args.chunk_size, args.force)
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)

