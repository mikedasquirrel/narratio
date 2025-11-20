"""Minimal test - just verify transformers work with 10 records"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("MINIMAL TEST - 10 RECORDS ONLY")
print("="*80)
print()

import json
import time
import traceback

def debug_print(msg, flush=True):
    """Print with timestamp for debugging"""
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}", flush=flush)

try:
    debug_print("STEP 1: Importing modules...")
    from src.pipelines.pipeline_composer import PipelineComposer
    from src.pipelines.domain_config import DomainConfig
    debug_print("✓ Imports successful")
    
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'domains' / 'golf' / 'config.yaml'
    data_path = project_root.parent / 'data' / 'domains' / 'golf_enhanced_narratives.json'
    
    debug_print(f"\nSTEP 2: Loading config from {config_path}...")
    config = DomainConfig.from_yaml(config_path)
    debug_print(f"✓ Config loaded: п={config.pi:.3f}, type={config.type.value}")
    
    debug_print(f"\nSTEP 3: Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    debug_print(f"✓ Loaded {len(all_data)} total records")
    
    # Take just 10 records
    test_data = all_data[:10]
    debug_print(f"✓ Using {len(test_data)} records for test")
    
    # Create temp file
    import tempfile
    import os
    debug_print("\nSTEP 4: Creating temporary data file...")
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(test_data, temp_file)
    temp_file.close()
    debug_print(f"✓ Temp file created: {temp_file.name}")
    
    debug_print("\nSTEP 5: Initializing PipelineComposer...")
    composer = PipelineComposer(project_root)
    debug_print("✓ Composer ready")
    
    debug_print("\nSTEP 6: Configuring pipeline...")
    config.sample_size = 10
    debug_print(f"✓ Sample size set to {config.sample_size}")
    debug_print(f"✓ Target feature count: 100")
    debug_print(f"✓ Cache: disabled")
    
    debug_print("\n" + "="*80)
    debug_print("STEP 7: RUNNING PIPELINE (this is where it might hang)")
    debug_print("="*80)
    start = time.time()
    
    results = composer.run_pipeline(
        config,
        data_path=Path(temp_file.name),
        target_feature_count=100,  # Smaller target
        use_cache=False  # No cache for test
    )
    
    elapsed = time.time() - start
    
    debug_print("\n" + "="*80)
    debug_print("STEP 8: PIPELINE COMPLETED!")
    debug_print("="*80)
    debug_print(f"✓ Completed in {elapsed:.1f}s")
    debug_print(f"✓ Features shape: {results.get('features', {}).get('shape', 'unknown')}")
    debug_print(f"✓ Analysis keys: {list(results.get('analysis', {}).keys())[:5]}")
    
    print("\n" + "="*80)
    print("SUCCESS! Pipeline works. Now you can scale up.")
    print("="*80)
    
except KeyboardInterrupt:
    elapsed = time.time() - start if 'start' in locals() else 0
    debug_print(f"\n✗ INTERRUPTED after {elapsed:.1f}s")
    debug_print("Pipeline was interrupted by user")
    raise
except Exception as e:
    elapsed = time.time() - start if 'start' in locals() else 0
    debug_print(f"\n✗ ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
    debug_print("\nFull traceback:")
    traceback.print_exc()
    raise
finally:
    if 'temp_file' in locals():
        try:
            os.unlink(temp_file.name)
            debug_print(f"✓ Cleaned up temp file")
        except:
            pass

